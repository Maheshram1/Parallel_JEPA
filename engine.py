"""
Training and evaluation loop functions.
"""

import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging

from config import Config 
from model import VisionTransformer, VisionTransformer1 
from utils import is_main_process 

def train(student_model: torch.nn.Module, teacher_model: torch.nn.Module,
          dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
          scheduler, device: torch.device, epoch: int, config, # config should be Config type
          scaler: GradScaler, rank: int, world_size: int) -> tuple[float, float]:
    """
    Runs one epoch of training for the student model using the teacher model.

    Args:
        student_model (torch.nn.Module): The student model (potentially DDP wrapped).
        teacher_model (torch.nn.Module): The teacher model (potentially DDP wrapped, frozen).
        dataloader (DataLoader): DataLoader for the training set.
        criterion (nn.Module): Loss function (e.g., MSELoss).
        optimizer (torch.optim.Optimizer): Optimizer for the student model parameters.
        scheduler: Learning rate scheduler (stepped per iteration).
        device (torch.device): The device to train on (GPU for the current process).
        epoch (int): The current epoch number (0-based).
        config: Configuration object (e.g., instance of Config class) holding hyperparameters like use_amp, num_epochs.
        scaler (GradScaler, optional): Gradient scaler for mixed-precision training. Required if config.use_amp is True.
        rank (int): Rank of the current process (global rank).
        world_size (int): Total number of processes.

    Returns:
        tuple[float, float]: Average training loss across all batches for the epoch,
                             final learning rate at the end of the epoch.
    """
    logger = logging.getLogger(__name__)
    student_model.train() # Set student model to training mode
    teacher_model.eval()  # Teacher model is always in evaluation mode

    total_loss_accumulated = 0.0
    batch_count = 0
    data_load_start_time = time.time() # Track data loading time separately

    # Set epoch for DistributedSampler to ensure proper shuffling across epochs
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
        logger.debug(f"Rank {rank}: Set DistributedSampler epoch to {epoch}")

    # Progress bar only on the main process
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]",
                        leave=False, disable=not is_main_process(rank), mininterval=1.0, dynamic_ncols=True)

    for batch_idx, (imgs, _) in enumerate(progress_bar):
        data_load_time = time.time() - data_load_start_time
        forward_backward_start_time = time.time() # Time the computation part

        imgs = imgs.to(device, non_blocking=True) # Move data to device
        local_batch_size = imgs.shape[0]
        if local_batch_size <= 0:
             logger.warning(f"Rank {rank}, Epoch {epoch+1}, Batch {batch_idx}: Skipping batch with local_batch_size <= 0.")
             data_load_start_time = time.time() # Reset data load timer
             continue
        global_batch_size = local_batch_size * world_size

        optimizer.zero_grad(set_to_none=True) # Reset gradients

        # Mixed Precision Context
        autocast_enabled = config.use_amp and scaler is not None
        with autocast(enabled=autocast_enabled):
            # Student forward pass
            _, student_output = student_model(imgs) # student_output shape: (B * num_parts, N, D)

            # Teacher forward pass (no gradients needed)
            with torch.no_grad():
                teacher_intermediate, _ = teacher_model(imgs) # teacher_intermediate shape: (B, N, D)

                # --- Prepare Teacher Target ---
                num_parts = student_output.shape[0] // local_batch_size
                expected_parts = getattr(config, 'num_parts', 4)
                if num_parts != expected_parts and local_batch_size > 0 : # Avoid warning if batch size was 0
                     logger.warning(f"Rank {rank}, Epoch {epoch+1}, Batch {batch_idx}: Calculated num_parts ({num_parts}) != expected ({expected_parts}). Student out: {student_output.shape}, Local BS: {local_batch_size}")
                     # Use calculated num_parts, assumes student model structure dictates this

                if teacher_intermediate.shape[0] != local_batch_size:
                     logger.error(f"Rank {rank}, Epoch {epoch+1}, Batch {batch_idx}: Mismatch teacher batch size ({teacher_intermediate.shape[0]}) vs input ({local_batch_size}). Skipping.")
                     data_load_start_time = time.time() # Reset timer
                     continue

                teacher_target = teacher_intermediate.repeat_interleave(num_parts, dim=0)

            # Ensure shapes match before loss calculation
            if student_output.shape != teacher_target.shape:
                 logger.error(f"Rank {rank}, Epoch {epoch+1}, Batch {batch_idx}: Shape mismatch! Student: {student_output.shape}, Teacher Target: {teacher_target.shape}. Skipping.")
                 data_load_start_time = time.time() # Reset timer
                 continue

            loss = criterion(student_output, teacher_target)

        # Check for NaN/inf loss before backward
        if not torch.isfinite(loss):
             logger.error(f"Rank {rank}, Epoch {epoch+1}, Batch {batch_idx}: Non-finite loss ({loss.item()}) detected BEFORE backward. Skipping step.")
             if scaler: scaler.update()
             # scheduler.step() # Optional: step scheduler even on bad batch
             data_load_start_time = time.time() # Reset timer
             continue

        # Backward pass and optimizer step
        if autocast_enabled:
            scaler.scale(loss).backward()
            # Optional: Gradient Clipping (applied *before* scaler.step)
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            scaler.step(optimizer) # Checks for inf/NaN grads scaled by scaler
            scaler.update()
        else:
            loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        step_loss = loss.item()
        total_loss_accumulated += step_loss
        batch_count += 1
        current_lr = scheduler.get_last_lr()[0]
        forward_backward_time = time.time() - forward_backward_start_time

        # Update progress bar postfix on main process
        if is_main_process(rank):
            postfix_dict = {
                'loss': f'{step_loss:.4f}',
                'lr': f'{current_lr:.3e}', # Compact LR format
                'gb': global_batch_size,
                # 'dl': f'{data_load_time:.2f}s', # Optional: data load time
                'iter': f'{forward_backward_time:.2f}s' # Iteration time (fw/bw/optim)
            }
            progress_bar.set_postfix(postfix_dict)

        data_load_start_time = time.time() # Reset data load timer for next batch

    avg_loss_epoch = total_loss_accumulated / batch_count if batch_count > 0 else 0.0
    avg_loss_tensor = torch.tensor(avg_loss_epoch, device=device)

    if world_size > 1:
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)

    final_avg_loss = avg_loss_tensor.item()
    final_lr = scheduler.get_last_lr()[0]

    if is_main_process(rank):
        logger.info(f"Epoch {epoch+1} [Train] Completed. Avg Loss: {final_avg_loss:.4f}, Final LR: {final_lr:.6e}")

    return final_avg_loss, final_lr


def evaluate(student_model: torch.nn.Module, teacher_model: torch.nn.Module,
             dataloader: DataLoader, criterion: nn.Module, device: torch.device,
             epoch: int, config, rank: int, world_size: int) -> float: # config should be Config type
    """
    Runs one epoch of evaluation on the validation set.

    Args:
        student_model (torch.nn.Module): The student model (potentially DDP wrapped).
        teacher_model (torch.nn.Module): The teacher model (potentially DDP wrapped, frozen).
        dataloader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function (e.g., MSELoss) to calculate validation loss.
        device (torch.device): The device to evaluate on (GPU for the current process).
        epoch (int): The current epoch number (0-based), used for logging and sampler.
        config: Configuration object (e.g., instance of Config class).
        rank (int): Rank of the current process (global rank).
        world_size (int): Total number of processes.

    Returns:
        float: Average validation loss across all batches for the epoch.
    """
    logger = logging.getLogger(__name__)
    student_model.eval() # Set student model to evaluation mode
    teacher_model.eval() # Teacher model is always in evaluation mode

    total_loss_accumulated = 0.0
    batch_count = 0

    # Set epoch for DistributedSampler if used
    if isinstance(dataloader.sampler, DistributedSampler):
       dataloader.sampler.set_epoch(epoch)

    # Progress bar only on the main process
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False,
                         disable=not is_main_process(rank), mininterval=1.0, dynamic_ncols=True)

    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, (imgs, _) in enumerate(progress_bar):
            imgs = imgs.to(device, non_blocking=True) # Move data to device
            local_batch_size = imgs.shape[0]
            if local_batch_size <= 0:
                 logger.warning(f"Rank {rank}, Epoch {epoch+1}, Val Batch {batch_idx}: Skipping batch with local_batch_size <= 0.")
                 continue
            global_batch_size = local_batch_size * world_size

            # Mixed Precision Context for forward pass
            autocast_enabled = config.use_amp
            with autocast(enabled=autocast_enabled):
                # Student forward pass
                _, student_output = student_model(imgs)
                # Teacher forward pass
                teacher_intermediate, _ = teacher_model(imgs)

                num_parts = student_output.shape[0] // local_batch_size
                expected_parts = getattr(config, 'num_parts', 4)
                if num_parts != expected_parts and local_batch_size > 0:
                     logger.warning(f"Rank {rank}, Epoch {epoch+1}, Val Batch {batch_idx}: Val num_parts ({num_parts}) != expected ({expected_parts}). Student out: {student_output.shape}, Local BS: {local_batch_size}")

                if teacher_intermediate.shape[0] != local_batch_size:
                     logger.error(f"Rank {rank}, Epoch {epoch+1}, Val Batch {batch_idx}: Mismatch val teacher batch size ({teacher_intermediate.shape[0]}) vs input ({local_batch_size}). Skipping.")
                     continue

                teacher_target = teacher_intermediate.repeat_interleave(num_parts, dim=0)

                # Ensure shapes match before loss calculation
                if student_output.shape != teacher_target.shape:
                    logger.error(f"Rank {rank}, Epoch {epoch+1}, Val Batch {batch_idx}: Val shape mismatch! Student: {student_output.shape}, Teacher Target: {teacher_target.shape}. Skipping.")
                    continue

                loss = criterion(student_output, teacher_target)

            # Check for NaN/inf loss
            if not torch.isfinite(loss):
                 logger.error(f"Rank {rank}, Epoch {epoch+1}, Val Batch {batch_idx}: Non-finite loss ({loss.item()}) detected during validation. Skipping batch accumulation.")
                 continue # Skip accumulating loss for this batch

            # --- Accumulate Loss ---
            step_loss = loss.item()
            total_loss_accumulated += step_loss
            batch_count += 1

            # Update progress bar postfix on main process
            if is_main_process(rank):
                progress_bar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'gb': global_batch_size
                })

    # --- Epoch End: Calculate and Synchronize Average Loss ---
    avg_loss_epoch = total_loss_accumulated / batch_count if batch_count > 0 else 0.0
    avg_loss_tensor = torch.tensor(avg_loss_epoch, device=device)

    if world_size > 1:
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)

    final_avg_loss = avg_loss_tensor.item()

    if is_main_process(rank):
        logger.info(f"Epoch {epoch+1} [Val] Completed. Avg Loss: {final_avg_loss:.4f}")

    return final_avg_loss
