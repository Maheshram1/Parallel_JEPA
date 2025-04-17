# -*- coding: utf-8 -*-
"""
Main training script using Distributed Data Parallel (DDP).

Sets up environment, models, data loaders, optimizer, and runs the training loop.
Handles checkpointing and logging. Uses torchrun for launching.
"""

import torch
import torch.nn as nn
# import torch.optim as optim # Optimizer is imported from optimizer.py
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time
import math
# from tqdm import tqdm # tqdm is used within engine.py
import logging

# --- Imports for Distributed Training ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# --- End Imports ---

# --- Imports for Mixed Precision ---
from torch.cuda.amp import GradScaler
# --- End Mixed Precision ---

# --- Project specific imports ---
from config import Config
from model import VisionTransformer, VisionTransformer1
from optimizer import SOAP # Import custom optimizer
from utils import (setup, cleanup, is_main_process, save_checkpoint,
                   load_checkpoint, log_to_file, load_pretrained_teacher_weights, refresh_teacher)
from engine import train, evaluate # Import train/eval functions
# --- End Project Imports ---


# --- Set up logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)
logger = logging.getLogger(__name__) # Get logger for this module (__main__)

# Adjust logging level for non-main processes AFTER basicConfig
if 'RANK' in os.environ:
    try:
        rank = int(os.environ['RANK'])
        if rank != 0:
            logger.setLevel(logging.WARNING)
    except (ValueError, KeyError):
        logger.info("RANK env var not found/invalid, running in single-process logging mode.")
        pass

logger.info("Script started")

# --- Directory creation (only on main process) ---
if os.environ.get('RANK', '0') == '0':
    try:
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        logger.info("Checked/created 'checkpoints' and 'logs' directories.")
    except OSError as e:
        logger.error(f"Error creating directories: {e}")

# --- Main DDP worker function ---
def main_worker(local_rank: int, world_size: int):
    """
    Main function executed by each DDP process.

    Args:
        local_rank (int): Local rank of the current process on its node.
        world_size (int): Total number of processes across all nodes.
    """
    setup(local_rank, world_size) # Initialize DDP and set device
    global_rank = dist.get_rank() # Get global rank after setup
    logger.info(f"Process Initialized: Global Rank {global_rank}/{world_size}, Local Rank {local_rank}.")

    # --- Configuration ---
    config = Config()
    config.device = torch.device(f'cuda:{local_rank}') # Assign device based on local rank
    logger.info(f"Rank {global_rank}: Config loaded. Device: {config.device}")

    # --- Data Transforms ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.img_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    logger.info(f"Rank {global_rank}: Data transforms defined.")

    # --- Dataset Loading ---
    # !!! IMPORTANT: Replace these placeholder paths with actual ImageNet paths !!!
    imagenet_train_path = '/nobackup/Imagenet/train' # e.g., '/path/to/imagenet/train'
    imagenet_val_path = '/nobackup/Imagenet/ILSVRC2012_img_val' # e.g., '/path/to/imagenet/val'

    if not os.path.isdir(imagenet_train_path): logger.error(f"Rank {global_rank}: Train path not found: '{imagenet_train_path}'"); cleanup(); return
    if not os.path.isdir(imagenet_val_path): logger.error(f"Rank {global_rank}: Val path not found: '{imagenet_val_path}'"); cleanup(); return

    try:
        train_dataset = torchvision.datasets.ImageFolder(root=imagenet_train_path, transform=train_transform)
        val_dataset = torchvision.datasets.ImageFolder(root=imagenet_val_path, transform=val_transform)
        logger.info(f"Rank {global_rank}: Loaded ImageNet - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    except Exception as e:
         logger.error(f"Rank {global_rank}: Error loading ImageNet dataset: {e}", exc_info=True); cleanup(); return

    # --- Distributed Samplers ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=False) # No drop_last for val
    logger.info(f"Rank {global_rank}: Distributed samplers created.")

    # --- Model Initialization ---
    # Initialize models on the correct device *before* wrapping with DDP
    student_model = VisionTransformer(
        img_size=config.img_size, patch_size=config.patch_size, embed_dim=config.embed_dim,
        depth=config.num_layers, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio
    ).to(config.device)
    teacher_model = VisionTransformer1(
        img_size=config.img_size, patch_size=config.patch_size, embed_dim=config.embed_dim,
        depth=config.num_layers, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio
    ).to(config.device)
    logger.info(f"Rank {global_rank}: Student/Teacher models initialized on {config.device}.")

    # --- Compile Models (Optional) ---
    use_compile = True # Set to False to disable torch.compile
    if use_compile:
        try:
            start_compile_time = time.time()
            student_model = torch.compile(student_model)
            teacher_model = torch.compile(teacher_model)
            compile_time = time.time() - start_compile_time
            logger.info(f"Rank {global_rank}: Models compiled successfully ({compile_time:.2f}s).")
        except Exception as e:
            logger.warning(f"Rank {global_rank}: Model compilation failed: {e}. Proceeding without compilation.", exc_info=True)

    # --- Load Pretrained Teacher (Optional, before DDP wrapping) ---
    # pretrained_teacher_path = '/path/to/your/pretrained_teacher_checkpoint.pth'
    pretrained_teacher_path = None # Set path or None
    # pretrained_teacher_path = '/u/m/a/maheshram/mahesh/checkpoints_imagenet_1/checkpoint_epoch_91.pth' # Example

    if pretrained_teacher_path:
        logger.info(f"Rank {global_rank}: Attempting to load pretrained teacher weights from: {pretrained_teacher_path}")
        # Pass the unwrapped model instance if compile was used
        teacher_model_uncompiled = getattr(teacher_model, '_orig_mod', teacher_model)
        teacher_model_uncompiled = load_pretrained_teacher_weights(teacher_model_uncompiled, pretrained_teacher_path)
        # If compile was used, the original teacher_model variable still points to the compiled wrapper
        # No need to reassign unless load_pretrained_teacher_weights returns a new instance (it shouldn't)
        for p in teacher_model.parameters(): p.requires_grad = False # Freeze after loading
        logger.info(f"Rank {global_rank}: Teacher parameters frozen after attempting pretrained load.")
    else:
        logger.info(f"Rank {global_rank}: No pretrained teacher path. Teacher will be initialized from student later.")


    # --- Wrap models with DDP ---
    # find_unused_parameters can be True if loss doesn't use all outputs, potentially slower
    # Set to False if sure all parameters contribute to loss computation for potential speedup
    student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    teacher_model = DDP(teacher_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    logger.info(f"Rank {global_rank}: Models wrapped with DDP.")

    # --- Initialize Teacher Weights (if not loaded) & Freeze ---
    if not pretrained_teacher_path:
        logger.info(f"Rank {global_rank}: Initializing teacher weights from student (after DDP wrap)...")
        refresh_teacher(student_model, teacher_model) # Copies weights and freezes teacher
        logger.info(f"Rank {global_rank}: Teacher weights initialized from student and frozen.")
    else:
        # Ensure teacher is frozen even if loading happened before DDP wrap
        for p in teacher_model.parameters(): p.requires_grad = False
        logger.info(f"Rank {global_rank}: Ensured teacher parameters remain frozen.")


    # --- Loss Function ---
    criterion = nn.MSELoss()
    logger.info(f"Rank {global_rank}: Loss function (MSELoss) defined.")

    # --- Optimizer ---
    # Optimize only the student model's parameters
    # Pass the DDP-wrapped student model; optimizer needs to handle unwrapping if necessary,
    # or optimize student_model.module.parameters(). SOAP seems designed to handle params directly.
    # Let's pass the DDP model's parameters directly.
    optimizer = SOAP(student_model.parameters(), lr=config.base_learning_rate)
    logger.info(f"Rank {global_rank}: SOAP optimizer initialized for student model parameters.")


    # --- Learning Rate Scheduler ---
    # Calculate total steps based on dataset size, global batch size, and epochs
    global_initial_batch_size = config.get_batch_size(0)
    per_process_initial_batch = max(1, global_initial_batch_size // world_size)
    effective_global_batch = per_process_initial_batch * world_size

    if effective_global_batch == 0:
         logger.error(f"Rank {global_rank}: Effective global batch size is 0. Cannot calculate steps."); cleanup(); return
    batches_per_epoch = len(train_dataset) // effective_global_batch
    if batches_per_epoch == 0:
         logger.warning(f"Rank {global_rank}: Batches per epoch is 0 (Dataset: {len(train_dataset)}, Eff. Global Batch: {effective_global_batch}). LR schedule might be incorrect.")
         batches_per_epoch = 1 # Avoid total_steps=0

    total_steps = config.num_epochs * batches_per_epoch
    warmup_steps = config.warmup_epochs * batches_per_epoch
    logger.info(f"Rank {global_rank}: Scheduler Params - Batches/Epoch: {batches_per_epoch}, Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")

    # LR Lambda function for cosine decay with linear warmup
    def lr_lambda(current_step: int) -> float:
        """Calculates the LR multiplier based on the current step."""
        current_step = min(current_step, total_steps) # Clamp step
        eff_warmup_steps = max(1, warmup_steps)
        eff_total_steps = max(1, total_steps)
        min_lr_ratio = config.min_lr / config.base_learning_rate if config.base_learning_rate > 0 else 0.0

        if current_step < eff_warmup_steps:
            warmup_factor = float(current_step) / float(eff_warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * warmup_factor
        else:
            progress = float(current_step - eff_warmup_steps) / float(max(1, eff_total_steps - eff_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"Rank {global_rank}: LambdaLR scheduler initialized.")

    # --- Mixed Precision Scaler ---
    scaler = GradScaler() if config.use_amp else None
    logger.info(f"Rank {global_rank}: GradScaler {'enabled' if config.use_amp else 'disabled'}.")

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_dir = 'checkpoints'
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

    # Attempt to load checkpoint on all ranks
    start_epoch, best_loss_loaded = load_checkpoint(latest_checkpoint_path, student_model, optimizer, scheduler, config.device)
    best_loss = best_loss_loaded if best_loss_loaded != float('inf') else best_loss
    # load_checkpoint includes a barrier

    if start_epoch > 0:
        logger.info(f"Rank {global_rank}: Checkpoint loaded. Resuming from epoch {start_epoch}. Best loss: {best_loss:.4f}")
        # Ensure teacher model is synced with the loaded student state
        logger.info(f"Rank {global_rank}: Refreshing teacher model from loaded student state...")
        refresh_teacher(student_model, teacher_model)
        logger.info(f"Rank {global_rank}: Teacher model synced after loading checkpoint.")
    else:
        logger.info(f"Rank {global_rank}: No checkpoint found or loaded, starting training from scratch.")

    # Barrier after potential teacher refresh (though refresh is local, ensure state is consistent)
    if world_size > 1: dist.barrier()

    # --- Log File Setup (after potential checkpoint loading) ---
    loss_log_file = os.path.join('logs', 'loss_log.csv')
    if is_main_process(global_rank):
        if start_epoch == 0: # Only clear/create logs if starting from scratch
             if os.path.exists(config.log_file): os.remove(config.log_file)
             if os.path.exists(loss_log_file): os.remove(loss_log_file)
             with open(loss_log_file, 'w') as f: f.write("Epoch,TrainLoss,ValLoss,LearningRate\n")

    # --- Training Loop ---
    logger.info(f"Rank {global_rank}: Starting training loop from epoch {start_epoch} to {config.num_epochs-1}")
    epoch_times = []

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.time()
        logger.info(f"Rank {global_rank}: Starting Epoch {epoch+1}/{config.num_epochs}")

        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        if isinstance(val_sampler, DistributedSampler): val_sampler.set_epoch(epoch)

        # --- Dynamic Batch Size ---
        # Note: Creating DataLoader inside the loop is necessary if batch size changes per epoch.
        current_global_batch_size = config.get_batch_size(epoch)
        if world_size > 0 and current_global_batch_size % world_size != 0:
            adjusted_global_batch_size = (current_global_batch_size // world_size) * world_size
            if adjusted_global_batch_size == 0 and current_global_batch_size > 0: adjusted_global_batch_size = world_size
            if current_global_batch_size != adjusted_global_batch_size and is_main_process(global_rank):
                 logger.warning(f"Epoch {epoch+1}: Adjusted global batch size from {current_global_batch_size} to {adjusted_global_batch_size} (world_size={world_size})")
            current_global_batch_size = adjusted_global_batch_size
        if current_global_batch_size <= 0:
             logger.error(f"Rank {global_rank}: Global batch size <= 0 for epoch {epoch+1}. Skipping epoch."); continue
        per_process_batch_size = max(1, current_global_batch_size // world_size) if world_size > 0 else current_global_batch_size
        if is_main_process(global_rank): logger.info(f"Epoch {epoch+1}: Global BS={current_global_batch_size}, Per-Process BS={per_process_batch_size}")
        # --- End Dynamic Batch Size ---

        # --- DataLoaders ---
        # Determine num_workers based on available CPUs per GPU
        cpus_per_gpu = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else 2
        num_workers_per_gpu = min(4, cpus_per_gpu) # Use a reasonable max like 4
        if is_main_process(global_rank): logger.info(f"Using {num_workers_per_gpu} DataLoader workers per GPU.")

        train_loader = DataLoader(
            train_dataset, batch_size=per_process_batch_size, sampler=train_sampler,
            num_workers=num_workers_per_gpu, pin_memory=True, drop_last=True,
            persistent_workers=True if num_workers_per_gpu > 0 else False )
        val_loader = DataLoader(
            val_dataset, batch_size=per_process_batch_size, sampler=val_sampler, # Use same BS for val
            num_workers=num_workers_per_gpu, pin_memory=True, drop_last=False,
            persistent_workers=True if num_workers_per_gpu > 0 else False )
        logger.debug(f"Rank {global_rank}: DataLoaders created for epoch {epoch+1}.")

        # --- Train & Evaluate ---
        train_loss_avg, current_lr = train(student_model, teacher_model, train_loader, criterion, optimizer, scheduler, config.device, epoch, config, scaler, global_rank, world_size)
        val_loss_avg = evaluate(student_model, teacher_model, val_loader, criterion, config.device, epoch, config, global_rank, world_size)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # --- Logging and Checkpointing (Main Process Only) ---
        if is_main_process(global_rank):
            log_message = (
                 f"Epoch [{epoch+1}/{config.num_epochs}] | Time: {epoch_duration:.2f}s | "
                 f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | LR: {current_lr:.6e}" )
            print(log_message) # Print to console
            log_to_file(config.log_file, log_message) # Log to main text file
            with open(loss_log_file, 'a') as f: f.write(f"{epoch+1},{train_loss_avg:.4f},{val_loss_avg:.4f},{current_lr:.6f}\n") # Log to CSV

            # Save latest checkpoint
            save_checkpoint(student_model, optimizer, scheduler, epoch, val_loss_avg, latest_checkpoint_path)

            # Save best checkpoint
            if val_loss_avg < best_loss:
                best_loss = val_loss_avg
                save_checkpoint(student_model, optimizer, scheduler, epoch, best_loss, best_checkpoint_path)
                logger.info(f"Epoch {epoch+1}: New best val loss: {best_loss:.4f}. Saved best model.")

            # Save periodic checkpoints
            save_freq = 10 # Save every 10 epochs
            if (epoch + 1) % save_freq == 0:
                 periodic_filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                 save_checkpoint(student_model, optimizer, scheduler, epoch, val_loss_avg, periodic_filename)

        # --- Teacher Refresh ---
        # Refresh teacher weights from student (local op, happens on all ranks)
        refresh_teacher(student_model, teacher_model)
        logger.debug(f"Rank {global_rank}: Teacher refreshed at end of epoch {epoch+1}")
        # Barrier ensures all ranks finish epoch + refresh before next one starts
        if world_size > 1: dist.barrier()

    # --- End of Training Loop ---
    if is_main_process(global_rank):
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        final_message = f"Training finished after {config.num_epochs} epochs. Best Val Loss: {best_loss:.4f}. Avg Epoch Time: {avg_epoch_time:.2f}s"
        print(final_message)
        log_to_file(config.log_file, final_message)

    cleanup() # Clean up DDP resources

# --- Entry Point ---
if __name__ == "__main__":
    # Check if running under torchrun/DDP
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
        try:
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"]) # torchrun provides LOCAL_RANK
            global_rank = int(os.environ["RANK"])     # torchrun provides RANK (global)

            print(f"Starting DDP training via torchrun: WORLD_SIZE={world_size}, GLOBAL_RANK={global_rank}, LOCAL_RANK={local_rank}")
            logger.info(f"DDP Env Vars: WORLD_SIZE={world_size}, RANK={global_rank}, LOCAL_RANK={local_rank}, MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")

            # Pass local_rank to main_worker for device assignment in setup
            main_worker(local_rank, world_size)

        except KeyError as e:
            print(f"Error: Missing DDP env var ({e}). Launch with torchrun.")
            logger.error(f"Missing DDP env var: {e}")
        except ValueError as e:
             print(f"Error: Invalid DDP env var ({e}). Ensure ranks/size are integers.")
             logger.error(f"Invalid DDP env var: {e}")
    else:
        # Fallback to single-process mode
        print("WARNING: DDP env vars not found. Running in single-process mode.")
        logger.warning("Running in single-process mode.")
        # In single process, local_rank=0, world_size=1
        main_worker(local_rank=0, world_size=1)
