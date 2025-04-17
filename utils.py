"""
Utility functions for DDP setup, checkpointing, logging, and teacher model operations.
"""

import os
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import copy

# --- Distributed Helper Functions ---

def setup(local_rank: int, world_size: int):
    """
    Initializes the distributed process group using torchrun environment variables.

    Args:
        local_rank (int): Local rank of the current process on its node.
        world_size (int): Total number of processes across all nodes.
    """
    # torchrun usually sets these, but provide defaults just in case
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355') # Default port if not set

    # Construct the init_method URL
    init_method = f"tcp://{master_addr}:{master_port}"

    # Get global rank from environment (torchrun provides RANK)
    global_rank = int(os.environ['RANK'])

    # Initialize the process group
    dist.init_process_group(backend="nccl",
                            init_method=init_method,
                            world_size=world_size,
                            rank=global_rank)

    # Set the CUDA device to the local rank to ensure each process uses a different GPU
    torch.cuda.set_device(local_rank)
    logging.info(f"Rank {global_rank}: Initialized process group. Device set to cuda:{local_rank}")


def cleanup():
    """Destroys the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.info("Distributed process group destroyed.")


def is_main_process(global_rank: int = -1) -> bool:
    """
    Checks if the current process is the main process (global rank 0).

    Args:
        global_rank (int, optional): The global rank if known. If -1, attempts to get rank from dist.

    Returns:
        bool: True if the process is global rank 0, False otherwise.
    """
    if not dist.is_initialized():
        # If DDP not initialized, assume single process, rank 0 is main
        return True # Or check if global_rank == 0 if passed? Safer to assume True.
    # If initialized, use the definitive global rank
    current_global_rank = dist.get_rank()
    return current_global_rank == 0


# --- Logging Helper ---

def log_to_file(log_file: str, message: str):
    """
    Logs a message to a file, only from the main process (global rank 0).

    Creates the directory if it doesn't exist.

    Args:
        log_file (str): Path to the log file.
        message (str): Message to log.
    """
    # Only log from the main process
    if not is_main_process():
        return
    try:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir: # Create directory only if log_file includes a path
             os.makedirs(log_dir, exist_ok=True)
        # Append message to the file
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    except IOError as e:
        # Use standard logging to report error if file logging fails
        logging.error(f"[Rank 0]: Could not write to log file {log_file}: {e}")


# --- Checkpointing Functions ---

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler,
                    epoch: int, loss: float, filename: str):
    """
    Saves model, optimizer, scheduler state, epoch, and loss to a checkpoint file.

    Only the main process (global rank 0) saves the file.
    Handles DDP and torch.compile wrappers.
    Moves tensors to CPU before saving for portability.

    Args:
        model (torch.nn.Module): The model to save (can be DDP/compiled).
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        scheduler: The learning rate scheduler state to save.
        epoch (int): The current epoch number (epoch just completed).
        loss (float): The loss value to save (e.g., validation loss).
        filename (str): Path to save the checkpoint file.
    """
    # Only save from the main process
    if not is_main_process():
        return

    try:
        # Ensure the checkpoint directory exists
        ckpt_dir = os.path.dirname(filename)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        # --- Get the underlying module state dict, handling DDP and compile ---
        model_to_save = model
        # 1. Unwrap torch.compile (if applied)
        if hasattr(model_to_save, '_orig_mod'):
            model_to_save = model_to_save._orig_mod
        # 2. Unwrap DDP (if applied)
        if isinstance(model_to_save, DDP):
            model_to_save = model_to_save.module
        model_state_dict = model_to_save.state_dict()
        # --- End unwrap ---

        # Prepare checkpoint data dictionary
        checkpoint = {
            'epoch': epoch, # Save the epoch *just completed*
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss, # Typically the validation loss at this epoch
        }
        logging.debug("[Rank 0]: Prepared checkpoint dictionary for epoch %d.", epoch)

        # --- Move tensors to CPU before saving ---
        cpu_checkpoint = {}
        for key, value in checkpoint.items():
            if isinstance(value, dict): # Handle nested dicts (like optimizer state)
                cpu_nested_dict = {}
                for k_sub, v_sub in value.items():
                    if isinstance(v_sub, dict): # Handle state dict within optimizer state (e.g., 'state')
                         cpu_state_val = {}
                         # Ensure state values (like momentum buffers) are moved
                         for k_state, v_state in v_sub.items():
                              if isinstance(v_state, torch.Tensor):
                                  cpu_state_val[k_state] = v_state.cpu()
                              # Handle potential nested structures within optimizer state if needed
                              elif isinstance(v_state, dict):
                                   cpu_nested_state = {}
                                   for k_n, v_n in v_state.items():
                                        if isinstance(v_n, torch.Tensor):
                                             cpu_nested_state[k_n] = v_n.cpu()
                                        else:
                                             cpu_nested_state[k_n] = v_n
                                   cpu_state_val[k_state] = cpu_nested_state
                              else:
                                  cpu_state_val[k_state] = v_state # Keep non-tensor leaves
                         cpu_nested_dict[k_sub] = cpu_state_val
                    elif isinstance(v_sub, torch.Tensor): # Tensors directly within the optimizer dict
                         cpu_nested_dict[k_sub] = v_sub.cpu()
                    else: # Other non-tensor values in optimizer state
                         cpu_nested_dict[k_sub] = v_sub
                cpu_checkpoint[key] = cpu_nested_dict
            elif isinstance(value, torch.Tensor): # Tensors at the top level (model state_dict)
                cpu_checkpoint[key] = value.cpu()
            else: # Keep other types as is (epoch, loss)
                cpu_checkpoint[key] = value
        # --- End CPU move ---
        logging.debug("[Rank 0]: Moved checkpoint tensors to CPU.")

        # Save the CPU checkpoint atomically (save to temp, then rename)
        tmp_filename = filename + ".tmp"
        torch.save(cpu_checkpoint, tmp_filename)
        os.rename(tmp_filename, filename) # Atomic rename
        logging.info(f"[Rank 0]: Checkpoint saved successfully to {filename} (Epoch {epoch+1})") # Log epoch+1 for clarity

    except Exception as e:
        logging.error(f"[Rank 0]: Failed to save checkpoint to {filename}: {e}", exc_info=True)
        # Clean up temporary file if it exists
        if 'tmp_filename' in locals() and os.path.exists(tmp_filename):
            try:
                os.remove(tmp_filename)
            except OSError:
                pass


def load_checkpoint(filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, device: torch.device) -> tuple[int, float]:
    """
    Loads model, optimizer, and scheduler state from a checkpoint file.

    All processes load the checkpoint state to ensure consistency.
    Handles DDP and torch.compile wrappers.
    Loads to CPU first, then moves state to the specified process's device.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load state into (can be DDP/compiled).
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        scheduler: The learning rate scheduler to load state into.
        device (torch.device): The target device for the current process.

    Returns:
        tuple[int, float]: (start_epoch, best_loss) loaded from checkpoint.
                           Returns (0, float('inf')) if loading fails or file not found.
    """
    start_epoch = 0
    best_loss = float('inf')
    rank = dist.get_rank() if dist.is_initialized() else 0 # Get current process rank

    if not os.path.exists(filename):
        logging.warning(f"Rank {rank}: Checkpoint file {filename} not found. Starting from scratch.")
        return start_epoch, best_loss

    try:
        # Load checkpoint to CPU first for safety and memory efficiency
        checkpoint = torch.load(filename, map_location='cpu')
        logging.info(f"Rank {rank}: Loading checkpoint from {filename} (loaded to CPU).")

        # --- Load Model State ---
        model_to_load = model
        if hasattr(model_to_load, '_orig_mod'): model_to_load = model_to_load._orig_mod
        if isinstance(model_to_load, DDP): model_to_load = model_to_load.module
        logging.debug(f"Rank {rank}: Identified underlying model module for loading: {type(model_to_load).__name__}")

        try:
            missing_keys, unexpected_keys = model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys: logging.warning(f"Rank {rank}: Missing keys when loading model state_dict: {missing_keys}")
            if unexpected_keys: logging.warning(f"Rank {rank}: Unexpected keys when loading model state_dict: {unexpected_keys}")
            model.to(device) # Move the potentially wrapped model to the correct device *after* loading state dict
            logging.info(f"Rank {rank}: Model state loaded and model moved to {device}.")
        except Exception as e:
            logging.error(f"Rank {rank}: Error loading model state_dict: {e}. Model weights may be incorrect.", exc_info=True)

        # --- Load Optimizer State ---
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer_state_dict = checkpoint['optimizer_state_dict']
                # Move optimizer state tensors to the correct device *before* loading
                optimizer_state_dict_device = copy.deepcopy(optimizer_state_dict) # Modify copy
                for state_id, state_val in optimizer_state_dict_device.get('state', {}).items():
                    for k, v in state_val.items():
                        if isinstance(v, torch.Tensor):
                            try:
                                state_val[k] = v.to(device)
                            except Exception as e_opt_tensor:
                                logging.error(f"Rank {rank}: Failed to move optimizer state tensor {k} (state {state_id}) to {device}: {e_opt_tensor}")
                        # Handle potential nested dicts in optimizer state if necessary
                        elif isinstance(v, dict):
                             for k_n, v_n in v.items():
                                  if isinstance(v_n, torch.Tensor):
                                       try:
                                            v[k_n] = v_n.to(device)
                                       except Exception as e_opt_tensor_n:
                                            logging.error(f"Rank {rank}: Failed to move nested optimizer state tensor {k_n} to {device}: {e_opt_tensor_n}")

                optimizer.load_state_dict(optimizer_state_dict_device)
                logging.info(f"Rank {rank}: Optimizer state loaded successfully.")
            except Exception as e:
                logging.error(f"Rank {rank}: Error loading optimizer state_dict: {e}. Optimizer state may be incorrect.", exc_info=True)
        else:
            logging.warning(f"Rank {rank}: Optimizer state ('optimizer_state_dict') not found in checkpoint.")

        # --- Load Scheduler State ---
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logging.info(f"Rank {rank}: Scheduler state loaded successfully.")
            except Exception as e:
                 logging.error(f"Rank {rank}: Error loading scheduler state_dict: {e}. Scheduler state may be incorrect.", exc_info=True)
        else:
             logging.warning(f"Rank {rank}: Scheduler state ('scheduler_state_dict') not found in checkpoint.")

        # --- Load Epoch and Loss ---
        start_epoch = checkpoint.get('epoch', -1) + 1 # Resume from the next epoch
        best_loss = checkpoint.get('loss', float('inf'))
        logging.info(f"Rank {rank}: Checkpoint loaded. Resuming from epoch {start_epoch}. Previous best loss: {best_loss:.4f}")

    except FileNotFoundError:
        # This case is already handled by the initial check, but keep for robustness
        logging.warning(f"Rank {rank}: Checkpoint file {filename} not found during load attempt. Starting from scratch.")
        start_epoch = 0
        best_loss = float('inf')
    except Exception as e:
        logging.error(f"Rank {rank}: General error loading checkpoint from {filename}: {e}. Starting from scratch.", exc_info=True)
        start_epoch = 0
        best_loss = float('inf')

    # Barrier ensures all processes sync up after attempting to load checkpoint
    if dist.is_initialized():
        logging.debug(f"Rank {rank}: Waiting at barrier after load_checkpoint.")
        dist.barrier()
        logging.debug(f"Rank {rank}: Passed barrier after load_checkpoint.")

    return start_epoch, best_loss


# --- Teacher Model Utilities ---

def load_pretrained_teacher_weights(model: torch.nn.Module, pretrained_path: str) -> torch.nn.Module:
    """
    Loads weights into a teacher model from a checkpoint, handling various state_dict keys
    and potential DDP/compile prefixes. Should be called on the model *before* DDP wrapping
    if loading from a standard checkpoint.

    Args:
        model (torch.nn.Module): The teacher model instance (unwrapped).
        pretrained_path (str): Path to the pretrained checkpoint file.

    Returns:
        torch.nn.Module: The model instance (modified in-place) with loaded weights.
                         Returns the original model if loading failed.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger(__name__)

    if not pretrained_path or not os.path.exists(pretrained_path):
        logger.warning(f"Rank {rank} [LOAD_TEACHER]: No pretrained weights path/file: '{pretrained_path}'. Teacher using initial weights.")
        return model

    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        logger.info(f"Rank {rank} [LOAD_TEACHER]: Loaded checkpoint from {pretrained_path}")

        # --- Extract State Dictionary ---
        state_dict = None
        if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict): state_dict = checkpoint
        else:
             logger.error(f"Rank {rank} [LOAD_TEACHER]: Checkpoint is not a dict. Cannot load.")
             return model
        if not state_dict:
             logger.error(f"Rank {rank} [LOAD_TEACHER]: Could not extract state_dict from checkpoint.")
             return model
        logger.info(f"Rank {rank} [LOAD_TEACHER]: Using state_dict with {len(state_dict)} keys.")
        # --- End Extract ---

        # --- Get Target Model's State Dict (Should be unwrapped) ---
        if isinstance(model, (DDP, torch.nn.modules.lazy.LazyModuleMixin)): # Check if accidentally wrapped
            logger.warning(f"Rank {rank} [LOAD_TEACHER]: `load_pretrained_teacher_weights` called on a potentially wrapped model ({type(model).__name__}). Unwrapping first.")
            if hasattr(model, '_orig_mod'): model = model._orig_mod
            if isinstance(model, DDP): model = model.module

        model_dict = model.state_dict()
        logger.info(f"Rank {rank} [LOAD_TEACHER]: Target teacher ({type(model).__name__}) expects {len(model_dict)} keys.")
        if not model_dict:
             logger.error(f"Rank {rank} [LOAD_TEACHER]: Target model state_dict is empty.")
             return model
        # --- End Target Prep ---

        # --- Process Checkpoint Keys & Load ---
        # Remove 'module.' prefix if present in checkpoint keys, as target model is assumed unwrapped
        processed_state_dict = {}
        has_module_prefix = False
        for k, v in state_dict.items():
            if k.startswith('module.'):
                processed_state_dict[k[len('module.'):]] = v
                has_module_prefix = True
            else:
                processed_state_dict[k] = v
        if has_module_prefix:
            logger.info(f"Rank {rank} [LOAD_TEACHER]: Removed 'module.' prefix from checkpoint keys.")

        # Load state dict (strict=False allows mismatches)
        load_result = model.load_state_dict(processed_state_dict, strict=False)

        if load_result.missing_keys:
            logger.warning(f"Rank {rank} [LOAD_TEACHER]: Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
             logger.warning(f"Rank {rank} [LOAD_TEACHER]: Unexpected keys: {load_result.unexpected_keys}")

        logger.info(f"Rank {rank} [LOAD_TEACHER]: Successfully loaded weights into teacher model!")
        return model

    except Exception as e:
        logger.error(f"Rank {rank} [LOAD_TEACHER]: Error loading weights from {pretrained_path}: {e}. Teacher not loaded.", exc_info=True)
        return model


def refresh_teacher(student_model: torch.nn.Module, teacher_model: torch.nn.Module):
    """
    Copies weights from the student model to the teacher model in-place.

    Handles DDP and torch.compile wrappers for both models.
    Ensures the teacher model's parameters do not require gradients afterwards.
    This operation is performed locally on each rank.

    Args:
        student_model (torch.nn.Module): The student model (source, potentially wrapped).
        teacher_model (torch.nn.Module): The teacher model (destination, potentially wrapped).
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger(__name__)

    def _unwrap(m):
        """Helper to unwrap DDP and torch.compile."""
        uncompiled = getattr(m, "_orig_mod", m)
        unwrapped = uncompiled.module if isinstance(uncompiled, DDP) else uncompiled
        return unwrapped

    try:
        s_core = _unwrap(student_model)
        t_core = _unwrap(teacher_model)
        logging.debug(f"Rank {rank}: Refreshing teacher ({type(t_core).__name__}) from student ({type(s_core).__name__})")

        # --- Copy weights using state_dict ---
        with torch.no_grad():
            student_state = s_core.state_dict()
            load_result = t_core.load_state_dict(student_state, strict=True) # Use strict=True for refresh
            # Log warnings only if strict loading fails (shouldn't happen if architectures match)
            # if load_result.missing_keys or load_result.unexpected_keys:
            #      logger.warning(f"Rank {rank}: Mismatch during teacher refresh. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

        # --- Freeze teacher parameters ---
        # Freeze parameters of the core teacher model
        for p in t_core.parameters():
            p.requires_grad = False
        # Also ensure parameters of the potentially wrapped teacher model are frozen
        for p in teacher_model.parameters():
             p.requires_grad = False
        logging.debug(f"Rank {rank}: Teacher model weights refreshed and parameters frozen.")

    except Exception as e:
        logger.error(f"Rank {rank}: Failed to refresh teacher model weights: {e}", exc_info=True)
        # Attempt to freeze teacher anyway if copy failed but model exists
        try:
            for p in teacher_model.parameters():
                 p.requires_grad = False
            logger.warning(f"Rank {rank}: Attempted to freeze teacher parameters after refresh error.")
        except Exception as freeze_e:
             logger.error(f"Rank {rank}: Failed to freeze teacher parameters after refresh error: {freeze_e}")
