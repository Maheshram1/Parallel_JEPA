# Parallel JEPA

Parallel JEPA is a scalable PyTorch framework for training Vision Transformer‑based student‑teacher models under the Joint Embedding Predictive Architecture (JEPA) paradigm. It splits image patches into multiple parts, shuffles and masks them in the student encoder, and reconstructs full patch representations in parallel decoder streams, using mean‑squared error against a frozen “teacher” ViT’s intermediate features. The code is fully distributed via PyTorch DDP, supports mixed precision with `torch.cuda.amp`, and employs a custom SOAP optimizer for adaptive gradient preconditioning.

## Features

*   **Student & Teacher ViTs**: Implements both the masked/shuffled student JEPA model (`VisionTransformer`) and a standard teacher ViT (`VisionTransformer1`), sharing patch embedding but differing in masking and decoder depth ([model.py](https://github.com/Maheshram1/Parallel_JEPA/blob/main/model.py)).
*   **Parallel Reconstruction**: Divides the full sequence of patches into `num_parts` blocks processed independently by decoder streams, enabling parallel JEPA reconstruction ([model.py#L138-L168](https://github.com/Maheshram1/Parallel_JEPA/blob/main/model.py#L138)). <!-- Update link/lines as needed -->
*   **Distributed Training**: Leverages `torch.distributed` and DDP for multi‑GPU scaling, with helper utilities for setup/cleanup, synchronized checkpointing, and main‑process logging ([utils.py](https://github.com/Maheshram1/Parallel_JEPA/blob/main/utils.py), [main.py](https://github.com/Maheshram1/Parallel_JEPA/blob/main/main.py)).
*   **Mixed Precision**: Optional AMP via `GradScaler` for faster throughput and lower memory usage ([main.py#L188](https://github.com/Maheshram1/Parallel_JEPA/blob/main/main.py#L188)). <!-- Update link/lines as needed -->
*   **Custom SOAP Optimizer**: Integrates a Shampoo‑style optimizer with second‑order gradient preconditioning for the student model ([optimizer.py](https://github.com/Maheshram1/Parallel_JEPA/blob/main/optimizer.py)).
*   **Flexible Hyperparameters**: All core settings (image size, patch size, embedding dim, batch‑size warmup, learning‑rate schedule) are centralized in `config.py` with dynamic batch‑size support ([config.py](https://github.com/Maheshram1/Parallel_JEPA/blob/main/config.py)).

## Installation

1.  **Clone the repo**
    ```bash
    git clone https://github.com/Maheshram1/Parallel_JEPA.git
    cd Parallel_JEPA
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    Requires `torch>=1.13.0`, `torchvision>=0.14.0`, and `tqdm` ([requirements.txt](https://github.com/Maheshram1/Parallel_JEPA/blob/main/requirements.txt)).

## Data Preparation

*   **ImageNet**: Place training and validation sets in directories and update paths in `main.py`:
    ```python
    # main.py
    imagenet_train_path = '/path/to/imagenet/train' # UPDATE THIS
    imagenet_val_path   = '/path/to/imagenet/val'   # UPDATE THIS
    ```
    ([main.py#L94-L95](https://github.com/Maheshram1/Parallel_JEPA/blob/main/main.py#L94)) <!-- Update link/lines as needed -->

## Configuration

All hyperparameters live in `config.py`:

```python
# config.py
config = Config()
config.img_size = 224
config.patch_size = 14
config.embed_dim = 1280
config.num_layers = 16
config.num_heads  = 32
config.initial_batch_size = 128 # Global batch size start
config.final_batch_size   = 128 # Global batch size end
config.num_epochs = 100
config.base_learning_rate = 1e-3
config.warmup_epochs = 4
config.use_amp = True
# ... and others
```


You can modify settings like learning rates, warmup, batch size schedule, and AMP usage directly in the Config class (config.py).

Training

Launch distributed training with torchrun (or torch.distributed.run):

# Example: Single node, 4 GPUs
torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

The script (main.py) will:

Initialize DDP and set the CUDA device based on LOCAL_RANK (utils.py#L15-L30). <!-- Update link/lines as needed -->

Build student (VisionTransformer) and teacher (VisionTransformer1) models and optionally compile them with torch.compile (main.py#L120-L147). <!-- Update link/lines as needed -->

Load or initialize teacher weights, freeze its parameters, and wrap both models in DDP (main.py#L150-L175). <!-- Update link/lines as needed -->

Create MSE loss, SOAP optimizer, and LambdaLR scheduler with cosine decay & linear warmup (main.py#L178-L220). <!-- Update link/lines as needed -->

Run epochs: dynamic batch resizing, forwarding through student & teacher, computing loss, backward pass with AMP, optimizer & scheduler steps, teacher refresh, and periodic checkpointing/logging (main.py#L253-L350, engine.py). <!-- Update link/lines as needed -->

Evaluation

Validation occurs at the end of each training epoch using the same DDP setup. It reports the average MSE loss between the student's reconstructed patch representations and the teacher’s intermediate features across the validation set (engine.py#L167-L258). <!-- Update link/lines as needed -->

Checkpoints & Logging

Checkpoints: Saved by the main process (rank 0) into the checkpoints/ directory:

latest_checkpoint.pth: Overwritten after every epoch.

best_model.pth: Overwritten when validation loss improves.

checkpoint_epoch_*.pth: Saved periodically (default: every 10 epochs).
(main.py#L325-L341) <!-- Update link/lines as needed -->

Logs: Saved by the main process into the logs/ directory:

training_log.txt: Detailed text log including epoch times, losses, and LR.

loss_log.csv: CSV file tracking Epoch, TrainLoss, ValLoss, LearningRate for easier analysis.
(main.py#L317-L324) <!-- Update link/lines as needed -->

Utilities

DDP Setup & Cleanup: setup(), cleanup(), and is_main_process() in utils.py manage the distributed environment (utils.py#L15-L46). <!-- Update link/lines as needed -->

Checkpointing Helpers: save_checkpoint() and load_checkpoint() in utils.py provide robust saving/loading, handling DDP/compile wrappers and ensuring CPU-based saving for portability (utils.py#L63-L219). <!-- Update link/lines as needed -->

Teacher Utilities: load_pretrained_teacher_weights() handles loading external weights, while refresh_teacher() copies student weights to the teacher (simulating momentum update) and freezes the teacher each epoch (utils.py#L224-L368). <!-- Update link/lines as needed -->

Feel free to file issues or contribute enhancements—Parallel JEPA is designed for extensibility to new architectures, datasets, and optimization strategies.

IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
