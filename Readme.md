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
