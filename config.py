"""
Configuration settings for the training process.
"""

class Config:
    """Holds hyperparameters and settings for the model and training."""
    def __init__(self):
        """Initializes the configuration settings."""
        self.img_size = 224
        self.patch_size = 14 # ViT patch size
        self.in_channels = 3
        self.embed_dim = 1280
        self.num_layers = 16
        self.num_heads = 32
        self.mlp_ratio = 4.0
        self.initial_batch_size = 128 # Global batch size
        self.final_batch_size = 128 # Global batch size
        self.batch_size_warmup_epochs = 4
        self.num_epochs = 100
        self.base_learning_rate = 1e-3
        self.warmup_epochs = 4
        self.min_lr = 1e-4
        self.log_file = 'logs/training_log.txt'
        self.use_amp = True
        self.device = None # Will be set in main_worker

    def get_batch_size(self, epoch: int) -> int:
        """
        Calculates the global batch size for a given epoch, supporting warmup.

        Args:
            epoch (int): The current epoch number.

        Returns:
            int: The calculated global batch size for the epoch.
        """
        if epoch >= self.batch_size_warmup_epochs:
            return self.final_batch_size
        progress = epoch / (self.batch_size_warmup_epochs - 1) if self.batch_size_warmup_epochs > 1 else 1.0
        current_batch_size = int(self.initial_batch_size +
                                 progress * (self.final_batch_size - self.initial_batch_size))
        return max(1, current_batch_size)
