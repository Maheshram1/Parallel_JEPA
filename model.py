"""
Vision Transformer model definitions (Student and Teacher) and building blocks.
"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Image to Patch Embedding block."""
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        """
        Initializes the PatchEmbedding layer.

        Args:
            img_size (int): Size of the input image (assumed square).
            patch_size (int): Size of each patch (assumed square).
            in_channels (int): Number of input image channels.
            embed_dim (int): Dimension of the embedding vectors.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PatchEmbedding.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).

        Returns:
            torch.Tensor: Embedded patches (B, N, D), where N is num_patches, D is embed_dim.
        """
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        """
        Initializes the Attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to include bias in Q, K, V projections.
            attn_drop (float): Dropout rate for attention weights.
            proj_drop (float): Dropout rate for the output projection.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for Attention.

        Args:
            x (torch.Tensor): Input tensor (B, N, C).
            mask (torch.Tensor, optional): Attention mask (B, N, N) or (B, H, N, N). Defaults to None.

        Returns:
            torch.Tensor: Output tensor (B, N, C).
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.dim() == 3: # B, N, N -> B, H, N, N
                 expanded_mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            else: # Assume B, H, N, N
                 expanded_mask = mask
            # Use float('-inf') for masking in attention softmax
            attn = attn.masked_fill(expanded_mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        # attn_weights = attn # Store if needed later
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FeedForward(nn.Module):
    """Feed-forward network (MLP) block."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0., activation=nn.GELU):
        """
        Initializes the FeedForward module.

        Args:
            dim (int): Input and output dimension.
            hidden_dim (int): Dimension of the hidden layer.
            dropout (float): Dropout rate.
            activation: Activation function class. Defaults to nn.GELU.
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FeedForward.

        Args:
            x (torch.Tensor): Input tensor (B, N, C).

        Returns:
            torch.Tensor: Output tensor (B, N, C).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-normalization."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False, drop: float = 0., attn_drop: float = 0., norm_layer=nn.LayerNorm):
        """
        Initializes the TransformerBlock.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
            qkv_bias (bool): Whether to use bias in QKV projections.
            drop (float): Dropout rate for MLP and projection layers.
            attn_drop (float): Dropout rate for attention.
            norm_layer: Normalization layer class. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)


    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor (B, N, C).
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor (B, N, C).
        """
        attn_output = self.attn(self.norm1(x), mask=mask) # Pass mask to attention
        x = x + attn_output # Residual connection 1
        x = x + self.mlp(self.norm2(x)) # Residual connection 2
        return x

class VisionTransformer(nn.Module):
    """Student Vision Transformer model implementing the shuffling/masking strategy."""
    def __init__(self, img_size: int = 224, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1280, depth: int = 16,
                 num_heads: int = 32, mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.):
        """
        Initializes the Student VisionTransformer model.

        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_channels (int): Number of input channels.
            embed_dim (int): Embedding dimension.
            depth (int): Number of transformer blocks (encoder).
            num_heads (int): Number of attention heads.
            mlp_ratio (float): MLP hidden dimension ratio.
            qkv_bias (bool): Enable bias in QKV projections.
            drop_rate (float): Dropout rate for MLP and projection layers.
            attn_drop_rate (float): Attention dropout rate.
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.num_parts = 4 # Number of parts to divide patches into
        assert self.num_patches % self.num_parts == 0, \
            f"num_patches ({self.num_patches}) must be divisible by num_parts ({self.num_parts})"
        self.block_size = num_patches // self.num_parts
        self.embed_dim = embed_dim

        # Learnable parameters
        self.decoder_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # Encoder positional embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # Decoder positional embedding

        # Encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        # Decoder blocks (fewer than encoder)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth // 4) # Decoder depth is 1/4 of encoder
        ])

        self.norm = nn.LayerNorm(embed_dim) # Final normalization layer for decoder output
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using truncated normal distribution."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._init_weights_general)

    def _init_weights_general(self, m):
        """Applies weight initialization to linear and layernorm layers."""
        if isinstance(m, nn.Linear):
            # Initialize linear layers with truncated normal
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Initialize layernorm layers with constants
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Student VisionTransformer.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Encoder output tensor after shuffling and masking (B, N, D).
                - Decoder output tensor after reconstruction (B * num_parts, N, D).
        """
        B, C, H, W = x.shape
        x = self.patch_embed(x) # (B, N, D)
        x = x + self.pos_embed # Add positional embedding: (B, N, D)

        # --- Shuffling and Masking Logic ---
        # Shuffle patch order per image
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, self.num_patches) # (B, N)
        random_indices = torch.rand(B, self.num_patches, device=x.device).argsort(dim=1) # (B, N) random permutation
        x = x[batch_indices, random_indices] # Apply shuffling: (B, N, D)

        # Create block-diagonal attention mask
        mask = torch.zeros(self.num_patches, self.num_patches, device=x.device)
        for i in range(self.num_parts):
            start = i * self.block_size
            end = start + self.block_size
            mask[start:end, start:end] = 1 # Allow attention within each block (value 1 means allow)
        mask = mask.unsqueeze(0).expand(B, -1, -1) # (B, N, N)
        # --- End Shuffling and Masking ---

        # Encoder blocks with masking
        for block in self.blocks:
            x = block(x, mask=mask) # Pass mask to encoder blocks: (B, N, D)

        encoder_output = x # Store encoder output if needed later

        # --- Decoder Logic ---
        # Initialize decoder input with decoder tokens and scatter encoded patches
        # Start with decoder tokens repeated for each part and each patch position
        decoder_input = self.decoder_token.repeat(B, self.num_parts, self.num_patches, 1) # (B, num_parts, N, D)

        # Scatter the encoded features (from encoder_output) back into the decoder input
        # based on their *original* positions before shuffling.
        for k in range(self.num_parts):
            # Identify the section of the shuffled sequence corresponding to this part
            start_idx = k * self.block_size
            end_idx = (k + 1) * self.block_size
            # Get the encoded features for this part from the shuffled encoder output
            current_block_encoded = encoder_output[:, start_idx:end_idx, :] # (B, block_size, D)
            # Get the original positions of these patches from the random indices
            positions = random_indices[:, start_idx:end_idx] # (B, block_size)
            # Prepare indices for scatter operation
            scatter_indices = positions.unsqueeze(-1).expand(-1, -1, self.embed_dim) # (B, block_size, D)
            # Scatter the encoded features into the k-th part of the decoder input at the correct positions
            # Note: scatter_ operates in-place
            decoder_input[:, k, :, :].scatter_(dim=1, index=scatter_indices, src=current_block_encoded)

        # Add decoder positional embedding (applied to all parts)
        decoder_input = decoder_input + self.decoder_pos_embed.unsqueeze(1) # (B, num_parts, N, D)

        # Reshape for decoder blocks: merge batch and num_parts dimensions
        merged_input = decoder_input.view(-1, self.num_patches, self.embed_dim) # (B * num_parts, N, D)

        # Decoder blocks (process each reconstruction attempt independently)
        for block in self.decoder_blocks:
            # Decoder does not use masking, it tries to reconstruct the full sequence
            merged_input = block(merged_input, mask=None) # (B * num_parts, N, D)

        # --- End Decoder Logic ---

        # Return encoder output (after shuffling/masking) and normalized decoder output
        return encoder_output, self.norm(merged_input)


class VisionTransformer1(nn.Module):
    """Teacher Vision Transformer model (standard ViT)."""
    def __init__(self, img_size: int = 224, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1280, depth: int = 16,
                 num_heads: int = 32, mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.):
        """
        Initializes the Teacher VisionTransformer model (standard ViT).

        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_channels (int): Number of input channels.
            embed_dim (int): Embedding dimension.
            depth (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): MLP hidden dimension ratio.
            qkv_bias (bool): Enable bias in QKV projections.
            drop_rate (float): Dropout rate for MLP and projection layers.
            attn_drop_rate (float): Attention dropout rate.
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_channels = in_channels
        # Standard ViT components
        self.num_parts = 4 # Kept for consistency if needed elsewhere, but not used in forward
        self.block_size = num_patches // self.num_parts # Kept for consistency
        self.embed_dim = embed_dim

        # Learnable parameters (standard ViT only needs pos_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # Standard positional embedding
        # Decoder token/embed might not be strictly necessary if only using encoder output, but kept for structural consistency
        self.decoder_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # Potentially unused
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # Potentially unused

        # Main transformer blocks (encoder)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        # Teacher also has 'decoder' blocks, acting as more encoder layers here
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth // 4) # Same depth ratio as student's decoder
        ])

        self.norm = nn.LayerNorm(embed_dim) # Final normalization layer
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using truncated normal distribution."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_token, std=0.02) # Initialize even if unused
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02) # Initialize even if unused
        self.apply(self._init_weights_general)

    def _init_weights_general(self, m):
        """Applies weight initialization to linear and layernorm layers."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Teacher VisionTransformer (standard ViT).

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output after the main encoder blocks (B, N, D). This is used as the target.
                - Output after the 'decoder' blocks (B, N, D). This is normalized by self.norm.
        """
        B, C, H, W = x.shape
        x = self.patch_embed(x) # (B, N, D)
        x = x + self.pos_embed # Add positional embedding: (B, N, D)

        # Pass through main blocks (standard ViT encoder, no mask)
        for block in self.blocks:
            x = block(x, mask=None) # (B, N, D)

        # Store the output after main blocks - this is the target intermediate representation
        intermediate_output = x

        # Pass through 'decoder' blocks (acting as further encoding layers, no mask)
        for block in self.decoder_blocks:
            x = block(x, mask=None) # (B, N, D)

        # Return the intermediate representation and the final normalized output
        # The target for the student is intermediate_output (before normalization)
        # The second element is the final output after decoder blocks and normalization
        return intermediate_output, self.norm(x) # Use self.norm on the final output
