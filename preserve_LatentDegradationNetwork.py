import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalNoiseEmbedding(nn.Module):
    """
    Fixed sinusoidal embedding for noise level, similar to positional encoding in transformers.
    No learnable parameters.
    """
    def __init__(self, dim=128, max_noise=1.0):
        super().__init__()
        self.dim = dim
        self.max_noise = max_noise
        
    def forward(self, noise_level):
        # Ensure noise_level is in correct shape [B, 1]
        if noise_level.dim() == 1:
            noise_level = noise_level.unsqueeze(1)
        
        # Normalize noise level to [0, 1] range
        noise_level = noise_level / self.max_noise
        
        # Create sinusoidal embedding
        half_dim = self.dim // 2
        embeddings = torch.zeros(noise_level.shape[0], self.dim, device=noise_level.device)
        
        # Calculate frequency bands
        frequencies = torch.exp(
            torch.arange(0, half_dim, device=noise_level.device) * (-math.log(10000.0) / (half_dim - 1))
        )
        
        # Calculate input for sin and cos
        args = noise_level * frequencies.unsqueeze(0)
        
        # Apply sin to even indices and cos to odd indices
        embeddings[:, 0::2] = torch.sin(args)
        embeddings[:, 1::2] = torch.cos(args)
        
        return embeddings


class SpatialPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embedding for 2D spatial positions.
    No learnable parameters.
    """
    def __init__(self, channels=8, height=64, width=64, temperature=10000):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.temperature = temperature
        
        # Create and register position encoding - not a parameter
        pos_emb = self._create_embedding(height, width, channels)
        self.register_buffer('positional_embedding', pos_emb)
        
    def _create_embedding(self, height, width, channels):
        """Create the sinusoidal position encoding."""
        # Create position indices
        y_pos = torch.arange(height).float()
        x_pos = torch.arange(width).float()
        
        # Normalize positions to [0, 1]
        y_pos = y_pos / (height - 1)
        x_pos = x_pos / (width - 1)
        
        # Prepare output tensor [C, H, W]
        pos_emb = torch.zeros(channels, height, width)
        
        # Split channels between x and y dimensions
        y_channels = channels // 2
        x_channels = channels - y_channels
        
        # Create frequency bands for y dimension
        y_freq_bands = torch.exp(
            torch.linspace(0, math.log(self.temperature), y_channels // 2)
        )
        
        # Create embeddings for y dimension
        for i in range(y_channels // 2):
            freq = y_freq_bands[i]
            y_sin = torch.sin(y_pos * freq)
            y_cos = torch.cos(y_pos * freq)
            
            for x in range(width):
                pos_emb[2*i, :, x] = y_sin
                pos_emb[2*i+1, :, x] = y_cos
        
        # Create frequency bands for x dimension
        x_freq_bands = torch.exp(
            torch.linspace(0, math.log(self.temperature), x_channels // 2)
        )
        
        # Create embeddings for x dimension
        for i in range(x_channels // 2):
            freq = x_freq_bands[i]
            x_sin = torch.sin(x_pos * freq)
            x_cos = torch.cos(x_pos * freq)
            
            for y in range(height):
                pos_emb[y_channels + 2*i, y, :] = x_sin
                pos_emb[y_channels + 2*i+1, y, :] = x_cos
                
        return pos_emb
        
    def forward(self, x):
        """
        Concatenate positional embeddings to the input tensor.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Output tensor with positional embeddings concatenated [B, C+pos_C, H, W]
        """
        batch_size, channels, height, width = x.shape
        
        # Handle different spatial dimensions if needed
        if height != self.height or width != self.width:
            pos_emb = F.interpolate(
                self.positional_embedding.unsqueeze(0),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            pos_emb = self.positional_embedding
            
        # Expand positional embedding to batch dimension
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Concatenate along channel dimension
        return torch.cat([x, pos_emb], dim=1)


class NoiseConditionedConvBlock(nn.Module):
    """
    Convolutional block with noise level conditioning.
    Uses same-padding to maintain spatial dimensions.
    """
    def __init__(self, in_channels, out_channels, noise_dim=128, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = kernel_size // 2  # Same padding
        
        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        # Noise conditioning
        self.noise_proj = nn.Linear(noise_dim, out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        # Residual connection (1x1 conv if needed)
        self.skip_conv = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Activation
        self.act = nn.SiLU()
        
    def forward(self, x, noise_embedding):
        # Save input for residual connection
        residual = self.skip_conv(x)
        
        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Add noise conditioning
        noise_cond = self.noise_proj(noise_embedding)  # [B, out_channels]
        noise_cond = noise_cond.view(noise_cond.shape[0], noise_cond.shape[1], 1, 1)  # [B, out_channels, 1, 1]
        h = h + noise_cond  # Scale and add as bias
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Residual connection
        output = h + residual
        output = self.act(output)
        
        return output


class LatentDegradationNetwork(nn.Module):
    """
    Lightweight CNN for mimicking image degradations in the latent space.
    Maintains spatial dimensions throughout and uses skip connections for stable gradients.
    Input and output are both of shape [B, 4, 64, 64].
    """
    def __init__(self,
                 in_channels=4,
                 out_channels=4,
                 base_channels=32,  # Reduced from typical 64 to keep model small
                 noise_embed_dim=128,
                 pos_embed_channels=8,
                 num_blocks=3,
                 max_noise=1.0):
        super().__init__()
        
        # Configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Embeddings (no learnable parameters)
        self.noise_embedding = SinusoidalNoiseEmbedding(dim=noise_embed_dim, max_noise=max_noise)
        self.pos_embedding = SpatialPositionalEmbedding(channels=pos_embed_channels)
        
        # Initial projection conv (preserve dimensions)
        self.initial_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),  # 1x1 conv for color preservation
            nn.SiLU()
        )
        
        # Initial and final convolutions
        self.initial_conv = nn.Conv2d(in_channels + pos_embed_channels, base_channels, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
        # Main processing blocks that maintain spatial dimensions
        channels = [base_channels]
        for i in range(num_blocks):
            channels.append(min(base_channels * (2**(i)), 128))  # Cap at 128 to keep model small
        
        # Processing blocks with residual connections
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(NoiseConditionedConvBlock(
                channels[i], channels[i+1], noise_dim=noise_embed_dim
            ))
        
        # Bottleneck block
        self.bottleneck = NoiseConditionedConvBlock(
            channels[-1], channels[-1], noise_dim=noise_embed_dim
        )
        
        # Upsampling blocks with skip connections
        self.up_blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_ch = channels[num_blocks-i] + channels[num_blocks-i-1] if i > 0 else channels[num_blocks-i]
            out_ch = channels[num_blocks-i-1]
            self.up_blocks.append(NoiseConditionedConvBlock(
                in_ch, out_ch, noise_dim=noise_embed_dim
            ))
        
        # Learnable scaling for residual connection
        self.skip_scale = nn.Parameter(torch.ones(1))
        
        # Identity path - direct path from input to output
        # This helps the model learn to easily preserve colors in identity case
        self.identity_gate = nn.Sequential(
            nn.Linear(noise_embed_dim, 1),
            nn.Sigmoid()
        )
    
    def _init_weights(self):
        """Initialize weights for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, noise_level):
        """
        Forward pass through the network.
        
        Args:
            x: Latent image [B, 4, 64, 64]
            noise_level: Scalar noise level [B] or [B, 1]
            
        Returns:
            Degraded latent image [B, 4, 64, 64]
        """
        # Save original input for residual connection
        original_input = x
        
        # Apply initial projection to help with color preservation
        color_projected = self.initial_proj(x)
        
        # Get noise embedding
        noise_embedding = self.noise_embedding(noise_level)
        
        # Add spatial positional embeddings (concatenate)
        x = self.pos_embedding(x)
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Processing path with skip connections
        skip_connections = []
        for block in self.blocks:
            skip_connections.append(x)
            x = block(x, noise_embedding)
        
        # Bottleneck
        x = self.bottleneck(x, noise_embedding)
        
        # Upsampling path with skip connections
        for i, up_block in enumerate(self.up_blocks):
            if i > 0:  # Skip connection for all but the first up block
                skip = skip_connections[len(skip_connections)-i-1]
                x = torch.cat([x, skip], dim=1)
            x = up_block(x, noise_embedding)
        
        # Final convolution with additional color projection for better preservation
        x = self.final_conv(x)
        
        # Dynamic identity gating based on noise level
        # This creates a shortcut for identity mapping when noise is low
        identity_weight = self.identity_gate(noise_embedding)  # [B, 1]
        identity_weight = identity_weight.view(-1, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting
        process_weight = 1.0 - identity_weight
        
        # Combine processed path and color-projected input (helps maintain color accuracy)
        x = process_weight * x + identity_weight * color_projected
        
        # Add weighted original input for residual learning
        x = x + original_input * self.skip_scale
        
        return x


# Example usage
if __name__ == "__main__":
    # Create a batch of 2 latent images
    from silo_utils import get_model_size
    batch_size = 2
    latents = torch.randn(batch_size, 4, 64, 64)
    
    # Create noise levels for each image in batch
    noise_levels = torch.tensor([0.1, 0.5])
    
    # Create model
    model = LatentDegradationNetwork(
        in_channels=4,
        out_channels=4,
        base_channels=24,  # Smaller base channels
        num_blocks=2
    )
    
    # Forward pass
    degraded_latents = model(latents, noise_levels)
    
    # Check output shape
    print(f"Input shape: {latents.shape}")
    print(f"Output shape: {degraded_latents.shape}")
    
    # Model parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters")
    print(f"Model size is {get_model_size(model)}")