import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalNoiseEmbedding(nn.Module):
    """
    Fixed sinusoidal embedding for noise level, similar to positional encoding in transformers.
    This eliminates learned parameters for the embedding, making the network lighter.
    """
    def __init__(self, dim=256, max_noise=1.0, min_period=1.0):
        super().__init__()
        self.dim = dim
        self.max_noise = max_noise
        self.min_period = min_period
        
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
    This helps the network understand spatial relationships in the latent space.
    """
    def __init__(self, channels=16, height=64, width=64, temperature=10000):
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
        
        # Only use channels we have
        features_per_dim = channels // 2
        if features_per_dim < 1:
            features_per_dim = 1
            
        # Create appropriate number of frequencies
        freq_bands = torch.exp(
            torch.linspace(
                0, math.log(self.temperature), features_per_dim
            )
        )
        
        # For each frequency
        for feat_idx in range(features_per_dim):
            if feat_idx * 2 >= channels:
                break
                
            freq = freq_bands[feat_idx]
            
            # Calculate sin/cos for y-dimension
            y_sin = torch.sin(y_pos * freq)
            y_cos = torch.cos(y_pos * freq)
            
            # Fill y-dimension embeddings
            for x in range(width):
                pos_emb[feat_idx * 2, :, x] = y_sin
                
            if feat_idx * 2 + 1 < channels:
                for x in range(width):
                    pos_emb[feat_idx * 2 + 1, :, x] = y_cos
            
        # Calculate sin/cos for x-dimension (in remaining channels)
        x_offset = features_per_dim * 2
        remaining_features = (channels - x_offset) // 2
        
        if remaining_features > 0:
            # Create appropriate number of frequencies for remaining channels
            freq_bands = torch.exp(
                torch.linspace(
                    0, math.log(self.temperature), remaining_features
                )
            )
            
            # For each frequency
            for feat_idx in range(remaining_features):
                if x_offset + feat_idx * 2 >= channels:
                    break
                    
                freq = freq_bands[feat_idx]
                
                # Calculate sin/cos for x-dimension
                x_sin = torch.sin(x_pos * freq)
                x_cos = torch.cos(x_pos * freq)
                
                # Fill x-dimension embeddings
                for y in range(height):
                    pos_emb[x_offset + feat_idx * 2, y, :] = x_sin
                    
                if x_offset + feat_idx * 2 + 1 < channels:
                    for y in range(height):
                        pos_emb[x_offset + feat_idx * 2 + 1, y, :] = x_cos
                
        return pos_emb
        
    def forward(self, x):
        """
        Concatenate positional embeddings to the input tensor along the channel dimension.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Output tensor with positional embeddings concatenated [B, C+pos_C, H, W]
        """
        batch_size, channels, height, width = x.shape
        
        # Ensure dimensions match by using interpolation if needed
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


class ResidualBlock(nn.Module):
    """
    Residual block with noise level conditioning.
    """
    def __init__(self, in_channels, out_channels, noise_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.activation1 = nn.SiLU()
        
        # Simple projection for the noise embedding
        self.noise_proj = nn.Linear(noise_dim, out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.activation2 = nn.SiLU()
        
        # Residual connection if dimensions don't match
        self.skip_conv = nn.Identity()
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x, noise_embedding):
        # Save residual
        residual = self.skip_conv(x)
        
        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation1(h)
        
        # Apply noise conditioning
        noise_cond = self.noise_proj(noise_embedding)
        noise_cond = noise_cond.view(*noise_cond.shape, 1, 1)  # Add spatial dims
        h = h + noise_cond
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Add residual and apply final activation
        h = h + residual
        h = self.activation2(h)
        
        return h


class LatentDegradationNetwork(nn.Module):
    """
    A lightweight CNN that mimics image degradations in the latent space of Stable Diffusion.
    Input and output are both of shape [B, 4, 64, 64].
    The network is conditioned on a noise level.
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 base_channels=64, 
                 noise_embed_dim=256, 
                 num_blocks=3,
                 max_noise=1.0):
        super().__init__()
        
        # Store config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        
        # Fixed sinusoidal noise embedding (no learnable parameters)
        self.noise_embedding = SinusoidalNoiseEmbedding(dim=noise_embed_dim, max_noise=max_noise)
        
        # Spatial positional embedding (will be concatenated)
        self.pos_embedding = SpatialPositionalEmbedding(channels=16)  # Using smaller channel count
        
        # Initial convolution (accounting for concatenated positional embeddings)
        self.initial_conv = nn.Conv2d(in_channels + 16, base_channels, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
        # Encoder blocks (downsampling)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        enc_channels = [base_channels]
        
        for i in range(num_blocks):
            # Double the channels in each block (up to a limit)
            out_ch = min(base_channels * (2 ** (i + 1)), 512)
            enc_channels.append(out_ch)
            
            # Encoder block with residual connections and noise conditioning
            self.encoders.append(ResidualBlock(
                enc_channels[i], enc_channels[i+1], noise_dim=noise_embed_dim
            ))
            
            # Downsampling via stride-2 convolution
            self.downs.append(nn.Conv2d(
                enc_channels[i+1], enc_channels[i+1], kernel_size=4, stride=2, padding=1
            ))
            
        # Middle block
        self.middle = ResidualBlock(
            enc_channels[-1], enc_channels[-1], noise_dim=noise_embed_dim
        )
        
        # Decoder blocks (upsampling)
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for i in range(num_blocks):
            # Input channels include skip connection channels
            in_ch = enc_channels[-i-1] + enc_channels[-i-2]
            out_ch = enc_channels[-i-2]
            
            # Upsampling via transposed convolution
            self.ups.append(nn.ConvTranspose2d(
                enc_channels[-i-1], enc_channels[-i-1], kernel_size=4, stride=2, padding=1
            ))
            
            # Decoder block with residual connections and noise conditioning
            self.decoders.append(ResidualBlock(
                in_ch, out_ch, noise_dim=noise_embed_dim
            ))
            
        # Skip connection from input to output
        self.skip_scale = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
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
        
        # Get noise embedding
        noise_embedding = self.noise_embedding(noise_level)
        
        # Add spatial positional embeddings (concatenate)
        x = self.pos_embedding(x)
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder path
        skip_connections = [x]
        for i in range(self.num_blocks):
            # Apply encoder block
            x = self.encoders[i](x, noise_embedding)
            skip_connections.append(x)
            # Apply downsampling
            x = self.downs[i](x)
            
        # Middle block
        x = self.middle(x, noise_embedding)
        
        # Decoder path
        for i in range(self.num_blocks):
            # Apply upsampling
            x = self.ups[i](x)
            # Get skip connection and concatenate
            skip = skip_connections[-(i+2)]  # -1 is last, -2 is second last, etc.
            
            # Ensure spatial dimensions match (needed for odd input sizes)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Apply decoder block
            x = self.decoders[i](x, noise_embedding)
            
        # Final convolution to get back to output channels
        x = self.final_conv(x)
        
        # Add weighted skip connection from original input
        x = x + original_input * self.skip_scale
        
        return x


if __name__ == "__main__":
    from silo_utils import get_model_size
    batch_size = 2
    latents = torch.randn(batch_size, 4, 64, 64)

    # Create noise levels for each image in batch
    noise_levels = torch.tensor([0.1, 0.5])

    # Create model with smaller depth for testing
    model = LatentDegradationNetwork(
        in_channels=4,
        out_channels=4,
        base_channels=64,
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