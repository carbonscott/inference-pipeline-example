#!/usr/bin/env python3
"""
ViT Utilities for Profiling and Performance Testing

Provides modified ViT models optimized for different profiling scenarios,
particularly for testing memory transfer patterns and pipeline performance.
"""

import torch
from torch import nn
from einops import repeat

try:
    from vit_pytorch import ViT
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False
    print("Warning: vit-pytorch not available. Install with: pip install vit-pytorch")


class ViTForProfiling(ViT):
    """
    ViT variant optimized for profiling D2H memory transfers.

    Instead of returning classification output [batch_size, num_classes],
    returns the full transformer output [batch_size, num_patches + 1, dim].

    This creates much larger D2H transfers, useful for:
    - Testing memory bandwidth bottlenecks
    - Profiling pipeline performance with substantial D2H workloads
    - Understanding NUMA effects on large data transfers
    """

    def forward(self, img):
        """
        Forward pass that returns transformer output directly.

        Args:
            img: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Transformer output of shape [batch_size, num_patches + 1, dim]
            Instead of classification output [batch_size, num_classes]
        """
        # Standard ViT preprocessing
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Run transformer and return output directly (no classification head)
        return self.transformer(x)


def create_vit_for_profiling(
    image_size,
    patch_size,
    dim,
    depth,
    heads,
    mlp_dim,
    channels=3,
    dropout=0.0,
    emb_dropout=0.0,
    device='cuda:0'
):
    """
    Convenience function to create a ViT model optimized for profiling.

    Args:
        image_size: Size of input images
        patch_size: Size of patches
        dim: Embedding dimension
        depth: Number of transformer layers
        heads: Number of attention heads
        mlp_dim: MLP hidden dimension
        channels: Number of input channels
        dropout: Dropout rate
        emb_dropout: Embedding dropout rate
        device: Device to place model on

    Returns:
        ViTForProfiling model ready for inference
    """
    if not VIT_AVAILABLE:
        raise ImportError("vit-pytorch not available. Install with: pip install vit-pytorch")

    model = ViTForProfiling(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=1000,  # Not used since we return transformer output
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        channels=channels,
        dropout=dropout,
        emb_dropout=emb_dropout
    ).to(device)

    # Set to eval mode for consistent timing
    model.eval()

    return model


def get_output_shape(image_size, patch_size, dim, batch_size=1):
    """
    Calculate the output shape for ViTForProfiling.

    Args:
        image_size: Size of input images
        patch_size: Size of patches  
        dim: Embedding dimension
        batch_size: Batch size

    Returns:
        tuple: Output shape (batch_size, num_patches + 1, dim)
    """
    num_patches = (image_size // patch_size) ** 2
    return (batch_size, num_patches + 1, dim)


def estimate_transfer_size(image_size, patch_size, dim, batch_size=1):
    """
    Estimate the D2H transfer size for ViTForProfiling output.

    Args:
        image_size: Size of input images
        patch_size: Size of patches
        dim: Embedding dimension
        batch_size: Batch size

    Returns:
        dict: Transfer size information
    """
    shape = get_output_shape(image_size, patch_size, dim, batch_size)
    num_elements = shape[0] * shape[1] * shape[2]
    size_bytes = num_elements * 4  # float32 = 4 bytes
    size_mb = size_bytes / (1024 * 1024)

    return {
        'shape': shape,
        'num_elements': num_elements,
        'size_bytes': size_bytes,
        'size_mb': size_mb
    }


if __name__ == '__main__':
    """Demo and testing"""
    if VIT_AVAILABLE:
        print("=== ViT Utils Demo ===")

        # Test configuration
        config = {
            'image_size': 224,
            'patch_size': 16,
            'dim': 768,
            'depth': 12,
            'heads': 12,
            'mlp_dim': 3072,
            'batch_size': 4
        }

        # Show transfer size comparison
        print(f"Configuration: {config}")

        transfer_info = estimate_transfer_size(**config)
        print(f"\nViTForProfiling output:")
        print(f"  Shape: {transfer_info['shape']}")
        print(f"  Transfer size: {transfer_info['size_mb']:.2f} MB")

        # Compare with standard classification output
        standard_shape = (config['batch_size'], 1000)
        standard_size_mb = (standard_shape[0] * standard_shape[1] * 4) / (1024 * 1024)
        print(f"\nStandard ViT classification output:")
        print(f"  Shape: {standard_shape}")
        print(f"  Transfer size: {standard_size_mb:.4f} MB")

        ratio = transfer_info['size_mb'] / standard_size_mb
        print(f"\nTransfer size ratio: {ratio:.1f}x larger")

        # Test model creation if CUDA available
        if torch.cuda.is_available():
            print(f"\nTesting model creation...")
            model = create_vit_for_profiling(**{k: v for k, v in config.items() if k != 'batch_size'})
            print(f"Model created successfully on {next(model.parameters()).device}")

            # Test forward pass
            test_input = torch.randn(2, 3, 224, 224).cuda()
            with torch.no_grad():
                output = model(test_input)
                print(f"Forward pass successful: {test_input.shape} -> {output.shape}")
        else:
            print("CUDA not available, skipping model test")
    else:
        print("vit-pytorch not available, skipping demo")
