#!/usr/bin/env python3
"""
PeakNet Utilities for Profiling and Performance Testing

Provides modified PeakNet models optimized for different profiling scenarios,
particularly for testing memory transfer patterns and pipeline performance.
"""

import torch
from torch import nn
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from omegaconf import OmegaConf

# PeakNet imports
try:
    from peaknet.modeling.convnextv2_bifpn_net import (
        PeakNet, PeakNetConfig, SegHeadConfig
    )
    from peaknet.modeling.bifpn_config import (
        BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
    )
    from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
    PEAKNET_AVAILABLE = True
except ImportError:
    PEAKNET_AVAILABLE = False
    print("Warning: PeakNet not available. Make sure peaknet is installed and accessible")


class PeakNetForProfiling(nn.Module):
    """
    PeakNet wrapper optimized for profiling pipeline performance.

    Wraps a PeakNet model to provide consistent interface with the inference
    pipeline and optimize for profiling different memory transfer patterns.
    """

    def __init__(self, peaknet_model: nn.Module, num_classes: int = 2):
        """
        Initialize wrapper around PeakNet model.

        Args:
            peaknet_model: Initialized PeakNet model
            num_classes: Number of classes (default: 2 for peak/background)
        """
        super().__init__()
        self.peaknet_model = peaknet_model
        # Get num_classes from PeakNet model config (correct access path)
        try:
            if hasattr(peaknet_model, 'config') and hasattr(peaknet_model.config, 'seg_head'):
                self.num_classes = peaknet_model.config.seg_head.num_classes
                print(f"✓ Got num_classes from model config: {self.num_classes}")
            else:
                self.num_classes = num_classes
                print(f"⚠ Could not access model.config.seg_head.num_classes, using default: {num_classes}")
        except Exception as e:
            self.num_classes = num_classes
            print(f"⚠ Error accessing model config ({e}), using default num_classes: {num_classes}")
        
        # Add device verification
        try:
            model_device = next(peaknet_model.parameters()).device
            if model_device.type == 'cuda':
                print(f"✓ PeakNet model on GPU: {model_device}")
            else:
                print(f"⚠ PeakNet model on CPU: {model_device}")
        except Exception as e:
            print(f"⚠ Could not determine model device: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns segmentation output.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Segmentation output of shape [batch_size, num_classes, height, width]
        """
        # Run PeakNet inference
        output = self.peaknet_model(x)
        return output

    def eval(self):
        """Set model to evaluation mode"""
        self.peaknet_model.eval()
        return self

    def to(self, device):
        """Move model to device"""
        self.peaknet_model = self.peaknet_model.to(device)
        return self


def load_peaknet_from_yaml(yaml_path: str, weights_path: Optional[str] = None) -> PeakNet:
    """
    Load PeakNet model from YAML configuration file.

    Args:
        yaml_path: Path to YAML configuration file
        weights_path: Optional path to pre-trained weights

    Returns:
        Configured PeakNet model
    """
    if not PEAKNET_AVAILABLE:
        raise ImportError("PeakNet not available. Please install peaknet package")

    print(f"Loading PeakNet configuration from {yaml_path}")

    # Load YAML configuration
    config = OmegaConf.load(yaml_path)
    model_params = config.get("model")

    if model_params is None:
        raise ValueError(f"No 'model' section found in {yaml_path}")

    # Extract configuration parameters
    backbone_params = model_params.get("backbone", {})
    hf_model_config = backbone_params.get("hf_config", {})
    bifpn_params = model_params.get("bifpn", {})
    bifpn_block_params = bifpn_params.get("block", {})
    bifpn_block_bn_params = bifpn_block_params.get("bn", {})
    bifpn_block_fusion_params = bifpn_block_params.get("fusion", {})
    seghead_params = model_params.get("seg_head", {})

    # Build model configuration objects
    # Convert OmegaConf to regular dicts to avoid issues
    hf_model_config_dict = OmegaConf.to_container(hf_model_config, resolve=True)
    backbone_config = ConvNextV2Config(**hf_model_config_dict)

    # BiFPN configuration
    bifpn_block_bn_params_dict = OmegaConf.to_container(bifpn_block_bn_params, resolve=True)
    bifpn_block_fusion_params_dict = OmegaConf.to_container(bifpn_block_fusion_params, resolve=True)
    bifpn_block_params_dict = OmegaConf.to_container(bifpn_block_params, resolve=True)
    bifpn_params_dict = OmegaConf.to_container(bifpn_params, resolve=True)

    bifpn_block_params_dict["bn"] = BNConfig(**bifpn_block_bn_params_dict)
    bifpn_block_params_dict["fusion"] = FusionConfig(**bifpn_block_fusion_params_dict)
    bifpn_params_dict["block"] = BiFPNBlockConfig(**bifpn_block_params_dict)
    bifpn_config = BiFPNConfig(**bifpn_params_dict)

    # Segmentation head configuration
    seghead_params_dict = OmegaConf.to_container(seghead_params, resolve=True)
    seghead_config = SegHeadConfig(**seghead_params_dict)

    # Create PeakNet configuration
    peaknet_config = PeakNetConfig(
        backbone=backbone_config,
        bifpn=bifpn_config,
        seg_head=seghead_config,
    )

    # Create model
    model = PeakNet(peaknet_config)
    model.init_weights()

    # Load weights if provided
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"PeakNet model loaded: {num_params/1e6:.1f}M parameters")
    print(f"Backbone: ConvNextV2 {backbone_config.hidden_sizes}")
    print(f"BiFPN: {bifpn_config.num_blocks} blocks, {bifpn_config.block.num_features} features")
    print(f"Input size: {backbone_config.image_size}×{backbone_config.image_size}")

    return model


def create_peaknet_for_profiling(
    yaml_path: str,
    weights_path: Optional[str] = None,
    device: str = 'cuda:0'
) -> PeakNetForProfiling:
    """
    Convenience function to create a PeakNet model optimized for profiling.

    Args:
        yaml_path: Path to YAML configuration file
        weights_path: Optional path to pre-trained weights
        device: Device to place model on

    Returns:
        PeakNetForProfiling model ready for inference
    """
    # Load PeakNet model
    peaknet_model = load_peaknet_from_yaml(yaml_path, weights_path)

    # Get num_classes from YAML config
    config = OmegaConf.load(yaml_path)
    model_params = config.get("model", {})
    seghead_params = model_params.get("seg_head", {})
    num_classes = seghead_params.get("num_classes", 2)

    # Create profiling wrapper
    model = PeakNetForProfiling(peaknet_model, num_classes=num_classes)
    model = model.to(device)

    # Set to eval mode for consistent timing
    model.eval()

    return model


def get_peaknet_output_shape(yaml_path: str, batch_size: int = 1) -> Tuple[int, ...]:
    """
    Calculate the output shape for PeakNet segmentation.

    Args:
        yaml_path: Path to YAML configuration file
        batch_size: Batch size

    Returns:
        tuple: Output shape (batch_size, num_classes, height, width)
    """
    # Load configuration to get parameters
    config = OmegaConf.load(yaml_path)
    model_params = config.get("model", {})

    # Get segmentation head info
    seghead_params = model_params.get("seg_head", {})
    num_classes = seghead_params.get("num_classes", 2)

    # Get backbone info for image size
    backbone_params = model_params.get("backbone", {})
    hf_config = backbone_params.get("hf_config", {})
    image_size = hf_config.get("image_size", 512)

    return (batch_size, num_classes, image_size, image_size)


def get_peaknet_input_shape(yaml_path: str, batch_size: int = 1) -> Tuple[int, ...]:
    """
    Calculate the input shape for PeakNet.

    Args:
        yaml_path: Path to YAML configuration file  
        batch_size: Batch size

    Returns:
        tuple: Input shape (batch_size, channels, height, width)
    """
    # Load configuration to get parameters
    config = OmegaConf.load(yaml_path)
    model_params = config.get("model", {})

    # Get backbone info
    backbone_params = model_params.get("backbone", {})
    hf_config = backbone_params.get("hf_config", {})
    num_channels = hf_config.get("num_channels", 1)
    image_size = hf_config.get("image_size", 512)

    return (batch_size, num_channels, image_size, image_size)


def estimate_transfer_size(yaml_path: str, batch_size: int = 1) -> Dict[str, Any]:
    """
    Estimate the D2H transfer size for PeakNet segmentation output.

    Args:
        yaml_path: Path to YAML configuration file
        batch_size: Batch size

    Returns:
        dict: Transfer size information
    """
    output_shape = get_peaknet_output_shape(yaml_path, batch_size)
    input_shape = get_peaknet_input_shape(yaml_path, batch_size)

    output_elements = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
    input_elements = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]

    output_size_bytes = output_elements * 4  # float32 = 4 bytes
    output_size_mb = output_size_bytes / (1024 * 1024)

    input_size_bytes = input_elements * 4  # float32 = 4 bytes  
    input_size_mb = input_size_bytes / (1024 * 1024)

    return {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'input_size_bytes': input_size_bytes,
        'input_size_mb': input_size_mb,
        'output_size_bytes': output_size_bytes,
        'output_size_mb': output_size_mb,
        'total_transfer_mb': input_size_mb + output_size_mb
    }


if __name__ == '__main__':
    """Demo and testing"""
    if PEAKNET_AVAILABLE:
        print("=== PeakNet Utils Demo ===")

        # Test with a sample configuration file path
        # You would replace this with an actual path to a PeakNet config
        sample_yaml_path = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet/convnext_seg_config.yaml"

        if os.path.exists(sample_yaml_path):
            print(f"Using configuration: {sample_yaml_path}")

            # Test shape calculation
            batch_size = 4
            input_shape = get_peaknet_input_shape(sample_yaml_path, batch_size)
            output_shape = get_peaknet_output_shape(sample_yaml_path, batch_size)

            print(f"\nBatch size: {batch_size}")
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output_shape}")

            # Test transfer size estimation
            transfer_info = estimate_transfer_size(sample_yaml_path, batch_size)
            print(f"\nTransfer size estimation:")
            print(f"  Input: {transfer_info['input_size_mb']:.2f} MB")
            print(f"  Output: {transfer_info['output_size_mb']:.2f} MB")
            print(f"  Total: {transfer_info['total_transfer_mb']:.2f} MB")

            # Test model creation if CUDA available
            if torch.cuda.is_available():
                print(f"\nTesting model creation...")
                try:
                    model = create_peaknet_for_profiling(sample_yaml_path)
                    print(f"Model created successfully on {next(model.parameters()).device}")

                    # Test forward pass with random data
                    C, H, W = input_shape[1], input_shape[2], input_shape[3]
                    test_input = torch.randn(2, C, H, W).cuda()
                    with torch.no_grad():
                        output = model(test_input)
                        print(f"Forward pass successful: {test_input.shape} -> {output.shape}")
                except Exception as e:
                    print(f"Model test failed: {e}")
            else:
                print("CUDA not available, skipping model test")
        else:
            print(f"Sample configuration file not found: {sample_yaml_path}")
            print("Please provide a valid PeakNet YAML configuration file path")
    else:
        print("PeakNet not available, skipping demo")
