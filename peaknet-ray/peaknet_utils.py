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




def create_peaknet_model(
    peaknet_config: dict,
    weights_path: Optional[str] = None,
    device: str = 'cuda:0'
) -> PeakNetForProfiling:
    """
    Create PeakNet model from Hydra configuration.
    
    Args:
        peaknet_config: PeakNet configuration dict with model parameters
        weights_path: Optional path to pre-trained weights
        device: Device to place model on
        
    Returns:
        PeakNetForProfiling model ready for inference
    """
    if not PEAKNET_AVAILABLE:
        raise ImportError("PeakNet not available. Please install peaknet package")

    print(f"Creating PeakNet model from Hydra configuration")

    # Extract model configuration from PeakNet config
    model_config = peaknet_config.get("model", {})
    
    # Extract backbone configuration
    backbone_params = model_config.get("backbone", {})
    hf_model_config = backbone_params.get("hf_config", {})
    
    # Extract BiFPN configuration
    bifpn_params = model_config.get("bifpn", {})
    bifpn_block_params = bifpn_params.get("block", {})
    bifpn_block_bn_params = bifpn_block_params.get("bn", {})
    bifpn_block_fusion_params = bifpn_block_params.get("fusion", {})
    
    # Extract segmentation head configuration
    seghead_params = model_config.get("seg_head", {})
    
    print(f"Model image_size: {hf_model_config.get('image_size', 512)}")
    print(f"Model num_channels: {hf_model_config.get('num_channels', 1)}")
    print(f"Model num_classes: {seghead_params.get('num_classes', 2)}")

    # Build model configuration objects (same as load_peaknet_from_yaml)
    # Convert OmegaConf to regular Python dict to avoid ListConfig issues
    hf_model_config_dict = OmegaConf.to_container(hf_model_config, resolve=True)
    backbone_config = ConvNextV2Config(**hf_model_config_dict)
    
    # BiFPN configuration - convert OmegaConf to regular dicts
    bifpn_block_bn_params_dict = OmegaConf.to_container(bifpn_block_bn_params, resolve=True)
    bifpn_block_fusion_params_dict = OmegaConf.to_container(bifpn_block_fusion_params, resolve=True)
    bifpn_block_bn_config = BNConfig(**bifpn_block_bn_params_dict)
    bifpn_block_fusion_config = FusionConfig(**bifpn_block_fusion_params_dict)
    
    # Update block params with config objects
    bifpn_block_params_dict = OmegaConf.to_container(bifpn_block_params, resolve=True)
    bifpn_block_params_dict["bn"] = bifpn_block_bn_config
    bifpn_block_params_dict["fusion"] = bifpn_block_fusion_config
    bifpn_block_config = BiFPNBlockConfig(**bifpn_block_params_dict)
    
    # Update BiFPN params with block config
    bifpn_params_dict = OmegaConf.to_container(bifpn_params, resolve=True)
    bifpn_params_dict["block"] = bifpn_block_config
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
    image_size = hf_model_config.get('image_size', 512)
    print(f"PeakNet model created: {num_params/1e6:.1f}M parameters")
    print(f"Backbone: ConvNextV2 {backbone_config.hidden_sizes}")
    print(f"BiFPN: {bifpn_config.num_blocks} blocks, {bifpn_config.block.num_features} features")
    print(f"Input size: {image_size}×{image_size}")

    # Get num_classes for wrapper
    num_classes = seghead_params.get("num_classes", 2)
    
    # Create profiling wrapper
    wrapper = PeakNetForProfiling(model, num_classes=num_classes)
    wrapper = wrapper.to(device)
    
    # Set to eval mode for consistent timing
    wrapper.eval()

    return wrapper


def get_peaknet_shapes(peaknet_config: dict, batch_size: int = 1) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Calculate input and output shapes from PeakNet configuration.
    
    Args:
        peaknet_config: PeakNet configuration dict with model parameters
        batch_size: Batch size
        
    Returns:
        tuple: (input_shape, output_shape) both as (batch_size, channels, height, width)
    """
    model_config = peaknet_config.get("model", {})
    
    # Get input parameters
    backbone_params = model_config.get("backbone", {})
    hf_config = backbone_params.get("hf_config", {})
    num_channels = hf_config.get("num_channels", 1)
    image_size = hf_config.get("image_size", 512)
    
    # Get output parameters
    seghead_params = model_config.get("seg_head", {})
    num_classes = seghead_params.get("num_classes", 2)
    
    input_shape = (batch_size, num_channels, image_size, image_size)
    output_shape = (batch_size, num_classes, image_size, image_size)
    
    return input_shape, output_shape


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
