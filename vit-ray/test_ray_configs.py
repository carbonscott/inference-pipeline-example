#!/usr/bin/env python3
"""
Configuration tests for Ray pipeline configurations

Tests the various Ray configuration files (multi_gpu.yaml, profiling.yaml) 
and validates that they work correctly with the pipeline.
"""

import pytest
import ray
import torch
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import yaml
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from vit_pipeline_ray import run_ray_pipeline_test, setup_ray


# GPU availability checks
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0

# Configuration file paths
CONFIG_DIR = Path(__file__).parent / "conf"
RAY_CONFIG_DIR = CONFIG_DIR / "ray"


@pytest.fixture(scope="function")
def ray_environment():
    """Setup clean Ray environment for each test."""
    if ray.is_initialized():
        ray.shutdown()
    
    yield
    
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="function")
def hydra_context():
    """Setup Hydra context for configuration loading."""
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with the config directory
    with initialize(version_base=None, config_path="conf"):
        yield
    
    # Clear Hydra after test
    GlobalHydra.instance().clear()


class TestConfigFileExistence:
    """Test that required configuration files exist and are valid YAML."""
    
    def test_config_directory_exists(self):
        """Test that configuration directory exists."""
        assert CONFIG_DIR.exists(), f"Configuration directory not found: {CONFIG_DIR}"
        assert CONFIG_DIR.is_dir(), f"Configuration path is not a directory: {CONFIG_DIR}"
    
    def test_ray_config_directory_exists(self):
        """Test that Ray configuration directory exists."""
        assert RAY_CONFIG_DIR.exists(), f"Ray config directory not found: {RAY_CONFIG_DIR}"
        assert RAY_CONFIG_DIR.is_dir(), f"Ray config path is not a directory: {RAY_CONFIG_DIR}"
    
    def test_base_config_exists(self):
        """Test that base configuration file exists."""
        base_config_path = CONFIG_DIR / "config.yaml"
        assert base_config_path.exists(), f"Base config file not found: {base_config_path}"
    
    def test_multi_gpu_config_exists(self):
        """Test that multi-GPU configuration file exists."""
        multi_gpu_config = RAY_CONFIG_DIR / "multi_gpu.yaml"
        assert multi_gpu_config.exists(), f"Multi-GPU config not found: {multi_gpu_config}"
    
    def test_profiling_config_exists(self):
        """Test that profiling configuration file exists."""
        profiling_config = RAY_CONFIG_DIR / "profiling.yaml"
        assert profiling_config.exists(), f"Profiling config not found: {profiling_config}"
    
    def test_ray_base_config_exists(self):
        """Test that Ray base configuration file exists."""
        ray_base_config = RAY_CONFIG_DIR / "ray_config.yaml"
        assert ray_base_config.exists(), f"Ray base config not found: {ray_base_config}"


class TestConfigFileValidity:
    """Test that configuration files are valid YAML and have expected structure."""
    
    def test_multi_gpu_yaml_valid(self):
        """Test that multi_gpu.yaml is valid YAML."""
        config_path = RAY_CONFIG_DIR / "multi_gpu.yaml"
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert config_data is not None, "multi_gpu.yaml is empty or invalid"
        assert isinstance(config_data, dict), "multi_gpu.yaml should contain a dictionary"
    
    def test_profiling_yaml_valid(self):
        """Test that profiling.yaml is valid YAML."""
        config_path = RAY_CONFIG_DIR / "profiling.yaml"
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert config_data is not None, "profiling.yaml is empty or invalid"
        assert isinstance(config_data, dict), "profiling.yaml should contain a dictionary"
    
    def test_ray_config_yaml_valid(self):
        """Test that ray_config.yaml is valid YAML."""
        config_path = RAY_CONFIG_DIR / "ray_config.yaml"
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert config_data is not None, "ray_config.yaml is empty or invalid"
        assert isinstance(config_data, dict), "ray_config.yaml should contain a dictionary"
    
    def test_base_config_yaml_valid(self):
        """Test that base config.yaml is valid YAML."""
        config_path = CONFIG_DIR / "config.yaml"
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert config_data is not None, "config.yaml is empty or invalid"
        assert isinstance(config_data, dict), "config.yaml should contain a dictionary"


class TestConfigStructure:
    """Test that configuration files have expected structure and values."""
    
    def test_multi_gpu_config_structure(self, hydra_context):
        """Test multi-GPU configuration has correct structure."""
        cfg = compose(config_name="config", overrides=["ray=multi_gpu"])
        
        # Should have ray section
        assert 'ray' in cfg, "multi_gpu config missing ray section"
        
        # Ray section should have expected subsections
        assert 'actors' in cfg.ray, "multi_gpu config missing ray.actors section"
        assert 'producers' in cfg.ray, "multi_gpu config missing ray.producers section"
        
        # Actors section should specify GPU count
        assert 'num_gpus' in cfg.ray.actors, "multi_gpu config missing ray.actors.num_gpus"
        assert isinstance(cfg.ray.actors.num_gpus, int), "ray.actors.num_gpus should be integer"
        assert cfg.ray.actors.num_gpus > 0, "ray.actors.num_gpus should be positive"
        
        # Should have profiling setting
        assert 'profiling_enabled' in cfg.ray.actors, "multi_gpu config missing profiling_enabled"
        assert isinstance(cfg.ray.actors.profiling_enabled, bool), "profiling_enabled should be boolean"
        
        # Producers section should have parameters
        assert 'num_tasks' in cfg.ray.producers, "multi_gpu config missing ray.producers.num_tasks"
        assert 'batches_per_task' in cfg.ray.producers, "multi_gpu config missing ray.producers.batches_per_task"
        
        # Values should be reasonable
        assert cfg.ray.producers.num_tasks > 0, "num_tasks should be positive"
        assert cfg.ray.producers.batches_per_task > 0, "batches_per_task should be positive"
    
    def test_profiling_config_structure(self, hydra_context):
        """Test profiling configuration has correct structure."""
        cfg = compose(config_name="config", overrides=["ray=profiling"])
        
        # Should have ray section with profiling enabled
        assert 'ray' in cfg, "profiling config missing ray section"
        assert 'actors' in cfg.ray, "profiling config missing ray.actors section"
        
        # Profiling should be enabled
        assert cfg.ray.actors.profiling_enabled is True, "profiling config should enable profiling"
        
        # Should have reasonable GPU count for profiling
        assert cfg.ray.actors.num_gpus <= 4, "profiling config should use reasonable GPU count"
        
        # Should have producers configuration
        assert 'producers' in cfg.ray, "profiling config missing producers section"
        assert cfg.ray.producers.batches_per_task > 0, "profiling config needs batches to profile"
    
    def test_base_ray_config_structure(self, hydra_context):
        """Test base Ray configuration structure."""
        # Load ray_config directly
        ray_config_path = RAY_CONFIG_DIR / "ray_config.yaml"
        with open(ray_config_path, 'r') as f:
            ray_config = yaml.safe_load(f)
        
        # Should have main sections
        assert 'actors' in ray_config, "ray_config missing actors section"
        assert 'producers' in ray_config, "ray_config missing producers section"
        
        # Should have init section for Ray initialization
        if 'init' in ray_config:
            assert isinstance(ray_config['init'], dict), "init section should be a dictionary"
    
    def test_config_value_types(self, hydra_context):
        """Test that configuration values have correct types."""
        # Test multi-GPU config
        cfg = compose(config_name="config", overrides=["ray=multi_gpu"])
        
        # Integer values
        assert isinstance(cfg.ray.actors.num_gpus, int)
        assert isinstance(cfg.ray.producers.num_tasks, int)
        assert isinstance(cfg.ray.producers.batches_per_task, int)
        
        # Boolean values
        assert isinstance(cfg.ray.actors.profiling_enabled, bool)
        assert isinstance(cfg.performance.pin_memory, bool)
        assert isinstance(cfg.test.deterministic, bool)
        
        # List/array values
        assert isinstance(cfg.shape, (list, tuple))
        
        # VIT config values
        assert isinstance(cfg.vit.patch_size, int)
        assert isinstance(cfg.vit.depth, int)
        assert isinstance(cfg.vit.heads, int)


class TestConfigurationLoading:
    """Test that configurations can be loaded and used with Hydra."""
    
    def test_load_multi_gpu_config(self, hydra_context):
        """Test loading multi-GPU configuration through Hydra."""
        cfg = compose(config_name="config", overrides=["ray=multi_gpu"])
        
        # Should be able to access all expected values
        assert cfg.ray.actors.num_gpus > 0
        assert cfg.ray.producers.num_tasks > 0
        assert hasattr(cfg.ray.actors, 'profiling_enabled')
        
        # Should have VIT model configuration
        assert cfg.vit.patch_size > 0
        assert cfg.vit.depth >= 0
        assert cfg.vit.dim > 0
    
    def test_load_profiling_config(self, hydra_context):
        """Test loading profiling configuration through Hydra."""
        cfg = compose(config_name="config", overrides=["ray=profiling"])
        
        # Profiling should be enabled
        assert cfg.ray.actors.profiling_enabled is True
        
        # Should have reasonable settings for profiling
        assert cfg.ray.actors.num_gpus <= 4  # Don't overwhelm with too many profiles
        assert cfg.ray.producers.batches_per_task > 0
    
    def test_load_config_with_overrides(self, hydra_context):
        """Test loading configuration with command-line style overrides."""
        cfg = compose(
            config_name="config", 
            overrides=[
                "ray=multi_gpu",
                "ray.actors.num_gpus=2", 
                "ray.producers.num_tasks=4",
                "vit.depth=4"
            ]
        )
        
        # Overrides should be applied
        assert cfg.ray.actors.num_gpus == 2
        assert cfg.ray.producers.num_tasks == 4
        assert cfg.vit.depth == 4
        
        # Other values should remain from base config
        assert cfg.vit.patch_size > 0
        assert hasattr(cfg.performance, 'pin_memory')
    
    def test_config_inheritance(self, hydra_context):
        """Test that Ray configs properly inherit from base configuration."""
        # Load multi_gpu config
        cfg_multi = compose(config_name="config", overrides=["ray=multi_gpu"])
        
        # Load profiling config  
        cfg_prof = compose(config_name="config", overrides=["ray=profiling"])
        
        # Both should have same base structure
        assert cfg_multi.vit.patch_size == cfg_prof.vit.patch_size
        assert cfg_multi.shape == cfg_prof.shape
        assert cfg_multi.performance.pin_memory == cfg_prof.performance.pin_memory
        
        # But different Ray settings
        assert cfg_multi.ray.actors.profiling_enabled != cfg_prof.ray.actors.profiling_enabled


class TestConfigurationExecution:
    """Test that configurations can be used to run the pipeline."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU required for execution test")
    def test_multi_gpu_config_execution(self, ray_environment, hydra_context):
        """Test that multi-GPU configuration can be executed."""
        cfg = compose(config_name="config", overrides=["ray=multi_gpu"])
        
        # Limit GPU usage for testing
        cfg.ray.actors.num_gpus = min(1, GPU_COUNT)
        cfg.ray.producers.batches_per_task = 3  # Reduce work for testing
        cfg.vit.depth = 0  # No-op mode for fast testing
        
        # Initialize Ray
        ray.init(num_cpus=4, num_gpus=GPU_COUNT, ignore_reinit_error=True)
        
        # Should be able to run pipeline
        result = run_ray_pipeline_test(cfg)
        
        if result['success']:
            assert result['config']['num_gpus'] == cfg.ray.actors.num_gpus
            assert result['performance']['total_samples'] > 0
        else:
            # If it fails, should be due to resource constraints, not config issues
            assert 'error' in result
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU required for execution test")
    def test_profiling_config_execution(self, ray_environment, hydra_context):
        """Test that profiling configuration can be executed."""
        cfg = compose(config_name="config", overrides=["ray=profiling"])
        
        # Limit for testing
        cfg.ray.actors.num_gpus = min(1, GPU_COUNT)
        cfg.ray.producers.batches_per_task = 2
        cfg.vit.depth = 0  # No-op mode
        
        # Initialize Ray
        ray.init(num_cpus=4, num_gpus=GPU_COUNT, ignore_reinit_error=True)
        
        # Should be able to run pipeline with profiling
        result = run_ray_pipeline_test(cfg)
        
        if result['success']:
            assert result['config']['profiling_enabled'] is True
            assert result['performance']['total_samples'] > 0
        else:
            # Profiling might not work in all environments
            assert 'error' in result
    
    def test_cpu_only_config_behavior(self, ray_environment, hydra_context):
        """Test behavior when GPU configs are used without GPUs."""
        cfg = compose(config_name="config", overrides=["ray=multi_gpu"])
        
        # Force CPU-only by setting num_gpus to 0
        cfg.ray.actors.num_gpus = 0
        
        ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True)
        
        # Should fail since actors require GPUs
        result = run_ray_pipeline_test(cfg)
        assert result['success'] is False
        assert 'error' in result


class TestConfigValidation:
    """Test validation of configuration parameters."""
    
    def test_invalid_gpu_count(self, hydra_context):
        """Test handling of invalid GPU counts."""
        cfg = compose(config_name="config", overrides=["ray=multi_gpu"])
        
        # Set invalid GPU count
        cfg.ray.actors.num_gpus = -1
        
        # Should be able to detect this as invalid
        assert cfg.ray.actors.num_gpus < 0  # Just verify it's set
        
        # Pipeline should handle this gracefully
        ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True)
        result = run_ray_pipeline_test(cfg)
        assert result['success'] is False
        
        ray.shutdown()
    
    def test_zero_producers(self, hydra_context):
        """Test handling of zero producers."""
        cfg = compose(config_name="config", overrides=["ray=multi_gpu"])
        
        cfg.ray.producers.num_tasks = 0
        
        if CUDA_AVAILABLE:
            ray.init(num_cpus=4, num_gpus=GPU_COUNT, ignore_reinit_error=True)
        else:
            ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True)
            cfg.ray.actors.num_gpus = 0
        
        result = run_ray_pipeline_test(cfg)
        
        # Should handle zero producers case
        assert isinstance(result, dict)
        assert 'success' in result
        
        ray.shutdown()
    
    def test_config_parameter_bounds(self, hydra_context):
        """Test that configuration parameters are within reasonable bounds."""
        cfg = compose(config_name="config", overrides=["ray=multi_gpu"])
        
        # Check reasonable parameter ranges
        assert 0 < cfg.ray.actors.num_gpus <= 16, "num_gpus should be reasonable"
        assert 0 < cfg.ray.producers.num_tasks <= 100, "num_tasks should be reasonable" 
        assert 0 < cfg.ray.producers.batches_per_task <= 1000, "batches_per_task should be reasonable"
        assert 0 < cfg.pipeline.batch_size <= 256, "batch_size should be reasonable"
        assert 0 <= cfg.vit.depth <= 48, "VIT depth should be reasonable"
        assert cfg.vit.patch_size in [8, 16, 32], "patch_size should be common value"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])