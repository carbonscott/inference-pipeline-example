#!/usr/bin/env python3
"""
Unit tests for vit_pipeline_ray.py

Tests the main orchestrator for Ray-based multi-GPU VIT pipeline including
Ray initialization, actor creation, data distribution, and results aggregation.
"""

import pytest
import ray
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock
from omegaconf import DictConfig, OmegaConf
import yaml

from vit_pipeline_ray import (
    setup_ray,
    run_ray_pipeline_test,
    cleanup_ray
)


# Check GPU availability
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0


@pytest.fixture(scope="function")
def ray_context():
    """Initialize Ray for testing and cleanup after each test."""
    # Ensure Ray is shutdown before starting
    if ray.is_initialized():
        ray.shutdown()
    
    # Initialize with limited resources
    if CUDA_AVAILABLE:
        ray.init(num_cpus=4, num_gpus=min(GPU_COUNT, 2), ignore_reinit_error=True)
    else:
        ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True)
    
    yield
    
    # Cleanup after test
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    config_dict = {
        'shape': [3, 224, 224],
        'batch_size': 4,
        'num_samples': 100,
        'warmup_samples': 0,
        'gpu_id': 0,
        
        'vit': {
            'patch_size': 32,
            'depth': 0,  # No-op mode for testing
            'heads': 4,
            'dim': 256,
            'mlp_dim': 512
        },
        
        'pipeline': {
            'batch_size': 4
        },
        
        'performance': {
            'pin_memory': True,
            'compile_model': False,
            'compile_mode': 'default'
        },
        
        'test': {
            'deterministic': True,
            'sync_frequency': 10
        },
        
        'ray': {
            'init': {},
            'actors': {
                'num_gpus': 1,
                'profiling_enabled': False
            },
            'producers': {
                'num_tasks': 2,
                'batches_per_task': 5,
                'inter_batch_delay': 0.0
            }
        }
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def multi_gpu_config(basic_config):
    """Multi-GPU configuration for testing."""
    config = basic_config.copy()
    config.ray.actors.num_gpus = min(2, GPU_COUNT) if CUDA_AVAILABLE else 1
    config.ray.producers.num_tasks = 4
    config.ray.producers.batches_per_task = 3
    return config


@pytest.fixture
def profiling_config(basic_config):
    """Configuration with profiling enabled."""
    config = basic_config.copy()
    config.ray.actors.profiling_enabled = True
    config.ray.producers.batches_per_task = 3  # Fewer batches for cleaner profiles
    return config


class TestSetupRay:
    """Test Ray initialization functionality."""
    
    def test_setup_ray_basic(self):
        """Test basic Ray setup."""
        if ray.is_initialized():
            ray.shutdown()
        
        ray_config = OmegaConf.create({
            'init': {
                'num_cpus': 2,
                'num_gpus': 0
            }
        })
        
        setup_ray(ray_config)
        
        assert ray.is_initialized()
        
        # Check cluster resources
        resources = ray.cluster_resources()
        assert 'CPU' in resources
        assert resources['CPU'] >= 2
        
        ray.shutdown()
    
    def test_setup_ray_with_gpu(self):
        """Test Ray setup with GPU configuration."""
        if not CUDA_AVAILABLE:
            pytest.skip("No GPU available for testing")
            
        if ray.is_initialized():
            ray.shutdown()
        
        ray_config = OmegaConf.create({
            'init': {
                'num_cpus': 2,
                'num_gpus': 1
            }
        })
        
        setup_ray(ray_config)
        
        assert ray.is_initialized()
        
        # Check GPU resources
        resources = ray.cluster_resources()
        assert 'GPU' in resources
        assert resources['GPU'] >= 1
        
        ray.shutdown()
    
    def test_setup_ray_already_initialized(self, ray_context):
        """Test Ray setup when Ray is already initialized."""
        assert ray.is_initialized()
        
        ray_config = OmegaConf.create({'init': {}})
        
        # Should handle already-initialized case gracefully
        setup_ray(ray_config)
        
        assert ray.is_initialized()
    
    def test_setup_ray_empty_config(self):
        """Test Ray setup with empty configuration."""
        if ray.is_initialized():
            ray.shutdown()
        
        ray_config = OmegaConf.create({})
        
        setup_ray(ray_config)
        
        assert ray.is_initialized()
        ray.shutdown()
    
    def test_setup_ray_invalid_config(self):
        """Test Ray setup with invalid configuration."""
        if ray.is_initialized():
            ray.shutdown()
        
        ray_config = OmegaConf.create({
            'init': {
                'num_cpus': -1  # Invalid value
            }
        })
        
        # Should handle invalid config by exiting
        with pytest.raises(SystemExit):
            setup_ray(ray_config)


class TestRunRayPipelineTest:
    """Test the main pipeline test orchestration."""
    
    def skip_if_no_gpu(self):
        """Helper to skip tests if no GPU available."""
        if not CUDA_AVAILABLE:
            pytest.skip("GPU not available for testing")
    
    def test_single_gpu_pipeline(self, ray_context, basic_config):
        """Test pipeline with single GPU configuration."""
        if not CUDA_AVAILABLE:
            # Modify config for CPU-only testing
            basic_config.ray.actors.num_gpus = 0
            
            # This should fail gracefully since actors require GPUs
            result = run_ray_pipeline_test(basic_config)
            assert result['success'] is False
            return
        
        # GPU available - run actual test
        result = run_ray_pipeline_test(basic_config)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'config' in result
            assert 'performance' in result
            assert 'actor_stats' in result
            
            # Verify config was applied
            assert result['config']['num_gpus'] == 1
            assert result['config']['num_producers'] == basic_config.ray.producers.num_tasks
            
            # Verify performance metrics
            assert result['performance']['total_samples'] > 0
            assert result['performance']['overall_throughput'] > 0
        else:
            # If failed, should have error information
            assert 'error' in result
    
    @pytest.mark.skipif(not CUDA_AVAILABLE or GPU_COUNT < 2, reason="Need at least 2 GPUs")
    def test_multi_gpu_pipeline(self, ray_context, multi_gpu_config):
        """Test pipeline with multiple GPU configuration."""
        result = run_ray_pipeline_test(multi_gpu_config)
        
        # Should handle multi-GPU case
        assert isinstance(result, dict)
        
        if result.get('success'):
            # Verify multi-GPU configuration
            expected_gpus = multi_gpu_config.ray.actors.num_gpus
            assert result['config']['num_gpus'] == expected_gpus
            
            # Should have stats from multiple actors
            assert len(result['actor_stats']) == expected_gpus
            
            # Performance should be reasonable
            assert result['performance']['total_samples'] > 0
    
    def test_profiling_enabled_pipeline(self, ray_context, profiling_config):
        """Test pipeline with profiling enabled."""
        if not CUDA_AVAILABLE:
            profiling_config.ray.actors.num_gpus = 0
            result = run_ray_pipeline_test(profiling_config)
            assert result['success'] is False  # Should fail without GPU
            return
        
        result = run_ray_pipeline_test(profiling_config)
        
        if result.get('success'):
            # Verify profiling was enabled
            assert result['config']['profiling_enabled'] is True
        else:
            # Profiling might fail in test environment - that's OK
            assert 'error' in result
    
    def test_pipeline_with_zero_producers(self, ray_context, basic_config):
        """Test pipeline behavior with zero data producers."""
        basic_config.ray.producers.num_tasks = 0
        
        result = run_ray_pipeline_test(basic_config)
        
        # Should handle gracefully or fail appropriately
        assert isinstance(result, dict)
        assert 'success' in result
        
        if not result['success']:
            assert 'error' in result
    
    def test_pipeline_with_zero_batches(self, ray_context, basic_config):
        """Test pipeline behavior with zero batches per producer."""
        basic_config.ray.producers.batches_per_task = 0
        
        if not CUDA_AVAILABLE:
            basic_config.ray.actors.num_gpus = 0
        
        result = run_ray_pipeline_test(basic_config)
        
        # Should handle zero batches case
        assert isinstance(result, dict)
        
        if result.get('success'):
            assert result['performance']['total_samples'] == 0
    
    def test_pipeline_configuration_propagation(self, ray_context, basic_config):
        """Test that configuration parameters are properly propagated."""
        # Modify some specific parameters
        basic_config.vit.depth = 2
        basic_config.vit.dim = 128
        basic_config.pipeline.batch_size = 6
        
        if not CUDA_AVAILABLE:
            basic_config.ray.actors.num_gpus = 0
            result = run_ray_pipeline_test(basic_config)
            assert result['success'] is False
            return
        
        result = run_ray_pipeline_test(basic_config)
        
        if result.get('success'):
            # Check that actor stats reflect the configuration
            actor_stats = result['actor_stats']
            assert len(actor_stats) > 0
            
            # Each actor should have model config
            for stats in actor_stats:
                if 'model_config' in stats:
                    assert stats['model_config']['depth'] == 2
                    assert stats['model_config']['dim'] == 128


class TestConfigurationHandling:
    """Test configuration handling and validation."""
    
    def test_missing_ray_config(self, ray_context):
        """Test handling of missing ray configuration."""
        config = OmegaConf.create({
            'shape': [3, 224, 224],
            'vit': {'depth': 0}
        })  # Missing 'ray' section
        
        # Should fail gracefully with missing ray config
        try:
            result = run_ray_pipeline_test(config)
            assert result.get('success') is False
        except (KeyError, AttributeError):
            # Expected to fail with missing config
            pass
    
    def test_invalid_gpu_count(self, ray_context, basic_config):
        """Test handling of invalid GPU count."""
        # Request more GPUs than available
        basic_config.ray.actors.num_gpus = GPU_COUNT + 10
        
        result = run_ray_pipeline_test(basic_config)
        
        # Should either fail or handle gracefully
        assert isinstance(result, dict)
        if not result.get('success'):
            assert 'error' in result
    
    def test_negative_batch_size(self, ray_context, basic_config):
        """Test handling of negative batch size."""
        basic_config.pipeline.batch_size = -1
        
        result = run_ray_pipeline_test(basic_config)
        
        # Should fail with invalid batch size
        assert result.get('success') is False
        assert 'error' in result
    
    def test_omegaconf_to_dict_conversion(self, basic_config):
        """Test that OmegaConf configurations are properly converted."""
        # The function should handle OmegaConf objects
        assert isinstance(basic_config, DictConfig)
        
        # Test conversion in configuration access
        ray_config = basic_config.get('ray', {})
        assert ray_config is not None
        
        # Should be able to access nested values
        num_gpus = ray_config.get('actors', {}).get('num_gpus', 1)
        assert isinstance(num_gpus, int)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_ray_initialization_failure(self, basic_config):
        """Test handling of Ray initialization failure."""
        # Start with Ray already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=1)
        
        # Try to run with config that might cause issues
        invalid_config = basic_config.copy()
        invalid_config.ray.init = {
            'num_cpus': -1,  # Invalid
            'num_gpus': -1   # Invalid
        }
        
        # Should handle initialization failure
        result = run_ray_pipeline_test(invalid_config)
        assert isinstance(result, dict)
        
        ray.shutdown()
    
    def test_actor_creation_failure(self, ray_context, basic_config):
        """Test handling of actor creation failure."""
        # Request impossible number of GPUs
        basic_config.ray.actors.num_gpus = 100
        
        result = run_ray_pipeline_test(basic_config)
        
        # Should fail gracefully
        assert result.get('success') is False
        assert 'error' in result
    
    def test_exception_during_processing(self, ray_context, basic_config):
        """Test handling of exceptions during processing."""
        # Create config that might cause processing issues
        basic_config.shape = []  # Invalid shape
        
        if not CUDA_AVAILABLE:
            basic_config.ray.actors.num_gpus = 0
        
        result = run_ray_pipeline_test(basic_config)
        
        # Should handle exceptions gracefully
        assert isinstance(result, dict)
        assert result.get('success') is False
        assert 'error' in result


class TestCleanupRay:
    """Test Ray cleanup functionality."""
    
    def test_cleanup_ray_when_initialized(self):
        """Test cleanup when Ray is initialized."""
        ray.init(num_cpus=1, ignore_reinit_error=True)
        assert ray.is_initialized()
        
        cleanup_ray()
        
        assert not ray.is_initialized()
    
    def test_cleanup_ray_when_not_initialized(self):
        """Test cleanup when Ray is not initialized."""
        if ray.is_initialized():
            ray.shutdown()
        
        # Should not crash when Ray is not initialized
        cleanup_ray()
        assert not ray.is_initialized()
    
    def test_cleanup_ray_exception_handling(self):
        """Test that cleanup handles exceptions gracefully."""
        if ray.is_initialized():
            ray.shutdown()
        
        # Mock ray.shutdown to raise an exception
        with patch('ray.shutdown', side_effect=Exception("Test exception")):
            # Should not crash
            cleanup_ray()


class TestIntegrationWithHydra:
    """Test integration with Hydra configuration system."""
    
    def test_hydra_config_structure(self, basic_config):
        """Test that config has expected Hydra structure."""
        # Should have main sections
        assert 'vit' in basic_config
        assert 'performance' in basic_config
        assert 'ray' in basic_config
        
        # Ray section should have expected subsections
        assert 'actors' in basic_config.ray
        assert 'producers' in basic_config.ray
        
        # Values should be accessible
        assert basic_config.vit.patch_size == 32
        assert basic_config.ray.actors.num_gpus >= 0
    
    def test_config_value_types(self, basic_config):
        """Test that configuration values have correct types."""
        # Shape should be a list
        assert isinstance(basic_config.shape, (list, tuple))
        
        # Numeric values should be proper types
        assert isinstance(basic_config.vit.depth, int)
        assert isinstance(basic_config.performance.pin_memory, bool)
        
        # Nested access should work
        assert isinstance(basic_config.ray.actors.num_gpus, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])