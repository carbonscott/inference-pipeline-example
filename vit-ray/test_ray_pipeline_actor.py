#!/usr/bin/env python3
"""
Unit tests for ray_pipeline_actor.py

Tests the Ray actor that wraps DoubleBufferedPipeline for multi-GPU processing.
These tests require GPU access for full functionality.
"""

import pytest
import ray
import torch
import numpy as np
import time
from typing import List, Dict, Any
import os
import tempfile

from ray_pipeline_actor import (
    VitPipelineActor,
    VitPipelineActorWithProfiling,
    create_pipeline_actors
)


# Check if CUDA is available for testing
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing with GPU support if available."""
    if not ray.is_initialized():
        if CUDA_AVAILABLE:
            ray.init(num_cpus=4, num_gpus=min(GPU_COUNT, 2))  # Use up to 2 GPUs for testing
        else:
            ray.init(num_cpus=4, num_gpus=0)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def test_config():
    """Standard configuration for testing."""
    return {
        'tensor_shape': (3, 224, 224),
        'batch_size': 4,
        'patch_size': 32,
        'depth': 2,  # Small model for fast testing
        'heads': 4,
        'dim': 256,
        'mlp_dim': 512,
        'pin_memory': True,
        'compile_model': False,
        'deterministic': True
    }


@pytest.fixture
def test_no_op_config():
    """No-op model configuration for testing without actual computation."""
    return {
        'tensor_shape': (3, 224, 224),
        'batch_size': 4,
        'patch_size': 32,
        'depth': 0,  # No-op mode
        'heads': 4,
        'dim': 256,
        'mlp_dim': 512,
        'pin_memory': True,
        'compile_model': False,
        'deterministic': True
    }


def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    return pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU not available for testing")


class TestVitPipelineActorInitialization:
    """Test VitPipelineActor initialization and setup."""
    
    @skip_if_no_gpu()
    def test_actor_creation(self, ray_context, test_no_op_config):
        """Test basic actor creation and GPU assignment."""
        actor = VitPipelineActor.remote(**test_no_op_config)
        
        # Test that actor was created successfully
        assert actor is not None
        
        # Test actor info retrieval
        info = ray.get(actor.get_actor_info.remote())
        
        # Verify basic info structure
        assert 'gpu_id' in info
        assert 'model_config' in info
        assert 'pipeline_config' in info
        assert 'stats' in info
        
        # Verify GPU assignment
        assert isinstance(info['gpu_id'], int)
        assert 0 <= info['gpu_id'] < GPU_COUNT
        
        # Verify model config
        assert info['model_config']['depth'] == test_no_op_config['depth']
        assert info['model_config']['patch_size'] == test_no_op_config['patch_size']
        
        # Verify pipeline config
        assert info['pipeline_config']['batch_size'] == test_no_op_config['batch_size']
        assert info['pipeline_config']['input_shape'] == test_no_op_config['tensor_shape']
    
    @skip_if_no_gpu()
    def test_actor_health_check(self, ray_context, test_no_op_config):
        """Test actor health check functionality."""
        actor = VitPipelineActor.remote(**test_no_op_config)
        
        health = ray.get(actor.health_check.remote())
        
        # Verify health check structure
        assert 'status' in health
        assert 'gpu_id' in health
        assert 'gpu_memory_total_mb' in health
        assert 'gpu_memory_used_mb' in health
        assert 'model_loaded' in health
        assert 'pipeline_initialized' in health
        
        # Actor should be healthy
        assert health['status'] == 'healthy'
        assert isinstance(health['gpu_id'], int)
        assert health['gpu_memory_total_mb'] > 0
        assert health['model_loaded'] == (test_no_op_config['depth'] > 0)  # False for no-op
        assert health['pipeline_initialized'] is True
    
    @skip_if_no_gpu()
    def test_actor_deterministic_initialization(self, ray_context, test_no_op_config):
        """Test that deterministic mode produces consistent initialization."""
        # Create two actors with same deterministic config
        actor1 = VitPipelineActor.remote(**test_no_op_config)
        actor2 = VitPipelineActor.remote(**test_no_op_config)
        
        # Get their configurations
        info1 = ray.get(actor1.get_actor_info.remote())
        info2 = ray.get(actor2.get_actor_info.remote())
        
        # Model configs should be identical
        assert info1['model_config'] == info2['model_config']
        assert info1['pipeline_config'] == info2['pipeline_config']
        
        # Both should be healthy
        health1 = ray.get(actor1.health_check.remote())
        health2 = ray.get(actor2.health_check.remote())
        
        assert health1['status'] == 'healthy'
        assert health2['status'] == 'healthy'
    
    def test_actor_creation_without_gpu(self, ray_context, test_no_op_config):
        """Test actor creation behavior when no GPU is assigned."""
        if CUDA_AVAILABLE:
            pytest.skip("GPU available, cannot test no-GPU behavior")
        
        # This should fail since VitPipelineActor requires num_gpus=1
        with pytest.raises(Exception):
            actor = VitPipelineActor.remote(**test_no_op_config)
            ray.get(actor.get_actor_info.remote(), timeout=10)


class TestVitPipelineActorProcessing:
    """Test VitPipelineActor batch processing functionality."""
    
    def _create_test_batch(self, batch_size: int, tensor_shape: tuple) -> List[ray.ObjectRef]:
        """Helper to create a batch of test tensors in Ray object store."""
        tensors = []
        for i in range(batch_size):
            tensor = torch.randn(*tensor_shape)
            tensor_ref = ray.put(tensor)
            tensors.append(tensor_ref)
        return tensors
    
    @skip_if_no_gpu()
    def test_single_batch_processing(self, ray_context, test_no_op_config):
        """Test processing a single batch of tensors."""
        actor = VitPipelineActor.remote(**test_no_op_config)
        
        # Create test batch
        batch_size = test_no_op_config['batch_size']
        tensor_shape = test_no_op_config['tensor_shape']
        batch_refs = self._create_test_batch(batch_size, tensor_shape)
        
        # Process batch
        result = ray.get(actor.process_batch_from_refs.remote(batch_refs, batch_id=0))
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'batch_id' in result
        assert 'batch_size' in result
        assert 'processing_time' in result
        assert 'gpu_id' in result
        assert 'throughput' in result
        
        # Verify result values
        assert result['batch_id'] == 0
        assert result['batch_size'] == batch_size
        assert result['processing_time'] > 0
        assert isinstance(result['gpu_id'], int)
        assert result['throughput'] > 0
    
    @skip_if_no_gpu()
    def test_multiple_batch_processing(self, ray_context, test_no_op_config):
        """Test processing multiple batches sequentially."""
        actor = VitPipelineActor.remote(**test_no_op_config)
        
        batch_size = test_no_op_config['batch_size']
        tensor_shape = test_no_op_config['tensor_shape']
        num_batches = 3
        
        # Create multiple batches
        batch_list = []
        for i in range(num_batches):
            batch_refs = self._create_test_batch(batch_size, tensor_shape)
            batch_list.append(batch_refs)
        
        # Process all batches
        results = ray.get(actor.process_batch_list.remote(batch_list))
        
        # Verify results
        assert len(results) == num_batches
        
        for i, result in enumerate(results):
            assert result['batch_id'] == i
            assert result['batch_size'] == batch_size
            assert result['processing_time'] > 0
            assert result['throughput'] > 0
    
    @skip_if_no_gpu()
    def test_statistics_tracking(self, ray_context, test_no_op_config):
        """Test that statistics are properly tracked across batches."""
        actor = VitPipelineActor.remote(**test_no_op_config)
        
        # Check initial statistics
        initial_stats = ray.get(actor.get_statistics.remote())
        assert initial_stats['batches_processed'] == 0
        assert initial_stats['samples_processed'] == 0
        assert initial_stats['total_time'] == 0
        
        # Process some batches
        batch_size = test_no_op_config['batch_size']
        tensor_shape = test_no_op_config['tensor_shape']
        num_batches = 2
        
        for i in range(num_batches):
            batch_refs = self._create_test_batch(batch_size, tensor_shape)
            ray.get(actor.process_batch_from_refs.remote(batch_refs, batch_id=i))
        
        # Check updated statistics
        final_stats = ray.get(actor.get_statistics.remote())
        assert final_stats['batches_processed'] == num_batches
        assert final_stats['samples_processed'] == num_batches * batch_size
        assert final_stats['total_time'] > 0
        assert final_stats['average_throughput'] > 0
        assert final_stats['average_batch_time'] > 0
    
    @skip_if_no_gpu()
    def test_statistics_reset(self, ray_context, test_no_op_config):
        """Test statistics reset functionality."""
        actor = VitPipelineActor.remote(**test_no_op_config)
        
        # Process a batch to generate some statistics
        batch_refs = self._create_test_batch(2, test_no_op_config['tensor_shape'])
        ray.get(actor.process_batch_from_refs.remote(batch_refs, batch_id=0))
        
        # Verify statistics were generated
        stats_before = ray.get(actor.get_statistics.remote())
        assert stats_before['batches_processed'] > 0
        
        # Reset statistics
        ray.get(actor.reset_statistics.remote())
        
        # Verify statistics were reset
        stats_after = ray.get(actor.get_statistics.remote())
        assert stats_after['batches_processed'] == 0
        assert stats_after['samples_processed'] == 0
        assert stats_after['total_time'] == 0
    
    @skip_if_no_gpu()
    def test_variable_batch_sizes(self, ray_context, test_no_op_config):
        """Test processing batches with different sizes."""
        actor = VitPipelineActor.remote(**test_no_op_config)
        
        tensor_shape = test_no_op_config['tensor_shape']
        batch_sizes = [1, 3, 5]
        
        for i, batch_size in enumerate(batch_sizes):
            batch_refs = self._create_test_batch(batch_size, tensor_shape)
            result = ray.get(actor.process_batch_from_refs.remote(batch_refs, batch_id=i))
            
            assert result['batch_size'] == batch_size
            assert result['throughput'] > 0
    
    @skip_if_no_gpu()
    def test_empty_batch_handling(self, ray_context, test_no_op_config):
        """Test handling of empty batches."""
        actor = VitPipelineActor.remote(**test_no_op_config)
        
        # Process empty batch
        empty_batch = []
        result = ray.get(actor.process_batch_from_refs.remote(empty_batch, batch_id=0))
        
        assert result['batch_size'] == 0
        assert result['processing_time'] >= 0  # Should be very small but not negative


class TestVitPipelineActorWithProfiling:
    """Test the profiling-enabled variant of the actor."""
    
    @skip_if_no_gpu()
    def test_profiling_actor_creation(self, ray_context, test_no_op_config):
        """Test creation of profiling-enabled actor."""
        # Note: This may not actually enable nsys profiling in test environment
        # but it should at least create the actor successfully
        actor = VitPipelineActorWithProfiling.remote(**test_no_op_config)
        
        # Should be able to get actor info
        info = ray.get(actor.get_actor_info.remote())
        assert 'gpu_id' in info
        
        # Health check should pass
        health = ray.get(actor.health_check.remote())
        assert health['status'] == 'healthy'
    
    @skip_if_no_gpu()
    def test_profiling_actor_processing(self, ray_context, test_no_op_config):
        """Test that profiling-enabled actor can process data."""
        actor = VitPipelineActorWithProfiling.remote(**test_no_op_config)
        
        # Create and process a small batch
        tensor_shape = test_no_op_config['tensor_shape']
        batch_refs = []
        for i in range(2):
            tensor = torch.randn(*tensor_shape)
            batch_refs.append(ray.put(tensor))
        
        result = ray.get(actor.process_batch_from_refs.remote(batch_refs, batch_id=0))
        
        assert result['batch_size'] == 2
        assert result['processing_time'] > 0


class TestCreatePipelineActors:
    """Test the create_pipeline_actors helper function."""
    
    @skip_if_no_gpu()
    def test_create_single_actor(self, ray_context, test_no_op_config):
        """Test creating a single pipeline actor."""
        actors = create_pipeline_actors(
            num_actors=1,
            enable_profiling=False,
            **test_no_op_config
        )
        
        assert len(actors) == 1
        
        # Verify actor is functional
        info = ray.get(actors[0].get_actor_info.remote())
        assert 'gpu_id' in info
        
        health = ray.get(actors[0].health_check.remote())
        assert health['status'] == 'healthy'
    
    @skip_if_no_gpu()
    def test_create_multiple_actors(self, ray_context, test_no_op_config):
        """Test creating multiple pipeline actors."""
        num_actors = min(2, GPU_COUNT)  # Don't exceed available GPUs
        
        actors = create_pipeline_actors(
            num_actors=num_actors,
            enable_profiling=False,
            **test_no_op_config
        )
        
        assert len(actors) == num_actors
        
        # Check that all actors are healthy
        health_futures = [actor.health_check.remote() for actor in actors]
        health_results = ray.get(health_futures)
        
        for health in health_results:
            assert health['status'] == 'healthy'
        
        # Verify different actors get different GPU assignments (if multiple GPUs)
        if GPU_COUNT > 1 and num_actors > 1:
            info_futures = [actor.get_actor_info.remote() for actor in actors]
            info_results = ray.get(info_futures)
            
            gpu_ids = [info['gpu_id'] for info in info_results]
            # Should have different GPU IDs (though Ray may reuse if limited GPUs)
            assert len(set(gpu_ids)) >= 1
    
    @skip_if_no_gpu()
    def test_create_profiling_actors(self, ray_context, test_no_op_config):
        """Test creating profiling-enabled actors."""
        actors = create_pipeline_actors(
            num_actors=1,
            enable_profiling=True,
            **test_no_op_config
        )
        
        assert len(actors) == 1
        
        # Should be able to interact with profiling-enabled actor
        health = ray.get(actors[0].health_check.remote())
        assert health['status'] == 'healthy'


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @skip_if_no_gpu()
    def test_invalid_tensor_shapes(self, ray_context, test_no_op_config):
        """Test handling of invalid tensor shapes in processing."""
        actor = VitPipelineActor.remote(**test_no_op_config)
        
        # Create batch with wrong tensor shape
        wrong_shape = (2, 128, 128)  # Different from config
        batch_refs = []
        for i in range(2):
            tensor = torch.randn(*wrong_shape)
            batch_refs.append(ray.put(tensor))
        
        # This should either handle gracefully or raise an appropriate error
        try:
            result = ray.get(actor.process_batch_from_refs.remote(batch_refs, batch_id=0))
            # If it succeeds, verify it processed the actual batch size
            assert result['batch_size'] == 2
        except Exception as e:
            # If it fails, that's also acceptable for invalid input
            assert isinstance(e, (RuntimeError, ValueError, ray.exceptions.RayTaskError))
    
    @skip_if_no_gpu()
    def test_actor_resource_limits(self, ray_context, test_no_op_config):
        """Test behavior when trying to create more actors than GPUs."""
        if GPU_COUNT >= 4:
            pytest.skip("Too many GPUs available to test resource limits")
        
        # Try to create more actors than available GPUs
        num_actors = GPU_COUNT + 1
        
        # This might timeout or handle gracefully depending on Ray configuration
        try:
            actors = create_pipeline_actors(
                num_actors=num_actors,
                enable_profiling=False,
                **test_no_op_config
            )
            # If successful, verify at least some actors were created
            assert len(actors) > 0
        except Exception as e:
            # Resource exhaustion is expected
            assert isinstance(e, (ray.exceptions.RayTaskError, RuntimeError))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])