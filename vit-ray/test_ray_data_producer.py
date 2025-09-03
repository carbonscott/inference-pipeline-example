#!/usr/bin/env python3
"""
Unit tests for ray_data_producer.py

Tests the Ray-based data producer tasks that generate streaming tensor data
and store it in Ray's object store for consumption by pipeline actors.
"""

import pytest
import ray
import torch
import numpy as np
import time
from typing import List, Tuple

from ray_data_producer import (
    generate_data_batch,
    generate_streaming_batches,
    RayDataProducerManager
)


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing and cleanup after tests."""
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=0)  # CPU-only for data producer tests
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def test_tensor_shape():
    """Standard tensor shape for testing."""
    return (3, 224, 224)


@pytest.fixture
def test_batch_size():
    """Standard batch size for testing."""
    return 4


class TestGenerateDataBatch:
    """Test the generate_data_batch Ray task."""
    
    def test_basic_data_generation(self, ray_context, test_tensor_shape, test_batch_size):
        """Test basic tensor generation with correct shapes."""
        batch_id = 0
        
        # Generate batch
        future = generate_data_batch.remote(
            batch_size=test_batch_size,
            tensor_shape=test_tensor_shape,
            batch_id=batch_id,
            pin_memory=False,
            deterministic=True
        )
        
        # Get result
        object_refs = ray.get(future)
        
        # Verify we got the right number of object references
        assert len(object_refs) == test_batch_size
        assert all(isinstance(ref, ray.ObjectRef) for ref in object_refs)
        
        # Get actual tensors and verify shapes
        tensors = ray.get(object_refs)
        for tensor in tensors:
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == test_tensor_shape
            assert tensor.dtype == torch.float32
    
    def test_object_store_integration(self, ray_context, test_tensor_shape):
        """Test that tensors are properly stored in Ray object store."""
        batch_size = 2
        batch_id = 1
        
        # Generate batch
        future = generate_data_batch.remote(
            batch_size=batch_size,
            tensor_shape=test_tensor_shape,
            batch_id=batch_id,
            deterministic=True
        )
        
        object_refs = ray.get(future)
        
        # Verify object refs are valid Ray objects
        assert len(object_refs) == batch_size
        
        # Test that we can retrieve tensors from object store
        tensor_0 = ray.get(object_refs[0])
        tensor_1 = ray.get(object_refs[1])
        
        assert tensor_0.shape == test_tensor_shape
        assert tensor_1.shape == test_tensor_shape
        
        # Tensors should be different (random generation)
        assert not torch.equal(tensor_0, tensor_1)
    
    def test_deterministic_mode(self, ray_context, test_tensor_shape):
        """Test that deterministic mode produces reproducible results."""
        batch_size = 2
        batch_id = 42  # Same batch_id should produce same results
        
        # Generate first batch
        future1 = generate_data_batch.remote(
            batch_size=batch_size,
            tensor_shape=test_tensor_shape,
            batch_id=batch_id,
            deterministic=True
        )
        
        # Generate second batch with same parameters
        future2 = generate_data_batch.remote(
            batch_size=batch_size,
            tensor_shape=test_tensor_shape,
            batch_id=batch_id,
            deterministic=True
        )
        
        # Get results
        refs1 = ray.get(future1)
        refs2 = ray.get(future2)
        
        # Get actual tensors
        tensors1 = ray.get(refs1)
        tensors2 = ray.get(refs2)
        
        # Should be identical due to deterministic mode
        for t1, t2 in zip(tensors1, tensors2):
            assert torch.equal(t1, t2), "Deterministic mode should produce identical tensors"
    
    def test_pin_memory_functionality(self, ray_context, test_tensor_shape):
        """Test memory pinning functionality."""
        batch_size = 2
        
        # Test with pin_memory=True
        future_pinned = generate_data_batch.remote(
            batch_size=batch_size,
            tensor_shape=test_tensor_shape,
            batch_id=0,
            pin_memory=True,
            deterministic=True
        )
        
        # Test with pin_memory=False
        future_unpinned = generate_data_batch.remote(
            batch_size=batch_size,
            tensor_shape=test_tensor_shape,
            batch_id=0,
            pin_memory=False,
            deterministic=True
        )
        
        # Get results
        refs_pinned = ray.get(future_pinned)
        refs_unpinned = ray.get(future_unpinned)
        
        tensors_pinned = ray.get(refs_pinned)
        tensors_unpinned = ray.get(refs_unpinned)
        
        # Both should have same content (deterministic)
        for t_pin, t_unpin in zip(tensors_pinned, tensors_unpinned):
            assert torch.equal(t_pin, t_unpin), "Content should be same regardless of pinning"
            
        # Note: is_pinned() check is tricky in Ray context, so we mainly test functionality
    
    def test_different_tensor_shapes(self, ray_context):
        """Test generation with different tensor shapes."""
        test_shapes = [
            (1, 64, 64),
            (3, 224, 224),
            (4, 128, 128)
        ]
        
        for shape in test_shapes:
            future = generate_data_batch.remote(
                batch_size=2,
                tensor_shape=shape,
                batch_id=0,
                deterministic=True
            )
            
            refs = ray.get(future)
            tensors = ray.get(refs)
            
            for tensor in tensors:
                assert tensor.shape == shape


class TestGenerateStreamingBatches:
    """Test the generate_streaming_batches Ray task."""
    
    def test_streaming_batch_generation(self, ray_context, test_tensor_shape):
        """Test streaming generation of multiple batches."""
        num_batches = 3
        batch_size = 2
        producer_id = 0
        
        future = generate_streaming_batches.remote(
            num_batches=num_batches,
            batch_size=batch_size,
            tensor_shape=test_tensor_shape,
            producer_id=producer_id,
            inter_batch_delay=0.0,
            deterministic=True
        )
        
        all_batches = ray.get(future)
        
        # Verify structure
        assert len(all_batches) == num_batches
        
        for batch in all_batches:
            assert len(batch) == batch_size
            assert all(isinstance(ref, ray.ObjectRef) for ref in batch)
            
            # Verify tensor shapes
            tensors = ray.get(batch)
            for tensor in tensors:
                assert tensor.shape == test_tensor_shape
    
    def test_inter_batch_delay(self, ray_context, test_tensor_shape):
        """Test that inter-batch delay works correctly."""
        num_batches = 2
        batch_size = 1
        delay = 0.1  # 100ms delay
        
        start_time = time.time()
        
        future = generate_streaming_batches.remote(
            num_batches=num_batches,
            batch_size=batch_size,
            tensor_shape=test_tensor_shape,
            producer_id=0,
            inter_batch_delay=delay
        )
        
        result = ray.get(future)
        end_time = time.time()
        
        # Should take at least the delay time (num_batches - 1) * delay
        expected_min_time = (num_batches - 1) * delay
        actual_time = end_time - start_time
        
        assert actual_time >= expected_min_time, f"Expected at least {expected_min_time}s, got {actual_time}s"
        assert len(result) == num_batches
    
    def test_producer_id_determinism(self, ray_context, test_tensor_shape):
        """Test that different producer IDs generate different content."""
        num_batches = 2
        batch_size = 1
        
        # Generate with producer_id=0
        future1 = generate_streaming_batches.remote(
            num_batches=num_batches,
            batch_size=batch_size,
            tensor_shape=test_tensor_shape,
            producer_id=0,
            deterministic=True
        )
        
        # Generate with producer_id=1
        future2 = generate_streaming_batches.remote(
            num_batches=num_batches,
            batch_size=batch_size,
            tensor_shape=test_tensor_shape,
            producer_id=1,
            deterministic=True
        )
        
        batches1 = ray.get(future1)
        batches2 = ray.get(future2)
        
        # Get first tensors from each
        tensor1 = ray.get(batches1[0][0])
        tensor2 = ray.get(batches2[0][0])
        
        # Should be different due to different producer_id seeds
        assert not torch.equal(tensor1, tensor2), "Different producer IDs should generate different content"


class TestRayDataProducerManager:
    """Test the RayDataProducerManager coordination class."""
    
    def test_single_producer_launch(self, ray_context, test_tensor_shape):
        """Test launching a single producer."""
        manager = RayDataProducerManager()
        
        producer_futures = manager.launch_producers(
            num_producers=1,
            batches_per_producer=2,
            batch_size=2,
            tensor_shape=test_tensor_shape,
            deterministic=True
        )
        
        assert len(producer_futures) == 1
        assert all(isinstance(future, ray.ObjectRef) for future in producer_futures)
        
        # Get all batches
        all_batches = manager.get_all_batches()
        
        # Should have 1 producer * 2 batches = 2 total batches
        assert len(all_batches) == 2
        
        # Each batch should have 2 tensors
        for batch in all_batches:
            assert len(batch) == 2
            tensors = ray.get(batch)
            for tensor in tensors:
                assert tensor.shape == test_tensor_shape
    
    def test_multiple_producer_launch(self, ray_context, test_tensor_shape):
        """Test launching multiple producers."""
        manager = RayDataProducerManager()
        
        num_producers = 3
        batches_per_producer = 2
        batch_size = 2
        
        producer_futures = manager.launch_producers(
            num_producers=num_producers,
            batches_per_producer=batches_per_producer,
            batch_size=batch_size,
            tensor_shape=test_tensor_shape,
            deterministic=True
        )
        
        assert len(producer_futures) == num_producers
        
        # Get all batches
        all_batches = manager.get_all_batches()
        
        # Should have num_producers * batches_per_producer total batches
        expected_total = num_producers * batches_per_producer
        assert len(all_batches) == expected_total
        
        # Verify all batches have correct structure
        for batch in all_batches:
            assert len(batch) == batch_size
            tensors = ray.get(batch)
            for tensor in tensors:
                assert tensor.shape == test_tensor_shape
    
    def test_batch_iterator(self, ray_context, test_tensor_shape):
        """Test the batch iterator functionality."""
        manager = RayDataProducerManager()
        
        manager.launch_producers(
            num_producers=2,
            batches_per_producer=2,
            batch_size=1,
            tensor_shape=test_tensor_shape,
            deterministic=True
        )
        
        # Test iterator
        batch_iterator = manager.get_batch_iterator()
        batches_from_iterator = list(batch_iterator)
        
        # Should have same content as get_all_batches
        all_batches = manager.get_all_batches()
        
        assert len(batches_from_iterator) == len(all_batches)
        assert len(batches_from_iterator) == 4  # 2 producers * 2 batches each
    
    def test_manager_state_isolation(self, ray_context, test_tensor_shape):
        """Test that different manager instances are isolated."""
        manager1 = RayDataProducerManager()
        manager2 = RayDataProducerManager()
        
        # Launch different numbers of producers
        manager1.launch_producers(
            num_producers=1,
            batches_per_producer=1,
            batch_size=1,
            tensor_shape=test_tensor_shape
        )
        
        manager2.launch_producers(
            num_producers=2,
            batches_per_producer=1,
            batch_size=1,
            tensor_shape=test_tensor_shape
        )
        
        batches1 = manager1.get_all_batches()
        batches2 = manager2.get_all_batches()
        
        # Should have different numbers of batches
        assert len(batches1) == 1
        assert len(batches2) == 2


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_tensor_shape(self, ray_context):
        """Test handling of invalid tensor shapes."""
        with pytest.raises(Exception):
            future = generate_data_batch.remote(
                batch_size=1,
                tensor_shape=(),  # Empty shape should fail
                batch_id=0
            )
            ray.get(future)
    
    def test_zero_batch_size(self, ray_context, test_tensor_shape):
        """Test handling of zero batch size."""
        future = generate_data_batch.remote(
            batch_size=0,
            tensor_shape=test_tensor_shape,
            batch_id=0
        )
        
        refs = ray.get(future)
        assert len(refs) == 0
    
    def test_zero_batches(self, ray_context, test_tensor_shape):
        """Test handling of zero batches in streaming generation."""
        future = generate_streaming_batches.remote(
            num_batches=0,
            batch_size=1,
            tensor_shape=test_tensor_shape,
            producer_id=0
        )
        
        batches = ray.get(future)
        assert len(batches) == 0
    
    def test_negative_delay(self, ray_context, test_tensor_shape):
        """Test handling of negative inter-batch delay."""
        # Should not crash, should treat as zero delay
        future = generate_streaming_batches.remote(
            num_batches=2,
            batch_size=1,
            tensor_shape=test_tensor_shape,
            producer_id=0,
            inter_batch_delay=-0.1
        )
        
        batches = ray.get(future)
        assert len(batches) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])