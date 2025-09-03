#!/usr/bin/env python3
"""
Ray Data Producer - Simulate streaming data source using Ray tasks

Generates random tensors and puts them into Ray's object store to simulate
streaming data that will be consumed by Ray pipeline actors.
"""

import ray
import torch
import numpy as np
import time
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@ray.remote
def generate_data_batch(
    batch_size: int,
    tensor_shape: Tuple[int, int, int],
    batch_id: int,
    pin_memory: bool = True,
    deterministic: bool = False
) -> List[ray.ObjectRef]:
    """
    Generate a batch of random tensors and put them in Ray object store.
    
    Args:
        batch_size: Number of tensors to generate in this batch
        tensor_shape: Shape of each tensor (C, H, W)
        batch_id: Unique identifier for this batch
        pin_memory: Whether to pin memory for faster GPU transfers
        deterministic: Use fixed seed for reproducible results
        
    Returns:
        List of Ray object references to the generated tensors
    """
    if deterministic:
        np.random.seed(42 + batch_id)
        torch.manual_seed(42 + batch_id)
    
    logging.info(f"Producer generating batch {batch_id} with {batch_size} tensors")
    
    # Generate tensors in host memory
    tensors = []
    for i in range(batch_size):
        tensor = torch.randn(*tensor_shape)
        
        if pin_memory and torch.cuda.is_available():
            tensor = tensor.pin_memory()
            
        tensors.append(tensor)
    
    # Put tensors into Ray object store
    object_refs = []
    for i, tensor in enumerate(tensors):
        obj_ref = ray.put(tensor)
        object_refs.append(obj_ref)
    
    logging.info(f"Producer batch {batch_id} complete: {len(object_refs)} tensors in object store")
    return object_refs


@ray.remote
def generate_streaming_batches(
    num_batches: int,
    batch_size: int, 
    tensor_shape: Tuple[int, int, int],
    producer_id: int,
    inter_batch_delay: float = 0.0,
    pin_memory: bool = True,
    deterministic: bool = False
) -> List[List[ray.ObjectRef]]:
    """
    Generate multiple batches with optional delays to simulate streaming.
    
    Args:
        num_batches: Number of batches to generate
        batch_size: Number of tensors per batch
        tensor_shape: Shape of each tensor (C, H, W)  
        producer_id: Unique identifier for this producer
        inter_batch_delay: Delay between batches in seconds
        pin_memory: Whether to pin memory for faster GPU transfers
        deterministic: Use fixed seed for reproducible results
        
    Returns:
        List of batches, each batch being a list of object references
    """
    if deterministic:
        np.random.seed(1000 + producer_id)
        torch.manual_seed(1000 + producer_id)
    
    logging.info(f"Streaming producer {producer_id} starting: {num_batches} batches")
    
    all_batches = []
    
    for batch_idx in range(num_batches):
        global_batch_id = producer_id * num_batches + batch_idx
        
        # Generate batch
        batch_object_refs = []
        tensors = []
        
        for i in range(batch_size):
            tensor = torch.randn(*tensor_shape)
            if pin_memory and torch.cuda.is_available():
                tensor = tensor.pin_memory()
            tensors.append(tensor)
        
        # Put all tensors from this batch into object store
        for tensor in tensors:
            obj_ref = ray.put(tensor)
            batch_object_refs.append(obj_ref)
        
        all_batches.append(batch_object_refs)
        
        logging.info(f"Producer {producer_id} completed batch {batch_idx}/{num_batches-1}")
        
        # Simulate streaming delay
        if inter_batch_delay > 0 and batch_idx < num_batches - 1:
            time.sleep(inter_batch_delay)
    
    logging.info(f"Streaming producer {producer_id} finished: {len(all_batches)} batches total")
    return all_batches


class RayDataProducerManager:
    """
    Manager for coordinating multiple Ray data producer tasks.
    """
    
    def __init__(self):
        self.producer_tasks = []
        self.all_batch_refs = []
    
    def launch_producers(
        self,
        num_producers: int,
        batches_per_producer: int,
        batch_size: int,
        tensor_shape: Tuple[int, int, int],
        inter_batch_delay: float = 0.0,
        pin_memory: bool = True,
        deterministic: bool = False
    ) -> List[ray.ObjectRef]:
        """
        Launch multiple producer tasks to generate data concurrently.
        
        Args:
            num_producers: Number of concurrent producer tasks
            batches_per_producer: Number of batches each producer generates
            batch_size: Number of tensors per batch
            tensor_shape: Shape of each tensor (C, H, W)
            inter_batch_delay: Delay between batches in seconds
            pin_memory: Whether to pin memory for faster GPU transfers  
            deterministic: Use fixed seed for reproducible results
            
        Returns:
            List of Ray object references to producer task futures
        """
        logging.info(f"Launching {num_producers} data producers")
        logging.info(f"Each producer will generate {batches_per_producer} batches of {batch_size} tensors")
        logging.info(f"Tensor shape: {tensor_shape}")
        
        # Launch all producer tasks
        producer_futures = []
        for producer_id in range(num_producers):
            future = generate_streaming_batches.remote(
                num_batches=batches_per_producer,
                batch_size=batch_size,
                tensor_shape=tensor_shape,
                producer_id=producer_id,
                inter_batch_delay=inter_batch_delay,
                pin_memory=pin_memory,
                deterministic=deterministic
            )
            producer_futures.append(future)
        
        self.producer_tasks = producer_futures
        return producer_futures
    
    def get_all_batches(self) -> List[List[ray.ObjectRef]]:
        """
        Wait for all producers to complete and return all generated batches.
        
        Returns:
            Flattened list of all batches from all producers
        """
        logging.info("Waiting for all producers to complete...")
        
        # Get results from all producers
        producer_results = ray.get(self.producer_tasks)
        
        # Flatten the results - each producer returns a list of batches
        all_batches = []
        for producer_batches in producer_results:
            all_batches.extend(producer_batches)
        
        logging.info(f"All producers completed: {len(all_batches)} total batches available")
        return all_batches
    
    def get_batch_iterator(self):
        """
        Return an iterator that yields batches as they become available.
        This enables streaming consumption without waiting for all producers.
        """
        # For now, implement simple version - can be enhanced for true streaming
        all_batches = self.get_all_batches()
        return iter(all_batches)


def test_data_producer():
    """Simple test of data producer functionality."""
    if not ray.is_initialized():
        ray.init()
    
    logging.info("Testing Ray data producer...")
    
    # Test parameters
    tensor_shape = (3, 224, 224)
    batch_size = 4
    num_producers = 2
    batches_per_producer = 3
    
    # Launch producers
    manager = RayDataProducerManager()
    producer_futures = manager.launch_producers(
        num_producers=num_producers,
        batches_per_producer=batches_per_producer,
        batch_size=batch_size,
        tensor_shape=tensor_shape,
        inter_batch_delay=0.1,
        deterministic=True
    )
    
    # Get all batches
    all_batches = manager.get_all_batches()
    
    # Verify results
    expected_total_batches = num_producers * batches_per_producer
    assert len(all_batches) == expected_total_batches, f"Expected {expected_total_batches} batches, got {len(all_batches)}"
    
    # Verify first batch
    first_batch = all_batches[0]
    assert len(first_batch) == batch_size, f"Expected batch size {batch_size}, got {len(first_batch)}"
    
    # Verify tensor shape by getting one tensor
    first_tensor = ray.get(first_batch[0])
    assert first_tensor.shape == tensor_shape, f"Expected tensor shape {tensor_shape}, got {first_tensor.shape}"
    
    logging.info("✅ Data producer test passed!")
    logging.info(f"Generated {len(all_batches)} batches with {batch_size} tensors each")


if __name__ == "__main__":
    test_data_producer()