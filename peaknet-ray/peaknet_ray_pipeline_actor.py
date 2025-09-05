#!/usr/bin/env python3
"""
Ray Pipeline Actor - Wrap DoubleBufferedPipeline in Ray actor for multi-GPU scaling

Each actor maintains a DoubleBufferedPipeline instance and processes data from
Ray's object store. Preserves all nvtx annotations for nsys profiling.

Adapted for PeakNet segmentation models.
"""

import ray
import torch
import torch.cuda.nvtx as nvtx
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import psutil
import os

# Import existing pipeline components
from peaknet_pipeline import DoubleBufferedPipeline, create_peaknet_model, get_numa_info, get_gpu_info
from peaknet_utils import get_peaknet_input_shape, get_peaknet_output_shape
# GPU health validation now handled at system level before Ray initialization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PeakNetPipelineActorBase:
    """
    Ray actor that wraps DoubleBufferedPipeline for distributed processing.
    
    Each actor maintains:
    - A loaded PeakNet model 
    - DoubleBufferedPipeline instance
    - GPU assignment from Ray
    - Statistics tracking
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 512, 512),
        batch_size: int = 10,
        yaml_path: str = None,
        weights_path: str = None,
        peaknet_config: dict = None,
        pin_memory: bool = True,
        compile_model: bool = False,
        compile_mode: str = 'default',
        deterministic: bool = False,
        gpu_id: int = None
    ):
        """
        Initialize the pipeline actor.
        
        Args:
            input_shape: Input tensor shape (C, H, W)
            batch_size: Batch size for processing
            yaml_path: Path to PeakNet YAML configuration (legacy)
            weights_path: Path to PeakNet model weights
            peaknet_config: Hydra PeakNet configuration dict (new integrated mode)
            pin_memory: Use pinned memory
            compile_model: Whether to compile the model
            compile_mode: Torch compile mode
            deterministic: Use deterministic operations
            gpu_id: Explicit GPU ID to use (None for Ray auto-assignment)
        """
        logging.info("=== Initializing PeakNetPipelineActor ===")
        
        # Set deterministic behavior if requested
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(42)
        
        # GPU assignment: Trust Ray's resource allocation
        # Ray automatically manages GPU assignment via CUDA_VISIBLE_DEVICES
        if gpu_id is not None:
            # Explicit GPU ID provided (for testing/debugging only)
            self.gpu_id = gpu_id
            logging.info(f"Using explicitly assigned GPU {self.gpu_id}")
        else:
            # Ray-native approach: assigned GPU always appears as cuda:0
            self.gpu_id = 0
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
            if cuda_visible != 'not set':
                physical_gpu = cuda_visible.split(',')[0]
                logging.info(f"Using Ray-assigned GPU device 0 (physical GPU {physical_gpu})")
            else:
                logging.warning("CUDA_VISIBLE_DEVICES not set by Ray - using device 0")
        
        logging.info(f"✅ Actor GPU assignment complete - using CUDA device {self.gpu_id}")
        
        # Verify CUDA_VISIBLE_DEVICES is set correctly
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        logging.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # CUDA context establishment: Minimal warmup to ensure GPU is accessible
        try:
            # Simple GPU context establishment
            with torch.cuda.device(self.gpu_id):
                # Create small tensor to initialize CUDA context
                warmup_tensor = torch.ones(2, 2, device=f'cuda:{self.gpu_id}')
                _ = warmup_tensor + warmup_tensor  # Simple operation
                torch.cuda.synchronize(self.gpu_id)
                del warmup_tensor
            
            logging.info(f"✅ CUDA context established on device {self.gpu_id}")
            
        except Exception as e:
            logging.error(f"❌ CUDA context initialization failed: {e}")
            raise RuntimeError(f"CUDA context failed: {e}")
        
        # Store configuration
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.yaml_path = yaml_path
        self.weights_path = weights_path
        self.peaknet_config = peaknet_config
        self.pin_memory = pin_memory
        self.deterministic = deterministic
        
        # Get system info
        self.numa_info = get_numa_info()
        self.gpu_info = get_gpu_info(self.gpu_id)
        
        logging.info(f"Actor GPU: {self.gpu_info.get('name', 'Unknown')} ({self.gpu_info.get('memory_mb', 0):.0f} MB)")
        logging.info(f"Actor CPU affinity: {self.numa_info.get('cpu_ranges', 'unknown')}")
        
        # Create PeakNet model
        self.peaknet_model, model_input_shape, model_output_shape = create_peaknet_model(
            yaml_path=yaml_path,
            weights_path=weights_path,
            gpu_id=self.gpu_id,
            compile_model=compile_model,
            compile_mode=compile_mode,
            hydra_config=peaknet_config
        )
        
        # Calculate shapes for pipeline
        if self.peaknet_model is None:
            # No-op mode: output shape same as input shape
            self.input_shape = input_shape
            self.output_shape = input_shape
            logging.info(f"No-op mode: input_shape={self.input_shape}, output_shape={self.output_shape}")
        else:
            # PeakNet mode: use shapes from model configuration
            self.input_shape = model_input_shape
            self.output_shape = model_output_shape
            logging.info(f"PeakNet mode: input_shape={self.input_shape}, output_shape={self.output_shape}")
        
        # Create double buffered pipeline - use the same GPU device as assigned to this actor
        pipeline_gpu_id = self.gpu_id
        
        self.pipeline = DoubleBufferedPipeline(
            model=self.peaknet_model,
            batch_size=batch_size,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            gpu_id=pipeline_gpu_id,
            pin_memory=pin_memory
        )
        
        # Initialize statistics
        self.stats = {
            'batches_processed': 0,
            'samples_processed': 0,
            'total_time': 0.0,
            'gpu_id': self.gpu_id,
            'actor_id': f"actor_{self.gpu_id}_{os.getpid()}",
            'model_config': {
                'yaml_path': yaml_path,
                'weights_path': weights_path,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape
            }
        }
        
        logging.info(f"✅ PeakNetPipelineActor initialized successfully on GPU {self.gpu_id}")
        logging.info(f"Model: yaml_path={yaml_path}, input_shape={self.input_shape}, output_shape={self.output_shape}")
    
    def get_actor_info(self) -> Dict[str, Any]:
        """Return actor information and current statistics."""
        return {
            'gpu_id': self.gpu_id,
            'gpu_info': self.gpu_info,
            'numa_info': self.numa_info,
            'model_config': self.stats['model_config'],
            'pipeline_config': {
                'batch_size': self.batch_size,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'pin_memory': self.pin_memory
            },
            'stats': self.stats.copy()
        }
    
    def process_batch_from_refs(self, batch_object_refs: List[ray.ObjectRef], batch_id: int) -> Dict[str, Any]:
        """
        Process a batch of tensors from Ray object references.
        
        Args:
            batch_object_refs: List of Ray object references to input tensors
            batch_id: Unique batch identifier
            
        Returns:
            Dictionary with processing results and statistics
        """
        with nvtx.range(f"ray_actor_process_batch_{batch_id}"):
            start_time = time.time()
            
            # Get tensors from Ray object store
            with nvtx.range(f"ray_get_tensors_{batch_id}"):
                cpu_tensors = ray.get(batch_object_refs)
            
            actual_batch_size = len(cpu_tensors)
            
            logging.debug(f"Actor {self.gpu_id}: Processing batch {batch_id} with {actual_batch_size} tensors")
            
            # Process through pipeline
            with nvtx.range(f"pipeline_process_{batch_id}"):
                # Swap to next buffer (except for first batch)
                if self.stats['batches_processed'] > 0:
                    self.pipeline.swap()
                
                # Process the batch through the full pipeline
                self.pipeline.process_batch(
                    cpu_batch=cpu_tensors,
                    batch_idx=batch_id,
                    current_batch_size=actual_batch_size,
                    nvtx_prefix=f"ray_actor_{self.gpu_id}"
                )
            
            # Note: Removed pipeline sync to allow overlap between batches
            # Pipeline completion will be handled at batch list level
            
            end_time = time.time()
            batch_time = end_time - start_time
            
            # Update statistics
            self.stats['batches_processed'] += 1
            self.stats['samples_processed'] += actual_batch_size
            self.stats['total_time'] += batch_time
            
            result = {
                'batch_id': batch_id,
                'batch_size': actual_batch_size,
                'processing_time': batch_time,
                'gpu_id': self.gpu_id,
                'throughput': actual_batch_size / batch_time if batch_time > 0 else 0
            }
            
            logging.debug(f"Actor {self.gpu_id}: Batch {batch_id} completed in {batch_time:.4f}s ({result['throughput']:.1f} samples/s)")
            
            return result
    
    def process_batch_list(self, batch_list: List[List[ray.ObjectRef]]) -> List[Dict[str, Any]]:
        """
        Process multiple batches sequentially.
        
        Args:
            batch_list: List of batches, each batch is a list of object references
            
        Returns:
            List of processing results for each batch
        """
        with nvtx.range(f"ray_actor_process_batch_list"):
            logging.info(f"Actor {self.gpu_id}: Processing {len(batch_list)} batches")
            
            results = []
            for batch_idx, batch_refs in enumerate(batch_list):
                result = self.process_batch_from_refs(batch_refs, batch_idx)
                results.append(result)
            
            # Final synchronization - only at the very end to allow pipeline overlap
            with nvtx.range("final_pipeline_sync"):
                self.pipeline.wait_for_completion()
                # Note: Removed explicit GPU sync to let Ray handle synchronization
            
            logging.info(f"Actor {self.gpu_id}: Completed {len(batch_list)} batches")
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return current processing statistics."""
        stats = self.stats.copy()
        
        if stats['total_time'] > 0:
            stats['average_throughput'] = stats['samples_processed'] / stats['total_time']
            stats['average_batch_time'] = stats['total_time'] / max(stats['batches_processed'], 1)
        else:
            stats['average_throughput'] = 0
            stats['average_batch_time'] = 0
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats.update({
            'batches_processed': 0,
            'samples_processed': 0,
            'total_time': 0.0
        })
        logging.info(f"Actor {self.gpu_id}: Statistics reset")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        try:
            # Check GPU availability
            torch.cuda.is_available()
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory
            gpu_memory_used = torch.cuda.memory_allocated(self.gpu_id)
            gpu_memory_cached = torch.cuda.memory_reserved(self.gpu_id)
            
            # Simple GPU computation test
            with torch.cuda.device(self.gpu_id):
                test_tensor = torch.randn(100, 100, device=f'cuda:{self.gpu_id}')
                _ = test_tensor.sum()
            
            return {
                'status': 'healthy',
                'gpu_id': self.gpu_id,
                'gpu_memory_total_mb': gpu_memory / (1024 * 1024),
                'gpu_memory_used_mb': gpu_memory_used / (1024 * 1024),
                'gpu_memory_cached_mb': gpu_memory_cached / (1024 * 1024),
                'model_loaded': self.peaknet_model is not None,
                'pipeline_initialized': self.pipeline is not None
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'gpu_id': self.gpu_id
            }


# Create Ray actor classes from the base
@ray.remote(num_gpus=1)
class PeakNetPipelineActor(PeakNetPipelineActorBase):
    """Ray actor for PeakNet pipeline processing without profiling."""
    pass


@ray.remote(num_gpus=1, runtime_env={"nsight": "default"})
class PeakNetPipelineActorWithProfiling(PeakNetPipelineActorBase):
    """Ray actor for PeakNet pipeline processing with nsys profiling enabled."""
    pass


def create_pipeline_actors(
    num_actors: int,
    enable_profiling: bool = False,
    validate_gpus: bool = True,
    **pipeline_kwargs
) -> List[ray.actor.ActorHandle]:
    """
    Create multiple PeakNet pipeline actors with optional GPU health validation.
    
    Args:
        num_actors: Number of actors to create
        enable_profiling: Whether to enable nsys profiling
        validate_gpus: Whether to pre-validate GPU health before actor creation
        **pipeline_kwargs: Arguments passed to actor constructor
        
    Returns:
        List of Ray actor handles
    """
    logging.info(f"Creating {num_actors} PeakNet pipeline actors (profiling={'enabled' if enable_profiling else 'disabled'})")
    
    # GPU validation is now handled at system level before Ray initialization
    # All GPUs Ray sees are guaranteed healthy
    if validate_gpus:
        logging.info("GPU health validation handled at system level - all Ray GPUs are pre-validated")
    
    # Choose actor class based on profiling preference
    actor_class = PeakNetPipelineActorWithProfiling if enable_profiling else PeakNetPipelineActor
    
    actors = []
    for i in range(num_actors):
        try:
            actor = actor_class.remote(**pipeline_kwargs)
            actors.append(actor)
            logging.info(f"✅ Created actor {i+1}/{num_actors}")
        except Exception as e:
            logging.error(f"❌ Failed to create actor {i+1}/{num_actors}: {e}")
            # Continue trying to create remaining actors
    
    if len(actors) == 0:
        raise RuntimeError("Failed to create any pipeline actors")
    elif len(actors) < num_actors:
        logging.warning(f"Only created {len(actors)}/{num_actors} requested actors")
    
    logging.info(f"✅ Successfully created {len(actors)} pipeline actors")
    return actors


def test_pipeline_actor():
    """Simple test of pipeline actor functionality."""
    if not ray.is_initialized():
        ray.init()
    
    logging.info("Testing Ray pipeline actor...")
    
    # Create single actor
    actor = PeakNetPipelineActor.remote(
        input_shape=(1, 512, 512),
        batch_size=4,
        yaml_path=None,  # No-op mode for testing
        deterministic=True
    )
    
    # Test health check
    health = ray.get(actor.health_check.remote())
    logging.info(f"Actor health: {health['status']}")
    
    # Generate test data
    test_tensors = []
    for i in range(4):
        tensor = torch.randn(1, 512, 512)
        test_tensors.append(ray.put(tensor))
    
    # Process batch
    result = ray.get(actor.process_batch_from_refs.remote(test_tensors, 0))
    logging.info(f"Batch result: {result}")
    
    # Get statistics
    stats = ray.get(actor.get_statistics.remote())
    logging.info(f"Actor stats: {stats}")
    
    logging.info("✅ Pipeline actor test passed!")


if __name__ == "__main__":
    test_pipeline_actor()