#!/usr/bin/env python3
"""
Ray Pipeline Actor - Wrap DoubleBufferedPipeline in Ray actor for multi-GPU scaling

Each actor maintains a DoubleBufferedPipeline instance and processes data from
Ray's object store. Preserves all nvtx annotations for nsys profiling.
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
from vit_pipeline import DoubleBufferedPipeline, create_vit_model, get_numa_info, get_gpu_info
# GPU health validation now handled at system level before Ray initialization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VitPipelineActorBase:
    """
    Ray actor that wraps DoubleBufferedPipeline for distributed processing.
    
    Each actor maintains:
    - A loaded VIT model 
    - DoubleBufferedPipeline instance
    - GPU assignment from Ray
    - Statistics tracking
    """
    
    def __init__(
        self,
        tensor_shape: Tuple[int, int, int] = (3, 224, 224),
        batch_size: int = 10,
        patch_size: int = 32,
        depth: int = 6,
        heads: int = 8,
        dim: int = 512,
        mlp_dim: int = 2048,
        pin_memory: bool = True,
        compile_model: bool = False,
        compile_mode: str = 'default',
        deterministic: bool = False,
        gpu_id: int = None
    ):
        """
        Initialize the pipeline actor.
        
        Args:
            tensor_shape: Input tensor shape (C, H, W)
            batch_size: Batch size for processing
            patch_size: ViT patch size
            depth: ViT depth (0 for no-op mode)
            heads: ViT attention heads
            dim: ViT dimension
            mlp_dim: ViT MLP dimension
            pin_memory: Use pinned memory
            compile_model: Whether to compile the model
            compile_mode: Torch compile mode
            deterministic: Use deterministic operations
            gpu_id: Explicit GPU ID to use (None for Ray auto-assignment)
        """
        logging.info("=== Initializing VitPipelineActor ===")
        
        # Set deterministic behavior if requested
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(42)
        
        # GPU assignment with our pre-Ray health validation architecture
        if gpu_id is not None:
            # Explicit GPU ID provided (for testing/debugging)
            self.gpu_id = gpu_id
            torch.cuda.set_device(self.gpu_id)
            logging.info(f"Using explicitly assigned GPU {self.gpu_id}")
        else:
            # With pre-Ray GPU validation, CUDA_VISIBLE_DEVICES is always set to healthy GPUs
            # PyTorch sees a clean device space: 0, 1, 2, ... (mapped from healthy physical GPUs)
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            if cuda_visible:
                # CUDA_VISIBLE_DEVICES is set - trust the pre-Ray validation
                # Always use device 0, which maps to the first healthy physical GPU
                self.gpu_id = 0  # Logical device 0 in the filtered device space
                torch.cuda.set_device(0)
                logging.info(f"Using CUDA device 0 (pre-validated healthy GPU via CUDA_VISIBLE_DEVICES={cuda_visible})")
            else:
                # Fallback: No CUDA_VISIBLE_DEVICES set, use Ray's GPU assignment
                try:
                    # Try Ray's new runtime context first
                    runtime_context = ray.get_runtime_context()
                    assigned_gpus = runtime_context.get_accelerator_ids().get("GPU", [])
                    
                    if assigned_gpus:
                        self.gpu_id = assigned_gpus[0]
                        torch.cuda.set_device(self.gpu_id)
                        logging.info(f"Using Ray runtime assigned GPU: {self.gpu_id}")
                    else:
                        # Fall back to legacy Ray method
                        legacy_gpu_ids = ray.get_gpu_ids()
                        if not legacy_gpu_ids:
                            raise RuntimeError("No GPU assigned to this actor by Ray")
                        self.gpu_id = legacy_gpu_ids[0]
                        torch.cuda.set_device(self.gpu_id)
                        logging.info(f"Using Ray legacy assigned GPU: {self.gpu_id}")
                        
                except Exception as e:
                    logging.error(f"Failed to get Ray GPU assignment: {e}")
                    raise RuntimeError(f"GPU assignment failed: {e}")
        
        logging.info(f"✅ Actor GPU assignment complete - using CUDA device {self.gpu_id}")
        
        # Verify CUDA_VISIBLE_DEVICES is set correctly
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        logging.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # Store configuration
        self.tensor_shape = tensor_shape
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.depth = depth
        self.pin_memory = pin_memory
        self.deterministic = deterministic
        
        # Get system info
        self.numa_info = get_numa_info()
        self.gpu_info = get_gpu_info(self.gpu_id)
        
        logging.info(f"Actor GPU: {self.gpu_info.get('name', 'Unknown')} ({self.gpu_info.get('memory_mb', 0):.0f} MB)")
        logging.info(f"Actor CPU affinity: {self.numa_info.get('cpu_ranges', 'unknown')}")
        
        # Create VIT model
        self.vit_model, self.image_size = create_vit_model(
            tensor_shape=tensor_shape,
            patch_size=patch_size,
            depth=depth,
            heads=heads,
            dim=dim,
            mlp_dim=mlp_dim,
            gpu_id=self.gpu_id,
            compile_model=compile_model,
            compile_mode=compile_mode
        )
        
        # Calculate shapes for pipeline
        self.input_shape = tensor_shape
        if self.vit_model is None:
            # No-op mode: output shape same as input shape
            self.output_shape = tensor_shape
        else:
            # ViT mode: output shape is transformer output (num_patches + 1, dim)
            num_patches = (self.image_size // patch_size) ** 2
            self.output_shape = (num_patches + 1, dim)
        
        # Create double buffered pipeline
        # When using CUDA_VISIBLE_DEVICES, always use device 0 for pipeline operations
        pipeline_gpu_id = 0 if os.environ.get('CUDA_VISIBLE_DEVICES') else self.gpu_id
        
        self.pipeline = DoubleBufferedPipeline(
            model=self.vit_model,
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
                'patch_size': patch_size,
                'depth': depth,
                'heads': heads,
                'dim': dim,
                'mlp_dim': mlp_dim
            }
        }
        
        logging.info(f"✅ VitPipelineActor initialized successfully on GPU {self.gpu_id}")
        logging.info(f"Model: depth={depth}, input_shape={self.input_shape}, output_shape={self.output_shape}")
    
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
            
            # Wait for pipeline completion to get accurate timing
            with nvtx.range(f"pipeline_sync_{batch_id}"):
                self.pipeline.wait_for_completion()
            
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
            
            # Final synchronization
            with nvtx.range("final_pipeline_sync"):
                self.pipeline.wait_for_completion()
                torch.cuda.synchronize(device=self.gpu_id)
            
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
                'model_loaded': self.vit_model is not None,
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
class VitPipelineActor(VitPipelineActorBase):
    """Ray actor for VIT pipeline processing without profiling."""
    pass


@ray.remote(num_gpus=1, runtime_env={"nsight": "default"})
class VitPipelineActorWithProfiling(VitPipelineActorBase):
    """Ray actor for VIT pipeline processing with nsys profiling enabled."""
    pass


def create_pipeline_actors(
    num_actors: int,
    enable_profiling: bool = False,
    validate_gpus: bool = True,
    **pipeline_kwargs
) -> List[ray.actor.ActorHandle]:
    """
    Create multiple VIT pipeline actors with optional GPU health validation.
    
    Args:
        num_actors: Number of actors to create
        enable_profiling: Whether to enable nsys profiling
        validate_gpus: Whether to pre-validate GPU health before actor creation
        **pipeline_kwargs: Arguments passed to actor constructor
        
    Returns:
        List of Ray actor handles
    """
    logging.info(f"Creating {num_actors} VIT pipeline actors (profiling={'enabled' if enable_profiling else 'disabled'})")
    
    # GPU validation is now handled at system level before Ray initialization
    # All GPUs Ray sees are guaranteed healthy
    if validate_gpus:
        logging.info("GPU health validation handled at system level - all Ray GPUs are pre-validated")
    
    # Choose actor class based on profiling preference
    actor_class = VitPipelineActorWithProfiling if enable_profiling else VitPipelineActor
    
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
    actor = VitPipelineActor.remote(
        tensor_shape=(3, 224, 224),
        batch_size=4,
        depth=2,  # Small model for testing
        deterministic=True
    )
    
    # Test health check
    health = ray.get(actor.health_check.remote())
    logging.info(f"Actor health: {health['status']}")
    
    # Generate test data
    test_tensors = []
    for i in range(4):
        tensor = torch.randn(3, 224, 224)
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