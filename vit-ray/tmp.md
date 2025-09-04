####
# ray_pipeline_actor.py
####

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
        
        # GPU assignment: Let Ray handle device assignment properly
        if gpu_id is not None:
            # Explicit GPU ID provided (for testing/debugging)
            self.gpu_id = gpu_id
            torch.cuda.set_device(self.gpu_id)
            logging.info(f"Using explicitly assigned GPU {self.gpu_id}")
        else:
            # Let Ray's scheduler assign GPUs - Ray sets CUDA_VISIBLE_DEVICES per actor
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            try:
                # Try Ray's new runtime context first
                runtime_context = ray.get_runtime_context()
                assigned_gpus = runtime_context.get_accelerator_ids().get("GPU", [])
                
                if cuda_visible:
                    # Ray manages CUDA_VISIBLE_DEVICES per actor - use device 0 in the visible space
                    self.gpu_id = 0  # Always use first visible device in filtered space
                    torch.cuda.set_device(self.gpu_id)
                    visible_devices = cuda_visible.split(',')
                    physical_gpu = visible_devices[0] if visible_devices else 'unknown'
                    logging.info(f"Using GPU device 0 in Ray's filtered space (physical GPU {physical_gpu})")
                elif assigned_gpus:
                    # Fallback to runtime context if CUDA_VISIBLE_DEVICES not set
                    try:
                        self.gpu_id = int(assigned_gpus[0])
                        torch.cuda.set_device(self.gpu_id)
                        logging.info(f"Using Ray runtime assigned GPU: {self.gpu_id}")
                    except ValueError:
                        # If conversion fails, use device 0
                        self.gpu_id = 0
                        torch.cuda.set_device(self.gpu_id)
                        logging.info(f"Using fallback GPU device 0 (runtime context parse failed)")  
                else:
                    # Fall back to legacy Ray method
                    legacy_gpu_ids = ray.get_gpu_ids()
                    if not legacy_gpu_ids:
                        raise RuntimeError("No GPU assigned to this actor by Ray")
                    self.gpu_id = int(legacy_gpu_ids[0])
                    torch.cuda.set_device(self.gpu_id)
                    logging.info(f"Using Ray legacy assigned GPU: {self.gpu_id}")
                    
            except Exception as e:
                logging.error(f"Failed to get Ray GPU assignment: {e}")
                raise RuntimeError(f"GPU assignment failed: {e}")
        
        logging.info(f"✅ Actor GPU assignment complete - using CUDA device {self.gpu_id}")
        
        # Verify CUDA_VISIBLE_DEVICES is set correctly
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        logging.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # CRITICAL: Force CUDA context initialization and verify GPU is working
        try:
            # Explicitly initialize CUDA and verify context
            torch.cuda.init()
            
            # Create and use tensors to force GPU context establishment
            with torch.cuda.device(self.gpu_id):
                # Create test tensors on GPU for warmup
                test_a = torch.randn(256, 256, device=f'cuda:{self.gpu_id}')
                test_b = torch.randn(256, 256, device=f'cuda:{self.gpu_id}')
                
                # Perform GPU computation to establish context
                for i in range(5):
                    test_c = torch.matmul(test_a, test_b)
                
                # Force synchronization to ensure GPU work completes
                torch.cuda.synchronize(self.gpu_id)
                
                # Cleanup test tensors
                del test_a, test_b, test_c
            
            logging.info(f"✅ CUDA context established - current device: {torch.cuda.current_device()}")
            
        except Exception as e:
            logging.error(f"❌ CUDA context initialization failed: {e}")
            raise RuntimeError(f"CUDA context failed: {e}")
        
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
        
        # Create double buffered pipeline - use the same GPU device as assigned to this actor
        pipeline_gpu_id = self.gpu_id
        
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

####
# vit_pipeline.py
####

#!/usr/bin/env python3
"""
GPU NUMA Pipeline Test with ViT and Double Buffering - EVENT-BASED SYNC VERSION

Test script to evaluate end-to-end pipeline performance with overlapping
H2D, Compute, and D2H stages using double buffering across NUMA nodes.

This version uses fine-grained CUDA events for synchronization instead of
stream dependencies for better parallelism and performance.

Usage with numactl:
  numactl --cpunodebind=0 --membind=0 python gpu_numa_pipeline_test.py --gpu-id=5
  numactl --cpunodebind=2 --membind=2 python gpu_numa_pipeline_test.py --gpu-id=3
"""

import torch
import torch.cuda.nvtx as nvtx
import time
import hydra
from omegaconf import DictConfig
import numpy as np
import psutil
import sys
import os

# Check for vit-pytorch availability
try:
    from vit_pytorch import ViT
    from vit_utils import ViTForProfiling
    VIT_AVAILABLE = True
except ImportError:
    print("ERROR: vit-pytorch not found. Please install with: pip install vit-pytorch")
    sys.exit(1)

def check_torch_compile_available():
    """Check if torch.compile is available (PyTorch 2.0+)"""
    try:
        import torch
        if hasattr(torch, 'compile'):
            return True
        else:
            return False
    except:
        return False

def get_numa_info():
    """Get current process NUMA binding info"""
    try:
        pid = os.getpid()
        proc = psutil.Process(pid)
        cpu_affinity = proc.cpu_affinity()
        return {
            'pid': pid,
            'cpu_affinity': cpu_affinity,
            'cpu_count': len(cpu_affinity),
            'cpu_ranges': _get_cpu_ranges(cpu_affinity)
        }
    except:
        return {'pid': os.getpid(), 'cpu_affinity': 'unknown', 'cpu_count': 'unknown'}

def _get_cpu_ranges(cpu_list):
    """Convert CPU list to readable ranges"""
    if not cpu_list or cpu_list == 'unknown':
        return 'unknown'

    sorted_cpus = sorted(cpu_list)
    ranges = []
    start = sorted_cpus[0]
    end = start

    for cpu in sorted_cpus[1:]:
        if cpu == end + 1:
            end = cpu
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = cpu

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ','.join(ranges)

def get_gpu_info(gpu_id):
    """Get GPU information"""
    try:
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}

        if gpu_id >= torch.cuda.device_count():
            return {'error': f'GPU {gpu_id} not available. Available: 0-{torch.cuda.device_count()-1}'}

        with torch.cuda.device(gpu_id):
            props = torch.cuda.get_device_properties(gpu_id)
            return {
                'name': props.name,
                'major': props.major,
                'minor': props.minor,
                'total_memory': props.total_memory,
                'multi_processor_count': props.multi_processor_count,
                'memory_mb': props.total_memory / (1024 * 1024),
                'compute_capability': f"{props.major}.{props.minor}"
            }
    except Exception as e:
        return {'error': str(e)}

def create_vit_model(tensor_shape, patch_size, depth, heads, dim, mlp_dim, gpu_id, compile_model=False, compile_mode='default'):
    """Create ViT model for compute simulation, or None for no-op"""
    C, H, W = tensor_shape

    # Ensure image size is compatible with patch size
    image_size = max(H, W)
    # Round up to nearest multiple of patch_size
    image_size = ((image_size + patch_size - 1) // patch_size) * patch_size

    # Handle no-op case
    if depth == 0:
        print("No-op compute mode: depth=0, skipping ViT model creation")
        return None, image_size

    # Normal ViT creation - using ViTForProfiling for larger D2H transfers
    vit_model = ViTForProfiling(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=1000,  # Standard ImageNet classes
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        channels=C,
        dropout=0.0,  # No dropout for consistent timing
        emb_dropout=0.0
    ).to(f'cuda:{gpu_id}')

    # Set to eval mode for consistent inference timing
    vit_model.eval()

    # Add torch.compile if requested and available
    if compile_model and check_torch_compile_available():
        print(f"Compiling ViT model (depth={depth}, dim={dim}) with mode={compile_mode}...")
        try:
            # Use specified compilation mode
            vit_model = torch.compile(vit_model, mode=compile_mode)
            print(f"Model compilation successful (mode={compile_mode})")
        except Exception as e:
            print(f"Warning: Model compilation failed with mode={compile_mode} ({e}), using non-compiled model")
    elif compile_model and not check_torch_compile_available():
        print("Warning: torch.compile not available (requires PyTorch 2.0+), using non-compiled model")

    return vit_model, image_size

class DoubleBufferedPipeline:
    """
    Generic double buffered pipeline for H2D -> Model Compute -> D2H.

    Provides a clean API with process_batch() method that handles the full pipeline.
    Internal methods are private to encourage proper encapsulation.
    """

    def __init__(self, model, batch_size, input_shape, output_shape, gpu_id, pin_memory=True):
        self.model = model
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gpu_id = gpu_id
        self.pin_memory = pin_memory

        # Check if model is None (no-op mode)
        self.is_noop = (self.model is None)
        
        # CRITICAL: Explicit CUDA device management for pipeline
        # Force device context establishment
        torch.cuda.set_device(gpu_id)
        
        # Verify GPU memory allocation capability (silent verification)
        initial_memory = torch.cuda.memory_allocated(gpu_id)

        # Create CUDA streams for pipeline stages
        self.h2d_stream = torch.cuda.Stream(device=gpu_id)
        self.compute_stream = torch.cuda.Stream(device=gpu_id)
        self.d2h_stream = torch.cuda.Stream(device=gpu_id)

        # CUDA events for fine-grained synchronization between all pipeline stages
        self.h2d_done_event = {
            'A': torch.cuda.Event(enable_timing=False),
            'B': torch.cuda.Event(enable_timing=False)
        }
        self.compute_done_event = {
            'A': torch.cuda.Event(enable_timing=False),
            'B': torch.cuda.Event(enable_timing=False)
        }
        self.d2h_done_event = {
            'A': torch.cuda.Event(enable_timing=False),
            'B': torch.cuda.Event(enable_timing=False)
        }
        # Prime all events so wait_event() never deadlocks on first use
        for events in [self.h2d_done_event, self.compute_done_event, self.d2h_done_event]:
            for ev in events.values():
                ev.record()  # Record on default stream makes them signaled immediately

        # GPU input buffers (use input_shape)
        self.gpu_input_buffers = {
            'A': torch.zeros(batch_size, *input_shape, device=f'cuda:{gpu_id}'),
            'B': torch.zeros(batch_size, *input_shape, device=f'cuda:{gpu_id}')
        }

        # GPU output buffers (use output_shape)
        self.gpu_output_buffers = {
            'A': torch.zeros(batch_size, *output_shape, device=f'cuda:{gpu_id}'),
            'B': torch.zeros(batch_size, *output_shape, device=f'cuda:{gpu_id}')
        }

        # CPU output buffers (use output_shape)
        self.cpu_output_buffers = {
            'A': torch.empty((batch_size, *output_shape), pin_memory=pin_memory),
            'B': torch.empty((batch_size, *output_shape), pin_memory=pin_memory)
        }

        # Pipeline state
        self.current = 'A'

    def swap(self):
        """Swap current buffer"""
        self.current = 'B' if self.current == 'A' else 'A'

    def _h2d_transfer(self, cpu_batch, batch_idx, current_batch_size, nvtx_prefix):
        """Perform H2D transfer with fine-grained event-based synchronization"""
        gpu_buffer = self.gpu_input_buffers[self.current]
        d2h_event = self.d2h_done_event[self.current]
        h2d_event = self.h2d_done_event[self.current]

        with torch.cuda.stream(self.h2d_stream):
            with nvtx.range(f"{nvtx_prefix}_h2d_batch_{batch_idx}"):
                # Fine-grained synchronization: wait only for THIS buffer's D2H completion
                if batch_idx > 0:
                    self.h2d_stream.wait_event(d2h_event)

                # Direct copy - no preprocessing (user responsible for correct input shape)
                for i in range(current_batch_size):
                    gpu_buffer[i].copy_(cpu_batch[i], non_blocking=True)

                # Record H2D completion event for this specific buffer
                self.h2d_stream.record_event(h2d_event)

    def _compute_workload(self, batch_idx, current_batch_size, nvtx_prefix):
        """Perform compute workload: generic model inference or no-op"""
        gpu_input_buffer = self.gpu_input_buffers[self.current]
        gpu_output_buffer = self.gpu_output_buffers[self.current]
        h2d_event = self.h2d_done_event[self.current]
        compute_event = self.compute_done_event[self.current]

        with torch.cuda.device(self.gpu_id):  # Explicit device context
            with torch.cuda.stream(self.compute_stream):
                with nvtx.range(f"{nvtx_prefix}_compute_batch_{batch_idx}"):
                    # EVENT-BASED: Wait only for THIS buffer's H2D completion
                    self.compute_stream.wait_event(h2d_event)

                    if self.is_noop:
                        # No-op compute: minimal operation for stream ordering
                        with nvtx.range(f"noop_compute_{batch_idx}"):
                            # Touch the data to ensure H2D completed and maintain stream dependencies
                            valid_input_slice = gpu_input_buffer[:current_batch_size]
                            _ = valid_input_slice.sum()  # Minimal compute operation
                            # For no-op, copy input to output (identity operation)
                            gpu_output_buffer[:current_batch_size].copy_(valid_input_slice)
                    else:
                        # Generic model inference
                        valid_input_slice = gpu_input_buffer[:current_batch_size]
                        with torch.no_grad():  # Restored for efficiency
                            with nvtx.range(f"{nvtx_prefix}_model_forward_{batch_idx}"):
                                predictions = self.model(valid_input_slice)
                                # Store model output in output buffer
                                gpu_output_buffer[:current_batch_size].copy_(predictions)
                                # CRITICAL: Force compute completion for CUDA synchronization
                                _ = predictions.sum()

                    # Record compute completion event for this specific buffer
                    self.compute_stream.record_event(compute_event)

    def _d2h_transfer(self, batch_idx, current_batch_size, nvtx_prefix):
        """Perform D2H transfer from current buffer (only valid slice)"""
        gpu_output_buffer = self.gpu_output_buffers[self.current]
        cpu_buffer = self.cpu_output_buffers[self.current]
        compute_event = self.compute_done_event[self.current]
        d2h_event = self.d2h_done_event[self.current]

        with torch.cuda.stream(self.d2h_stream):
            with nvtx.range(f"{nvtx_prefix}_d2h_batch_{batch_idx}"):
                # EVENT-BASED: Wait only for THIS buffer's compute completion
                self.d2h_stream.wait_event(compute_event)

                # Direct copy - no postprocessing (model output already in correct shape)
                for i in range(current_batch_size):
                    cpu_buffer[i].copy_(gpu_output_buffer[i], non_blocking=True)

                # Record D2H completion event for this specific buffer
                self.d2h_stream.record_event(d2h_event)

    def process_batch(self, cpu_batch, batch_idx, current_batch_size, nvtx_prefix):
        """Process a batch through the full H2D -> compute -> D2H pipeline"""
        self._h2d_transfer(cpu_batch, batch_idx, current_batch_size, nvtx_prefix)
        self._compute_workload(batch_idx, current_batch_size, nvtx_prefix)
        self._d2h_transfer(batch_idx, current_batch_size, nvtx_prefix)

    def wait_for_completion(self):
        """Wait for all pipeline stages to complete"""
        self.h2d_stream.synchronize()
        self.compute_stream.synchronize()
        self.d2h_stream.synchronize()

def run_pipeline_test(
    gpu_id=0,
    tensor_shape=(3, 224, 224),
    num_samples=1000,
    batch_size=10,
    warmup_samples=100,
    patch_size=32,
    depth=6,
    heads=8,
    dim=512,
    mlp_dim=2048,
    skip_warmup=False,
    deterministic=False,
    pin_memory=True,
    sync_frequency=10,
    compile_model=False,
    compile_mode='default'
):
    """
    Run comprehensive pipeline performance test with double buffering

    Simple double buffered pipeline test with synthetic random data.
    When depth=0, runs in no-op mode testing only H2D/D2H performance.
    When depth>0, runs full ViT inference pipeline.
    """

    # Set deterministic behavior if requested
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(42)
        np.random.seed(42)

    numa_info = get_numa_info()
    gpu_info = get_gpu_info(gpu_id)

    print(f"=== GPU NUMA Pipeline Performance Test ===")
    print(f"Process PID: {numa_info['pid']}")
    print(f"CPU Affinity: {numa_info['cpu_ranges']}")
    print(f"GPU ID: {gpu_id}")
    if 'error' in gpu_info:
        print(f"GPU Error: {gpu_info['error']}")
        sys.exit(1)
    print(f"GPU: {gpu_info['name']} ({gpu_info['memory_mb']:.0f} MB)")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print(f"Tensor Shape: {tensor_shape}")
    print(f"Batch Size: {batch_size}")
    print(f"Total Samples: {num_samples}")
    print(f"Warmup Samples: {warmup_samples if not skip_warmup else 0}")
    print(f"ViT Config: patch_size={patch_size}, depth={depth}, heads={heads}, dim={dim}, mlp_dim={mlp_dim}")
    print(f"Pin Memory: {pin_memory}")
    print(f"Sync Frequency: {sync_frequency}")
    print(f"Deterministic: {deterministic}")
    print(f"Compile Model: {compile_model} (mode: {compile_mode})")
    print("=" * 60)

    # Check vit-pytorch availability for non-no-op mode
    if depth > 0 and not VIT_AVAILABLE:
        print("ERROR: vit-pytorch not found and depth > 0. Install with: pip install vit-pytorch")
        print("Or use --vit-depth 0 for no-op compute mode.")
        sys.exit(1)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    torch.cuda.set_device(gpu_id)

    # Increase warmup for aggressive compilation modes
    if compile_model and compile_mode in ['reduce-overhead', 'max-autotune']:
        original_warmup = warmup_samples
        warmup_samples = max(warmup_samples, 1000)
        if warmup_samples > original_warmup:
            print(f"Increased warmup samples to {warmup_samples} for {compile_mode} compilation mode")

    # Pre-generate test data
    print("Pre-generating test data...")
    total_samples = (0 if skip_warmup else warmup_samples) + num_samples

    cpu_tensors = []
    for i in range(total_samples):
        tensor = torch.randn(*tensor_shape)

        if pin_memory:
            tensor = tensor.pin_memory()

        cpu_tensors.append(tensor)

    print(f"Generated {len(cpu_tensors)} CPU tensors")

    # Create ViT model separately
    vit_model, image_size = create_vit_model(
        tensor_shape, patch_size, depth, heads, dim, mlp_dim, gpu_id, compile_model, compile_mode
    )

    # Calculate input and output shapes
    input_shape = tensor_shape  # Original input shape
    if vit_model is None:
        # No-op mode: output shape same as input shape
        output_shape = tensor_shape
    else:
        # ViT mode: output shape is transformer output (num_patches + 1, dim)
        num_patches = (image_size // patch_size) ** 2
        output_shape = (num_patches + 1, dim)

    # Create generic pipeline
    pipeline = DoubleBufferedPipeline(
        model=vit_model,
        batch_size=batch_size,
        input_shape=input_shape,
        output_shape=output_shape,
        gpu_id=gpu_id,
        pin_memory=pin_memory
    )

    # Warmup phase
    if not skip_warmup and warmup_samples > 0:
        print(f"Warmup phase: {warmup_samples} samples...")
        _run_double_buffer_pipeline(
            pipeline, cpu_tensors[:warmup_samples], batch_size, "warmup", sync_frequency, is_warmup=True
        )
        # CRITICAL: Ensure all warmup GPU work completes before test timing
        pipeline.wait_for_completion()
        torch.cuda.synchronize()
        print("Warmup completed, GPU synchronized")

    # Main test phase with accurate total timing
    print(f"Test phase: {num_samples} samples...")
    start_idx = 0 if skip_warmup else warmup_samples
    test_tensors = cpu_tensors[start_idx:start_idx + num_samples]

    # Start timing AFTER warmup synchronization
    start_time = time.time()

    # Process all test batches (without individual timing)
    _run_double_buffer_pipeline(pipeline, test_tensors, batch_size, "test", sync_frequency, is_warmup=False)

    # End timing AFTER all GPU work completes
    pipeline.wait_for_completion()
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate accurate throughput
    total_time = end_time - start_time
    throughput = num_samples / total_time

    # Print results summary
    print(f"\n=== Pipeline Results Summary ===")
    print(f"Test Samples: {num_samples}")
    print(f"Total Time: {total_time:.6f}s")
    print(f"Average Throughput: {throughput:.2f} samples/s")

    print("\n=== Pipeline Test Completed ===")
    print("Use nsys GUI or stats to analyze the detailed profiling data.")


def _run_double_buffer_pipeline(pipeline, tensors, batch_size, nvtx_prefix, sync_frequency, is_warmup):
    """Run fully overlapped double buffered pipeline without individual timing"""

    with nvtx.range(f"{nvtx_prefix}_double_buffer"):
        num_batches = (len(tensors) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(tensors))
            current_batch_size = batch_end - batch_start
            batch_tensors = tensors[batch_start:batch_end]

            with nvtx.range(f"{nvtx_prefix}_batch_{batch_idx}"):
                # Swap to next buffer for all batches except the first
                if batch_idx > 0:
                    pipeline.swap()

                # Process the batch through the full pipeline
                pipeline.process_batch(batch_tensors, batch_idx, current_batch_size, nvtx_prefix)

                # Progress reporting
                if (batch_idx + 1) % sync_frequency == 0:
                    progress = batch_end / len(tensors) * 100
                    print(f"  Progress: {progress:.1f}% ({batch_end}/{len(tensors)})")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_pipeline_test(
        gpu_id=cfg.gpu_id,
        tensor_shape=tuple(cfg.shape),
        num_samples=cfg.num_samples,
        batch_size=cfg.batch_size,
        warmup_samples=cfg.warmup_samples,
        patch_size=cfg.vit.patch_size,
        depth=cfg.vit.depth,
        heads=cfg.vit.heads,
        dim=cfg.vit.dim,
        mlp_dim=cfg.vit.mlp_dim,
        skip_warmup=cfg.test.skip_warmup,
        deterministic=cfg.test.deterministic,
        pin_memory=cfg.performance.pin_memory,
        sync_frequency=cfg.test.sync_frequency,
        compile_model=cfg.performance.compile_model,
        compile_mode=cfg.performance.compile_mode
    )

if __name__ == '__main__':
    main()

####
# vit_pipeline_ray.py
####

#!/usr/bin/env python3
"""
VIT Ray Pipeline - Multi-GPU scaling with Ray

Orchestrates distributed VIT inference using Ray actors and tasks.
Supports multi-GPU scaling with optional nsys profiling.

Usage:
    python vit_pipeline_ray.py ray=multi_gpu
    python vit_pipeline_ray.py ray=profiling
    python vit_pipeline_ray.py ray.actors.num_gpus=4 ray.producers.num_tasks=8
"""

import ray
import torch
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from typing import List, Dict, Any
import sys
import os

# Import Ray components
from ray_data_producer import RayDataProducerManager
from ray_pipeline_actor import create_pipeline_actors
from gpu_health_validator import get_healthy_gpus_for_ray

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_ray(ray_config: DictConfig, min_gpus: int = 1) -> List[int]:
    """
    Initialize Ray cluster with pre-validated healthy GPUs only.
    
    Args:
        ray_config: Ray configuration
        min_gpus: Minimum number of healthy GPUs required
        
    Returns:
        List of healthy GPU IDs that Ray can use
        
    Raises:
        RuntimeError: If insufficient healthy GPUs or Ray initialization fails
    """
    # Step 1: Validate GPUs BEFORE Ray initialization
    logging.info("=== Pre-Ray GPU Health Validation ===")
    try:
        healthy_gpus = get_healthy_gpus_for_ray(min_gpus=min_gpus)
        logging.info(f"✅ Pre-validation complete: {len(healthy_gpus)} healthy GPUs configured")
    except RuntimeError as e:
        logging.error(f"❌ GPU validation failed: {e}")
        raise RuntimeError(f"Cannot start Ray: {e}")
    
    # Step 2: Initialize Ray (will now only see healthy GPUs)
    if ray.is_initialized():
        logging.info("Ray already initialized")
        # Verify Ray sees the right number of GPUs
        cluster_resources = ray.cluster_resources()
        ray_gpu_count = int(cluster_resources.get('GPU', 0))
        if ray_gpu_count != len(healthy_gpus):
            logging.warning(f"Ray sees {ray_gpu_count} GPUs but expected {len(healthy_gpus)}")
        return healthy_gpus
    
    init_config = ray_config.get('init', {})
    ray_init_kwargs = OmegaConf.to_container(init_config, resolve=True) if init_config else {}
    
    logging.info(f"Initializing Ray with config: {ray_init_kwargs}")
    
    try:
        ray.init(**ray_init_kwargs)
        logging.info("✅ Ray initialized successfully")
        
        # Verify Ray cluster sees correct GPU configuration
        cluster_resources = ray.cluster_resources()
        ray_gpu_count = int(cluster_resources.get('GPU', 0))
        
        logging.info(f"Ray cluster resources: {cluster_resources}")
        logging.info(f"Ray GPU count: {ray_gpu_count}, Expected: {len(healthy_gpus)}")
        
        if ray_gpu_count != len(healthy_gpus):
            raise RuntimeError(f"Ray sees {ray_gpu_count} GPUs but {len(healthy_gpus)} were validated")
        
        logging.info(f"✅ Ray GPU configuration verified: {len(healthy_gpus)} healthy GPUs available")
        return healthy_gpus
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize Ray: {e}")
        raise RuntimeError(f"Ray initialization failed: {e}")


def run_ray_pipeline_test(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run Ray-based VIT pipeline test with multiple GPUs.
    
    Args:
        cfg: Hydra configuration containing ray, pipeline, and model settings
        
    Returns:
        Dictionary with test results and performance metrics
    """
    logging.info("=== Starting Ray VIT Pipeline Test ===")
    
    # Extract configurations
    ray_config = cfg.get('ray', {})
    actor_config = ray_config.get('actors', {})
    producer_config = ray_config.get('producers', {})
    
    num_gpus = actor_config.get('num_gpus', 1)
    profiling_enabled = actor_config.get('profiling_enabled', False)
    
    num_producers = producer_config.get('num_tasks', 2)
    batches_per_producer = producer_config.get('batches_per_task', 10)
    inter_batch_delay = producer_config.get('inter_batch_delay', 0.0)
    
    logging.info(f"Configuration:")
    logging.info(f"  GPUs: {num_gpus}")
    logging.info(f"  Profiling: {'enabled' if profiling_enabled else 'disabled'}")
    logging.info(f"  Data producers: {num_producers}")
    logging.info(f"  Batches per producer: {batches_per_producer}")
    logging.info(f"  Total batches: {num_producers * batches_per_producer}")
    
    start_time = time.time()
    
    try:
        # Setup Ray with pre-validated healthy GPUs
        healthy_gpus = setup_ray(ray_config, min_gpus=num_gpus)
        
        # All GPUs Ray sees are now guaranteed healthy
        actual_num_gpus = len(healthy_gpus)
        
        if actual_num_gpus < num_gpus:
            logging.warning(f"Requested {num_gpus} GPUs but only {actual_num_gpus} healthy GPUs available")
        
        logging.info(f"Using {actual_num_gpus} pre-validated healthy GPUs")
        num_gpus = actual_num_gpus
        
        # Create pipeline actors (Ray will assign healthy GPUs automatically)
        logging.info(f"Creating {num_gpus} pipeline actors...")
        
        pipeline_kwargs = {
            'tensor_shape': tuple(cfg.shape),
            'batch_size': cfg.get('batch_size', cfg.pipeline.batch_size),
            'patch_size': cfg.vit.patch_size,
            'depth': cfg.vit.depth,
            'heads': cfg.vit.heads,
            'dim': cfg.vit.dim,
            'mlp_dim': cfg.vit.mlp_dim,
            'pin_memory': cfg.performance.pin_memory,
            'compile_model': cfg.performance.compile_model,
            'compile_mode': cfg.performance.compile_mode,
            'deterministic': cfg.test.deterministic
        }
        
        actors = create_pipeline_actors(
            num_actors=num_gpus,
            enable_profiling=profiling_enabled,
            validate_gpus=True,  # Enable GPU validation during actor creation
            **pipeline_kwargs
        )
        
        # Step 3: Verify actor health
        logging.info("Checking actor health...")
        health_futures = [actor.health_check.remote() for actor in actors]
        health_results = ray.get(health_futures)
        
        for i, health in enumerate(health_results):
            if health['status'] != 'healthy':
                logging.error(f"Actor {i} is unhealthy: {health}")
                return {'error': f'Actor {i} failed health check'}
            else:
                logging.info(f"✅ Actor {i} healthy (GPU {health['gpu_id']})")
        
        # Step 4: Launch data producers
        logging.info("Launching data producers...")
        
        producer_manager = RayDataProducerManager()
        producer_futures = producer_manager.launch_producers(
            num_producers=num_producers,
            batches_per_producer=batches_per_producer,
            batch_size=cfg.get('batch_size', cfg.pipeline.batch_size),
            tensor_shape=tuple(cfg.shape),
            inter_batch_delay=inter_batch_delay,
            pin_memory=cfg.performance.pin_memory,
            deterministic=cfg.test.deterministic
        )
        
        # Step 5: Wait for data generation to complete
        logging.info("Waiting for data production to complete...")
        all_batches = producer_manager.get_all_batches()
        
        total_batches = len(all_batches)
        logging.info(f"✅ Data production complete: {total_batches} batches available")
        
        # Step 6: Distribute batches across actors
        logging.info("Distributing batches across actors...")
        
        # Simple round-robin distribution (handle edge case of 0 actors)
        if num_gpus == 0:
            return {
                'success': False,
                'error': 'No GPU actors available - pipeline requires at least 1 GPU actor',
                'total_time': time.time() - start_time
            }
        
        actor_batch_assignments = [[] for _ in range(num_gpus)]
        for batch_idx, batch in enumerate(all_batches):
            actor_id = batch_idx % num_gpus
            actor_batch_assignments[actor_id].append(batch)
        
        for i, batches in enumerate(actor_batch_assignments):
            logging.info(f"Actor {i}: assigned {len(batches)} batches")
        
        # Step 7: Process batches in parallel across all actors
        logging.info("Starting parallel processing across all actors...")
        processing_start = time.time()
        
        processing_futures = []
        for actor_id, (actor, assigned_batches) in enumerate(zip(actors, actor_batch_assignments)):
            if assigned_batches:  # Only process if there are batches assigned
                future = actor.process_batch_list.remote(assigned_batches)
                processing_futures.append((actor_id, future))
        
        # Wait for all processing to complete
        processing_results = []
        for actor_id, future in processing_futures:
            try:
                result = ray.get(future)
                processing_results.append((actor_id, result))
                logging.info(f"✅ Actor {actor_id} completed {len(result)} batches")
            except Exception as e:
                logging.error(f"❌ Actor {actor_id} failed: {e}")
                processing_results.append((actor_id, {'error': str(e)}))
        
        processing_end = time.time()
        processing_time = processing_end - processing_start
        
        # Step 8: Collect statistics from all actors
        logging.info("Collecting statistics from all actors...")
        
        stats_futures = [actor.get_statistics.remote() for actor in actors]
        actor_stats = ray.get(stats_futures)
        
        # Aggregate results
        total_samples = sum(stats.get('samples_processed', 0) for stats in actor_stats)
        total_batches_processed = sum(stats.get('batches_processed', 0) for stats in actor_stats)
        
        overall_throughput = total_samples / processing_time if processing_time > 0 else 0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Compile final results
        results = {
            'success': True,
            'config': {
                'num_gpus': num_gpus,
                'num_gpus_requested': original_num_gpus,
                'healthy_gpus': healthy_gpus,
                'num_producers': num_producers,
                'total_batches_generated': total_batches,
                'profiling_enabled': profiling_enabled,
                'gpu_health_validated': True
            },
            'performance': {
                'total_samples': total_samples,
                'total_batches_processed': total_batches_processed,
                'processing_time': processing_time,
                'total_time': total_time,
                'overall_throughput': overall_throughput,
                'samples_per_gpu_per_sec': overall_throughput / num_gpus if num_gpus > 0 else 0
            },
            'actor_stats': actor_stats,
            'processing_results': dict(processing_results)
        }
        
        # Print summary
        logging.info("=== Ray Pipeline Results Summary ===")
        logging.info(f"GPU Health: {len(healthy_gpus)} healthy GPUs found")
        if original_num_gpus != num_gpus:
            logging.info(f"GPU Adjustment: Reduced from {original_num_gpus} to {num_gpus} actors due to unhealthy GPUs")
        logging.info(f"Total samples processed: {total_samples}")
        logging.info(f"Processing time: {processing_time:.3f}s")
        logging.info(f"Overall throughput: {overall_throughput:.1f} samples/s")
        logging.info(f"Per-GPU throughput: {results['performance']['samples_per_gpu_per_sec']:.1f} samples/s")
        logging.info(f"Profiling: {'enabled' if profiling_enabled else 'disabled'}")
        
        return results
        
    except Exception as e:
        logging.error(f"Ray pipeline test failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'total_time': time.time() - start_time
        }


def cleanup_ray():
    """Cleanup Ray resources."""
    try:
        if ray.is_initialized():
            ray.shutdown()
            logging.info("Ray shutdown complete")
    except Exception as e:
        logging.warning(f"Ray shutdown warning: {e}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for Ray VIT pipeline."""
    
    # Check if Ray configuration is provided
    if not cfg.get('ray'):
        logging.error("Ray configuration not found. Please specify ray config:")
        logging.error("  python vit_pipeline_ray.py ray=multi_gpu")
        logging.error("  python vit_pipeline_ray.py ray=profiling")
        sys.exit(1)
    
    # Print configuration
    logging.info("Starting Ray VIT Pipeline with configuration:")
    logging.info(OmegaConf.to_yaml(cfg))
    
    try:
        # Run the Ray pipeline test
        results = run_ray_pipeline_test(cfg)
        
        if results.get('success'):
            logging.info("🎉 Ray pipeline test completed successfully!")
            
            if results['config']['profiling_enabled']:
                logging.info("💡 Profiling was enabled - check for nsys-rep files in Ray working directory")
                logging.info("    Use 'nsys-ui' or 'nsys stats' to analyze the profiles")
        else:
            logging.error(f"❌ Ray pipeline test failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        cleanup_ray()


if __name__ == '__main__':
    main()

####
# vit_utils.py
####

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


####
# Summary
# Processed files: 4
# Skipped files: 1
####
