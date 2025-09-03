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
                    with torch.no_grad():
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
