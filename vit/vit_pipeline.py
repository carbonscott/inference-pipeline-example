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
import argparse
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
    """Double buffered pipeline for H2D -> ViT Compute -> D2H"""

    def __init__(self, batch_size, tensor_shape, gpu_id, patch_size, depth, heads, dim, mlp_dim, pin_memory=True, compile_model=False, compile_mode='default'):
        self.batch_size = batch_size
        self.tensor_shape = tensor_shape
        self.gpu_id = gpu_id
        self.pin_memory = pin_memory

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

        # Create ViT model (or None for no-op) - NOW WITH COMPILATION SUPPORT
        self.vit_model, self.image_size = create_vit_model(
            tensor_shape, patch_size, depth, heads, dim, mlp_dim, gpu_id, compile_model, compile_mode
        )
        self.is_noop = (self.vit_model is None)

        # Double buffers on GPU (with ViT-compatible size)
        C = tensor_shape[0]
        self.gpu_buffers = {
            'A': torch.zeros(batch_size, C, self.image_size, self.image_size, device=f'cuda:{gpu_id}'),
            'B': torch.zeros(batch_size, C, self.image_size, self.image_size, device=f'cuda:{gpu_id}')
        }

        # Single contiguous pinned host buffers (better D2H bandwidth)
        shape = (batch_size, *tensor_shape)
        self.cpu_buffers = {
            'A': torch.empty(shape, pin_memory=pin_memory),
            'B': torch.empty(shape, pin_memory=pin_memory)
        }

        # Pipeline state
        self.current = 'A'

    def swap(self):
        """Swap current buffer"""
        self.current = 'B' if self.current == 'A' else 'A'

    def h2d_transfer(self, cpu_batch, batch_idx, current_batch_size, nvtx_prefix):
        """Perform H2D transfer with fine-grained event-based synchronization"""
        gpu_buffer = self.gpu_buffers[self.current]
        d2h_event = self.d2h_done_event[self.current]
        h2d_event = self.h2d_done_event[self.current]

        with torch.cuda.stream(self.h2d_stream):
            with nvtx.range(f"{nvtx_prefix}_h2d_batch_{batch_idx}"):
                # Fine-grained synchronization: wait only for THIS buffer's D2H completion
                if batch_idx > 0:
                    self.h2d_stream.wait_event(d2h_event)

                # Copy only the valid batch size
                for i in range(current_batch_size):
                    tensor = cpu_batch[i]
                    # Resize tensor to ViT-compatible size if needed
                    if tensor.shape[-2:] != (self.image_size, self.image_size):
                        resized_tensor = torch.nn.functional.interpolate(
                            tensor.unsqueeze(0),
                            size=(self.image_size, self.image_size),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        gpu_buffer[i].copy_(resized_tensor, non_blocking=True)
                    else:
                        gpu_buffer[i].copy_(tensor, non_blocking=True)

                # Record H2D completion event for this specific buffer
                self.h2d_stream.record_event(h2d_event)

    def compute_workload(self, batch_idx, current_batch_size, nvtx_prefix):
        """Perform compute workload: ViT inference or no-op"""
        gpu_buffer = self.gpu_buffers[self.current]
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
                        valid_gpu_slice = gpu_buffer[:current_batch_size]
                        _ = valid_gpu_slice.sum()  # Minimal compute operation
                else:
                    # Normal ViT inference
                    valid_gpu_slice = gpu_buffer[:current_batch_size]
                    # Run ViT inference
                    with torch.no_grad():
                        with nvtx.range(f"{nvtx_prefix}_vit_forward_{batch_idx}"):
                            predictions = self.vit_model(valid_gpu_slice)
                            # Force compute completion with a small operation
                            _ = predictions.sum()

                # Record compute completion event for this specific buffer
                self.compute_stream.record_event(compute_event)

    def d2h_transfer(self, batch_idx, current_batch_size, nvtx_prefix):
        """Perform D2H transfer from current buffer (only valid slice)"""
        gpu_buffer = self.gpu_buffers[self.current]
        cpu_buffer = self.cpu_buffers[self.current]
        compute_event = self.compute_done_event[self.current]
        d2h_event = self.d2h_done_event[self.current]

        with torch.cuda.stream(self.d2h_stream):
            with nvtx.range(f"{nvtx_prefix}_d2h_batch_{batch_idx}"):
                # EVENT-BASED: Wait only for THIS buffer's compute completion
                self.d2h_stream.wait_event(compute_event)

                # Copy back only the valid slice
                for i in range(current_batch_size):
                    gpu_tensor = gpu_buffer[i]
                    # Resize back to original shape if needed
                    if gpu_tensor.shape[-2:] != self.tensor_shape[-2:]:
                        resized_tensor = torch.nn.functional.interpolate(
                            gpu_tensor.unsqueeze(0),
                            size=self.tensor_shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        cpu_buffer[i].copy_(resized_tensor, non_blocking=True)
                    else:
                        cpu_buffer[i].copy_(gpu_tensor, non_blocking=True)

                # Record D2H completion event for this specific buffer
                self.d2h_stream.record_event(d2h_event)

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
    memory_size_mb=512,
    patch_size=32,
    depth=6,
    heads=8,
    dim=512,
    mlp_dim=2048,
    skip_warmup=False,
    deterministic=False,
    pin_memory=True,
    fill_pattern='random',
    sync_frequency=10,
    compile_model=False,
    compile_mode='default'
):
    """
    Run comprehensive pipeline performance test with double buffering

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
    print(f"CPU Memory Pool: {memory_size_mb} MB ({fill_pattern})")
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

    # Allocate CPU memory pool
    print(f"Allocating {memory_size_mb} MB CPU memory pool...")
    if fill_pattern == 'random':
        memory_pool = torch.randn(memory_size_mb * 1024 * 1024 // 4)
    elif fill_pattern == 'sequential':
        memory_pool = torch.arange(memory_size_mb * 1024 * 1024 // 4, dtype=torch.float32)
    else:  # zeros
        memory_pool = torch.zeros(memory_size_mb * 1024 * 1024 // 4)

    print(f"CPU memory pool allocated: {memory_pool.element_size() * memory_pool.nelement() / 1024 / 1024:.2f} MB")

    # Pre-generate test data
    print("Pre-generating test data...")
    total_samples = (0 if skip_warmup else warmup_samples) + num_samples

    cpu_tensors = []
    for i in range(total_samples):
        if fill_pattern == 'random':
            tensor = torch.randn(*tensor_shape)
        elif fill_pattern == 'sequential':
            tensor = torch.arange(np.prod(tensor_shape), dtype=torch.float32).reshape(tensor_shape) + i
        else:  # zeros
            tensor = torch.zeros(*tensor_shape)

        if pin_memory:
            tensor = tensor.pin_memory()

        cpu_tensors.append(tensor)

        # Touch memory pool periodically
        if i % 50 == 0:
            _ = memory_pool[:(1024*1024)].sum()

    print(f"Generated {len(cpu_tensors)} CPU tensors")

    # Free memory pool after warmup to save RAM
    del memory_pool

    # Create pipeline - NOW WITH COMPILATION SUPPORT
    pipeline = DoubleBufferedPipeline(
        batch_size, tensor_shape, gpu_id, patch_size, depth, heads, dim, mlp_dim, pin_memory, compile_model, compile_mode
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
                if batch_idx == 0:
                    # First batch: just start the pipeline
                    pipeline.h2d_transfer(batch_tensors, batch_idx, current_batch_size, nvtx_prefix)
                    pipeline.compute_workload(batch_idx, current_batch_size, nvtx_prefix)
                    pipeline.d2h_transfer(batch_idx, current_batch_size, nvtx_prefix)
                else:
                    # Overlapped execution: start next batch while finishing previous
                    # Swap to next buffer
                    pipeline.swap()

                    # Start H2D for current batch (will wait for THIS buffer's D2H via event)
                    pipeline.h2d_transfer(batch_tensors, batch_idx, current_batch_size, nvtx_prefix)

                    # Start compute for current batch (will wait for H2D)
                    pipeline.compute_workload(batch_idx, current_batch_size, nvtx_prefix)

                    # Start D2H for current batch (will wait for compute and record event)
                    pipeline.d2h_transfer(batch_idx, current_batch_size, nvtx_prefix)

                # Progress reporting
                if (batch_idx + 1) % sync_frequency == 0:
                    progress = batch_end / len(tensors) * 100
                    print(f"  Progress: {progress:.1f}% ({batch_end}/{len(tensors)})")


def main():
    parser = argparse.ArgumentParser(description='GPU NUMA Pipeline Performance Test with ViT Double Buffering')

    # Core parameters
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--shape', nargs=3, type=int, default=[3, 224, 224],
                        help='Tensor shape as C H W (default: 3 224 224)')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to process (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size (default: 10)')
    parser.add_argument('--warmup-samples', type=int, default=100,
                        help='Number of warmup samples (default: 100)')
    parser.add_argument('--memory-size-mb', type=int, default=512,
                        help='CPU memory pool size in MB (default: 512)')

    # ViT configuration parameters
    parser.add_argument('--vit-patch-size', type=int, default=32,
                        help='ViT patch size (default: 32)')
    parser.add_argument('--vit-depth', type=int, default=6,
                        help='ViT number of transformer blocks (default: 6)')
    parser.add_argument('--vit-heads', type=int, default=8,
                        help='ViT number of attention heads (default: 8)')
    parser.add_argument('--vit-dim', type=int, default=512,
                        help='ViT embedding dimension (default: 512)')
    parser.add_argument('--vit-mlp-dim', type=int, default=2048,
                        help='ViT MLP dimension (default: 2048)')

    # Test control
    parser.add_argument('--sync-frequency', type=int, default=10,
                        help='Synchronization frequency for progress reporting (default: 10)')
    parser.add_argument('--skip-warmup', action='store_true',
                        help='Skip warmup phase for quick tests')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic algorithms in PyTorch')

    # Memory and performance options
    parser.add_argument('--no-pin-memory', action='store_true',
                        help='Disable pinned memory (default: enabled)')
    parser.add_argument('--fill-pattern', choices=['random', 'sequential', 'zeros'], default='random',
                        help='Memory fill pattern (default: random)')

    # Compilation options
    parser.add_argument('--compile-model', action='store_true',
                        help='Enable torch.compile for reduced kernel overhead (requires PyTorch 2.0+)')
    parser.add_argument('--compile-mode', choices=['default', 'reduce-overhead', 'max-autotune'], 
                        default='default', help='torch.compile optimization mode (default: default)')

    # Profiling and automation
    parser.add_argument('--nsys-report', type=str, default=None,
                        help='NSYS report name for automation (deprecated - handled by shell script)')

    args = parser.parse_args()

    run_pipeline_test(
        gpu_id=args.gpu_id,
        tensor_shape=tuple(args.shape),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        warmup_samples=args.warmup_samples,
        memory_size_mb=args.memory_size_mb,
        patch_size=args.vit_patch_size,
        depth=args.vit_depth,
        heads=args.vit_heads,
        dim=args.vit_dim,
        mlp_dim=args.vit_mlp_dim,
        skip_warmup=args.skip_warmup,
        deterministic=args.deterministic,
        pin_memory=not args.no_pin_memory,
        fill_pattern=args.fill_pattern,
        sync_frequency=args.sync_frequency,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode
    )

if __name__ == '__main__':
    main()
