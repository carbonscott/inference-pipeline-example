#!/usr/bin/env python3
"""
GPU NUMA Pipeline Test with PeakNet and Double Buffering - EVENT-BASED SYNC VERSION

Test script to evaluate end-to-end pipeline performance with overlapping
H2D, Compute, and D2H stages using double buffering across NUMA nodes.

This version uses fine-grained CUDA events for synchronization instead of
stream dependencies for better parallelism and performance.

Usage with numactl:
  numactl --cpunodebind=0 --membind=0 python peaknet_pipeline.py --gpu-id=5
  numactl --cpunodebind=2 --membind=2 python peaknet_pipeline.py --gpu-id=3
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

# Check for PeakNet availability
try:
    from peaknet_utils import (
        PeakNetForProfiling, create_peaknet_model as create_peaknet_from_utils,
        get_peaknet_shapes
    )
    PEAKNET_AVAILABLE = True
except ImportError:
    print("ERROR: PeakNet utilities not found. Make sure peaknet is installed and peaknet_utils.py is available")
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


def create_peaknet_model(peaknet_config, weights_path, gpu_id, compile_model=False, compile_mode='default'):
    """
    Create PeakNet model for compute simulation, or None for no-op.
    
    Args:
        peaknet_config: PeakNet configuration dict with model parameters (None for no-op)
        weights_path: Path to model weights
        gpu_id: GPU device ID
        compile_model: Whether to compile the model
        compile_mode: Compilation mode
    
    Returns:
        tuple: (model, input_shape, output_shape) or (None, None, None) for no-op
    """

    # Handle no-op case (when no config is provided)
    if not peaknet_config or not peaknet_config.get('model'):
        print("No-op compute mode: no PeakNet config specified, skipping model creation")
        return None, None, None

    # Create PeakNet model from Hydra configuration
    try:
        peaknet_model = create_peaknet_from_utils(
            peaknet_config=peaknet_config,
            weights_path=weights_path,
            device=f'cuda:{gpu_id}'
        )

        # Get input and output shapes from config
        input_shape, output_shape = get_peaknet_shapes(peaknet_config, batch_size=1)

        # Remove batch dimension for pipeline usage
        input_shape = input_shape[1:]  # (C, H, W)
        output_shape = output_shape[1:]  # (num_classes, H, W)
        print(f"[DEBUG] Input shape: {input_shape}, Output shape: {output_shape}")
        
        # COMPREHENSIVE MODEL DEVICE VALIDATION
        print(f"[DEBUG] Verifying PeakNet model components on GPU...")
        model_device = next(peaknet_model.parameters()).device
        print(f"[DEBUG] Overall model device: {model_device}")
        
        # Verify model is on correct GPU
        assert model_device.type == 'cuda', f"Model not on GPU: {model_device}"
        assert model_device.index == gpu_id, f"Model on wrong GPU: {model_device}, expected cuda:{gpu_id}"
        print(f"✓ PeakNet model verified on GPU {gpu_id}")

        # Add torch.compile if requested and available
        if compile_model and check_torch_compile_available():
            print(f"Compiling PeakNet model with mode={compile_mode}...")
            try:
                # Use specified compilation mode
                peaknet_model = torch.compile(peaknet_model, mode=compile_mode)
                print(f"Model compilation successful (mode={compile_mode})")
            except Exception as e:
                print(f"Warning: Model compilation failed with mode={compile_mode} ({e}), using non-compiled model")
        elif compile_model and not check_torch_compile_available():
            print("Warning: torch.compile not available (requires PyTorch 2.0+), using non-compiled model")

        return peaknet_model, input_shape, output_shape
        
    except Exception as e:
        print(f"Error creating PeakNet model: {e}")
        return None, None, None


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
        
        # DEVICE VALIDATION: Verify all GPU buffers are on correct device
        print(f"[DEBUG] Pipeline buffer validation:")
        for key in ['A', 'B']:
            input_device = self.gpu_input_buffers[key].device
            output_device = self.gpu_output_buffers[key].device
            print(f"[DEBUG]   Buffer {key}: input {input_device}, output {output_device}")
            assert input_device == torch.device(f'cuda:{gpu_id}'), f"Input buffer {key} on wrong device: {input_device}"
            assert output_device == torch.device(f'cuda:{gpu_id}'), f"Output buffer {key} on wrong device: {output_device}"
        print(f"✓ All pipeline buffers verified on GPU {gpu_id}")
        
        # Log buffer memory usage
        if torch.cuda.is_available():
            input_mem = self.gpu_input_buffers['A'].numel() * self.gpu_input_buffers['A'].element_size() * 2 / (1024**3)
            output_mem = self.gpu_output_buffers['A'].numel() * self.gpu_output_buffers['A'].element_size() * 2 / (1024**3)
            total_buffer_mem = input_mem + output_mem
            print(f"[DEBUG] Pipeline buffer memory: {total_buffer_mem:.3f} GB ({input_mem:.3f} GB input + {output_mem:.3f} GB output)")

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
                        # Handle shape mismatch between input and output for segmentation
                        if valid_input_slice.shape == gpu_output_buffer[:current_batch_size].shape:
                            gpu_output_buffer[:current_batch_size].copy_(valid_input_slice)
                        else:
                            # Create dummy segmentation output (all zeros)
                            gpu_output_buffer[:current_batch_size].fill_(0.0)
                else:
                    # PeakNet model inference
                    valid_input_slice = gpu_input_buffer[:current_batch_size]
                    
                    # GPU DEBUGGING: Add comprehensive device validation
                    input_device = valid_input_slice.device
                    model_device = next(self.model.parameters()).device
                    print(f"[DEBUG] Batch {batch_idx}: input tensor device: {input_device}")
                    print(f"[DEBUG] Batch {batch_idx}: model device: {model_device}")
                    
                    # CRITICAL: Ensure both model and input are on GPU
                    assert input_device.type == 'cuda', f"Input tensor on {input_device}, expected CUDA"
                    assert model_device.type == 'cuda', f"Model on {model_device}, expected CUDA"
                    assert input_device == model_device, f"Device mismatch: input {input_device}, model {model_device}"
                    
                    with torch.no_grad():
                        with nvtx.range(f"{nvtx_prefix}_model_forward_{batch_idx}"):
                            # Monitor GPU memory before inference
                            if torch.cuda.is_available():
                                gpu_mem_before = torch.cuda.memory_allocated(input_device) / (1024**3)
                                print(f"[DEBUG] GPU memory before inference: {gpu_mem_before:.2f} GB")
                            
                            predictions = self.model(valid_input_slice)
                            
                            # Verify output is on GPU
                            output_device = predictions.device
                            print(f"[DEBUG] Batch {batch_idx}: output tensor device: {output_device}")
                            assert output_device.type == 'cuda', f"Output tensor on {output_device}, expected CUDA"
                            
                            # Monitor GPU memory after inference
                            if torch.cuda.is_available():
                                gpu_mem_after = torch.cuda.memory_allocated(input_device) / (1024**3)
                                print(f"[DEBUG] GPU memory after inference: {gpu_mem_after:.2f} GB")
                            
                            # Store model output in output buffer
                            gpu_output_buffer[:current_batch_size].copy_(predictions)
                            # CRITICAL: Force compute completion for CUDA synchronization
                            _ = predictions.sum()
                            
                            print(f"[DEBUG] Batch {batch_idx}: PeakNet forward pass completed on GPU")

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
    tensor_shape=(1, 512, 512),
    num_samples=1000,
    batch_size=10,
    warmup_samples=100,
    peaknet_config=None,
    weights_path=None,
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
    When peaknet_config is None, runs in no-op mode testing only H2D/D2H performance.
    When peaknet_config is provided, runs full PeakNet inference pipeline.
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
    print(f"PeakNet Config: peaknet_config={peaknet_config is not None}, weights_path={weights_path}")
    print(f"Pin Memory: {pin_memory}")
    print(f"Sync Frequency: {sync_frequency}")
    print(f"Deterministic: {deterministic}")
    print(f"Compile Model: {compile_model} (mode: {compile_mode})")
    print("=" * 60)

    # Check PeakNet availability for non-no-op mode
    if peaknet_config and not PEAKNET_AVAILABLE:
        print("ERROR: PeakNet not found and PeakNet config provided. Make sure peaknet is installed.")
        print("Or use peaknet_config=None for no-op compute mode.")
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

    # Create PeakNet model separately
    peaknet_model, input_shape, output_shape = create_peaknet_model(
        peaknet_config, weights_path, gpu_id, compile_model, compile_mode
    )

    # Calculate input and output shapes
    if peaknet_model is None:
        # No-op mode: use provided tensor_shape for both input and output
        input_shape = tensor_shape
        output_shape = tensor_shape
        print(f"No-op mode: input_shape={input_shape}, output_shape={output_shape}")
    else:
        print(f"PeakNet mode: input_shape={input_shape}, output_shape={output_shape}")

    # Create generic pipeline
    pipeline = DoubleBufferedPipeline(
        model=peaknet_model,
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
        yaml_path=cfg.peaknet.yaml_path,
        weights_path=cfg.peaknet.weights_path,
        skip_warmup=cfg.test.skip_warmup,
        deterministic=cfg.test.deterministic,
        pin_memory=cfg.performance.pin_memory,
        sync_frequency=cfg.test.sync_frequency,
        compile_model=cfg.performance.compile_model,
        compile_mode=cfg.performance.compile_mode
    )


if __name__ == '__main__':
    main()
