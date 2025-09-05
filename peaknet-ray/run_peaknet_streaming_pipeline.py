#!/usr/bin/env python3
"""
Ray Multi-GPU Streaming Pipeline for PeakNet

A production-ready interface for running PeakNet segmentation model inference at scale with 
streaming data sources across multiple GPUs using Ray.

Features:
- Pre-Ray GPU health validation (filters out faulty GPUs)
- Automatic round-robin GPU assignment across healthy GPUs
- Streaming data generation and multi-GPU processing
- NSys profiling support for performance analysis
- Comprehensive performance metrics and reporting
- Fail-fast error handling (no CPU fallback)

Example Usage:

    ## Hydra Configuration Style (Recommended):
    # Use predefined experiment configurations
    python run_peaknet_streaming_pipeline.py experiment=peaknet_fast_test --max-actors 2
    python run_peaknet_streaming_pipeline.py experiment=peaknet_single_channel_512px --total-samples 512000 --enable-profiling
    python run_peaknet_streaming_pipeline.py experiment=peaknet_production --max-actors 4
    
    # Mix model templates with runtime overrides
    python run_peaknet_streaming_pipeline.py shape=[1,512,512] --max-actors 4 --total-samples 512000
    python run_peaknet_streaming_pipeline.py streaming.runtime.batch_size=16 --enable-profiling
    
    ## Traditional CLI Style (Legacy):
    # Auto-discover and use all healthy GPUs with 4 data producers, small tensors
    python run_peaknet_streaming_pipeline.py --num-producers 4 --tensor-size 64
    
    # Limit to 2 actors max with custom configuration
    python run_peaknet_streaming_pipeline.py --max-actors 2 --batch-size 8 --batches-per-producer 10 --verbose
"""

import os
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

# Import our pipeline components
from gpu_health_validator import get_healthy_gpus_for_ray
from peaknet_ray_data_producer import RayDataProducerManager
from peaknet_ray_pipeline_actor import create_pipeline_actors
import ray


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(levelname)s - %(message)s' if verbose else '%(message)s'

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_banner():
    """Print a nice banner for the tool."""
    print("🚀 Ray Multi-GPU PeakNet Streaming Pipeline")
    print("=" * 55)


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration parameters."""
    runtime = cfg.streaming.runtime
    system = cfg.streaming.system
    data = cfg.streaming.data
    
    if runtime.max_actors is not None and runtime.max_actors <= 0:
        raise ValueError("max_actors must be positive")

    if system.min_gpus <= 0:
        raise ValueError("min_gpus must be positive")

    if runtime.num_producers <= 0:
        raise ValueError("num_producers must be positive")

    if runtime.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if data.input_size <= 0:
        raise ValueError("input_size must be positive")

    if data.input_channels <= 0:
        raise ValueError("input_channels must be positive")

    if data.input_height is not None and data.input_height <= 0:
        raise ValueError("input_height must be positive")

    if data.input_width is not None and data.input_width <= 0:
        raise ValueError("input_width must be positive")

    if runtime.inter_batch_delay < 0:
        raise ValueError("inter_batch_delay cannot be negative")

    # Note: NSys profiling files are automatically saved by Ray to $TMPDIR/ray/session_*/logs/nsight/


def setup_gpu_environment(min_gpus: int, skip_validation: bool = False, verbose: bool = False) -> List[int]:
    """
    Set up GPU environment with optional health validation.

    Args:
        min_gpus: Minimum number of healthy GPUs required
        skip_validation: Skip health validation for production clusters
        verbose: Enable detailed output

    Returns:
        List of healthy GPU IDs that Ray will use
    """
    if skip_validation:
        print("\n⚡ Step 1: GPU Environment Setup (Production Mode)")
        print(f"✅ Skipping validation - trusting Ray cluster provides {min_gpus}+ healthy GPUs")
        # Return dummy list since we're trusting the cluster
        return list(range(min_gpus))

    print("\n🔍 Step 1: GPU Health Validation")
    if verbose:
        print("   (Use --skip-gpu-validation for faster startup in production)")

    try:
        healthy_gpus = get_healthy_gpus_for_ray(min_gpus=min_gpus)

        print(f"✅ Found {len(healthy_gpus)} healthy GPUs")
        if verbose:
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
            print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")

        return healthy_gpus

    except RuntimeError as e:
        print(f"❌ GPU validation failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Ensure CUDA is available: nvidia-smi")
        print("   - Check for GPU hardware issues")
        print("   - Try reducing --min-gpus or --max-actors")
        print("   - Use --skip-gpu-validation for production clusters")
        sys.exit(1)


def setup_ray_cluster(cfg: DictConfig) -> None:
    """Initialize Ray cluster connection."""
    print("\n⚡ Step 2: Ray Cluster Setup")

    max_actors = cfg.streaming.runtime.max_actors

    if not ray.is_initialized():
        try:
            ray.init()
            cluster_resources = ray.cluster_resources()
            gpu_count = int(cluster_resources.get('GPU', 0))

            print(f"✅ Ray cluster initialized")
            print(f"   Available GPUs: {gpu_count}")
            print(f"   CPU cores: {int(cluster_resources.get('CPU', 0))}")

            print(f"   Ray cluster GPU resources: {gpu_count}")
            if max_actors:
                print(f"   Will create up to {max_actors} actors (user limit)")
            else:
                print(f"   Will auto-scale to use all healthy GPUs")

        except Exception as e:
            print(f"❌ Ray initialization failed: {e}")
            sys.exit(1)
    else:
        cluster_resources = ray.cluster_resources()
        gpu_count = int(cluster_resources.get('GPU', 0))

        print(f"✅ Ray cluster already running")
        print(f"   Available GPUs: {gpu_count}")
        print(f"   CPU cores: {int(cluster_resources.get('CPU', 0))}")

        print(f"   Ray cluster GPU resources: {gpu_count}")
        if max_actors:
            print(f"   Will create up to {max_actors} actors (user limit)")
        else:
            print(f"   Will auto-scale to use all healthy GPUs")


def create_gpu_actors(cfg: DictConfig, healthy_gpus: List[int]) -> List[Any]:
    """Create GPU pipeline actors with automatic scaling based on healthy GPUs."""
    # Determine actual number of actors to create
    max_possible_actors = len(healthy_gpus)
    actual_num_actors = max_possible_actors
    
    max_actors = cfg.streaming.runtime.max_actors
    if max_actors is not None:
        actual_num_actors = min(max_actors, max_possible_actors)
        if max_actors > max_possible_actors:
            print(f"⚠️  Requested {max_actors} actors but only {max_possible_actors} healthy GPUs available")
    
    enable_profiling = cfg.streaming.profiling.enable_profiling
    profiling_text = " (with profiling)" if enable_profiling else ""
    print(f"\n🎭 Step 3: Creating {actual_num_actors} GPU Pipeline Actors{profiling_text}")
    print(f"   Available healthy GPUs: {max_possible_actors}")
    if max_actors:
        print(f"   User-specified actor limit: {max_actors}")

    if enable_profiling and cfg.streaming.output.verbose:
        print("   📊 NSys profiling enabled - profile files will be generated per actor")

    # Calculate input shape based on config
    data = cfg.streaming.data
    height = data.input_height if data.input_height is not None else data.input_size
    width = data.input_width if data.input_width is not None else data.input_size
    input_shape = (data.input_channels, height, width)

    try:
        actors = create_pipeline_actors(
            num_actors=actual_num_actors,
            enable_profiling=enable_profiling,
            validate_gpus=False,  # Already validated at system level
            # Pipeline configuration
            input_shape=input_shape,
            batch_size=cfg.streaming.runtime.batch_size,
            # PeakNet configuration from Hydra config
            weights_path=cfg.peaknet.weights_path,
            peaknet_config=cfg.peaknet if (hasattr(cfg.peaknet, 'model') and not cfg.streaming.processing.no_compute) else None,
            compile_model=cfg.performance.compile_model,
            compile_mode=cfg.performance.compile_mode,
            deterministic=True,
            pin_memory=cfg.performance.pin_memory and not cfg.streaming.processing.no_pin_memory
        )

        print(f"✅ Successfully created {len(actors)} GPU actors")

        # Verify actor health
        if cfg.streaming.system.verify_actors:
            print("   Verifying actor health...")
            health_futures = [actor.health_check.remote() for actor in actors]

            try:
                health_results = ray.get(health_futures, timeout=30)
                healthy_count = sum(1 for h in health_results if h.get('status') == 'healthy')
                print(f"   ✅ {healthy_count}/{len(actors)} actors are healthy")

                if cfg.streaming.output.verbose:
                    for i, health in enumerate(health_results):
                        gpu_id = health.get('gpu_id', 'unknown')
                        status = health.get('status', 'unknown')
                        print(f"      Actor {i}: GPU {gpu_id} - {status}")

            except Exception as e:
                print(f"   ⚠️  Actor health check failed: {e}")
                print("   Continuing anyway...")

        return actors

    except Exception as e:
        print(f"❌ Failed to create pipeline actors: {e}")
        sys.exit(1)


def generate_streaming_data(cfg: DictConfig) -> List[Any]:
    """Generate streaming data using Ray tasks."""
    print(f"\n📊 Step 4: Generating Streaming Data")
    
    # Extract config values
    runtime = cfg.streaming.runtime
    data = cfg.streaming.data
    
    # Calculate actual production parameters based on total_samples if provided
    if runtime.total_samples is not None:
        # Calculate required batches to reach total_samples
        total_batches_needed = (runtime.total_samples + runtime.batch_size - 1) // runtime.batch_size
        batches_per_producer = max(1, total_batches_needed // runtime.num_producers)
        # Adjust if we need more producers or batches
        if total_batches_needed > runtime.num_producers * batches_per_producer:
            batches_per_producer += 1
        print(f"   Using total_samples={runtime.total_samples}")
        print(f"   Adjusted batches per producer: {batches_per_producer}")
    else:
        batches_per_producer = runtime.batches_per_producer
        
    print(f"   Producers: {runtime.num_producers}")
    print(f"   Batches per producer: {batches_per_producer}")
    print(f"   Total batches: {runtime.num_producers * batches_per_producer}")
    print(f"   Batch size: {runtime.batch_size} samples")
    print(f"   Total samples: {runtime.num_producers * batches_per_producer * runtime.batch_size}")
    # Calculate input shape based on config
    height = data.input_height if data.input_height is not None else data.input_size
    width = data.input_width if data.input_width is not None else data.input_size
    input_shape = (data.input_channels, height, width)
    print(f"   Input shape: {input_shape}")

    manager = RayDataProducerManager()

    start_time = time.time()

    try:
        producer_futures = manager.launch_producers(
            num_producers=runtime.num_producers,
            batches_per_producer=batches_per_producer,
            batch_size=runtime.batch_size,
            tensor_shape=input_shape,
            inter_batch_delay=runtime.inter_batch_delay,
            deterministic=False  # Random data for realistic streaming
        )

        all_batches = manager.get_all_batches()
        generation_time = time.time() - start_time

        total_samples = len(all_batches) * runtime.batch_size
        generation_rate = total_samples / generation_time

        print(f"✅ Data generation complete:")
        print(f"   Generated: {len(all_batches)} batches ({total_samples} samples)")
        print(f"   Time: {generation_time:.2f}s")
        print(f"   Rate: {generation_rate:.1f} samples/s")

        return all_batches

    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        sys.exit(1)


def process_streaming_data(actors: List[Any], all_batches: List[Any], cfg: DictConfig) -> Dict[str, Any]:
    """Process streaming data across multiple GPU actors."""
    print(f"\n⚡ Step 5: Multi-GPU Streaming Processing")
    print(f"   Processing {len(all_batches)} batches across {len(actors)} GPUs")

    # Distribute batches across actors (round-robin)
    processing_futures = []
    actor_assignments = []

    for batch_idx, batch in enumerate(all_batches):
        actor_idx = batch_idx % len(actors)
        actor = actors[actor_idx]

        future = actor.process_batch_from_refs.remote(batch, batch_idx)
        processing_futures.append(future)
        actor_assignments.append(actor_idx)

    # Process batches and collect results
    print("   Processing batches...")
    processing_start = time.time()

    results = []
    completed = 0
    total_batches = len(processing_futures)

    try:
        for i, future in enumerate(processing_futures):
            result = ray.get(future, timeout=30)
            results.append(result)
            completed += 1

            if cfg.streaming.output.verbose or (completed % max(1, total_batches // 10) == 0):
                progress = (completed / total_batches) * 100
                actor_idx = actor_assignments[i]
                samples = result['batch_size']
                proc_time = result['processing_time']

                print(f"   [{progress:5.1f}%] Batch {completed}/{total_batches} "
                      f"→ GPU Actor {actor_idx}: {samples} samples ({proc_time:.3f}s)")

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return {'success': False, 'error': str(e)}

    total_processing_time = time.time() - processing_start

    # Calculate performance metrics
    total_samples = sum(r['batch_size'] for r in results)
    overall_throughput = total_samples / total_processing_time

    # Per-actor statistics
    actor_stats = {}
    for i, result in enumerate(results):
        actor_idx = actor_assignments[i]
        if actor_idx not in actor_stats:
            actor_stats[actor_idx] = {'batches': 0, 'samples': 0, 'total_time': 0.0}

        actor_stats[actor_idx]['batches'] += 1
        actor_stats[actor_idx]['samples'] += result['batch_size']
        actor_stats[actor_idx]['total_time'] += result['processing_time']

    return {
        'success': True,
        'total_samples': total_samples,
        'total_batches': len(results),
        'total_processing_time': total_processing_time,
        'overall_throughput': overall_throughput,
        'actor_stats': actor_stats,
        'results': results
    }


def print_results(performance: Dict[str, Any], cfg: DictConfig) -> None:
    """Print comprehensive performance results."""
    print("\n📈 Performance Results")
    print("=" * 30)

    if not performance['success']:
        print(f"❌ Processing failed: {performance.get('error', 'Unknown error')}")
        return

    # Overall metrics
    print(f"✅ Overall Performance:")
    print(f"   Total samples processed: {performance['total_samples']:,}")
    print(f"   Total batches: {performance['total_batches']:,}")
    print(f"   Processing time: {performance['total_processing_time']:.2f}s")
    print(f"   Overall throughput: {performance['overall_throughput']:.1f} samples/s")

    # Per-actor breakdown
    print(f"\n🎭 Per-Actor Performance:")
    actor_stats = performance['actor_stats']

    for actor_idx, stats in actor_stats.items():
        actor_throughput = stats['samples'] / stats['total_time'] if stats['total_time'] > 0 else 0
        avg_batch_time = stats['total_time'] / stats['batches'] if stats['batches'] > 0 else 0

        print(f"   GPU Actor {actor_idx}:")
        print(f"      Batches: {stats['batches']}")
        print(f"      Samples: {stats['samples']:,}")
        print(f"      Throughput: {actor_throughput:.1f} samples/s")
        print(f"      Avg batch time: {avg_batch_time:.3f}s")

    # Profiling information
    if cfg.streaming.profiling.enable_profiling:
        import os
        tmpdir = os.environ.get('TMPDIR', '/tmp')
        print(f"\n📊 Profiling Information:")
        print(f"   NSys profiling: enabled")
        print(f"   Profile files: generated per actor (nsys-rep format)")
        print(f"   📁 Files saved to: {tmpdir}/ray/session_latest/logs/nsight/")
        print(f"   💡 Find your .nsys-rep files in Ray's logs directory")
        print(f"   💡 Copy files locally: cp {tmpdir}/ray/session_latest/logs/nsight/*.nsys-rep ./")
        print(f"   💡 Analyze with: nsys-ui <file.nsys-rep> or nsys stats <file.nsys-rep>")

    # Configuration summary
    if cfg.streaming.output.verbose:
        runtime = cfg.streaming.runtime
        data = cfg.streaming.data
        print(f"\n⚙️  Configuration Used:")
        print(f"   Actor limit: {'auto-scale' if runtime.max_actors is None else runtime.max_actors}")
        print(f"   Min GPUs required: {cfg.streaming.system.min_gpus}")
        print(f"   Producers: {runtime.num_producers}")
        print(f"   Batch size: {runtime.batch_size}")
        print(f"   Input size: {data.input_size}x{data.input_size}")
        print(f"   Inter-batch delay: {runtime.inter_batch_delay}s")
        print(f"   Profiling: {'enabled' if cfg.streaming.profiling.enable_profiling else 'disabled'}")


def save_results(performance: Dict[str, Any], cfg: DictConfig) -> None:
    """Save results to output directory if specified."""
    output_dir_path = cfg.streaming.output.output_dir
    if not output_dir_path or not performance['success']:
        return

    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save performance metrics
    import json
    from datetime import datetime

    results_file = output_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'configuration': OmegaConf.to_container(cfg, resolve=True),
        'performance': performance
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {results_file}")




@hydra.main(version_base=None, config_path="conf", config_name="streaming_config")
def main(cfg: DictConfig) -> None:
    """Main entry point for streaming pipeline."""
    
    # Set up logging
    if not cfg.streaming.output.quiet:
        setup_logging(cfg.streaming.output.verbose)
    
    # Print configuration if verbose
    if cfg.streaming.output.verbose:
        print("🔧 Configuration:")
        print(OmegaConf.to_yaml(cfg))
        print("=" * 50)
    
    # Run the pipeline logic directly with config
    run_streaming_pipeline_main(cfg)


def run_streaming_pipeline_main(cfg: DictConfig) -> None:
    """Core streaming pipeline execution logic."""
    # Validate configuration
    try:
        validate_config(cfg)
    except ValueError as e:
        print(f"❌ Invalid configuration: {e}")
        sys.exit(1)

    # Print banner
    if not cfg.streaming.output.quiet:
        print_banner()
        if cfg.streaming.output.verbose:
            print(f"Configuration summary:")
            print(f"  Model: PeakNet (Hydra config: {hasattr(cfg.peaknet, 'model')})")
            print(f"  Data: {cfg.streaming.data.input_channels} channels, {cfg.streaming.data.input_size}px")
            print(f"  Runtime: {cfg.streaming.runtime.batch_size} batch, {cfg.streaming.runtime.num_producers} producers")

    try:
        # Step 1: GPU Environment Setup
        healthy_gpus = setup_gpu_environment(
            cfg.streaming.system.min_gpus,
            cfg.streaming.system.skip_gpu_validation,
            cfg.streaming.output.verbose
        )

        # Step 2: Ray Cluster Setup  
        setup_ray_cluster(cfg)

        # Step 3: Create GPU Actors
        actors = create_gpu_actors(cfg, healthy_gpus)

        # Step 4: Generate Streaming Data
        all_batches = generate_streaming_data(cfg)

        # Step 5: Process Data
        performance = process_streaming_data(actors, all_batches, cfg)

        # Step 6: Results
        if not cfg.streaming.output.quiet:
            print_results(performance, cfg)

        # Step 7: Save Results
        if cfg.streaming.output.output_dir:
            save_results(performance, cfg)

        if performance['success']:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"   Processed {performance['total_samples']:,} samples at {performance['overall_throughput']:.1f} samples/s")
        else:
            print(f"\n💥 Pipeline failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {e}")
        if cfg.streaming.output.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()