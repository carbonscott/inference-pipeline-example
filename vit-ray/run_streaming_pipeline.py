#!/usr/bin/env python3
"""
Ray Multi-GPU Streaming Pipeline

A production-ready interface for running ML model inference at scale with 
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
    python run_streaming_pipeline.py experiment=fast_streaming_test --max-actors 2
    python run_streaming_pipeline.py experiment=large_single_channel_streaming --total-samples 512000 --enable-profiling
    python run_streaming_pipeline.py experiment=production_streaming --max-actors 4
    
    # Mix model templates with runtime overrides
    python run_streaming_pipeline.py model=vit_base_patch32 vit.channels=1 shape=[1,512,512] --max-actors 4 --total-samples 512000
    python run_streaming_pipeline.py model=vit_tiny streaming.runtime.batch_size=16 --enable-profiling
    
    ## Traditional CLI Style (Legacy):
    # Auto-discover and use all healthy GPUs with 4 data producers, small tensors
    python run_streaming_pipeline.py --num-producers 4 --tensor-size 64
    
    # Limit to 2 actors max with custom configuration
    python run_streaming_pipeline.py --max-actors 2 --batch-size 8 --batches-per-producer 10 --verbose
"""

import argparse
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
from ray_data_producer import RayDataProducerManager
from ray_pipeline_actor import create_pipeline_actors
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
    print("🚀 Ray Multi-GPU Streaming Pipeline")
    print("=" * 50)


def validate_args(args) -> None:
    """Validate command line arguments."""
    if args.max_actors is not None and args.max_actors <= 0:
        raise ValueError("--max-actors must be positive")

    if args.min_gpus <= 0:
        raise ValueError("--min-gpus must be positive")

    if args.num_producers <= 0:
        raise ValueError("--num-producers must be positive")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if args.tensor_size <= 0:
        raise ValueError("--tensor-size must be positive")

    if args.tensor_channels <= 0:
        raise ValueError("--tensor-channels must be positive")

    if args.tensor_height is not None and args.tensor_height <= 0:
        raise ValueError("--tensor-height must be positive")

    if args.tensor_width is not None and args.tensor_width <= 0:
        raise ValueError("--tensor-width must be positive")

    if args.inter_batch_delay < 0:
        raise ValueError("--inter-batch-delay cannot be negative")

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


def setup_ray_cluster(args) -> None:
    """Initialize Ray cluster connection."""
    print("\n⚡ Step 2: Ray Cluster Setup")

    if not ray.is_initialized():
        try:
            ray.init()
            cluster_resources = ray.cluster_resources()
            gpu_count = int(cluster_resources.get('GPU', 0))

            print(f"✅ Ray cluster initialized")
            print(f"   Available GPUs: {gpu_count}")
            print(f"   CPU cores: {int(cluster_resources.get('CPU', 0))}")

            print(f"   Ray cluster GPU resources: {gpu_count}")
            if args.max_actors:
                print(f"   Will create up to {args.max_actors} actors (user limit)")
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
        if args.max_actors:
            print(f"   Will create up to {args.max_actors} actors (user limit)")
        else:
            print(f"   Will auto-scale to use all healthy GPUs")


def create_gpu_actors(args, healthy_gpus: List[int]) -> List[Any]:
    """Create GPU pipeline actors with automatic scaling based on healthy GPUs."""
    # Determine actual number of actors to create
    max_possible_actors = len(healthy_gpus)
    actual_num_actors = max_possible_actors
    
    if args.max_actors is not None:
        actual_num_actors = min(args.max_actors, max_possible_actors)
        if args.max_actors > max_possible_actors:
            print(f"⚠️  Requested {args.max_actors} actors but only {max_possible_actors} healthy GPUs available")
    
    profiling_text = " (with profiling)" if args.enable_profiling else ""
    print(f"\n🎭 Step 3: Creating {actual_num_actors} GPU Pipeline Actors{profiling_text}")
    print(f"   Available healthy GPUs: {max_possible_actors}")
    if args.max_actors:
        print(f"   User-specified actor limit: {args.max_actors}")

    if args.enable_profiling and args.verbose:
        print("   📊 NSys profiling enabled - profile files will be generated per actor")
        if args.profiling_output_dir:
            print(f"   📁 Profiling output directory: {args.profiling_output_dir}")

    # Calculate tensor shape based on arguments
    height = args.tensor_height if args.tensor_height is not None else args.tensor_size
    width = args.tensor_width if args.tensor_width is not None else args.tensor_size
    tensor_shape = (args.tensor_channels, height, width)

    try:
        actors = create_pipeline_actors(
            num_actors=actual_num_actors,
            enable_profiling=args.enable_profiling,
            validate_gpus=False,  # Already validated at system level
            # Pipeline configuration
            tensor_shape=tensor_shape,
            batch_size=args.batch_size,
            # Use config values if available (Hydra), otherwise hardcoded defaults (argparse)
            patch_size=getattr(args, 'vit_patch_size', 16),
            depth=getattr(args, 'vit_depth', 0 if args.no_compute else 6),
            heads=getattr(args, 'vit_heads', 8),
            dim=getattr(args, 'vit_dim', 384),
            mlp_dim=getattr(args, 'vit_mlp_dim', 1536),
            channels=getattr(args, 'vit_channels', args.tensor_channels),
            dropout=getattr(args, 'vit_dropout', 0.0),
            emb_dropout=getattr(args, 'vit_emb_dropout', 0.0),
            compile_model=getattr(args, 'compile_model', False),
            compile_mode=getattr(args, 'compile_mode', 'default'),
            deterministic=True,
            pin_memory=getattr(args, 'pin_memory', not args.no_pin_memory)
        )

        print(f"✅ Successfully created {len(actors)} GPU actors")

        # Verify actor health
        if args.verify_actors:
            print("   Verifying actor health...")
            health_futures = [actor.health_check.remote() for actor in actors]

            try:
                health_results = ray.get(health_futures, timeout=30)
                healthy_count = sum(1 for h in health_results if h.get('status') == 'healthy')
                print(f"   ✅ {healthy_count}/{len(actors)} actors are healthy")

                if args.verbose:
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


def generate_streaming_data(args) -> List[Any]:
    """Generate streaming data using Ray tasks."""
    print(f"\n📊 Step 4: Generating Streaming Data")
    
    # Calculate actual production parameters based on total_samples if provided
    if args.total_samples is not None:
        # Calculate required batches to reach total_samples
        total_batches_needed = (args.total_samples + args.batch_size - 1) // args.batch_size
        batches_per_producer = max(1, total_batches_needed // args.num_producers)
        # Adjust if we need more producers or batches
        if total_batches_needed > args.num_producers * batches_per_producer:
            batches_per_producer += 1
        print(f"   Using --total-samples={args.total_samples}")
        print(f"   Adjusted batches per producer: {batches_per_producer}")
    else:
        batches_per_producer = args.batches_per_producer
        
    print(f"   Producers: {args.num_producers}")
    print(f"   Batches per producer: {batches_per_producer}")
    print(f"   Total batches: {args.num_producers * batches_per_producer}")
    print(f"   Batch size: {args.batch_size} samples")
    print(f"   Total samples: {args.num_producers * batches_per_producer * args.batch_size}")
    # Calculate tensor shape based on arguments
    height = args.tensor_height if args.tensor_height is not None else args.tensor_size
    width = args.tensor_width if args.tensor_width is not None else args.tensor_size
    tensor_shape = (args.tensor_channels, height, width)
    print(f"   Tensor shape: {tensor_shape}")

    manager = RayDataProducerManager()

    start_time = time.time()

    try:
        producer_futures = manager.launch_producers(
            num_producers=args.num_producers,
            batches_per_producer=batches_per_producer,
            batch_size=args.batch_size,
            tensor_shape=tensor_shape,
            inter_batch_delay=args.inter_batch_delay,
            deterministic=False  # Random data for realistic streaming
        )

        all_batches = manager.get_all_batches()
        generation_time = time.time() - start_time

        total_samples = len(all_batches) * args.batch_size
        generation_rate = total_samples / generation_time

        print(f"✅ Data generation complete:")
        print(f"   Generated: {len(all_batches)} batches ({total_samples} samples)")
        print(f"   Time: {generation_time:.2f}s")
        print(f"   Rate: {generation_rate:.1f} samples/s")

        return all_batches

    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        sys.exit(1)


def process_streaming_data(actors: List[Any], all_batches: List[Any], args) -> Dict[str, Any]:
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

            if args.verbose or (completed % max(1, total_batches // 10) == 0):
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


def print_results(performance: Dict[str, Any], args) -> None:
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
    if args.enable_profiling:
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
    if args.verbose:
        print(f"\n⚙️  Configuration Used:")
        print(f"   Actor limit: {'auto-scale' if args.max_actors is None else args.max_actors}")
        print(f"   Min GPUs required: {args.min_gpus}")
        print(f"   Producers: {args.num_producers}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Tensor size: {args.tensor_size}x{args.tensor_size}")
        print(f"   Inter-batch delay: {args.inter_batch_delay}s")
        print(f"   Profiling: {'enabled' if args.enable_profiling else 'disabled'}")


def save_results(performance: Dict[str, Any], args) -> None:
    """Save results to output directory if specified."""
    if not args.output_dir or not performance['success']:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save performance metrics
    import json
    from datetime import datetime

    results_file = output_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'configuration': vars(args),
        'performance': performance
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {results_file}")


def config_to_args(cfg: DictConfig) -> argparse.Namespace:
    """Convert Hydra config to argparse.Namespace for backward compatibility."""
    args = argparse.Namespace()
    
    # Runtime parameters
    runtime = cfg.streaming.runtime
    args.max_actors = runtime.max_actors
    args.num_producers = runtime.num_producers
    args.batches_per_producer = runtime.batches_per_producer
    args.total_samples = runtime.total_samples
    args.batch_size = runtime.batch_size
    args.inter_batch_delay = runtime.inter_batch_delay
    
    # System parameters
    system = cfg.streaming.system
    args.min_gpus = system.min_gpus
    args.skip_gpu_validation = system.skip_gpu_validation
    args.verify_actors = system.verify_actors
    
    # Data parameters
    data = cfg.streaming.data
    args.tensor_channels = data.tensor_channels
    args.tensor_size = data.tensor_size
    args.tensor_height = data.tensor_height
    args.tensor_width = data.tensor_width
    
    # Processing options
    processing = cfg.streaming.processing
    args.no_compute = processing.no_compute
    args.no_pin_memory = processing.no_pin_memory
    
    # Output options
    output = cfg.streaming.output
    args.output_dir = output.output_dir
    args.verbose = output.verbose
    args.quiet = output.quiet
    
    # Profiling options
    profiling = cfg.streaming.profiling
    args.enable_profiling = profiling.enable_profiling
    
    # Model parameters from Hydra config
    args.vit_patch_size = cfg.vit.patch_size
    args.vit_depth = cfg.vit.depth
    args.vit_heads = cfg.vit.heads
    args.vit_dim = cfg.vit.dim
    args.vit_mlp_dim = cfg.vit.mlp_dim
    args.vit_channels = cfg.vit.channels
    args.vit_dropout = cfg.vit.get('dropout', 0.0)
    args.vit_emb_dropout = cfg.vit.get('emb_dropout', 0.0)
    
    # Performance settings
    args.pin_memory = cfg.performance.pin_memory
    args.compile_model = cfg.performance.compile_model
    args.compile_mode = cfg.performance.compile_mode
    
    return args


@hydra.main(version_base=None, config_path="conf", config_name="streaming_config")
def hydra_main(cfg: DictConfig) -> None:
    """Hydra-based main entry point for streaming pipeline."""
    
    # Print configuration if verbose
    if cfg.streaming.output.verbose:
        print("🔧 Configuration:")
        print(OmegaConf.to_yaml(cfg))
        print("=" * 50)
    
    # Convert config to args for backward compatibility
    args = config_to_args(cfg)
    
    # Run the existing pipeline logic
    run_streaming_pipeline_main(args)


def run_streaming_pipeline_main(args) -> None:
    """Core streaming pipeline execution logic (shared between Hydra and argparse)."""
    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        print(f"❌ Invalid arguments: {e}")
        sys.exit(1)

    # Print banner
    if not args.quiet:
        print_banner()
        if args.verbose:
            print(f"Configuration: {vars(args)}")

    try:
        # Step 1: GPU Environment Setup
        healthy_gpus = setup_gpu_environment(args.min_gpus, args.skip_gpu_validation, args.verbose)

        # Step 2: Ray Cluster Setup  
        setup_ray_cluster(args)

        # Step 3: Create GPU Actors
        actors = create_gpu_actors(args, healthy_gpus)

        # Step 4: Generate Streaming Data
        all_batches = generate_streaming_data(args)

        # Step 5: Process Data
        performance = process_streaming_data(actors, all_batches, args)

        # Step 6: Results
        if not args.quiet:
            print_results(performance, args)

        # Step 7: Save Results
        if args.output_dir:
            save_results(performance, args)

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
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Ray Multi-GPU Streaming Pipeline for ML Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and use all healthy GPUs
  python run_streaming_pipeline.py --total-samples 2000 --enable-profiling

  # High throughput test with limited actors
  python run_streaming_pipeline.py --max-actors 4 --num-producers 8 --batches-per-producer 20

  # Quick test with small data (auto-scale)
  python run_streaming_pipeline.py --batch-size 2 --tensor-size 64 --verbose

  # Save results for analysis
  python run_streaming_pipeline.py --output-dir ./results --verbose

  # Enable NSys profiling with specific actor count
  python run_streaming_pipeline.py --max-actors 2 --enable-profiling --verbose

  # Performance test requiring at least 3 GPUs
  python run_streaming_pipeline.py --min-gpus 3 --enable-profiling --batches-per-producer 10
  
  # Production mode with validation skipped for faster startup
  python run_streaming_pipeline.py --skip-gpu-validation --total-samples 10000
        """
    )

    # Core pipeline parameters
    parser.add_argument('--max-actors', type=int, default=None,
                       help='Maximum number of GPU actors to create (default: use all healthy GPUs)')
    parser.add_argument('--num-producers', type=int, default=4,
                       help='Number of data producer tasks (default: 4)')
    parser.add_argument('--batches-per-producer', type=int, default=5,
                       help='Batches each producer generates (default: 5)')
    parser.add_argument('--total-samples', type=int, default=None,
                       help='Total number of samples to generate (overrides num-producers * batches-per-producer * batch-size)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Number of samples per batch (default: 4)')

    # Data parameters  
    parser.add_argument('--tensor-size', type=int, default=224,
                       help='Height/width of generated tensors (default: 224)')
    parser.add_argument('--tensor-channels', type=int, default=3,
                       help='Number of input channels (default: 3 for RGB)')
    parser.add_argument('--tensor-height', type=int, default=None,
                       help='Tensor height (default: same as tensor-size)')
    parser.add_argument('--tensor-width', type=int, default=None,
                       help='Tensor width (default: same as tensor-size)')
    parser.add_argument('--inter-batch-delay', type=float, default=0.1,
                       help='Delay between batches in seconds (default: 0.1)')

    # System parameters
    parser.add_argument('--min-gpus', type=int, default=1,
                       help='Minimum healthy GPUs required (default: 1)')
    parser.add_argument('--skip-gpu-validation', action='store_true',
                       help='Skip GPU health validation (faster startup for production clusters)')

    # Processing options
    parser.add_argument('--no-compute', action='store_true',
                       help='Use no-op mode for speed testing (depth=0)')
    parser.add_argument('--no-pin-memory', action='store_true',
                       help='Disable pinned memory for compatibility')
    parser.add_argument('--verify-actors', action='store_true', default=True,
                       help='Verify actor health after creation')

    # Profiling options
    parser.add_argument('--enable-profiling', action='store_true',
                       help='Enable nsys profiling for GPU actors (files saved to Ray logs directory)')

    # Output and debugging
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save results (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable detailed logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output (overrides --verbose)')

    args = parser.parse_args()

    # Set up logging
    if not args.quiet:
        setup_logging(args.verbose)

    # Use shared pipeline execution logic
    run_streaming_pipeline_main(args)


if __name__ == "__main__":
    # Check if user is trying to use Hydra-style arguments
    import sys
    hydra_style_args = any(
        arg.startswith(('model=', 'experiment=', 'vit.', 'streaming.', '+')) or '=' in arg 
        for arg in sys.argv[1:]
    )
    
    if hydra_style_args:
        # Use Hydra main
        hydra_main()
    else:
        # Use argparse main
        main()
