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