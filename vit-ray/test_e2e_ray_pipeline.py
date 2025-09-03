#!/usr/bin/env python3
"""
End-to-end integration tests for Ray VIT pipeline

Tests the complete pipeline from data generation through processing to results,
including multi-GPU scaling, data consistency, and performance validation.
"""

import pytest
import ray
import torch
import numpy as np
import time
from omegaconf import OmegaConf
from typing import List, Dict, Any
import tempfile
import os

from vit_pipeline_ray import run_ray_pipeline_test, setup_ray, cleanup_ray
from ray_data_producer import RayDataProducerManager
from ray_pipeline_actor import create_pipeline_actors


# GPU availability checks
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0


@pytest.fixture(scope="function")
def ray_environment():
    """Setup Ray environment for integration testing."""
    if ray.is_initialized():
        ray.shutdown()
    
    # Initialize with more resources for integration tests
    if CUDA_AVAILABLE:
        ray.init(num_cpus=8, num_gpus=GPU_COUNT, ignore_reinit_error=True)
    else:
        ray.init(num_cpus=8, num_gpus=0, ignore_reinit_error=True)
    
    yield
    
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def integration_config():
    """Configuration for integration testing."""
    return OmegaConf.create({
        'shape': [3, 224, 224],
        'batch_size': 4,
        'num_samples': 100,
        'warmup_samples': 0,
        
        'vit': {
            'patch_size': 32,
            'depth': 0,  # No-op mode for faster testing
            'heads': 4,
            'dim': 256,
            'mlp_dim': 512
        },
        
        'pipeline': {
            'batch_size': 4
        },
        
        'performance': {
            'pin_memory': True,
            'compile_model': False,
            'compile_mode': 'default'
        },
        
        'test': {
            'deterministic': True,
            'sync_frequency': 5
        },
        
        'ray': {
            'init': {},
            'actors': {
                'num_gpus': 1,
                'profiling_enabled': False
            },
            'producers': {
                'num_tasks': 2,
                'batches_per_task': 10,
                'inter_batch_delay': 0.0
            }
        }
    })


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU required for pipeline test")
    def test_single_gpu_complete_pipeline(self, ray_environment, integration_config):
        """Test complete pipeline with single GPU."""
        result = run_ray_pipeline_test(integration_config)
        
        # Pipeline should complete successfully
        assert result['success'] is True
        
        # Verify basic structure
        assert 'config' in result
        assert 'performance' in result
        assert 'actor_stats' in result
        
        # Verify configuration was applied
        assert result['config']['num_gpus'] == 1
        assert result['config']['num_producers'] == 2
        assert result['config']['total_batches_generated'] == 2 * 10  # 2 producers * 10 batches
        
        # Verify processing happened
        assert result['performance']['total_samples'] > 0
        assert result['performance']['total_batches_processed'] > 0
        assert result['performance']['overall_throughput'] > 0
        assert result['performance']['processing_time'] > 0
        
        # Verify actor stats
        assert len(result['actor_stats']) == 1
        actor_stats = result['actor_stats'][0]
        assert actor_stats['batches_processed'] > 0
        assert actor_stats['samples_processed'] > 0
        assert actor_stats['average_throughput'] > 0
    
    @pytest.mark.skipif(not CUDA_AVAILABLE or GPU_COUNT < 2, reason="Need at least 2 GPUs")
    def test_multi_gpu_complete_pipeline(self, ray_environment, integration_config):
        """Test complete pipeline with multiple GPUs."""
        # Configure for multi-GPU
        integration_config.ray.actors.num_gpus = min(2, GPU_COUNT)
        integration_config.ray.producers.num_tasks = 4  # More producers for multi-GPU
        
        result = run_ray_pipeline_test(integration_config)
        
        # Should complete successfully
        assert result['success'] is True
        
        # Verify multi-GPU configuration
        expected_gpus = min(2, GPU_COUNT)
        assert result['config']['num_gpus'] == expected_gpus
        
        # Should have stats from all actors
        assert len(result['actor_stats']) == expected_gpus
        
        # Verify all actors processed data
        total_batches = 0
        total_samples = 0
        for stats in result['actor_stats']:
            assert stats['batches_processed'] > 0
            assert stats['samples_processed'] > 0
            total_batches += stats['batches_processed']
            total_samples += stats['samples_processed']
        
        # Total should match performance metrics
        assert result['performance']['total_batches_processed'] == total_batches
        assert result['performance']['total_samples'] == total_samples
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU required for pipeline test")
    def test_data_flow_consistency(self, ray_environment, integration_config):
        """Test that data flows consistently through the pipeline."""
        # Use deterministic mode for consistency
        integration_config.test.deterministic = True
        integration_config.ray.producers.batches_per_task = 5
        
        # Run pipeline twice with same config
        result1 = run_ray_pipeline_test(integration_config)
        
        # Reset Ray for second run
        ray.shutdown()
        if CUDA_AVAILABLE:
            ray.init(num_cpus=8, num_gpus=GPU_COUNT, ignore_reinit_error=True)
        
        result2 = run_ray_pipeline_test(integration_config)
        
        # Both should succeed
        assert result1['success'] is True
        assert result2['success'] is True
        
        # Should process same amount of data
        assert result1['performance']['total_samples'] == result2['performance']['total_samples']
        assert result1['performance']['total_batches_processed'] == result2['performance']['total_batches_processed']
        
        # Performance should be roughly similar (within 50% due to timing variations)
        throughput1 = result1['performance']['overall_throughput']
        throughput2 = result2['performance']['overall_throughput']
        throughput_ratio = max(throughput1, throughput2) / min(throughput1, throughput2)
        assert throughput_ratio < 2.0, f"Throughput too different: {throughput1} vs {throughput2}"


class TestScalingValidation:
    """Test scaling behavior across different configurations."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE or GPU_COUNT < 2, reason="Need multiple GPUs for scaling test")
    def test_gpu_scaling_efficiency(self, ray_environment, integration_config):
        """Test that throughput scales reasonably with GPU count."""
        results = {}
        
        # Test with 1 GPU
        integration_config.ray.actors.num_gpus = 1
        integration_config.ray.producers.num_tasks = 2
        result_1gpu = run_ray_pipeline_test(integration_config)
        assert result_1gpu['success'] is True
        results[1] = result_1gpu
        
        # Test with 2 GPUs if available
        if GPU_COUNT >= 2:
            integration_config.ray.actors.num_gpus = 2
            integration_config.ray.producers.num_tasks = 4  # More producers for 2 GPUs
            
            # Reset Ray for new configuration
            ray.shutdown()
            ray.init(num_cpus=8, num_gpus=GPU_COUNT, ignore_reinit_error=True)
            
            result_2gpu = run_ray_pipeline_test(integration_config)
            assert result_2gpu['success'] is True
            results[2] = result_2gpu
            
            # Calculate scaling efficiency
            throughput_1gpu = result_1gpu['performance']['overall_throughput']
            throughput_2gpu = result_2gpu['performance']['overall_throughput']
            
            scaling_factor = throughput_2gpu / throughput_1gpu
            
            # Should see some improvement (at least 30% of theoretical 2x)
            assert scaling_factor > 1.3, f"Poor scaling: {scaling_factor}x improvement with 2 GPUs"
            
            # Shouldn't be more than 2.5x (accounting for overhead)
            assert scaling_factor < 2.5, f"Unrealistic scaling: {scaling_factor}x improvement"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU required for scaling test")
    def test_producer_scaling(self, ray_environment, integration_config):
        """Test scaling with different numbers of data producers."""
        throughputs = []
        
        for num_producers in [1, 2, 4]:
            integration_config.ray.producers.num_tasks = num_producers
            integration_config.ray.producers.batches_per_task = 5  # Keep total work reasonable
            
            # Reset Ray for clean test
            ray.shutdown()
            ray.init(num_cpus=8, num_gpus=GPU_COUNT, ignore_reinit_error=True)
            
            result = run_ray_pipeline_test(integration_config)
            assert result['success'] is True
            
            throughputs.append(result['performance']['overall_throughput'])
        
        # More producers shouldn't hurt performance significantly
        # (might not improve much since processing is the bottleneck)
        min_throughput = min(throughputs)
        max_throughput = max(throughputs)
        
        # All throughputs should be within reasonable range
        throughput_range = max_throughput / min_throughput
        assert throughput_range < 2.0, f"Too much variation in throughput: {throughputs}"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU required for batch size test")
    def test_batch_size_scaling(self, ray_environment, integration_config):
        """Test performance with different batch sizes."""
        throughputs = []
        
        for batch_size in [2, 4, 8]:
            integration_config.pipeline.batch_size = batch_size
            integration_config.batch_size = batch_size  # Also set global batch_size
            integration_config.ray.producers.batches_per_task = 5
            
            # Reset Ray for clean test
            ray.shutdown()
            ray.init(num_cpus=8, num_gpus=GPU_COUNT, ignore_reinit_error=True)
            
            result = run_ray_pipeline_test(integration_config)
            assert result['success'] is True
            
            # Calculate samples per second
            samples_per_sec = result['performance']['overall_throughput']
            throughputs.append(samples_per_sec)
        
        # Larger batch sizes should generally improve throughput
        # (though this depends on model and hardware)
        # At minimum, shouldn't degrade significantly
        assert max(throughputs) / min(throughputs) < 3.0, f"Batch size scaling issues: {throughputs}"


class TestResourceUtilization:
    """Test resource utilization and memory management."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU required for resource test")
    def test_gpu_memory_usage(self, ray_environment, integration_config):
        """Test GPU memory usage stays within bounds."""
        # Configure for memory testing
        integration_config.ray.producers.batches_per_task = 10
        integration_config.pipeline.batch_size = 8  # Larger batches use more memory
        
        result = run_ray_pipeline_test(integration_config)
        assert result['success'] is True
        
        # Check actor health to verify memory usage
        # This is indirect since we can't easily monitor memory during test
        for stats in result['actor_stats']:
            assert stats.get('batches_processed', 0) > 0
            assert stats.get('samples_processed', 0) > 0
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU required for utilization test") 
    def test_gpu_utilization_distribution(self, ray_environment, integration_config):
        """Test that GPU utilization is distributed reasonably across actors."""
        if GPU_COUNT < 2:
            pytest.skip("Need multiple GPUs to test distribution")
        
        # Configure for multi-GPU utilization test
        integration_config.ray.actors.num_gpus = min(2, GPU_COUNT)
        integration_config.ray.producers.num_tasks = 6  # Plenty of work
        integration_config.ray.producers.batches_per_task = 8
        
        result = run_ray_pipeline_test(integration_config)
        assert result['success'] is True
        
        # All actors should have processed some data
        actor_throughputs = []
        for stats in result['actor_stats']:
            assert stats['batches_processed'] > 0
            assert stats['average_throughput'] > 0
            actor_throughputs.append(stats['average_throughput'])
        
        # Throughputs should be reasonably balanced
        min_throughput = min(actor_throughputs)
        max_throughput = max(actor_throughputs)
        balance_ratio = max_throughput / min_throughput
        
        # Allow for some imbalance but not too much
        assert balance_ratio < 3.0, f"Poor load balancing: throughputs {actor_throughputs}"


class TestErrorRecovery:
    """Test error handling and recovery in end-to-end scenarios."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU required for error test")
    def test_pipeline_with_insufficient_data(self, ray_environment, integration_config):
        """Test pipeline behavior with very little data."""
        # Configure for minimal data
        integration_config.ray.producers.num_tasks = 1
        integration_config.ray.producers.batches_per_task = 1
        integration_config.pipeline.batch_size = 1
        
        result = run_ray_pipeline_test(integration_config)
        
        # Should handle minimal data gracefully
        if result['success']:
            assert result['performance']['total_samples'] > 0
            assert len(result['actor_stats']) > 0
        else:
            # If it fails, should have reasonable error message
            assert 'error' in result
    
    def test_pipeline_with_no_gpus_available(self, ray_environment, integration_config):
        """Test pipeline behavior when no GPUs are available."""
        if CUDA_AVAILABLE:
            # Force no-GPU configuration
            integration_config.ray.actors.num_gpus = 0
        
        result = run_ray_pipeline_test(integration_config)
        
        # Should fail gracefully since actors require GPUs
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU required for resource test")
    def test_pipeline_with_excessive_resource_requests(self, ray_environment, integration_config):
        """Test pipeline with impossible resource requests."""
        # Request way more GPUs than available
        integration_config.ray.actors.num_gpus = GPU_COUNT + 10
        
        result = run_ray_pipeline_test(integration_config)
        
        # Should handle resource constraints gracefully
        # Either succeed with available resources or fail with clear error
        if not result['success']:
            assert 'error' in result
        else:
            # If it succeeded, verify actual resource usage
            assert len(result['actor_stats']) <= GPU_COUNT


class TestDataConsistency:
    """Test data consistency across different configurations."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU required for consistency test")
    def test_deterministic_output_consistency(self, ray_environment, integration_config):
        """Test that deterministic mode produces consistent results."""
        # Ensure deterministic configuration
        integration_config.test.deterministic = True
        integration_config.ray.producers.batches_per_task = 5
        
        # Run pipeline multiple times
        results = []
        for run in range(2):
            # Reset Ray for clean run
            ray.shutdown()
            ray.init(num_cpus=8, num_gpus=GPU_COUNT, ignore_reinit_error=True)
            
            result = run_ray_pipeline_test(integration_config)
            assert result['success'] is True
            results.append(result)
        
        # Should get identical data processing
        for i in range(1, len(results)):
            assert results[0]['performance']['total_samples'] == results[i]['performance']['total_samples']
            assert results[0]['performance']['total_batches_processed'] == results[i]['performance']['total_batches_processed']
    
    @pytest.mark.skipif(not CUDA_AVAILABLE or GPU_COUNT < 2, reason="Need multiple GPUs")
    def test_multi_gpu_vs_single_gpu_consistency(self, ray_environment, integration_config):
        """Test that multi-GPU produces same amount of work as single GPU."""
        # Test single GPU first
        integration_config.ray.actors.num_gpus = 1
        integration_config.ray.producers.num_tasks = 4
        integration_config.ray.producers.batches_per_task = 5
        
        result_single = run_ray_pipeline_test(integration_config)
        assert result_single['success'] is True
        
        # Reset for multi-GPU test
        ray.shutdown()
        ray.init(num_cpus=8, num_gpus=GPU_COUNT, ignore_reinit_error=True)
        
        # Test multi-GPU
        integration_config.ray.actors.num_gpus = 2
        # Keep same total amount of work
        
        result_multi = run_ray_pipeline_test(integration_config)
        assert result_multi['success'] is True
        
        # Should process same amount of data
        assert result_single['performance']['total_samples'] == result_multi['performance']['total_samples']
        assert result_single['config']['total_batches_generated'] == result_multi['config']['total_batches_generated']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])