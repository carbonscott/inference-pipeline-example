#!/usr/bin/env python3
"""
Test script for GPU health check functionality.

This script tests the GPU health validation functions independently
and as part of the Ray pipeline initialization.
"""

import torch
import logging
import sys
from vit_utils import validate_gpu_health, get_healthy_gpus, get_gpu_health_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def test_individual_gpu_health():
    """Test health validation for individual GPUs."""
    logging.info("=== Testing Individual GPU Health Validation ===")
    
    if not torch.cuda.is_available():
        logging.warning("CUDA not available - cannot test GPU health")
        return False
    
    total_gpus = torch.cuda.device_count()
    logging.info(f"Found {total_gpus} GPUs to test")
    
    healthy_count = 0
    unhealthy_count = 0
    
    for gpu_id in range(total_gpus):
        logging.info(f"\nTesting GPU {gpu_id}...")
        
        # Test health validation
        is_healthy = validate_gpu_health(gpu_id)
        logging.info(f"GPU {gpu_id} health: {'✅ HEALTHY' if is_healthy else '❌ UNHEALTHY'}")
        
        # Get detailed health info
        health_info = get_gpu_health_info(gpu_id)
        if health_info['status'] == 'healthy':
            logging.info(f"  Name: {health_info['name']}")
            logging.info(f"  Memory: {health_info['total_memory_mb']:.0f} MB")
            logging.info(f"  Compute: {health_info['compute_capability']}")
            healthy_count += 1
        else:
            logging.warning(f"  Error: {health_info.get('error', 'Unknown')}")
            unhealthy_count += 1
    
    logging.info(f"\nGPU Health Summary: {healthy_count} healthy, {unhealthy_count} unhealthy")
    
    # Test passes if at least one GPU is healthy (pipeline can still work)
    if healthy_count > 0:
        logging.info("✅ GPU health validation working correctly - at least one healthy GPU found")
        return True
    else:
        logging.error("❌ No healthy GPUs found - system cannot run pipeline")
        return False


def test_healthy_gpu_detection():
    """Test the get_healthy_gpus function."""
    logging.info("\n=== Testing Healthy GPU Detection ===")
    
    healthy_gpus = get_healthy_gpus()
    logging.info(f"Healthy GPUs detected: {healthy_gpus}")
    
    if len(healthy_gpus) == 0:
        logging.warning("No healthy GPUs found!")
        return False
    
    # Verify each reported healthy GPU actually works
    all_verified = True
    for gpu_id in healthy_gpus:
        if not validate_gpu_health(gpu_id):
            logging.error(f"GPU {gpu_id} reported as healthy but failed validation!")
            all_verified = False
    
    if all_verified:
        logging.info("✅ All reported healthy GPUs validated successfully")
    
    return all_verified


def test_ray_integration_simulation():
    """Simulate how the GPU health check would work with Ray."""
    logging.info("\n=== Testing Ray Integration Simulation ===")
    
    healthy_gpus = get_healthy_gpus()
    
    # Simulate Ray actor creation logic
    requested_actors = 4
    actual_actors = min(len(healthy_gpus), requested_actors)
    
    logging.info(f"Requested actors: {requested_actors}")
    logging.info(f"Healthy GPUs available: {len(healthy_gpus)}")
    logging.info(f"Actual actors that would be created: {actual_actors}")
    
    if actual_actors < requested_actors:
        logging.warning(f"Would reduce from {requested_actors} to {actual_actors} actors due to unhealthy GPUs")
    
    if actual_actors == 0:
        logging.error("No actors could be created - no healthy GPUs available!")
        return False
    
    # Simulate actor health validation
    for i in range(actual_actors):
        gpu_id = healthy_gpus[i]
        logging.info(f"Actor {i} would use GPU {gpu_id}")
        
        # This is what happens during actor initialization
        if not validate_gpu_health(gpu_id):
            logging.error(f"Actor initialization would fail on GPU {gpu_id}")
            return False
    
    logging.info("✅ Ray integration simulation successful")
    return True


def main():
    """Run all GPU health check tests."""
    logging.info("Starting GPU Health Check Tests")
    
    success = True
    
    # Test 1: Individual GPU health
    success &= test_individual_gpu_health()
    
    # Test 2: Healthy GPU detection
    success &= test_healthy_gpu_detection()
    
    # Test 3: Ray integration simulation
    success &= test_ray_integration_simulation()
    
    # Summary
    logging.info("\n=== Test Results Summary ===")
    if success:
        logging.info("🎉 All GPU health check tests passed!")
        return 0
    else:
        logging.error("❌ Some tests failed - GPU health check may have issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())