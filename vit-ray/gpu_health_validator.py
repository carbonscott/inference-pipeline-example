"""
GPU Health Validator - Pre-Ray system-wide GPU validation

This module provides comprehensive GPU health validation that runs before Ray
initialization to ensure only healthy GPUs are made available to Ray actors.
"""

import torch
import logging
import time
import os
from typing import List, Dict, Any


def comprehensive_gpu_health_check(gpu_id: int) -> Dict[str, Any]:
    """
    Perform comprehensive health check on a specific GPU.
    
    Args:
        gpu_id: GPU device ID to test
        
    Returns:
        Dict with health check results including status, error info, and metrics
    """
    result = {
        'gpu_id': gpu_id,
        'healthy': False,
        'error': None,
        'tests_passed': [],
        'tests_failed': [],
        'metrics': {}
    }
    
    try:
        device = torch.device(f'cuda:{gpu_id}')
        
        # Test 1: Basic device creation and selection
        torch.cuda.set_device(gpu_id)
        result['tests_passed'].append('device_selection')
        
        # Test 2: Memory allocation test
        test_size_mb = 100
        test_tensor = torch.randn(test_size_mb * 256 * 256 // 4, device=device)  # ~100MB
        result['tests_passed'].append('memory_allocation')
        result['metrics']['allocated_memory_mb'] = test_size_mb
        
        # Test 3: Basic compute test
        start_time = time.time()
        compute_result = torch.mm(test_tensor.view(1024, -1), test_tensor.view(-1, 1024))
        torch.cuda.synchronize(device)
        compute_time = time.time() - start_time
        result['tests_passed'].append('basic_compute')
        result['metrics']['compute_time_ms'] = compute_time * 1000
        
        # Test 4: Memory transfer test
        cpu_tensor = torch.randn(1024, 1024)
        start_time = time.time()
        gpu_tensor = cpu_tensor.to(device)
        cpu_result = gpu_tensor.cpu()
        torch.cuda.synchronize(device)
        transfer_time = time.time() - start_time
        result['tests_passed'].append('memory_transfer')
        result['metrics']['transfer_time_ms'] = transfer_time * 1000
        
        # Test 5: Multiple operations stress test
        for i in range(3):
            stress_tensor = torch.randn(512, 512, device=device)
            _ = torch.mm(stress_tensor, stress_tensor.t())
        torch.cuda.synchronize(device)
        result['tests_passed'].append('stress_test')
        
        # Test 6: Memory cleanup
        del test_tensor, compute_result, gpu_tensor, cpu_result, stress_tensor
        torch.cuda.empty_cache()
        result['tests_passed'].append('cleanup')
        
        # All tests passed
        result['healthy'] = True
        
    except Exception as e:
        error_str = str(e)
        result['error'] = error_str
        result['healthy'] = False
        
        # Categorize the type of failure
        if 'CUDA error' in error_str:
            result['tests_failed'].append('cuda_error')
        elif 'out of memory' in error_str.lower():
            result['tests_failed'].append('memory_error')
        elif 'uncorrectable ECC error' in error_str:
            result['tests_failed'].append('ecc_error')
        else:
            result['tests_failed'].append('unknown_error')
    
    return result


def validate_all_gpus() -> Dict[str, Any]:
    """
    Validate all available GPUs and return comprehensive results.
    
    Returns:
        Dict with validation summary and per-GPU results
    """
    logging.info("=== Starting System-Wide GPU Health Validation ===")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        return {
            'success': False,
            'error': 'CUDA not available on this system',
            'healthy_gpus': [],
            'unhealthy_gpus': [],
            'gpu_results': {}
        }
    
    total_gpus = torch.cuda.device_count()
    logging.info(f"Found {total_gpus} CUDA devices to validate")
    
    healthy_gpus = []
    unhealthy_gpus = []
    gpu_results = {}
    
    # Test each GPU
    for gpu_id in range(total_gpus):
        logging.info(f"Testing GPU {gpu_id}...")
        
        result = comprehensive_gpu_health_check(gpu_id)
        gpu_results[gpu_id] = result
        
        if result['healthy']:
            healthy_gpus.append(gpu_id)
            tests_passed = len(result['tests_passed'])
            compute_time = result['metrics'].get('compute_time_ms', 0)
            logging.info(f"✅ GPU {gpu_id}: HEALTHY ({tests_passed} tests passed, "
                        f"{compute_time:.1f}ms compute)")
        else:
            unhealthy_gpus.append(gpu_id)
            error = result['error'][:100] if result['error'] else 'Unknown error'
            failed_tests = ', '.join(result['tests_failed'])
            logging.warning(f"❌ GPU {gpu_id}: UNHEALTHY - {error}")
            logging.warning(f"   Failed tests: {failed_tests}")
    
    # Summary
    validation_result = {
        'success': len(healthy_gpus) > 0,
        'total_gpus': total_gpus,
        'healthy_gpus': healthy_gpus,
        'unhealthy_gpus': unhealthy_gpus,
        'gpu_results': gpu_results
    }
    
    logging.info(f"\n=== GPU Validation Summary ===")
    logging.info(f"Total GPUs: {total_gpus}")
    logging.info(f"Healthy GPUs: {len(healthy_gpus)} {healthy_gpus}")
    logging.info(f"Unhealthy GPUs: {len(unhealthy_gpus)} {unhealthy_gpus}")
    
    if len(healthy_gpus) == 0:
        validation_result['error'] = 'No healthy GPUs found'
        logging.error("❌ VALIDATION FAILED: No healthy GPUs available")
    else:
        logging.info(f"✅ VALIDATION SUCCESS: {len(healthy_gpus)} healthy GPUs ready")
    
    return validation_result


def configure_cuda_for_healthy_gpus(healthy_gpu_ids: List[int]) -> bool:
    """
    Configure CUDA_VISIBLE_DEVICES to only expose healthy GPUs.
    
    Args:
        healthy_gpu_ids: List of healthy GPU IDs
        
    Returns:
        True if configuration was successful
    """
    if not healthy_gpu_ids:
        logging.error("Cannot configure CUDA: No healthy GPUs provided")
        return False
    
    # Convert GPU IDs to string for CUDA_VISIBLE_DEVICES
    cuda_devices = ','.join(map(str, healthy_gpu_ids))
    
    # Set environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
    
    logging.info(f"Configured CUDA_VISIBLE_DEVICES='{cuda_devices}'")
    logging.info(f"Ray will now only see {len(healthy_gpu_ids)} healthy GPUs")
    
    # Note: CUDA context verification will happen in child processes
    # PyTorch/CUDA needs to be restarted to see the new CUDA_VISIBLE_DEVICES
    logging.info(f"✅ CUDA_VISIBLE_DEVICES configured - child processes will see {len(healthy_gpu_ids)} GPUs")
    logging.warning("Note: Current process may still see old GPU layout - child processes will see new layout")
    return True


def get_healthy_gpus_for_ray(min_gpus: int = 1) -> List[int]:
    """
    Main function to get healthy GPUs and configure the environment for Ray.
    
    Args:
        min_gpus: Minimum number of healthy GPUs required
        
    Returns:
        List of healthy GPU IDs (in CUDA_VISIBLE_DEVICES space)
        
    Raises:
        RuntimeError: If insufficient healthy GPUs found
    """
    # Validate all GPUs
    validation_result = validate_all_gpus()
    
    if not validation_result['success']:
        error_msg = validation_result.get('error', 'GPU validation failed')
        raise RuntimeError(f"GPU validation failed: {error_msg}")
    
    healthy_gpus = validation_result['healthy_gpus']
    
    if len(healthy_gpus) < min_gpus:
        raise RuntimeError(f"Need at least {min_gpus} healthy GPUs, "
                          f"but only {len(healthy_gpus)} found: {healthy_gpus}")
    
    # Configure CUDA to only see healthy GPUs
    if not configure_cuda_for_healthy_gpus(healthy_gpus):
        raise RuntimeError("Failed to configure CUDA for healthy GPUs")
    
    # Return the remapped GPU IDs (0, 1, 2, ... in CUDA_VISIBLE_DEVICES space)
    remapped_gpu_ids = list(range(len(healthy_gpus)))
    
    logging.info(f"Ray will see GPUs as: {remapped_gpu_ids}")
    logging.info(f"These map to physical GPUs: {healthy_gpus}")
    
    return remapped_gpu_ids


if __name__ == "__main__":
    # Standalone testing
    logging.basicConfig(level=logging.INFO)
    
    try:
        healthy_gpus = get_healthy_gpus_for_ray(min_gpus=2)
        print(f"\n🎉 SUCCESS: {len(healthy_gpus)} healthy GPUs configured for Ray")
        print(f"Ray GPU IDs: {healthy_gpus}")
    except RuntimeError as e:
        print(f"\n💥 FAILED: {e}")