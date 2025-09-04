#!/usr/bin/env python3

import torch

def check_gpu_health():
    """Check GPU health using PyTorch"""

    if not torch.cuda.is_available():
        print("CUDA is not available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")
    print("\n=== GPU Health Check ===")

    for i in range(num_gpus):
        try:
            device = torch.device(f'cuda:{i}')
            torch.cuda.set_device(device)

            # Get GPU properties
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i} ({props.name}):")

            # Test memory allocation and computation
            test_tensor = torch.randn(1024, 1024, device=device)
            result = test_tensor @ test_tensor.T

            # Clear cache to free memory
            del test_tensor, result
            torch.cuda.empty_cache()

            print(f"  Status: ✅ Accessible and functional")

        except RuntimeError as e:
            error_msg = str(e).lower()
            if 'ecc' in error_msg:
                print(f"  Status: ❌ ECC Error - {e}")
            elif 'out of memory' in error_msg:
                print(f"  Status: ⚠️  Memory Error - {e}")
            else:
                print(f"  Status: ❌ Error - {e}")
        except Exception as e:
            print(f"  Status: ❌ Unexpected Error - {e}")

        print()

if __name__ == "__main__":
    check_gpu_health()
