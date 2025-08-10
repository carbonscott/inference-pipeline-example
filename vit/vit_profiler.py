#!/usr/bin/env python3
"""
VIT NSys Profiler - Thin wrapper for adding NSys profiling to VIT pipeline experiments

A minimal wrapper that adds NSys profiling to your existing Hydra workflow:
- Preserves all vit_pipeline.py arguments and functionality  
- Organizes NSys reports by experiment name and timestamp
- Supports multirun bulk profiling
- Simple implementation focused on doing one thing well

Usage:
    python vit_profiler.py experiment=vit32x6x384_compiled
    python vit_profiler.py -m experiment=vit32x6x384_compiled,vit32x12x768_compiled
    python vit_profiler.py experiment=vit32x6x384_compiled batch_size=16
"""

import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime

def extract_experiment_name(args):
    """Extract experiment name from command line arguments"""
    for arg in args:
        if arg.startswith('experiment='):
            return arg.split('=', 1)[1]
    return 'default'

def is_multirun(args):
    """Check if this is a multirun execution"""
    return '-m' in args or '--multirun' in args

def create_output_directory(base_name, is_multirun=False):
    """Create organized output directory for NSys reports"""
    base_dir = Path("nsys_reports")
    base_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if is_multirun:
        output_dir = base_dir / f"bulk_{timestamp}"
    else:
        output_dir = base_dir / base_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, timestamp

def check_nsys_available():
    """Check if nsys is available in PATH"""
    try:
        result = subprocess.run(['nsys', '--version'], 
                               capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def main():
    if not check_nsys_available():
        print("❌ Error: nsys not found in PATH. Please install NVIDIA Nsight Systems.")
        sys.exit(1)
    
    # Parse arguments
    args = sys.argv[1:]  # Remove script name
    
    if not args:
        print("Usage: python vit_profiler.py [same arguments as vit_pipeline.py]")
        print("Examples:")
        print("  python vit_profiler.py experiment=vit32x6x384_compiled")
        print("  python vit_profiler.py -m experiment=vit32x6x384_compiled,vit32x12x768_compiled")
        sys.exit(1)
    
    # Extract experiment info
    experiment_name = extract_experiment_name(args)
    multirun = is_multirun(args)
    
    # Create output directory
    output_dir, timestamp = create_output_directory(experiment_name, multirun)
    
    # Build NSys command
    if multirun:
        # For multirun, let Hydra handle the job numbering
        output_path = output_dir / f"profile_{timestamp}"
    else:
        output_path = output_dir / f"{experiment_name}_{timestamp}"
    
    nsys_cmd = [
        'nsys', 'profile',
        '-o', str(output_path),
        '--trace=cuda,nvtx',
        '--cuda-memory-usage=true',
        'python', 'vit_pipeline.py'
    ] + args
    
    print(f"🚀 Starting NSys profiling...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔬 Experiment: {experiment_name}")
    if multirun:
        print(f"📊 Multirun mode detected")
    print(f"⚡ Command: nsys profile [options] python vit_pipeline.py {' '.join(args)}")
    print()
    
    # Execute
    result = subprocess.run(nsys_cmd, cwd=Path.cwd())
    
    print()  # Add spacing after execution
    if result.returncode == 0:
        print(f"✅ Profiling completed successfully!")
        print(f"📁 NSys reports saved to: {output_dir}")
        
        # List generated files
        nsys_files = list(output_dir.glob("*.nsys-rep"))
        if nsys_files:
            print("📊 Generated files:")
            for f in sorted(nsys_files):
                file_size = f.stat().st_size / (1024*1024)  # MB
                print(f"   {f.name} ({file_size:.1f} MB)")
        
        print()
        print("💡 Next steps:")
        print("   • Open NSys reports in NVIDIA Nsight Systems GUI")
        print("   • Or use command line: nsys stats <report_file>")
        
    else:
        print(f"❌ Profiling failed with exit code: {result.returncode}")
        print("💡 Troubleshooting:")
        print("   • Check if CUDA/GPU is available")
        print("   • Verify vit_pipeline.py arguments are correct")
        print("   • Ensure sufficient disk space for profiling data")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()