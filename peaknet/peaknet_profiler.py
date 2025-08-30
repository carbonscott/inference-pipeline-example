#!/usr/bin/env python3
"""
PeakNet NSys Profiler - Individual experiment profiling for systematic analysis

Creates individual NSys profiles for each experiment, enabling:
- Separate analysis of each model configuration 
- Organized output by experiment name
- Support for both single and batch profiling
- Sequential execution for clean profiling data

Usage:
    python peaknet_profiler.py experiment=peaknet_small_compiled
    python peaknet_profiler.py -m experiment=peaknet_small_compiled,peaknet_base_compiled
    python peaknet_profiler.py experiment=peaknet_small_compiled batch_size=16 num_samples=500
"""

import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime


def parse_experiment_list(args):
    """Parse experiment list from command line arguments"""
    # Find experiment argument
    experiment_arg = None
    for arg in args:
        if arg.startswith('experiment='):
            experiment_arg = arg
            break
    
    if not experiment_arg:
        return ['default']
    
    # Extract experiments (handle comma-separated list)
    exp_value = experiment_arg.split('=', 1)[1]
    experiments = [exp.strip() for exp in exp_value.split(',')]
    return experiments


def get_other_args(args):
    """Get all arguments except experiment= and multirun flags"""
    other_args = []
    skip_next = False
    
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
            
        if arg in ['-m', '--multirun']:
            continue
        elif arg.startswith('experiment='):
            continue
        else:
            other_args.append(arg)
    
    return other_args


def run_individual_experiment(experiment_name, other_args, base_timestamp):
    """Run NSys profiling for a single experiment"""
    # Create individual output directory
    output_dir = Path("nsys_reports") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build NSys command for this specific experiment
    output_path = output_dir / f"{experiment_name}_{base_timestamp}"
    
    nsys_cmd = [
        'nsys', 'profile',
        '-o', str(output_path),
        '--trace=cuda,nvtx',
        '--cuda-memory-usage=true',
        'python', 'peaknet_pipeline.py',
        f'experiment={experiment_name}'
    ] + other_args
    
    print(f"🔬 Profiling: {experiment_name}")
    print(f"📁 Output: {output_path}.nsys-rep")
    
    # Execute the command
    result = subprocess.run(nsys_cmd, cwd=Path.cwd())
    
    if result.returncode == 0:
        # Check if file was actually created
        nsys_file = Path(str(output_path) + ".nsys-rep")
        if nsys_file.exists():
            file_size = nsys_file.stat().st_size / (1024*1024)  # MB
            print(f"✅ {experiment_name}: {file_size:.1f} MB")
            return True, nsys_file
        else:
            print(f"❌ {experiment_name}: NSys file not created")
            return False, None
    else:
        print(f"❌ {experiment_name}: Failed with exit code {result.returncode}")
        return False, None


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
        print("Usage: python peaknet_profiler.py [same arguments as peaknet_pipeline.py]")
        print("Examples:")
        print("  python peaknet_profiler.py experiment=peaknet_small_compiled")
        print("  python peaknet_profiler.py -m experiment=peaknet_small_compiled,peaknet_base_compiled")
        print("  python peaknet_profiler.py experiment=peaknet_small_compiled num_samples=500")
        sys.exit(1)
    
    # Parse experiments and other arguments
    experiments = parse_experiment_list(args)
    other_args = get_other_args(args)
    is_multirun = len(experiments) > 1
    
    # Create base timestamp for this profiling session
    base_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("🚀 Starting individual NSys profiling...")
    print(f"📊 Experiments to profile: {len(experiments)}")
    if is_multirun:
        print(f"🔬 Running in batch mode: {', '.join(experiments[:3])}{'...' if len(experiments) > 3 else ''}")
    else:
        print(f"🔬 Single experiment: {experiments[0]}")
    
    if other_args:
        print(f"⚙️  Additional args: {' '.join(other_args)}")
    print()
    
    # Profile each experiment individually
    successful_runs = []
    failed_runs = []
    
    for i, experiment in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] {experiment}")
        
        success, nsys_file = run_individual_experiment(experiment, other_args, base_timestamp)
        
        if success:
            successful_runs.append((experiment, nsys_file))
        else:
            failed_runs.append(experiment)
        
        print()  # Add spacing between experiments
    
    # Final summary
    print("=" * 60)
    print("🎯 PROFILING SUMMARY")
    print("=" * 60)
    
    if successful_runs:
        print(f"✅ Successfully profiled: {len(successful_runs)}/{len(experiments)} experiments")
        print()
        print("📁 Generated NSys reports:")
        
        for experiment, nsys_file in successful_runs:
            file_size = nsys_file.stat().st_size / (1024*1024)  # MB
            print(f"   📊 {experiment}: {nsys_file.name} ({file_size:.1f} MB)")
    
    if failed_runs:
        print()
        print(f"❌ Failed experiments: {len(failed_runs)}")
        for experiment in failed_runs:
            print(f"   ❌ {experiment}")
    
    print()
    print("💡 Next steps:")
    if successful_runs:
        print("   • Open individual NSys reports in NVIDIA Nsight Systems GUI")
        print("   • Compare experiments: nsys stats <report1> <report2>")
        print(f"   • Reports organized in: nsys_reports/<experiment_name>/")
    
    if failed_runs:
        print()
        print("🔧 Troubleshooting failed experiments:")
        print("   • Check GPU availability and CUDA installation")
        print("   • Verify experiment configs exist in conf/experiment/")
        print("   • Check PeakNet installation and YAML configuration paths")
        print("   • Ensure sufficient disk space for profiling data")
    
    # Exit with error code if any experiments failed
    if failed_runs:
        sys.exit(1)


if __name__ == "__main__":
    main()