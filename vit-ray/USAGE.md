# Ray Multi-GPU Streaming Pipeline - Usage Guide

This guide explains how to use the Ray-based multi-GPU streaming pipeline for ML model inference at scale.

## Overview

The pipeline implements a clean architecture with these components:

1. **Pre-Ray GPU Health Validation** - Filters out faulty GPUs before Ray starts
2. **Streaming Data Generation** - Simulates continuous data sources using Ray tasks  
3. **Multi-GPU Processing** - Distributes inference across healthy GPUs automatically
4. **Performance Monitoring** - Comprehensive metrics and reporting

## Quick Start

### Basic Usage

Run the pipeline with default settings (2 GPUs, 4 data producers):

```bash
python run_streaming_pipeline.py
```

### Custom Configuration

Auto-discover and use all healthy GPUs with 8 data producers:

```bash
python run_streaming_pipeline.py --num-producers 8
```

Limit to 4 actors maximum:

```bash
python run_streaming_pipeline.py --max-actors 4 --num-producers 8
```

### Verbose Output

See detailed progress and configuration:

```bash
python run_streaming_pipeline.py --verbose
```

## Command Line Arguments

### Core Pipeline Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-actors` | None | Maximum number of GPU actors to create (None = use all healthy GPUs) |
| `--num-producers` | 4 | Number of data producer tasks |
| `--batches-per-producer` | 5 | Batches each producer generates |
| `--total-samples` | None | Total samples to generate (overrides producers × batches × batch-size) |
| `--batch-size` | 4 | Number of samples per batch |

### Data Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--tensor-size` | 224 | Height/width of generated tensors |
| `--inter-batch-delay` | 0.1 | Delay between batches (seconds) |

### System Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--min-gpus` | 1 | Minimum healthy GPUs required |

### Processing Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-compute` | False | Use no-op mode for speed testing |
| `--no-pin-memory` | False | Disable pinned memory |
| `--verify-actors` | True | Verify actor health after creation |

### Output Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | None | Directory to save results |
| `--verbose`, `-v` | False | Enable detailed logging |
| `--quiet`, `-q` | False | Minimal output |

## Example Use Cases

### 1. Quick Test (Small Scale)

Test the pipeline quickly with minimal resources (auto-discover GPUs):

```bash
python run_streaming_pipeline.py \
    --batch-size 2 \
    --tensor-size 64 \
    --batches-per-producer 3 \
    --verbose
```

### 2. Performance Testing (Large Scale)

Run high-throughput test limiting to 4 actors:

```bash
python run_streaming_pipeline.py \
    --max-actors 4 \
    --num-producers 8 \
    --batches-per-producer 20 \
    --batch-size 8 \
    --inter-batch-delay 0.05
```

### 3. Speed Benchmark (No-Op Mode)

Test pure throughput without actual computation, auto-scale to all GPUs:

```bash
python run_streaming_pipeline.py \
    --no-compute \
    --batch-size 16 \
    --verbose
```

### 4. Results Collection

Save detailed results for analysis (auto-scale to available GPUs):

```bash
python run_streaming_pipeline.py \
    --output-dir ./results \
    --verbose
```

### 5. Streaming Simulation

Simulate realistic streaming with delays (ensure at least 2 GPUs):

```bash
python run_streaming_pipeline.py \
    --min-gpus 2 \
    --num-producers 6 \
    --batches-per-producer 15 \
    --inter-batch-delay 0.2 \
    --verbose
```

## Understanding the Output

### Execution Phases

The pipeline runs in these phases:

1. **🔍 GPU Health Validation** - Tests all GPUs, filters out unhealthy ones
2. **⚡ Ray Cluster Setup** - Initializes Ray with healthy GPUs only  
3. **🎭 GPU Actor Creation** - Creates pipeline actors on each GPU
4. **📊 Data Generation** - Generates streaming tensor data
5. **⚡ Multi-GPU Processing** - Processes data across GPUs in parallel
6. **📈 Results** - Shows performance metrics and statistics

### Success Indicators

Look for these signs of successful execution:

- ✅ **GPU validation**: `Found N healthy GPUs`  
- ✅ **Ray cluster**: `Ray cluster initialized`
- ✅ **Actors**: `Successfully created N GPU actors`
- ✅ **Data**: `Data generation complete`
- ✅ **Processing**: `[100.0%] Batch processing complete`
- ✅ **Results**: `Pipeline completed successfully!`

### Performance Metrics

The pipeline reports these metrics:

- **Overall throughput**: Samples processed per second across all GPUs
- **Per-actor performance**: Individual GPU actor statistics
- **Processing time**: Total time for inference across all batches  
- **Round-robin distribution**: Verification that work is distributed across GPUs

Example output:
```
📈 Performance Results
==============================
✅ Overall Performance:
   Total samples processed: 160
   Total batches: 40
   Processing time: 8.45s
   Overall throughput: 18.9 samples/s

🎭 Per-Actor Performance:
   GPU Actor 0:
      Batches: 20
      Samples: 80
      Throughput: 19.2 samples/s
      Avg batch time: 0.208s
   GPU Actor 1:
      Batches: 20
      Samples: 80  
      Throughput: 18.7 samples/s
      Avg batch time: 0.214s
```

## Troubleshooting

### Common Issues

#### 1. No GPUs Available
```
❌ GPU validation failed: CUDA not available on this system
```

**Solutions:**
- Ensure you're on a GPU node: `nvidia-smi`
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

#### 2. GPU Hardware Issues  
```
❌ GPU validation failed: Need at least 2 healthy GPUs, but only 1 found
```

**Solutions:**
- Use `--max-actors` to limit actor count to available healthy GPUs
- Check GPU health: Look for ECC errors in validation output
- Use `--min-gpus 1` to allow single GPU operation

#### 3. GPU 0 Faulty/Unavailable

If GPU 0 is faulty and you need to use other GPUs (1,2,3,etc.), manually exclude GPU 0:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9 python run_streaming_pipeline.py --enable-profiling --verbose
```

This forces the system to only see GPUs 1-9, making them appear as GPU 0-8 to the pipeline. The system will auto-discover and use all healthy visible GPUs.

#### 4. Ray Connection Issues
```
❌ Ray initialization failed: [connection error]
```

**Solutions:**
- Stop existing Ray clusters: `ray stop`
- Check available ports: Ray uses 6379 by default
- Try restarting: `ray start --head`

#### 5. Actor Creation Failures
```
❌ Failed to create any actors
```

**Solutions:**
- Use `--no-pin-memory` for compatibility issues
- Try `--no-compute` mode to isolate GPU computation issues
- Check `--verbose` output for detailed error messages

#### 6. Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `--batch-size` (default: 4)
- Reduce `--tensor-size` (default: 224)  
- Use `--no-pin-memory` to reduce memory usage

### Debugging Tips

1. **Use `--verbose`** to see detailed execution logs
2. **Start small** with `--batch-size 2 --tensor-size 64` (auto-scale GPUs)
3. **Check GPU health** in the validation phase output
4. **Monitor Ray dashboard** at http://localhost:8265
5. **Save results** with `--output-dir` for post-analysis

### Performance Optimization

1. **Increase batch size** for better GPU utilization
2. **Adjust producer count** to match number of GPUs
3. **Reduce inter-batch delay** for maximum throughput
4. **Use larger tensors** for realistic workloads
5. **Enable real computation** (remove `--no-compute`) for actual ML workloads

## Architecture Details

### GPU Health Validation

The pipeline uses a novel pre-Ray GPU validation approach:

1. **System-wide health check** - Tests all GPUs before Ray starts
2. **CUDA_VISIBLE_DEVICES** - Configures Ray to only see healthy GPUs  
3. **Fail-fast behavior** - Actors die immediately on GPU issues (no CPU fallback)

This is cleaner than per-actor health checks and ensures Ray's round-robin assignment works properly.

### Streaming Data Flow

```
Data Producers → Ray Object Store → GPU Pipeline Actors
     │                 │                     │
Ray Tasks          Distributed            Ray Actors
  (CPU)             Storage               (GPU + CPU)
```

### Multi-GPU Processing

- **Round-robin distribution**: Batches distributed evenly across GPU actors
- **Parallel processing**: Each GPU processes independently  
- **Automatic load balancing**: Ray handles GPU assignment and scheduling

## Advanced Usage

### Custom Model Configuration

To use your own ML model instead of the demo ViT:

1. Modify `ray_pipeline_actor.py` to load your model
2. Adjust tensor shapes to match your model's input requirements
3. Update the no-op mode (`depth=0`) to use your model's inference

### Integration with Real Data Sources

To connect real data sources:

1. Replace `RayDataProducerManager` with your data loading code
2. Ensure data is placed in Ray's object store using `ray.put()`
3. Maintain the same batch structure for GPU actor compatibility

### Scaling to Multiple Nodes

For multi-node Ray clusters:

1. Start Ray head node: `ray start --head --node-ip-address=<head-ip>`
2. Add worker nodes: `ray start --address=<head-ip>:6379`  
3. Run pipeline - it will automatically use all available GPUs across nodes

## File Structure

```
vit-ray-play/
├── run_streaming_pipeline.py     # Main user interface (this tool)
├── gpu_health_validator.py       # Pre-Ray GPU validation
├── ray_data_producer.py          # Streaming data generation  
├── ray_pipeline_actor.py         # GPU pipeline actors
├── vit_pipeline_ray.py           # Pipeline orchestration
└── USAGE.md                      # This documentation
```

## Support

For issues or questions:

1. Check this USAGE.md for common solutions
2. Run with `--verbose` for detailed debugging
3. Check Ray dashboard at http://localhost:8265
4. Review Ray logs for detailed error messages