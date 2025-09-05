# PeakNet Ray Streaming Pipeline

Simple Ray-based streaming pipeline for PeakNet segmentation model inference at scale.

## Quick Start

1. **Start Ray head node:**
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9 ray start --head --block
```

2. **Run PeakNet pipeline:**
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9 python run_peaknet_streaming_pipeline.py experiment=peaknet_test_run streaming.runtime.total_samples=10000
```

3. **Override max_actors (e.g., use all 9 GPUs):**
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9 python run_peaknet_streaming_pipeline.py experiment=peaknet_test_run streaming.runtime.max_actors=9 streaming.runtime.total_samples=10000
```

The pipeline processes 1920x1920 images with PeakNet segmentation model using double-buffered GPU inference across multiple actors.