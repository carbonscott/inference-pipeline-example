## Project Overview

- **Goal**: Run ML model (ViT or peaknet model to be more specific) inference at scale with streaming data source
- **Approach**: Pipelining
  - Stage 1: input data are streamed into Ray's object store
  - Stage 2: compute with double buffering based on the input data in Ray's object store
  - Stage 3: output data are streamed into Ray's object store
  - Stage 4: post processing based on the processed data in Ray's object store

### Past works

- **peaknet**
  - Path: /sdf/home/c/cwang31/codes/peaknet
  - The peaknet library
- **exp-peaknet**
  - Path: /sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet
  - train_convnext_seg.py is what I used for training.  It has information like
    how I like to handle auto cast for mixed precision.
- **crystfel_stream_parser**
- **crystfel_stream_parser**
  - Path: /sdf/home/c/cwang31/codes/crystfel_stream_parser
  - Stream file parser libray, and it's only needed for the post-processing
    stage.
- **peaknet-pipeline**
  - Path: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline`
  - The very first attempt of doing this.  However, it's a mix of mpi and ray,
    which is unnecessarily complicated.  It also doesn't have double buffering
    in the compute pipeline itself.
- **peaknet-pipeline-ray**
  - Path: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray`
  - This repo has a lot of issues, but it at least shows how nsys can work
    nicely with Ray.
- **inference-pipeline-examples**
  - Path: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/inference-pipeline-examples`
  - An example of compute pipeline with double buffering.

### Directories

- Experiment directory: `/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet-pipeline`
- Code development directory: 
  - `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/inference-pipeline-examples`

### Handling permission

Ask me now to add all relevant directories to working directories.

### Ray Documentation

Path: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/ray/doc/`

Ray APIs can be confusing.  Please look up the documentation before making your
decisions.
