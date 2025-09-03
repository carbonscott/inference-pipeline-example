# VIT NSys GPU Profiler

NSys profiling wrapper for systematic ViT performance analysis. Creates individual NSys reports for each experiment configuration.

## How it works

- `vit_profiler.py` - Simple wrapper script that handles experiment parsing
- `vit_pipeline.py` - Actual Hydra app that processes all configuration arguments  
- Arguments get passed through from wrapper → NSys → Hydra app

## Basic Usage

```bash
# Single experiment
python vit_profiler.py experiment=vit32x6x384_compiled

# Multiple experiments  
python vit_profiler.py -m experiment=vit32x6x384_compiled,vit16x12x768_compiled

# Pass any vit_pipeline.py arguments
python vit_profiler.py experiment=vit32x6x384_compiled num_samples=1000 warmup_samples=100

# Custom batch sizes and compilation modes work too
python vit_profiler.py experiment=vit32x12x768_compiled batch_size=16
```

## Results

Individual NSys reports saved to:
```
nsys_reports/{experiment_name}/{experiment_name}_{timestamp}.nsys-rep
```

View with NSys GUI or command line:
```bash
nsys stats nsys_reports/vit32x6x384_compiled/*.nsys-rep
```

## Available Experiments

32 pre-configured experiments covering:
- **Model sizes**: vit16x6x384, vit32x12x768, vit32x18x1024, etc.
- **Compilation**: compiled vs uncompiled
- **Optimization modes**: max-autotune, reduce-overhead  
- **Batch sizes**: batch1, batch32 variants

See `conf/experiment/` directory for full list.