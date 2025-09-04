# 1. Start Ray head node (one time)
ray start --head --port=6379 --num-cpus=8 --num-gpus=2

# 2. Run any tests - they'll automatically connect
python -c "import ray_data_producer; ray_data_producer.test_data_producer()"

# 3. When done
ray stop




# Job commands

python run_streaming_pipeline.py --max-actors 4 --total-samples 512000 --enable-profiling --batches-per-producer 128 --batch-size 128 --num-producers 20
