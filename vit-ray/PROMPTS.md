I want to be able to scale up the current vit pipeline while testing a streaming
data source.  I have used nsys to confirm that the pipeline is an actual double
buffering pipeline that runs on a single accelerator (gpu in this case).

The scaling direction is mostly accelerator parallel, basically, to map the same
operation on multiple accelerators (gpus).  In JAX, I understand it's a matter
of pmap or shard_map.  Now, we are working with PyTorch codes, obviusly as you
can see in this codebase.  I think we can scale it in two directions: (1) Using
Ray; (2) Using MPI.

Currently, our streaming data source is not fully ready yet.  And, you probably
need to simulate data being streamed into an accelerator equiped host device
using random tensors generated in the host memory.  Once the streaming data
source is ready, I believe a simpler option is just to use MPI since this is
a very simple mechanism to manage multiple devices.  

However, as of today, I have a one-day time budget to test it with Ray.  Why
Ray?  I think it's mainly because it's very easy to use its object store to
emulate the streaming (put and get methods).  Secondly, I am just personally
interested in how much differencely Ray and MPI can perform on our own HPC
system.  I think it's relatively trivial to adapt the current codes to MPI, but
I'm not so sure about Ray.  You need to think about actors and tasks, right?
What's more challenging is?  How do I profile it?  In MPI, I can just profile
each rank with nsys directly on the command line.  Ray is a task scheduling
system itself, and where I should put nsys?  Or, I guess it's very likely Ray
offers an entry point or APIs for users to interface with nsys.  To be clear,
I'm interested in the profiling the inference pipeline instead of what the Ray
Driver is doing (I guess scheduling Ray tasks).

Let's create a git worktree to play with this idea (vit-ray-play, wait, should
we create it in its parent directory?  let me know).


Now, after we have the worktree, let's talk about what we want to play with to
understand how to integrate the current pipeline into Ray.

- **data producer**: could we design the data producer actor/task (which one?)
  such that it generates data in memory and then puts generated data into Ray's
  object store.  Users can specify how many Ray actors/tasks they need in this
  emulated streaming envirnoment.  Also, I notice that current vit_pipeline.py
  file generates all tensors for the test upfront.  I guess we can do something
  like this, since this is a Ray task, users can specify how many of such tasks
  need to be executed, correct?
- **pre-processing**: no need to consider preprocessing for now.
- **pipeline as a consumer**: pipeline sounds like Ray actor thing because each
  accelerator (GPU) should only have one pipeline running on it.  Also, once the
  pipeline is running, its job is to keep processing data that is transferred to
  the device (accelerator)'s memory and process it.  If we use Ray task, I'm
  afraid Ray will reschedule it every time the processing is done, and it's a
  waste of time (adding latency of initialize the model in the pipeline, and
  what's worse, it kills the purpose to have a pipeline).  I think the tricky
  thing is how to use nsys to profile this.  I hope to retain all the nsys
  related lines of codes in the current vit_pipeline.py such that I can
  hopefully reproduce the profiling of these pipelines execet it's running in
  Ray now.
- **post-processing**: let's forget about post-processing right now.  Did
  vit_pipeline.py just do D2H and then forget about it (so that pipeline still
  runs).  Let's do the same essentially for now.


I understand I said a lot.  Please develop a ROADMAP.md file that focuses on
what we should focus on today and what we can improve later.


---


Please read the ROADMAP.md and understand what we have done.  I believe the new
codes ray_data_producer.py, ray_pipeline_actor.py, vit_pipeline_ray.py (I might
have missed a few others) have not been tested.  Could you come up with a
testing plan (including profiling) and write it in TEST_PLAN.md?  Btw, also add
one detail to the plan that Ray will output nsys's profiling results nsys-rep
files under its logging directory, which in our case is very like `$TMPDIR`.  So
you know where to find there files.  Also, please add a detail that nsys
supports turning a nsys-rep file into a sqlite file so that you can actually
look into it without the GUI to understand if GPU activities have actually been
recorded.


---

Okay, you have built all the test scripts.  Please test them out.  I am
concerned we are a bit ahead of ourselves in terms of testing.  We haven't
actually manually run any of the script, so testing them directly with unit
tests might be pre-mature.  Nonetheless, go ahead with the tests you have right
now, and plan to pivot to manual testing if things get stuck.

---

Let's adapt one change.  Either look up Ray's docs directory or online and
understand how many ways Ray can be launched.  I remember there're two ways to
launch Ray.  One a single node, what you have right now is fine.  However, on a
multi node system, I remember you need to launch a Ray head node (I forgot the
command, but if you find one, please add it to RAY_NOTES.md).

---

I prefer to launch ray head node before running any ray applications.  Do you
think all the codes now work with the Ray head node launching method?  Please
adjust the MANUAL_TEST_GUIDE.md assuming we are using the head node approach.


---

I know that the GPU infrastructure we have right now is very prone to
uncorrectable ECC errors (basically, this GPU is not useful).  There are
multiple GPUs in one node.  I think it's a good idea to have a quick check if
the GPU has such an error before even initializing any models or the pipeline on
this GPU.  We can skip it.  Ray should know to skip it.  It's possible that a
few lines of code like below can function as a test and you just have to catch
the error, and let Ray act accordingly.  

```
device = torch.device(f'cuda:{i}')
torch.cuda.set_device(device)
```

---

The MANUAL_TEST_GUIDE.md is great.  Now, could we actually run the data producer
continuously (since the test says it's working) and then run the pipeline to get
and process these data (I assume we have passed a focused test) to see if these
two things can work nicely together under Ray.  Could you try it now and
identify if there is any gap between where we are and where we are going?

---

okay, for testing, could you try to run multiple GPUs like at least 2, and if
there's GPU failure upfront, then this actor should not continue to live
(falling back to CPU is NOT an option)

---

I am a bit losing track of what you have done.  You have created too many python
scripts.  Which ones are truly useful and which ones have been obsolete?  I
wanted to understand where we are now in terms of integrating the data source
and the pipeline.

---

I don't quite understand the GPU assignment problem.  If you use a round robin
fashion, it won't always assign GPU 0 (which is the one current experiencing ECC
errors) to our pipeline actor.

---


Maybe a cleaner design is that before assigning GPUs to any actor, Ray will do a health
check upfront and rule out faulty GPUs before even getting into initializing any actor.

---

You could run `python run_streaming_pipeline.py --num-gpus 2 --enable-profiling
--profiling-output-dir ./profiles --verbose` even now, but it will fail because
of one faulty GPU.  I believe you actually removed the GPU health check codes.
Maybe a cleaner design is that before assigning GPUs to any actor, Ray will
do a health check upfront and rule out faulty GPUs before even getting into
initializing any pipeline actor.

---

I don't like the idea of reinitializing Ray.  I ran it in multi-node scenario
too, start a Ray head node should be a normal operation.  GPUs can fail in the
middle too.  I think a proper way to handle this is that the GPU check needs to
run every time when run_streaming_pipeline.py is launched.  But I'm not sure if
there is a good way to filter out bad GPUs at this stage.  Please investigate
the Ray doc and online.

---

It sounds like it's done at the node level, which is a good direction, but what
about encountering an actual faulty GPU on one node?  Do we label the entire
node as faulty node?  Or only that GPU will be excluded from being assigned to
tasks/actors?

---

make sure you document actually running `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9
python run_streaming_pipeline.py --num-gpus 2 --enable-profiling --verbose` if
GPU 0 is faulty into USAGE.md

Then, let's discuss what should go to the main worktree.  I am a bit concerned
that the way this work is created is a bit different than I have thought.  You
included basically created a worktree for the entire
`/sdf/data/lcls/ds/prj/prjcwang31/results/codes/inference-pipeline-examples/`
instead of just vit-ray, is it right?  so be careful about what we should merge
into vit-ray.

---

Firstly, could we add one option to set the total number of samples for the pipeline
tests in run_streaming_pipeline.py?  I want to make the sample size larger to
observe the GPU activities.  Actually, I feel it might not make senese for Ray
to only use 4 GPUs, because Ray is a task based scheduling system.  Argparse
like `--num-gpus 4` doesn't make sense.  How should I think about using Ray to
scale application like vit_pipeline.py at all?

Secondly, when I launch nvidia-smi while running `python
run_streaming_pipeline.py --num-gpus 4 --num-producers 8 --enable-profiling`.  I
see all 10 gpus have some GPU memory loaded, but the GPU utilization is 0%.
Something is wrong.

Finally, it did output nsys-rep files in $TMPDIR/ray/session_latest/logs/nsight/, and I
checked these files in the nvidia nisght viewer.  I don't see any GPU activities.
There's no "CUDA HW" timeline.  It seems to match what has been observed above.
The screenshot in profiler_v1 shows you somewhere near the end of the entire
timeline.  No GPU activities at all.

If it's a Ray problem, please look up the Ray doc to understand what's the Ray
way of doing things.

---

I am running `python run_streaming_pipeline.py --max-actors 4 --total-samples
100000 --enable-profiling`.  In profiler_v2.png, as you can see, the GPU
utilization is 0% for all GPUs.  The first 4 GPUs have higher GPU memory, which
I assume came from the pipeline actor.  I suspect actor doesn't really process
any data because the 0% GPU utilization.  Please investigate.

---

Do you agree with <text>Yes, absolutely. This is an excellent and non-trivial use of Ray that shows a solid understanding of the core concepts for scaling. The overall architecture is correct:

1.  **Actor Pool:** You create a pool of actors (`VitPipelineActor`), one for each GPU you intend to use.
2.  **Resource Declaration:** You correctly use `@ray.remote(num_gpus=1)` to tell Ray that each actor requires one dedicated GPU.
3.  **State Encapsulation:** Each actor correctly loads its own model and manages its own state (`DoubleBufferedPipeline`) in the `__init__` method, which is the right pattern.
4.  **Data Flow:** You use `ray.put` (implicitly via the data producers) to create object references and pass those to the actors, which is the right way to move data efficiently.

However, your question is "does it use Ray *correctly*?", and while the architecture is right, the implementation reveals a very common pattern: **you are "fighting the framework" in a few key places.** You've written a lot of code to manually do things that Ray is designed to handle for you automatically.

Let's break down the positives and then look at how we can simplify this code to be much more "Ray-native."

### What's Done Well

*   **Correct Parallelism Model:** Creating an actor per GPU is the canonical way to do distributed model inference in Ray.
*   **Encapsulating Work:** The `VitPipelineActorBase` class is a great example of encapsulating the model, pipeline, and stats for a single worker.
*   **Profiling-Aware:** Separating actors with and without the `nsight` runtime environment is a clever and useful pattern.
*   **Data Producers:** Using separate Ray tasks (`RayDataProducerManager`) to generate data in parallel with model execution is a good design.

### Areas for Major Simplification (The "Fighting the Framework" parts)

Your code can be made much simpler, more robust, and easier to maintain by letting Ray do more of the work.

#### 1. The Biggest Issue: Manual GPU ID Management

The most significant area for improvement is the GPU assignment logic inside your actor's `__init__` method. This entire complex block is unnecessary and works against Ray's core design.

```python
# ray_pipeline_actor.py -> in VitPipelineActorBase.__init__
# THIS IS AN ANTI-PATTERN IN RAY

# ... you have a giant block of code to find the gpu_id ...
if gpu_id is not None:
    # ...
else:
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        runtime_context = ray.get_runtime_context()
        # ... logic with get_accelerator_ids ...
    except Exception as e:
        # ... fallback to get_gpu_ids ...

# ... and then you use self.gpu_id everywhere ...
torch.cuda.set_device(self.gpu_id)
test_a = torch.randn(256, 256, device=f'cuda:{self.gpu_id}')
self.vit_model = create_vit_model(..., gpu_id=self.gpu_id)
```

**Why this is a problem:**

When you declare `@ray.remote(num_gpus=1)`, Ray's scheduler finds a machine with a free GPU. On that machine, it starts the actor's worker process and **automatically sets the `CUDA_VISIBLE_DEVICES` environment variable** to the specific ID of the assigned GPU.

This means that from within your actor's process, **PyTorch only sees one GPU, and its ID is always `0`**.

You don't need to find the "real" physical GPU ID. You just need to use `cuda:0`. Ray handles the complex mapping from `0` to the physical `3`, `5`, etc., for you.

#### 2. Over-Complication: External GPU Health Checks

In `vit_pipeline_ray.py`, you have a `setup_ray` function that calls `get_healthy_gpus_for_ray` *before* `ray.init()`. This manually sets `CUDA_VISIBLE_DEVICES` for the driver script, hoping to constrain which GPUs Ray sees.

This is also fighting the framework. The health of the nodes in the Ray cluster is a system-level concern, not an application-level one. If a GPU on a node is unhealthy, that Ray node should be configured to not offer that GPU to the cluster in the first place. Your application code should be ableto trust that any `GPU` resource reported by `ray.cluster_resources()` is usable.

### A Refactored, More "Ray-Native" Actor

Let's refactor `VitPipelineActorBase` to show how simple it can be when you trust the framework. I'll combine the `DoubleBufferedPipeline` logic directly into the actor for clarity, as the actor itself is the perfect container for this state.

:::artifact{identifier="simplified-ray-actor" type="text/x-python" title="Simplified Ray GPU Actor"}
```python
import ray
import torch
import torch.cuda.nvtx as nvtx
import time
import os
from typing import List, Dict, Any, Tuple

# Assume these functions are available from your other files
from vit_pipeline import create_vit_model

# A simplified DoubleBufferedPipeline logic integrated directly into the Actor
class InternalPipeline:
    def __init__(self, model, batch_size, input_shape, output_shape, pin_memory):
        self.model = model
        self.is_noop = model is None
        self.device = "cuda:0" # Always cuda:0 inside a Ray GPU actor

        # Buffers are created directly on the target device
        self.gpu_input_buffers = {
            'A': torch.zeros(batch_size, *input_shape, device=self.device),
            'B': torch.zeros(batch_size, *input_shape, device=self.device)
        }
        self.gpu_output_buffers = {
            'A': torch.zeros(batch_size, *output_shape, device=self.device),
            'B': torch.zeros(batch_size, *output_shape, device=self.device)
        }
        # The CPU output buffer can be created on-demand or pre-allocated
        # For simplicity, we'll create it during D2H transfer
        
        # Streams
        self.h2d_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        
        self.current_buffer = 'A'
        self.is_first_batch = True

    def swap(self):
        self.current_buffer = 'B' if self.current_buffer == 'A' else 'A'

    def process(self, cpu_batch: List[torch.Tensor]) -> torch.Tensor:
        if not self.is_first_batch:
            self.swap()
        self.is_first_batch = False

        batch_size = len(cpu_batch)
        
        # 1. H2D Transfer
        with torch.cuda.stream(self.h2d_stream):
            # Wait for the *previous* compute on the other buffer to finish
            # before overwriting this buffer's input
            self.compute_stream.wait_stream(self.h2d_stream)
            
            input_tensor = torch.stack(cpu_batch).to(self.device, non_blocking=True)
            self.gpu_input_buffers[self.current_buffer][:batch_size].copy_(input_tensor, non_blocking=True)

        # 2. Compute
        with torch.cuda.stream(self.compute_stream):
            # Wait for the current H2D to finish
            self.h2d_stream.wait_stream(self.compute_stream)
            
            input_data = self.gpu_input_buffers[self.current_buffer][:batch_size]
            
            if self.is_noop:
                output_data = input_data
            else:
                with torch.no_grad():
                    output_data = self.model(input_data)
            
            self.gpu_output_buffers[self.current_buffer][:batch_size].copy_(output_data, non_blocking=True)

        # 3. Synchronize and return result
        # The D2H copy will implicitly sync the compute stream
        result_cpu = self.gpu_output_buffers[self.current_buffer][:batch_size].cpu()
        torch.cuda.synchronize() # Ensure all work is done for timing
        return result_cpu


@ray.remote(num_gpus=1)
class SimplifiedVitActor:
    """
    A simplified, more Ray-native version of the pipeline actor.
    """
    def __init__(
        self,
        tensor_shape: Tuple[int, int, int] = (3, 224, 224),
        batch_size: int = 10,
        pin_memory: bool = True,
        # ... other model params (depth, heads, etc.)
        **model_kwargs
    ):
        # Ray automatically sets CUDA_VISIBLE_DEVICES.
        # The assigned GPU is always 'cuda:0' from the actor's perspective.
        self.device = "cuda:0"
        
        print(f"Actor PID {os.getpid()} Initializing on {self.device}...")
        print(f"CUDA_VISIBLE_DEVICES set by Ray: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        # 1. Create Model (passing 'gpu_id' is no longer needed)
        self.vit_model, image_size = create_vit_model(
            tensor_shape=tensor_shape, 
            gpu_id=0, # Always 0 now
            **model_kwargs
        )
        # The model is already moved to the correct device by create_vit_model

        # 2. Calculate shapes and create internal pipeline
        # (This replaces your separate DoubleBufferedPipeline class)
        input_shape = tensor_shape
        if self.vit_model is None:
            output_shape = tensor_shape
        else:
            patch_size = model_kwargs.get('patch_size', 32)
            dim = model_kwargs.get('dim', 512)
            num_patches = (image_size // patch_size) ** 2
            output_shape = (num_patches + 1, dim)

        self.pipeline = InternalPipeline(
            model=self.vit_model,
            batch_size=batch_size,
            input_shape=input_shape,
            output_shape=output_shape,
            pin_memory=pin_memory
        )
        
        # 3. Initialize statistics
        self.stats = {'samples_processed': 0, 'total_time': 0.0}
        print("Actor initialization complete.")

    def process_batch(self, batch_object_refs: List[ray.ObjectRef]) -> Dict[str, Any]:
        """Processes a single batch of data from object references."""
        start_time = time.time()
        
        # 1. Get data from the object store
        cpu_tensors = ray.get(batch_object_refs)
        actual_batch_size = len(cpu_tensors)
        
        # 2. Process through the internal pipeline
        # The pipeline handles H2D, compute, and D2H internally
        output_tensor = self.pipeline.process(cpu_tensors)
        
        processing_time = time.time() - start_time
        
        # 3. Update stats
        self.stats['samples_processed'] += actual_batch_size
        self.stats['total_time'] += processing_time
        
        return {
            'batch_size': actual_batch_size,
            'processing_time': processing_time,
            'output_shape': tuple(output_tensor.shape)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return current processing statistics."""
        return self.stats
```
:::

### Summary of the "Ray Mindset" Shift

1.  **Trust the Scheduler:** Don't write code to manage resources (`gpu_id`). Declare your resource needs (`num_gpus=1`) and let Ray's scheduler handle the "where" and "which".
2.  **Embrace Isolation:** Assume your actor has exclusive access to the resources it requested. For a GPU actor, this means `cuda:0` is *your* GPU, and you don't need to worry about other actors.
3.  **Simplify Application Logic:** Remove system-level concerns like health checks from your application logic. Your app should be able to run on any cluster that meets its resource requirements, and it's the cluster's job to provide healthy resources.
4.  **The Actor *is* the State:** The Actor class is the natural boundary for state like models, pipelines, and streams. You often don't need another layer of class abstraction inside the actor unless the logic is exceptionally complex.

Your original code is a fantastic, working piece of engineering. By applying
these simplifications, you can make it more robust, much easier to read, and
more aligned with Ray's design philosophy, which will pay dividends as your
projects grow in complexity.</text>
