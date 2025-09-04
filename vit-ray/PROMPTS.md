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

Please do the following and you will see what I see.

Run `python run_streaming_pipeline.py --max-actors 4 --total-samples 2000
--enable-profiling`.  Then, Launch `watch -n 1 nvidia-sm` and watch it for a
while.  You will see the GPU utilization is 0%.

---

Please read @profiler_v3.png.  You will see cudaStreamSynchronize.  And IO don't
really overlapp with compute.  I am 100% sure the same vit pipeline behaves
differently (H2D/D2H overlaps with compute, and there's no such
cudaStreamSynchronize).  I suspect Ray adds its own stuff, and my pipeline code
doesn't really work as intended.  The profiling result seems to suggest Ray is
launching my pipeline for each new batch of codes, whereas an ideal way is
pipeline processes all batches so overlap can happen.  What's going on?  Please
investigate.

---

Let's review all codes and identify places where things are hard-coded.  I think
it's important for users to be able to specify the input channel size and
spatial size they want.  The vit models should be able to be configured - patch
size, hidden size, number of blocks, number of heads, etc (everything about the
transformer).  Could you analyze the codes again about provide me a list of
concrete things we can do to address these issues.

---

Only consider priority 1 but exclude point 4 - no need to worry about configurable output modes

---

so I can do things like `python run_streaming_pipeline.py --max-actors 4
--total-samples 512000 --enable-profiling --batches-per-producer 128
--batch-size 128 --num-producers 20 --tensor-channels 1 --tensor-size 512`.
However, I do need to overwrite the patch size and channel size for the
underlying transformer, or don't I?

---

Git commit what you just had changed for now.

I see you try to support backward compatibility that uses argparse.  I think
hydra is the way forward and thus let's use hydra without considering backward
compatibility.

---

I don't like the name tensor_channel and tensor_size, let's call it
input_channel, and input_size.  This might be a good name for the data producer
(so maybe don't change it), but doesn't make sense for the consumer which is the
ML pipeline code.  Please identify places where tensor-like naming can be
repalced with input-like naming.
