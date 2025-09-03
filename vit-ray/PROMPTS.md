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
