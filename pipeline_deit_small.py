# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from typing import Any
import time
import numpy as np
from collections import defaultdict
from transformers import AutoImageProcessor, DeiTForImageClassification
from pippy.IR import Pipe
import pippy
import pippy.fx
from pippy import run_pippy
from pippy.hf import PiPPyHFTracer, inject_pipeline_forward

from PIL import Image
import requests
from accelerate import Accelerator


MODEL_NAME = "deit_small_distilled_patch16_224"
mn = DeiTForImageClassification.from_pretrained('facebook/deit-small-distilled-patch16-224')

# To run a distributed training job, we must launch the script in multiple
# different processes. We are using `torchrun` to do so in this example.
# `torchrun` defines two environment variables: `LOCAL_RANK` and `WORLD_SIZE`,
# which represent the index of this process within the set of processes and
# the total number of processes, respectively.
#
# To learn more about `torchrun`, see
# https://pytorch.org/docs/stable/elastic/run.html
import os
os.environ["LOCAL_RANK"]='0'
os.environ["WORLD_SIZE"]='4'
os.environ["MASTER_ADDR"]='192.168.1.100'
os.environ["MASTER_PORT"]='50000'
os.environ["GLOO_SOCKET_IFNAME"]='eth0'
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# PiPPy uses the PyTorch RPC interface. To use RPC, we must call `init_rpc`
# and inform the RPC framework of this process's rank and the total world
# size. We can directly pass values `torchrun` provided.`
#
# To learn more about the PyTorch RPC framework, see
# https://pytorch.org/docs/stable/rpc.html
import torch.distributed.rpc as rpc
import torch.distributed as dist
#dist.init_process_group(backend='gloo', init_method='tcp://192.168.1.100:50000', rank=local_rank, world_size=4)
rpc.init_rpc(f"worker{local_rank}", rank=local_rank, world_size=world_size, rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
         num_worker_threads=4,
         rpc_timeout=2000, # 2000 second timeout
	#init_method=f"tcp://192.168.1.100:50000",
    ))

# PiPPy relies on the concept of a "driver" process. The driver process
# should be a single process within the RPC group that instantiates the
# PipelineDriver and issues commands on that object. The other processes
# in the RPC group will receive commands from this process and execute
# the pipeline stages
print("**************** My Rank: %d ****************", (local_rank,))
if local_rank == 0:
    print("RANK=0")
    # We are going to use the PipelineDriverFillDrain class. This class
    # provides an interface for executing the `Pipe` in a style similar
    # to the GPipe fill-drain schedule. To learn more about GPipe and
    # the fill-drain schedule, see https://arxiv.org/abs/1811.06965
    from pippy.PipelineDriver import PipelineDriverFillDrain
    from pippy.microbatch import TensorChunkSpec

    # Pipelining relies on _micro-batching_--that is--the process of
    # dividing the program's input data into smaller chunks and
    # feeding those chunks through the pipeline sequentially. Doing
    # this requires that the data and operations be _separable_, i.e.
    # there should be at least one dimension along which data can be
    # split such that the program does not have interactions across
    # this dimension. PiPPy provides `chunk_spec` arguments for this
    # purpose, to specify the batch dimension for tensors in each of
    # the args, kwargs, and outputs. The structure of the `chunk_spec`s
    # should mirror that of the data type. Here, the program has a
    # single tensor input and single tensor output, so we specify
    # a single `TensorChunkSpec` instance indicating dimension 0
    # for args[0] and the output value.
    args_chunk_spec: Any = (TensorChunkSpec(0),)
    kwargs_chunk_spec: Any = {}
    output_chunk_spec: Any = TensorChunkSpec(0)

    # Finally, we instantiate the PipelineDriver. We pass in the pipe,
    # chunk specs, and world size, and the constructor will distribute
    # our code to the processes in the RPC group. `driver` is an object
    # we can invoke to run the pipeline.
    num_ranks = world_size
    split_policy = pippy.split_into_equal_size(num_ranks)
    bs = 1 * num_ranks
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    inputs = image_processor(images=image, return_tensors="pt")
    input_dict = {
        'pixel_values': inputs,
    }
    concrete_args = pippy.create_default_args(
        mn,
        except_keys=input_dict.keys(),
    )

    driver, stage_mod = pippy.all_compile(
        mn,
        num_ranks,
        64,
        split_policy=split_policy,
        tracer=PiPPyHFTracer(),
        concrete_args=concrete_args,
        index_filename=None,
        checkpoint_prefix=None,
    )

    #x = torch.randn(512, 512)
    x = inputs
    num_runs = 100
    timings = []
    with torch.no_grad():
        for i in range(1, num_runs+1):
            start_time = time.perf_counter()
            reference_output = mn(x)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            if i%(num_runs/5)==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, num_runs, np.mean(timings)*1000))

    print('Latency per query without pipeline: %.2f ms'%((np.mean(timings))*1000))
    
    
    output = driver(x)
    timings = []
    
    with torch.no_grad():
        for i in range(1, num_runs+1):
            start_time = time.perf_counter()
            output = driver(x)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            if i%(num_runs/5)==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, num_runs, np.mean(timings)*1000))

    print('Latency per query: %.2f ms'%((np.mean(timings))*1000))
    

    # Run the original code and get the output for comparison
    reference_output = mn(x)

    # Compare numerics of pipeline and original model
    torch.testing.assert_close(output, reference_output)

    print(" Pipeline parallel model ran successfully! ".center(80, "*"))


rpc.shutdown()
