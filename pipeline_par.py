# Copyright (c) Meta Platforms, Inc. and affiliates
import os
import pippy
from torch.distributed import rpc
from torch import nn
import torch
from typing import Any
import time
import numpy as np
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768*4*4, 768*4*4)
        self.fc2 = nn.Linear(768*4*4, 768*4*4)
        self.fc3 = nn.Linear(768*4*4, 768*4*4)
        # self.fc4 = nn.Linear(768, 128)
        
    def forward(self, x):
        # for i in range(300):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
            # x = self.fc4(x)
        return x

net = Net()
net.eval()


from pippy.IR import Pipe

pipe = Pipe.from_tracing(net)
print(pipe)
print(pipe.split_gm.submod_0)


from pippy.IR import annotate_split_points, PipeSplitWrapper

annotate_split_points(
    net,
    {
        "layer0": PipeSplitWrapper.SplitPoint.END,
        "layer1": PipeSplitWrapper.SplitPoint.END,
    },
)

pipe = Pipe.from_tracing(mn)
print(" pipe ".center(80, "*"))
print(pipe)
print(" submod0 ".center(80, "*"))
print(pipe.split_gm.submod_0)
print(" submod1 ".center(80, "*"))
print(pipe.split_gm.submod_1)
print(" submod2 ".center(80, "*"))
print(pipe.split_gm.submod_2)


# To run a distributed training job, we must launch the script in multiple
# different processes. We are using `torchrun` to do so in this example.
# `torchrun` defines two environment variables: `LOCAL_RANK` and `WORLD_SIZE`,
# which represent the index of this process within the set of processes and
# the total number of processes, respectively.
#
# To learn more about `torchrun`, see
# https://pytorch.org/docs/stable/elastic/run.html
import os

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# PiPPy uses the PyTorch RPC interface. To use RPC, we must call `init_rpc`
# and inform the RPC framework of this process's rank and the total world
# size. We can directly pass values `torchrun` provided.`
#
# To learn more about the PyTorch RPC framework, see
# https://pytorch.org/docs/stable/rpc.html
import torch.distributed.rpc as rpc

rpc.init_rpc(f"worker{local_rank}", rank=local_rank, world_size=world_size, rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
         num_worker_threads=16,
         rpc_timeout=2000 # 2000 second timeout
    ))

# PiPPy relies on the concept of a "driver" process. The driver process
# should be a single process within the RPC group that instantiates the
# PipelineDriver and issues commands on that object. The other processes
# in the RPC group will receive commands from this process and execute
# the pipeline stages
if local_rank == 0:
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
    driver = PipelineDriverFillDrain(
        pipe,
        64,
        world_size=world_size,
        args_chunk_spec=args_chunk_spec,
        kwargs_chunk_spec=kwargs_chunk_spec,
        output_chunk_spec=output_chunk_spec,
    )

    x = torch.randn(512, 512)

    # Run the pipeline with input `x`. Divide the batch into 64 micro-batches
    # and run them in parallel on the pipeline
    output = driver(x)
    timings = []
    num_runs = 100
    
    
    
    with torch.no_grad():
        for i in range(1, num_runs+1):
            start_time = time.perf_counter()
            reference_output = net(x)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            if i%(num_runs/5)==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, num_runs, np.mean(timings)*1000))
    nonpipe_time = ((np.mean(timings))*1000)
    print('Latency per query without pipeline: %.2f ms'%nonpipe_time)


    timings = []
    
    
    with torch.no_grad():
        for i in range(1, num_runs+1):
            start_time = time.perf_counter()
            output = driver(x)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            if i%(num_runs/5)==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, num_runs, np.mean(timings)*1000))
    pipe_time = ((np.mean(timings))*1000)
    print('Latency per query: %.2f ms'%pipe_time)
    
    # Run the original code and get the output for comparison
    reference_output = net(x)

    # Compare numerics of pipeline and original model
    torch.testing.assert_close(output, reference_output)

    print('Communication time: ', str(pipe_time - nonpipe_time))

    print('Speed up: ', str(nonpipe_time / pipe_time))

    print(" Pipeline parallel model ran successfully! ".center(80, "*"))


rpc.shutdown()