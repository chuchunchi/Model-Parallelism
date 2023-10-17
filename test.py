# Copyright (c) Meta Platforms, Inc. and affiliates
import torch, time, pippy, requests, os
from typing import Any
import numpy as np
from transformers import AutoImageProcessor, DeiTForImageClassification
import pippy.fx
from pippy.hf import PiPPyHFTracer
from PIL import Image
import torch.distributed.rpc as rpc
from pippy.microbatch import TensorChunkSpec
from pippy.PipelineDriver import PipelineDriverFillDrain

MODEL_NAME = "deit_small_distilled_patch16_224"
mn = DeiTForImageClassification.from_pretrained('facebook/deit-small-distilled-patch16-224')

os.environ["LOCAL_RANK"]='0'
os.environ["WORLD_SIZE"]='4'
os.environ["MASTER_ADDR"]='192.168.1.100'
os.environ["MASTER_PORT"]='50000'
os.environ["GLOO_SOCKET_IFNAME"]='eth0'
os.environ["TP_SOCKET_IFNAME"] = "eth0"
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Define device mappings
device_maps = {
    "worker0": {0: "cpu"},
    "worker1": {0: "cpu"},
    "worker2": {0: "cpu"},
    "worker3": {0: "cpu"}
}

# Define local devices for the RPC agent
devices = ["cpu", "cpu", "cpu", "cpu"]

# Set device mappings from this worker to other RPC callees
options = rpc.TensorPipeRpcBackendOptions(
    num_worker_threads=4,
    # rpc_timeout=2000, # 2000 second timeout
    #init_method=f"tcp://192.168.1.100:50000",
    # device_maps=device_maps,
    # devices=devices
)

rpc.init_rpc(f"worker{local_rank}", rank=local_rank, world_size=world_size, backend=rpc.BackendType.TENSORPIPE, rpc_backend_options=options)

print(f"**************** My Rank: {local_rank} ****************")

'''
split_policy = pippy.split_into_equal_size(world_size)
bs = 1 * world_size
driver, stage_mod = pippy.all_compile(
        mn,
        num_ranks=world_size,
        num_chunks=world_size,
        schedule="FillDrain",
        split_policy=split_policy,
    )
'''

if local_rank == 0:
    print("RANK=0")

    #args_chunk_spec: Any = (TensorChunkSpec(0),)
    #kwargs_chunk_spec: Any = {}
    #output_chunk_spec: Any = TensorChunkSpec(0)
    split_policy = pippy.split_into_equal_size(world_size)
    
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
        num_ranks=world_size,
        num_chunks=world_size,
        split_policy=split_policy,
        tracer=PiPPyHFTracer(),
        concrete_args=concrete_args,
        index_filename=None,
        checkpoint_prefix=None,
    )
    
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
