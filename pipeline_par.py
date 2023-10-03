import os
import pippy
from torch.distributed import rpc
from torch import nn
import torch
import time

RANK  = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD"])
HOST  = os.environ["HOST"]
PORT  = os.environ["PORT"]
NUM_THREADS = 4
NUM_INFERENCE = 100
torch.set_num_threads(4)

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

print(f"My rank is {RANK}")


# first thing to do is to init RCP
print("Waiting for all the nodes...")
rpc.init_rpc(
    f"worker{RANK}", # just an identifier
    rank=RANK,
    world_size=WORLD,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=4,
        rpc_timeout=500, # seconds
        init_method=f"tcp://{HOST}:{PORT}", # head node's address and port
    )
)

# split the model, each process materializes its pipeline stage
driver, stage = pippy.all_compile(
    net,
    num_ranks=WORLD,
    num_chunks=WORLD, # microbatching
    schedule="FillDrain", # feed chunks through the pipeline sequentially
    split_policy=pippy.split_into_equal_size(WORLD), # split the model into specified number of equal-size stages
)

print(stage)
print(f"Measuring {WORLD}-stage Pipelined elapsed time...")
print(f"num of threads: {torch.get_num_threads()} ")

if RANK == 0:
    x = torch.randn(4, 768*4*4)
    start_time = time.perf_counter()
    for i in range(NUM_INFERENCE):
        y = driver(x) # only rank 0 is able the call the pipeline's driver
    end_time = time.perf_counter()
    pipeline_time = end_time - start_time

del net
rpc.shutdown()
print("rpc Bye!")

print(f"Measuring Non-Pipelined elapsed time...")
print(f"num of threads: {torch.get_num_threads()} ")
net = Net()
x = torch.randn(4, 768*4*4)
start_time = time.perf_counter()
for i in range(NUM_INFERENCE):
    y = net(x)
end_time = time.perf_counter()
non_pipelined_time = end_time - start_time
print(f"\n\n**** Result ****")
print(f"{WORLD}-stage Pipelined Elapsed time = {pipeline_time} seconds")
print(f"      Non-Pipelined Elapsed time = {non_pipelined_time} seconds")

print(f"Speedup = {non_pipelined_time / pipeline_time}")
print(f"No-pipelined latency = {non_pipelined_time / NUM_INFERENCE} seconds")
print(f"Communication time = {(pipeline_time - non_pipelined_time) / NUM_INFERENCE } seconds")