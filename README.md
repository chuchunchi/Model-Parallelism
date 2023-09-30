# Model-Parallelism
modified config.yaml:
```
machine_rank = <MACHINE_NUM> (either 0 or 1)
main_process_ip = <IP_OF_MACHINE_0>
```
and run the following command in two machines.
```
export RANK=0
export WORLD_SIZE=2
export GLOO_SOCKET_IFNAME=eth0
accelerate launch resnet18.py
```
