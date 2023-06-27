# Model-Parallelism
modified config.yaml:
```
machine_rank = <MACHINE_NUM> (either 0 or 1)
main_process_ip = <IP_OF_MACHINE_0>
```
and run the following command in two machines.
```
accelerate launch --config_file config.yaml resnet18.py
```
