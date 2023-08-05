import torch
import pandas as pd
import torch.multiprocessing as mp
import accelerate
from collections import defaultdict
from transformers import DeiTForImageClassificationWithTeacher
from torch.utils import benchmark


def walltime(stmt, arg_dict, duration=10):
    t = benchmark.Timer(stmt=stmt, globals=arg_dict).blocked_autorange(min_run_time=duration).median
    print(t)
    return t

def layer_benchmark(layer, hidden_size, seq_lens, batch_sizes, cross_attention=False, device='cuda'):
    h = hidden_size
    results = defaultdict(lambda: {})    
    encoder_state = 'encoder_hidden_states=X' if cross_attention else ''
    
    layer = accelerate.DistributedDataParallel(layer)  # 使用 accelerate.DistributedDataParallel 進行平行化

    for s in seq_lens:
        for b in batch_sizes:            
            ffn = 16 * b * s * h * h / 1e9  # GFLOPS for the Feed-Forward Network
            atten = (4 * b * h * s * s + 8 * b * s * h * h) / 1e9  # GFLOPS for attention            
            forward = ffn + (2 if cross_attention else 1) * atten
            
            X = torch.randn(b, s, 224, 224).to(device)
            results[f'batch={b}'][f'fwd seq_len={s}'] = forward / walltime(
                f'layer(X, {encoder_state})', arg_dict={'layer': layer, 'X': X})
            results[f'batch={b}'][f'fwd+bwd seq_len={s}'] = 3 * forward / walltime(
                f'layer(X, {encoder_state})[0].sum().backward()', arg_dict={'layer': layer, 'X': X})            
    return pd.DataFrame(results)

def deit(rank, world_size):
    # Initialize distributed training
    accelerate.launch(backend='gloo', local_rank=rank, num_processes=world_size)

    # Initialize the accelerator
    accelerator = accelerate.Accelerator(num_processes=world_size)
    device = accelerator.device
    MODEL_NAME = "deit_small_distilled_patch16_224"
    model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-small-distilled-patch16-224')
    model = accelerator.prepare(model)
    model = model.to(device)

    # Assuming your DeiT model has hidden_size as 768, you can use the same benchmark function
    print(layer_benchmark(model, hidden_size=768, seq_lens=[3, 3], batch_sizes=[1], device=device))

def main():
    num_processes = 2
    mp.spawn(deit, args=(num_processes,), nprocs=num_processes)

if __name__ == '__main__':
    main()
