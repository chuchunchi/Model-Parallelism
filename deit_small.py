import torch
import pandas as pd
from collections import defaultdict
from transformers import AutoConfig, BertLayer
from torch.distributed import init_process_group
from transformers import DeiTForImageClassificationWithTeacher, Accelerator
from torch.utils import benchmark

def walltime(stmt, arg_dict, duration=10):
    return benchmark.Timer(stmt=stmt, globals=arg_dict).blocked_autorange(min_run_time=duration).median

def walltime(code, var_dict):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    eval(code, var_dict)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

def layer_benchmark(layer, hidden_size, seq_lens, batch_sizes, cross_attention=False):
    h = hidden_size
    results = defaultdict(lambda: {})    
    encoder_state = 'encoder_hidden_states=X' if cross_attention else ''
    for s in seq_lens:
        for b in batch_sizes:            
            ffn = 16*b*s*h*h / 1e12  # TFLOPS for the Feed-Forward Network
            atten = (4*b*h*s*s + 8*b*s*h*h) / 1e12  # TFLOPS for attention            
            forward = ffn + (2 if cross_attention else 1) * atten
            
            X = torch.randn(b, s, h).half().cuda()
            results[f'batch={b}'][f'fwd seq_len={s}'] = forward / walltime(
                f'layer(X, {encoder_state})', var_dict={'layer': layer, 'X': X})
            results[f'batch={b}'][f'fwd+bwd seq_len={s}'] = 3 * forward / walltime(
                f'layer(X, {encoder_state})[0].sum().backward()', var_dict={'layer': layer, 'X': X})            
    return pd.DataFrame(results)

def main():
    # Initialize distributed training
    init_process_group(backend='gloo')

    # Initialize the accelerator
    accelerator = Accelerator()
    device = accelerator.device
    MODEL_NAME = "deit_small_distilled_patch16_224"
    model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-small-distilled-patch16-224')
    model = accelerator.prepare(model)

    # Assuming your DeiT model has hidden_size as 768, you can use the same benchmark function
    layer_benchmark(model, hidden_size=768, seq_lens=[128, 512], batch_sizes=[2, 4, 8, 16, 32, 64, 128])

if __name__ == '__main__':
    main()
