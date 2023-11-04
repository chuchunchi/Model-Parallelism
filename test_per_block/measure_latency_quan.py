import timm, torch
import numpy as np
from PIL import Image

from urllib.request import urlopen
import time

from torch.quantization import quantize_dynamic

def quantize_model(model):
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    model_quantized = quantize_dynamic(model_prepared)
    return model_quantized
def quantize_proportion(model):
    fp32_param = 0.0
    total = 0.0
    q8_param = 0.0
    # print(model.state_dict())
    for name, param in model.state_dict().items():
        if(isinstance(param, torch.Tensor)):
            total += torch.numel(param)
            if(param.dtype == torch.float32):
                fp32_param += torch.numel(param)
            if(param.dtype == torch.qint8):
                q8_param += torch.numel(param)
        elif(isinstance(param, torch.dtype)):
            print(param)
        elif isinstance(param, tuple):
            for sub_param in param:
                if isinstance(sub_param, torch.Tensor):
                    total += torch.numel(sub_param)
                    if sub_param.dtype == torch.float32:
                        fp32_param += torch.numel(sub_param)
                    elif sub_param.dtype == torch.qint8:
                        q8_param += torch.numel(sub_param)
    print(fp32_param, total, q8_param, q8_param / total)
    return fp32_param / total

def measure_latency(
    model:torch.nn.modules, 
    measure_times=100, 
    warm_up=3, 
    compile_model=False, 
    num_threads=2,
    num_model_blocks=12,
    input_shape=(1, 3, 224, 224)
):
    torch.set_num_threads(num_threads)
    MEASURE_TIMES = measure_times
    WARM_UP = warm_up
    COMPILE_MODEL=compile_model
    MODEL_NAME = model.pretrained_cfg['architecture'] if hasattr(model, 'pretrained_cfg') else 'PatchEmbed'
    BATCH_SIZE = input_shape[0]
    

        
    if(COMPILE_MODEL):
        model = torch.compile(model) 
        output = model(input)   # Compiling model takes a long time to trace computational graph at the first inference run

    print(f"\n\nMeasuring average inference latecy of {MODEL_NAME} over {MEASURE_TIMES} run.")
    print(f"  - compile Model = {COMPILE_MODEL}")
    print(f"  - batch size = {BATCH_SIZE}")
    print(f"  - transformer blocks = {MODEL_NAME if MODEL_NAME=='PatchEmbed' else num_model_blocks}")
    print(f"  - nums_threads = {num_threads}")
    
    latency_list = []
    with torch.no_grad():
        for i in range(MEASURE_TIMES + WARM_UP):            
            input = torch.randn(BATCH_SIZE,3,224,224)
            
            start_time = time.perf_counter()
            output = model(input)
            end_time = time.perf_counter()
            
            if(i<WARM_UP):  # ignore the first 3 runs
                continue
            
            inference_time = end_time - start_time
            latency_list.append(inference_time)
            print(f"Inference time for {num_model_blocks} blocks: iteration_{i - WARM_UP} = {inference_time:.3f} || average = {np.mean(latency_list):.3f} seconds ")
    # print(latency_list)
    return np.mean(latency_list)

if __name__ == "__main__":  
    measure_times = 100
    num_trheads = 1
    model = timm.layers.PatchEmbed()
    #quantized_model = quantize_model(model)
    patch_embed_latency = measure_latency(model, measure_times=measure_times, num_threads=num_trheads)
    
    MODEL_NAME = 'deit_small_patch16_224'
    model = timm.create_model(MODEL_NAME, pretrained=True)
    #quantized_model = quantize_model(model)
    quantized_model = quantize_dynamic(model, dtype=torch.qint8)
    print(type(quantized_model))
    print(quantize_proportion(quantized_model))
    time.sleep(10)  # cool-down CPU for fair comparison
    num_blocks_to_keep = 12
    full_latency = measure_latency(model, measure_times=measure_times, num_model_blocks=num_blocks_to_keep, num_threads=num_trheads)
    
    time.sleep(10)  # cool-down CPU for fair comparison
    num_blocks_to_keep = 1
    model.blocks = torch.nn.Sequential(model.blocks[:num_blocks_to_keep])
    block_latency = measure_latency(model, measure_times=measure_times, num_model_blocks=num_blocks_to_keep, num_threads=num_trheads)
    
    print(f"--- Latency(sec) ---")
    print(f"patchEmbed = {full_latency-patch_embed_latency}\n")
    print(f"transformer block + patchEmbed: 1 block = {block_latency} || 12 block = {full_latency}")
    print(f" - speedup (12 block + 1 patchEmbed)/(1 block + 1 patchEmbed) = {(full_latency)/(block_latency):.2f}\n")
    print(f"transformer block: 1 block = {block_latency-patch_embed_latency} || 12 block = {full_latency-patch_embed_latency}")
    print(f" - speedup (12 block / 1 block) = {(full_latency - patch_embed_latency)/(block_latency - patch_embed_latency):.2f}\n")
    print(f" - speedup ((12 block + 1 patchEmbed) / 1 block) = {(full_latency)/(block_latency - patch_embed_latency):.2f}")
