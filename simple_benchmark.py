# from https://huggingface.co/microsoft/resnet-18/

from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
import torch
from datasets import load_dataset
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from datasets import load_dataset
from accelerate import Accelerator
import time
from torchvision.transforms import ToTensor
import numpy as np
import timm

def benchmark(model, device="cuda", input_shape=(1, 3, 224, 224), dtype='fp32', num_warmup=50, num_runs=1000):
    input_data = torch.randn(input_shape)
    input_data, moel = input_data.to(device), model.to(device)
    num_runs = int(num_runs/input_shape[0])
    # Warm-up GPU
    print("     Batch Size: {}".format(input_shape[0]))
    print("        Warm up: {} iteraion".format(num_warmup))
    with torch.no_grad():
        for _ in range(num_warmup):
            features = model(input_data)
    # Measure Latency
    # torch.cuda.synchronize()
    print("    Start timing: {} iteration".format(num_runs))
    timings = []
    with torch.no_grad():
        for i in range(1, num_runs+1):
            start_time = time.perf_counter()
            features = model(input_data)
            # torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            if i%(num_runs/10)==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, num_runs, np.mean(timings)*1000))

    print("      Input shape:", input_data.shape)
    print("     Output shape:", features[0].shape)
    print(   'Avg batch time: %.2f ms'%(np.mean(timings)*1000))
    print('Latency per query: %.2f ms'%((np.mean(timings)/input_shape[0])*1000))


def main():
    torch.distributed.init_process_group(backend='gloo')
    # Initialize the accelerator
    accelerator = Accelerator()
    device = accelerator.device
    MODEL_NAME = "deit_small_distilled_patch16_224"
    # model = timm.create_model(MODEL_NAME, pretrained=True, scriptable=True)
    # feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-small-distilled-patch16-224')
    model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-small-distilled-patch16-224')
    model = accelerator.prepare(model)
    batch_size = 1
    benchmark(model=model, device=device, input_shape=(batch_size, 3, 224, 224), dtype="fp32", num_warmup=50, num_runs=500)


    
if __name__ == '__main__':
    main()