# from https://huggingface.co/microsoft/resnet-18/

from transformers import AutoFeatureExtractor, ResNetForImageClassification
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
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    # Initialize the accelerator
    accelerator = Accelerator()
    device = accelerator.device
    transform = ToTensor()
    image = transform(image)
    # Move the image tensor to the appropriate device
    image = image.to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    model_o = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

    # Move the feature extractor and model to the appropriate device(s)
    feature_extractor = accelerator.prepare(feature_extractor)
    model = accelerator.prepare(model_o)
    batch_size = 1
    benchmark(model=model, device=device, input_shape=(batch_size, 3, 224, 224), dtype="fp32", nun_warmup=50, num_runs=1000)
    '''inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        # Use the `model.forward` method instead of `model.__call__` for compatibility with Accelerate
        logits = model(**inputs).logits
    print(logits)
    # Synchronize the results across devices
    logits = accelerator.gather(logits)
    
    # Model predicts one of the 1000 ImageNet classes
    print(logits)
    predicted_label = logits.argmax(-1).item()
    print(model_o.config.id2label[predicted_label])'''

    
if __name__ == '__main__':
    main()
