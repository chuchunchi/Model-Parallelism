from urllib.request import urlopen
from PIL import Image
import timm
import torch
import time

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

# Load the DeiT-tiny model with 5 million parameters
model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
model = model.eval()

num_blocks_to_keep = 1
model.blocks = torch.nn.Sequential(model.blocks[:num_blocks_to_keep])

# Load the quantization module from torch
from torch.quantization import quantize_dynamic

# Define a function to quantize the model
def quantize_model(model):
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    model_quantized = quantize_dynamic(model_prepared)
    return model_quantized

# Quantize the model
quantized_model = quantize_model(model)

data_config = timm.data.resolve_model_data_config(quantized_model)
transforms = timm.data.create_transform(**data_config, is_training=False)
input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    start_time = time.perf_counter()
    output = quantized_model(input)
    end_time = time.perf_counter()

inference_time = end_time - start_time
print(f"Inference time for 1 block with quantized model: {inference_time} seconds")
