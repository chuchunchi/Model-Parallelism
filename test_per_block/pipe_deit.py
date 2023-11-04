:from urllib.request import urlopen
from PIL import Image
import timm, torch, time

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('deit_base_patch16_224.fb_in1k', pretrained=True)
#model = torch.compile(model, backend="inductor")
model = model.eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    start_time = time.perf_counter()
    output = model(input) 
    end_time = time.perf_counter()

#top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

inference_time = end_time - start_time
print(f"Inference time for 12 blocks: {inference_time} seconds")
