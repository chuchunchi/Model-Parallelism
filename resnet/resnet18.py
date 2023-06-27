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

from torchvision.transforms import ToTensor
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
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

    # Move the feature extractor and model to the appropriate device(s)
    feature_extractor = accelerator.prepare(feature_extractor)
    model = accelerator.prepare(model)

    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        # Use the `model.forward` method instead of `model.__call__` for compatibility with Accelerate
        logits = model(**inputs).logits

    # Synchronize the results across devices
    logits = accelerator.gather(logits)

    # Model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])

    
if __name__ == '__main__':
    main()
