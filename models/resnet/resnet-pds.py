import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

# An instance of your model.
#model = torchvision.models.resnet18(pretrained=True)
modelPath='/home/schwarm/pds/DataServerTest/pdServe/pds-0520v3.pth'

model=torch.load(modelPath,map_location='cpu')

# Evaluation mode
model.eval()

# An example input you would normally provide to your model's forward() method.
#example = torch.rand(1,3,224,2688)



convert_tensor = transforms.ToTensor()
PNGPath = "/home/schwarm/pdServe/Serve/geeks.png"
scrollImage=Image.open(PNGPath)
t=convert_tensor(scrollImage)
example=t.float()
example=example.unsqueeze(0)

def export_cpu(model, example):
    model = model.to("cpu")
    example = example.to("cpu")

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Save traced model
    traced_script_module.save("pds_cpu.pth")
    output1 = model(example)
    output2 = traced_script_module(example)
    np.testing.assert_allclose(output1.detach().numpy(), output2.detach().numpy())


def export_gpu(model, example):
    model = model.to("cuda")
    example = example.to("cuda")

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Save traced model
    traced_script_module.save("pds_gpu.pth")

export_cpu(model, example)

if torch.cuda.is_available():
    export_gpu(model, example)
