import torch
import torchvision.models as models

# Download pretrained ResNet-18
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.eval()
torch.save(resnet18.state_dict(), "resnet18.pth")
print("ResNet-18 saved as resnet18.pth")

# Download pretrained MobileNetV3-Small
mobilenet_v3_small = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
mobilenet_v3_small.eval()
torch.save(mobilenet_v3_small.state_dict(), "mobilenet_v3_small.pth")
print("MobileNetV3-Small saved as mobilenet_v3_small.pth")

# Download pretrained AlexNet
alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
alexnet.eval()
torch.save(alexnet.state_dict(), "alexnet.pth")
print("AlexNet saved as alexnet.pth")
