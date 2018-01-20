import torch
from torchvision import transforms
from PIL import Image
from resnet import resnet50
from deepdream import dream

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_img = Image.open('./rem.jpg')
input_tensor = img_transform(input_img).unsqueeze(0)
input_np = input_tensor.numpy()

model = resnet50(pretrained=True)
if torch.cuda.is_available():
    model = model.cuda()
for param in model.parameters():
    param.requires_grad = False
	

dream(model, input_np)
