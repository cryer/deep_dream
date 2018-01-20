import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import os
from resnet import resnet50
from deepdream import dream
from PIL import Image
from util import showimage


img_transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_img = Image.open('./rem.jpg')
input_tensor = img_transform1(input_img).unsqueeze(0)
input_np = input_tensor.numpy()


img_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
inputs_control = Image.open('./image/3.jpg')
inputs_control = img_transform(inputs_control).unsqueeze(0)
inputs_control_np = inputs_control.numpy()

#showimage(inputs_control_np)

model = resnet50(pretrained=True)
if torch.cuda.is_available():
    model = model.cuda()
for param in model.parameters():
    param.requires_grad = False

if torch.cuda.is_available():
    x_variable = Variable(inputs_control.cuda())
else:
    x_variable = Variable(inputs_control)

control_features = model.forward(x_variable, end_layer=3)

def objective_guide(dst, guide_features):
    x = dst.data[0].cpu().numpy().copy()
    y = guide_features.data[0].cpu().numpy()
    ch, w, h = x.shape
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    result = y[:,A.argmax(1)] # select ones that match best
    result = torch.Tensor(np.array([result.reshape(ch, w, h)], dtype=np.float)).cuda()
    return result
	
	
dream(model, input_np, control=control_features, distance=objective_guide)
