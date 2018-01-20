import PIL.Image
import numpy as np
import matplotlib.pyplot as plt


def showimage(a):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    inp = a[0, :, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std * inp + mean
    inp *= 255	
    print("inp:",np.shape(inp))
    inp = np.uint8(np.clip(inp, 0, 255))
    plt.imshow(inp)
    plt.show()

