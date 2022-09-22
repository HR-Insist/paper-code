import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math

batch_size = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data = datasets.MNIST("download", train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# 同时显示一个patch的图片
def show_mnist(imgs, labels):
    imgs = imgs.numpy()  # FloatTensor转为ndarray
    imgs = np.transpose(imgs, (0, 2, 3, 1))  # 把channel那一维放到最后
    plt.figure(figsize=(batch_size, batch_size))
    for i in range(batch_size):
        plt.subplot(int(math.sqrt(batch_size)), int(math.sqrt(batch_size)), i + 1)
        plt.title(labels[i].numpy())
        plt.axis('off')
        plt.imshow(imgs[i], cmap='gray')
    plt.show()

# 测试
def test():
    mnist_model = torch.load('./mnist_model.pth')
    mnist_model.eval()
    with torch.no_grad():
        all_corrects = 0
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = mnist_model(batch_X)
            _, pred = torch.max(outputs.data, 1)
            print(pred)
            batch_correct = pred.eq(batch_Y.data).sum()
            all_corrects += batch_correct
            print("batch test correct:%.3f%%" % (100 * batch_correct / batch_size))
            # show_mnist(batch_X, batch_Y)
            # break
    print("test correct:%.3f%%" % (100 * all_corrects / len(test_data)))


if __name__ == '__main__':
    test()
