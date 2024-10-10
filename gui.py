import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from PIL import Image, ImageOps,ImageDraw



class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),
            
            #The size of the picture is 14x14
            torch.nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),
            
            #The size of the picture is 7x7
            torch.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 7 * 7 * 64,out_features = 128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 128,out_features = 10),
            torch.nn.Softmax(dim=1)
        )
    def forward(self,input):
        output = self.model(input)
        return output


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Path = './model.pth'
model = Net()
model.load_state_dict(torch.load(Path))
model.to(device)

class HandwriteApp:
    def __init__(self,model):
        self.model = model
        self.window = tk.Tk()
        self.canvas = tk.Canvas(self.window,width=200,height=200,bg='white')
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', self.draw)
        self.button = tk.Button(self.window,text='Predict',command=self.predict)
        self.button.pack(side=tk.LEFT)
        self.button_refresh = tk.Button(self.window,text='Refresh',command=self.refresh)
        self.button_refresh.pack(side=tk.RIGHT)
        self.image = Image.new('L',(28,28),color=255)
        self.draw_interface = ImageDraw.Draw(self.image) # 创建一个用于绘制的接口（在 `self.image` 上绘图）

    def draw(self,event):
        x,y = event.x,event.y
        self.canvas.create_oval(x,y,x+8,y+8,fill='black')# 在画布上绘制一个小黑色圆，模拟笔触效果
        self.draw_interface.ellipse([x//7, y//7, x//7+1, y//7+1], fill=0)# 在28x28的图像上绘制对应的黑色小点

    def predict(self):
        img = self.image.resize((28,28)).convert('L')
        img = ImageOps.invert(img)
        img = np.array(img)
        img = img/255.0
        img = torch.tensor(img,dtype=torch.float32).unsqueeze(0).unsqueeze(0) # 转换为PyTorch张量，并增加批次和通道维度

        with torch.no_grad():
            self.model.eval()
            output = self.model(img.to(device))
            pred = output.argmax(dim=1,keepdim=True)
            print(f"Predicted Digit: {pred.item()}")  # 打印预测的数字
    
    def refresh(self):
        self.canvas.delete('all')
        self.image = Image.new('L',(28,28),color=255)
        self.draw_interface = ImageDraw.Draw(self.image)
    def run(self):
        self.window.mainloop()  # 启动Tkinter的主事件循环


app = HandwriteApp(model)
app.run()
