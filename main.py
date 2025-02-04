import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"
transfrom = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5],std=[0.5])]
)
path = './data/'
trainData = torchvision.datasets.MNIST(path,train=True,transform=transfrom,download=True)
testData  = torchvision.datasets.MNIST(path,train=False,transform=transfrom)

BATCH_SIZE = 256

trainDataLoader = torch.utils.data.DataLoader(dataset=trainData,batch_size=BATCH_SIZE,shuffle=True)
testDataLoader =torch.utils.data.DataLoader(dataset=testData,batch_size=BATCH_SIZE)


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

net = Net()
print(net.to(device))

lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

EPOCHS = 10
history = {'Test Loss':[],'Test Accuracy':[]}
for epoch in range(1,EPOCHS+1):
    processBar = tqdm(trainDataLoader,unit='step')
    net.train(True)
    for step, (trainImgs,labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)
        net.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs,labels)
        predictions = torch.argmax(outputs,dim=1)
        accuracy = torch.sum(predictions==labels/labels.shape[0])
        loss.backward()
        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % 
                                   (epoch,EPOCHS,loss.item(),accuracy.item()))
        if step ==len(processBar)-1:
            correct,totalLoss = 0,0
            net.train(False)
            for testImgs,labels in testDataLoader:
                testImgs = testImgs.to(device)
                labels = labels.to(device)
                outputs = net(testImgs)
                loss = lossF(outputs,labels)
                predictions = torch.argmax(outputs,dim=1)
                totalLoss+=loss
                correct+= torch.sum(predictions==labels)
            testAccuracy = correct/(BATCH_SIZE *len(testDataLoader))
            testLoss = totalLoss /(len(testDataLoader))
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" % 
                                   (epoch,EPOCHS,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
    processBar.close()

#对测试Loss进行可视化
plt.plot(history['Test Loss'],label = 'Test Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#对测试准确率进行可视化
plt.plot(history['Test Accuracy'],color = 'red',label = 'Test Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

torch.save(net.state_dict(),'./model.pth')
