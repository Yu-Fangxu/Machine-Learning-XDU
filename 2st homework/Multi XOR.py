import torch
import torch.nn as nn
import  torch.optim as optim
from torch.utils.data import DataLoader,Dataset
###定义XOR的网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3,4)
        self.fc2 = nn.Linear(4,1)
        self.activate = nn.Sigmoid()
    def forward(self, x):
        x = self.activate(self.fc1(x))
        return self.activate(self.fc2(x))
###自定义数据集
class Mydata(Dataset):
    def __init__(self):
        super(Mydata, self).__init__()
        self.dataset = torch.tensor([[0.,0.,0.],[0.,0.,1.],[0.,1.,0.],[0.,1.,1.],[1.,0.,0.],
                                     [1.,0.,1,],[1.,1.,0.],[1.,1.,1.]]).to(device)
        self.labels = torch.tensor([0.,1.,1.,0.,1.,0.,0.,1.]).to(device)
    def __getitem__(self, index):
        return self.dataset[index, :], self.labels[index]
    def __len__(self):
        return len(self.dataset)
###放在哪个设备上运行
device = 'cpu' if torch.cuda.is_available() == False else 'cuda:0'
###定义dataset与dataloader
dataset = Mydata()
loader = DataLoader(dataset, shuffle=False,batch_size=1)
###定义损失函数
Loss = nn.BCELoss().to(device)
###定义网络
net = Net().to(device)
###设置优化器
optimizer = optim.Adam(net.parameters(), lr=1e-2)
###定义epochs
epochs = 3000
###训练部分
for epoch in range(epochs):
    total_loss = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        input, label = data[0], data[1]
        out = net(input[0])
        loss = Loss(out, label)
        loss.backward()
        optimizer.step()
    total_loss += loss
    
    # early stopping
    if total_loss < 0.001:
        break

###测试部分
for i, data in enumerate(loader):
    ###输入顺序依次为(0,1), (1,1) (0,0) (1,0)
    input, label = data[0], data[1]
    with torch.no_grad():
        out = net(input[0])
    print(out)
for name, param in net.named_parameters():
    print(name, '      ', param)
