import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from math import sqrt
from torch.autograd import Variable
import Net
class Parameters():
    def __init__(self):
        self.input_size = 512
        self.output_size = 512
        self.input_channels = 2
        self.channels = 64
parameters = Parameters()

class CNN(nn.Module):
    def __init__(self, parameters):
        super(CNN,self).__init__()
        self.input_size = parameters.input_size
        self.output_size = parameters.output_size
        self.input_channels = parameters.input_channels
        self.channels = parameters.channels
        self.blockl = self.make_layer(Net.BaseBlock, 2, self.channels)
        self.blockl1 = self.make_layer(Net.BaseBlock, self.channels*2, self.channels*2)
        self.blockl2 = self.make_layer(Net.BaseBlock, self.channels*4, self.channels*4)
        
        self.blocku2 = self.make_layer(Net.BaseBlock, self.channels*8, self.channels*4)
        self.blocku1 = self.make_layer(Net.BaseBlock, self.channels*4, self.channels*2)
        self.blocku = self.make_layer(Net.BaseBlock, self.channels*2, 1)
        #Convolution with stride
        self.convs = nn.Sequential(
            nn.Conv1d(self.channels, self.channels*2, 5, stride = 2, padding = 2),
            nn.BatchNorm1d(self.channels*2),
            nn.ReLU(inplace=True)
            )
        self.convs1 = nn.Sequential(
            nn.Conv1d(self.channels*2, self.channels*4, 5, stride = 2, padding = 2),
            nn.BatchNorm1d(self.channels*4),
            nn.ReLU(inplace=True)
            )
        self.convs2 = nn.Sequential(
            nn.Conv1d(self.channels*4, self.channels*8, 5, stride = 2, padding = 2),
            nn.BatchNorm1d(self.channels*8),
            nn.ReLU(inplace=True)
            )
        
        self.convt2 = nn.Sequential(
            nn.ConvTranspose1d(self.channels*8, self.channels*4, 5, stride = 2, padding=2, output_padding=1),
            nn.BatchNorm1d(self.channels*4),
            nn.ReLU(inplace=True)
            )
        self.convt1 = nn.Sequential(
            nn.ConvTranspose1d(self.channels*4, self.channels*2, 5, stride = 2, padding=2, output_padding=1),
            nn.BatchNorm1d(self.channels*2),
            nn.ReLU(inplace=True)
            )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(self.channels*2, self.channels, 5, stride = 2, padding=2, output_padding=1),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(inplace=True)
            )
        
    def make_layer(self, block, input_channels, output_channels, BlockNum = 1):
        layer = []
        for i in range(BlockNum):
            if i == 0:
                layer.append(block(input_channels, output_channels))
            else:
                layer.append(block(output_channels, output_channels))        
        return nn.Sequential(*layer)
    def forward(self, x):
        l = self.blockl(x)
        x1 = self.convs(l)
        l1 = self.blockl1(x1)
        x2 = self.convs1(l1)
        l2 = self.blockl2(x2)
        x3 = self.convs2(l2)
        y2 = self.convt2(x3)
        z2 = torch.cat((l2, y2), 1)
        u2 = self.blocku2(z2)
        y1 = self.convt1(u2)     
        z1 = torch.cat((l1, y1), 1)
        u1 = self.blocku1(z1)
        y = self.convt(u1)
        z = torch.cat((l,y), 1)
        u = self.blocku(z)
        p = u.view(-1, self.output_size)
        dia = torch.max(p,1)[0]
        sys = torch.min(p,1)[0]
        return torch.stack((dia, sys), 1)
cnn = CNN(parameters)
cnn.cuda()

train_data = []
test_data = []
train_label = []
test_label = []
tr = 0
te = 0
# 0 PPG , 1 Blood Pressure, 2 ECG
Mtrain = np.load('data/data1.npy')
Mtest  = np.load('data/data2.npy')
for i in range(3000):
    Mpeople = Mtrain[i]
    if Mpeople.shape[0] > 32* parameters.input_size:
        for j in range(32):
            M1 = Mpeople[j*parameters.input_size:(j+1)*parameters.input_size,0]
            M2 = Mpeople[j*parameters.input_size:(j+1)*parameters.input_size,2]
            Label = Mpeople[j*parameters.input_size:(j+1)*parameters.input_size,1]
            train_data.append([M1,M2])
            train_label.append([max(Label),min(Label)])
            tr +=1
for i in range(500):
    Mpeople = Mtrain[i]
    if Mpeople.shape[0] > 32* parameters.input_size:
        for j in range(32):
            M1 = Mpeople[j*parameters.input_size:(j+1)*parameters.input_size,0]
            M2 = Mpeople[j*parameters.input_size:(j+1)*parameters.input_size,2]
            Label = Mpeople[j*parameters.input_size:(j+1)*parameters.input_size,1]
            test_data.append([M1,M2])
            test_label.append([max(Label),min(Label)])
            te += 1

train_dataset = [torch.Tensor(train_data).float(), torch.Tensor(train_label).float()]
test_dataset = [torch.Tensor(test_data).float(), torch.Tensor(test_label).float()]

criterion = nn.MSELoss(size_average = False)
optimizer = torch.optim.Adam(cnn.parameters())

class MIMIC(data.Dataset):
    def __init__(self, dataset, train = True):
        self.train = train
        self.dataset = dataset
    def __getitem__(self, index):
        signals = self.dataset[0][index]
        label = self.dataset[1][index]
        return signals, label
    def __len__(self):
        if self.train:
            return tr
        else:
            return te
train_dataset = MIMIC(train_dataset,True)
test_dataset = MIMIC(test_dataset,False)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 32, shuffle = False)

i =1
for epoch in range(200):
    eloss = 0
    for signals, labels in train_loader:
        signals = Variable(signals.view(-1,parameters.input_channels,parameters.input_size)).cuda()
        labels = Variable(labels.view(-1,2)).cuda()

        optimizer.zero_grad()
        outputs = cnn(signals)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        eloss+=loss.cpu().detach().numpy()
    print(sqrt(eloss/tr))

testloss = 0
j = 0
_lsd = []
_osd = []
for signals, labels in test_loader:
    
    signals = torch.Tensor(signals.view(-1,parameters.input_channels,parameters.input_size)).cuda()
    labels = torch.Tensor(labels.view(-1,2)).cuda()
    outputs = cnn(signals)
    if j == 0:
        _lsd = labels.cpu().numpy()
        _osd = outputs.cpu().detach().numpy()
    else:
        _lsd = np.vstack((_lsd, labels.cpu().numpy()))
        _osd = np.vstack((_osd, outputs.cpu().detach().numpy()))
    j+=1
    testloss += criterion(outputs, labels).item()
    
np.savetxt('lsd', _lsd)
np.savetxt('osd', _osd)
torch.save(cnn, 'modelsd.pkl')