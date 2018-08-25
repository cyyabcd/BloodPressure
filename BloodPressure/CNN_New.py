import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import math
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
        return u.view(-1, self.output_size)
cnn = CNN(parameters)
cnn.cuda()

train_data = []
test_data = []
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
            M3 = Mpeople[j*parameters.input_size:(j+1)*parameters.input_size,1]
            train_data.append([M1,M2,M3])
            tr +=1
for i in range(500):
    Mpeople = Mtrain[i]
    if Mpeople.shape[0] > 32* parameters.input_size:
        for j in range(32):
            M1 = Mpeople[j*parameters.input_size:(j+1)*parameters.input_size,0]
            M2 = Mpeople[j*parameters.input_size:(j+1)*parameters.input_size,2]
            M3 = Mpeople[j*parameters.input_size:(j+1)*parameters.input_size,1]
            test_data.append([M1,M2,M3])
            te +=1


train_dataset = torch.Tensor(train_data).float()
test_dataset = torch.Tensor(test_data).float()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters())

class MIMIC(data.Dataset):
    def __init__(self, dataset, train = True):
        self.train = train
        self.dataset = dataset
    def __getitem__(self, index):
        signals = self.dataset[index][0:2]
        label = self.dataset[index][2]
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
cnn.train()
for epoch in range(200):
    eloss = 0
    for signals, labels in train_loader:
        signals = Variable(signals.view(-1,parameters.input_channels,parameters.input_size)).cuda()
        labels = Variable(labels.view(-1,parameters.output_size)).cuda()

        optimizer.zero_grad()
        outputs = cnn(signals)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        eloss+=loss.cpu().detach().numpy()
    print(math.sqrt(eloss/tr*64))

'''
def draw(f1,f2):
    t = np.arange(1,256,1)
    plt.plot(t,f1,'g')
    plt.plot(t,f2,'r')
    plt.show()
    '''
def test():
    cnn.eval()
    testloss = 0
    j = 0    
    _l = []
    _o = []
    for signals, labels in test_loader:
        signals = torch.Tensor(signals.view(-1,parameters.input_channels,parameters.input_size)).cuda()
        labels = torch.Tensor(labels.view(-1,parameters.output_size)).cuda()
        outputs = cnn(signals)
        testloss += criterion(outputs, labels).item()
        if j == 0:
            _l = labels.cpu().numpy()
            _o = outputs.cpu().detach().numpy()
        else:
            _l = np.vstack((_l, labels.cpu().numpy()))
            _o = np.vstack((_o, outputs.cpu().detach().numpy()))
        j+=1
    np.savetxt('l', _l)
    np.savetxt('o', _o)
    print('testloss %f'%(math.sqrt(testloss/te*32)))
torch.save(cnn, 'model.pkl')
test()