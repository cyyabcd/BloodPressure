import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import Net
class Parameters():
    def __init__(self):
        self.input_size = 512
        self.output_size = 512
        self.input_channels = 3
        self.channels = 64
parameters = Parameters()

class CNN(nn.Module):
    def __init__(self, parameters):
        super(CNN,self).__init__()
        self.input_size = parameters.input_size
        self.output_size = parameters.output_size
        self.input_channels = parameters.input_channels
        self.channels = parameters.channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.channels, self.channels, 5, padding = 2),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.channels, 2*self.channels, 5, padding = 2),
            nn.BatchNorm1d(2*self.channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(2*self.channels, self.channels, 5, padding = 2),
            nn.BatchNorm1d(self.channels)
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.channels, self.channels, 7, padding = 3),
            nn.BatchNorm1d(self.channels)
            )
        self.conv3 = nn.Sequential(
            nn.Conv1d(3*self.channels, self.channels, 5, padding = 2),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.channels, 2*self.channels, 5, padding = 2),
            nn.BatchNorm1d(2*self.channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(2*self.channels, self.channels, 5, padding = 2),
            nn.BatchNorm1d(self.channels)
            )
        self.conv4 = nn.Sequential(
            nn.Conv1d(3*self.channels, self.channels, 5, padding = 2),
            nn.BatchNorm1d(self.channels)
            )
        
        self.block1 = self.make_layer(Net.BaseBlock, 1, self.channels)
        self.block2 = self.make_layer(Net.BaseBlock, self.channels*self.input_channels, self.channels)
        
        self.Ul = nn.Sequential(
            nn.Conv1d(self.channels, self.channels, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm1d(self.channels)
            )
        self.Ur = nn.Sequential(
            nn.ConvTranspose1d(self.channels, self.channels, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm1d(self.channels)
            )
        self.output = nn.Conv1d(2*self.channels, 1, 9, padding = 4)

    def make_layer(self, block, input_channels, output_channels, BlockNum = 1):
        layer = []
        for i in range(BlockNum):
            if i == 0:
                layer.append(block(input_channels, output_channels))
            else:
                layer.append(block(output_channels, output_channels))        
        return nn.Sequential(*layer)
    def forward(self, x):
        x0 = x[:,0].view(-1,1,self.input_size)
        x1 = x[:,1].view(-1,1,self.input_size)
        x2 = x[:,2].view(-1,1,self.input_size)
        b0 = self.block1(x0)
        b1 = self.block1(x1)
        b2 = self.block1(x2)
        c10 = self.conv1(b0)
        c20 = self.conv2(b0)
        c11 = self.conv1(b1)
        c21 = self.conv2(b1)
        c12 = self.conv1(b2)
        c22 = self.conv2(b2)
        r0 = F.relu(c10+c20)
        r1 = F.relu(c11+c21)
        r2 = F.relu(c12+c22)
        r = torch.cat((r0,r1,r2),1)
        br = self.block2(r)
        fr = F.relu(self.conv3(r)+self.conv4(r))
        ul = self.Ul(fr)
        ur = self.Ur(ul)
        c = torch.cat((ur, fr), 1)
        out = self.output(c)
        return out.view(-1,self.output_size)

cnn = CNN(parameters)
cnn.cuda()

train_data = []
train_label = []
test_data = []
test_label = []
tr = 0
te = 0
for i in range(6):
    M = np.load('data/3000393/%d.npy'%(i))
    for j in range(M.shape[0]//parameters.input_size):
        M1 = M[j*parameters.input_size:(j+1)*parameters.input_size,1]
        M2 = M[j*parameters.input_size:(j+1)*parameters.input_size,2]
        M3 = M[j*parameters.input_size:(j+1)*parameters.input_size,3]
        Label = M[j*parameters.input_size:(j+1)*parameters.input_size,4]
        if i < 4 and tr < 9000:
            tr+=1
            train_data.append([M1,M2,M3])
            train_label.append([Label])
        elif te < 200:
            te+=1
            test_data.append([M1,M2,M3])
            test_label.append([Label])


train_dataset = [torch.Tensor(train_data).float(), torch.Tensor(train_label).float()]
test_dataset = [torch.Tensor(test_data).float(), torch.Tensor(test_label).float()]

criterion = nn.MSELoss()
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
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 200, shuffle = False)

i =1
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
    print(eloss/tr)

testloss = 0
'''
def draw(f1,f2):
    t = np.arange(1,256,1)
    plt.plot(t,f1,'g')
    plt.plot(t,f2,'r')
    plt.show()
    '''
j = 1
for signals, labels in test_loader:
    
    signals = Variable(signals.view(-1,parameters.input_channels,parameters.input_size)).cuda()
    labels = Variable(labels.view(-1,parameters.output_size)).cuda()
    outputs = cnn(signals)
    
    np.savetxt('l%d'%(j),labels.cpu().numpy())
    np.savetxt('t%d'%(j),outputs.cpu().detach().numpy())
    j+=1
#    draw(labels,signals)
    testloss += criterion(outputs, labels)
print('testloss %f'%(testloss/200))
    