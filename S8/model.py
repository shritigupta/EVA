import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc3 = nn.Linear(10, 10)
        self.conv11 = nn.Sequential(nn.Conv2d(3,32,3, padding=1),nn.BatchNorm2d(32))
        self.conv12 = nn.Sequential(nn.Conv2d(32,64,3, padding=1),nn.BatchNorm2d(64))

        self.conv21 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128))
        self.conv22 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128))

        self.conv3 = nn.Sequential(nn.Conv2d(128,256,3, dilation=2,padding=2),nn.BatchNorm2d(256))

        self.conv4 = nn.Sequential((nn.Conv2d(256, 512 ,3, padding=1, groups=32)),
                                   nn.BatchNorm2d(512),
                      nn.Conv2d(512 ,10,1),
                      nn.BatchNorm2d(10))

        self.gap = nn.AvgPool2d(3) 
    
    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.pool(F.relu(self.conv12(x)))

        x = F.relu(self.conv21(x))
        x = self.pool(F.relu(self.conv22(x)))

        x = self.pool(F.relu(self.conv3(x)))
       
        x = self.conv4(x)
        x = self.gap(x)

        x = x.view(-1, 10)

        x = self.fc3(x)
        return x
