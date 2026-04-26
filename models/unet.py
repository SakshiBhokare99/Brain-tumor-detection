import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,3,padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3,64)
        self.d2 = DoubleConv(64,128)
        self.d3 = DoubleConv(128,256)

        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(256,128,2,2)
        self.u1 = DoubleConv(256,128)

        self.up2 = nn.ConvTranspose2d(128,64,2,2)
        self.u2 = DoubleConv(128,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))

        x = self.up1(c3)
        x = torch.cat([x,c2],1)
        x = self.u1(x)

        x = self.up2(x)
        x = torch.cat([x,c1],1)
        x = self.u2(x)

        return torch.sigmoid(self.out(x))