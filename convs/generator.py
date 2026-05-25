import torch
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def clone(self):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        return clone.cuda()

class Generator1D(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=31, nc=1):
        super(Generator1D, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size
        
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(ngf * 2),
            
            nn.Conv1d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        
        # We need a linear layer to match the exact 31 dimensions if it gets reshaped 
        # or we just use 1D Convs. Since we want exactly 31 output size:
        self.final_linear = nn.Linear(31, 31)

    def forward(self, z):
        # z shape: [batch, nz]
        out = self.l1(z)
        # Reshape to [batch, features, length] => [batch, ngf*2, 31]
        out = out.unsqueeze(2).expand(-1, -1, self.init_size)
        
        # Pass through 1D Convs
        out = self.conv_blocks(out) # shape: [batch, 1, 31]
        
        # Flatten and adjust exact dimensions
        out = out.view(out.size(0), -1)
        out = self.final_linear(out)
        out = torch.sigmoid(out)
        
        # F2SCIL expects the generator output to match the input shape, 
        # so for 1D we can output [batch, 1, 31] or [batch, 31].
        # The real dataset yields [batch, 31], so we output [batch, 31]
        return out

    def clone(self):
        clone = Generator1D(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        return clone.cuda()