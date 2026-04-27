import torch
from torch import nn
import torch.nn.functional as F

class WaveformGenerator(nn.Module):
    def __init__(self, nz, output_len):
        super().__init__()

        self.init_len = output_len // 16  # downscale factor
        self.fc = nn.Linear(nz, 256 * self.init_len)

        self.net = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 16, 4, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.Conv1d(16, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, self.init_len)
        x = self.net(x)
        return x.squeeze(1)
    
    def train_generator(self, discriminator, optimizer, data_fake):
        optimizer.zero_grad()
        output = discriminator(data_fake)
        loss = -torch.mean(output)
        loss.backward()
        optimizer.step()
        return loss
        
class SpectrogramDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)
    
    def train(self, optimizer, data_real, data_fake):
        optimizer.zero_grad()
        output_real = self.forward(data_real)
        loss_real = torch.mean(F.relu(1. - output_real)) #Hinge loss
        output_fake = self.forward(data_fake)
        loss_fake = torch.mean(F.relu(1. + output_fake)) #Hinge loss
        loss_real.backward()
        loss_fake.backward()
        optimizer.step()
        return loss_real + loss_fake