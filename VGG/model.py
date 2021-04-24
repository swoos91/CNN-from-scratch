import torch
import torch.nn as nn

types = {
    "VGG16":[
        (64, 2),
        'M',
        (128, 2),
        'M',
        (256, 3),
        'M',
        (512, 3),
        'M',
        (512, 3),
        'M'
    ]
}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu( self.bn( self.conv(x) ) )

class FCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        return self.dropout( self.relu( self.linear(x) ) )

class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, architecture=types["VGG16"]):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(architecture)
        self.fcs = nn.Sequential(
            FCBlock(512 * 7 * 7, 4096),
            FCBlock(4096, 4096),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for module in architecture:
            if isinstance(module, tuple):
                out_channels, num_repeats = module
                for _ in range(num_repeats):
                    layers.append(
                        CNNBlock(
                            in_channels,
                            out_channels
                        )
                    )
                    in_channels = out_channels
            elif isinstance(module, str):
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2
                    )
                )

        return nn.Sequential(*layers)

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = VGG(in_channels=3, num_classes=1000).to(device)
#     x = torch.randn(3, 3, 224, 224).to(device) # Mini batch size=3
#     print(model(x).shape)