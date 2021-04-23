import torch
import torch.nn as nn

"""
Tuple: (filters, kernel_size, stride, pooling & reshape) for CNNBlock
List: [output_size, activation function]  
"""
config = [
    (6, 5, 1, False),
    (16, 5, 1, False),
    (120, 5, 1, True),
    [84, True],
    [10, False]
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_reshape=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool_reshape = pool_reshape

    def forward(self, x):
        x = self.pool( self.relu( self.conv(x) ) ) if not self.pool_reshape else self.relu( self.conv(x) ) 
        return x if not self.pool_reshape else x.reshape(x.shape[0], -1)


class FC_Layer(nn.Module):
    def __init__(self, input_size, output_size, activation=True):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        return self.relu( self.linear(x) ) if self.activation else self.linear(x)


class LeNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.layers = self._create_layers()

    def forward(self, x):
        return self.layers(x)

    def _create_layers(self):
        layers = []
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride, pool_reshape = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        pool_reshape=pool_reshape,
                        kernel_size=kernel_size,
                        stride=stride
                    )
                )
                in_channels = out_channels
            
            elif isinstance(module, list):
                out_channels, ac_fc = module
                layers.append(
                    FC_Layer(
                        in_channels,
                        out_channels,
                        activation=ac_fc
                    )
                )
                in_channels = out_channels

        return nn.Sequential(*layers)



# if __name__ == "__main__":
#     x = torch.randn(64, 1, 32, 32)
#     model = LeNet(in_channels=1)
#     print( model(x).shape )