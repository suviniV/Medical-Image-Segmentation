import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ParameterSet:
    """Simple parameter container"""
    def __init__(self):
        self.parameters = {}
        self.flat = None
    
    def __setitem__(self, key, value):
        self.parameters[key] = value
        # Update flat parameters
        self.flat = torch.cat([p.flatten() for p in self.parameters.values()])
    
    def __getitem__(self, key):
        return self.parameters[key]

class SupervisedModel(nn.Module):
    """Base class for supervised models"""
    def __init__(self):
        super().__init__()
        self.parameters = ParameterSet()
        self.exprs = {}
        
    def _init_exprs(self):
        raise NotImplementedError()

class SequentialModel(nn.Module):
    """PyTorch implementation of sequential 3D CNN model for segmentation"""
    def __init__(self, image_height=120, image_width=120, image_depth=78, 
                 n_channels=4, n_output=2, imp_weight=False,
                 batch_size=4, optimizer='adam', max_epochs=100,
                 l1=None, l2=None):
        super(SequentialModel, self).__init__()
        
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channels = n_channels
        self.n_output = n_output
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.l1 = l1
        self.l2 = l2
        
        # Simple encoder path matching saved weights dimensions
        self.enc1 = nn.Sequential(
            nn.Conv3d(n_channels, 32, kernel_size=3, padding=1)  # [32, 4, 3, 3, 3]
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1)  # [64, 32, 3, 3, 3]
        )
        
        # Final convolution
        self.final = nn.Conv3d(64, n_output, kernel_size=1)
        
    def forward(self, x):
        # Forward pass matching the simple architecture
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.final(x)
        return x

class FCN(SupervisedModel):
    def __init__(self, image_height, image_width, image_depth,
                 n_channel, n_output, n_hiddens_conv, down_filter_shapes,
                 hidden_transfers_conv, down_pools, n_hiddens_upconv,
                 up_filter_shapes, hidden_transfers_upconv, up_pools,
                 out_transfer, loss, optimizer='adam',
                 bm_up='same', bm_down='same',
                 batch_size=1, max_iter=1000,
                 strides_d=(1, 1, 1), up_factors=(2, 2, 2),
                 verbose=False, implementation=False):
        super().__init__()
        
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_output = n_output
        self.n_hiddens_conv = n_hiddens_conv
        self.down_filter_shapes = down_filter_shapes
        self.hidden_transfers_conv = hidden_transfers_conv
        self.down_pools = down_pools
        self.n_hiddens_upconv = n_hiddens_upconv
        self.up_filter_shapes = up_filter_shapes
        self.hidden_transfers_upconv = hidden_transfers_upconv
        self.up_pools = up_pools
        self.out_transfer = out_transfer
        self.loss_ident = loss
        self.optimizer = optimizer
        self.bm_down = bm_down
        self.bm_up = bm_up
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.implementation = implementation
        self.strides_d = strides_d
        self.up_factors = up_factors
        
        self._init_exprs()

    def _init_exprs(self):
        inpt = torch.randn(1, self.image_depth, self.n_channel, self.image_height, self.image_width)

        target = torch.randn(1, self.n_output)

        parameters = ParameterSet()

        self.conv_net = nn.Sequential(
            nn.Conv3d(self.n_channel, self.n_hiddens_conv[0], kernel_size=self.down_filter_shapes[0], padding=self.bm_down),
            nn.ReLU(),
            nn.MaxPool3d(self.down_pools[0]),
            nn.Conv3d(self.n_hiddens_conv[0], self.n_hiddens_conv[1], kernel_size=self.down_filter_shapes[1], padding=self.bm_down),
            nn.ReLU(),
            nn.MaxPool3d(self.down_pools[1]),
            nn.ConvTranspose3d(self.n_hiddens_conv[1], self.n_hiddens_upconv[0], kernel_size=self.up_filter_shapes[0], stride=self.up_factors[0], padding=self.bm_up),
            nn.ReLU(),
            nn.ConvTranspose3d(self.n_hiddens_upconv[0], self.n_hiddens_upconv[1], kernel_size=self.up_filter_shapes[1], stride=self.up_factors[1], padding=self.bm_up),
            nn.ReLU(),
            nn.Conv3d(self.n_hiddens_upconv[1], self.n_output, kernel_size=1)
        )

        output = self.conv_net(inpt)

        if self.imp_weight:
            imp_weight = torch.randn(1, self.n_output)
        else:
            imp_weight = None

        self.loss_layer = nn.CrossEntropyLoss(weight=imp_weight)

        SupervisedModel.__init__(self)
        self.exprs['imp_weight'] = imp_weight

    def forward(self, x):
        """Forward pass through the network"""
        for layer in self.conv_net:
            x = layer(x)
        return x


class ConvNet3d(SupervisedModel):
    def __init__(self, image_height, image_width, image_depth,
                 n_channel, n_hiddens_conv, filter_shapes, pool_shapes,
                 n_hiddens_full, n_output, hidden_transfers_conv,
                 hidden_transfers_full, out_transfer, loss, optimizer='adam',
                 batch_size=1, max_iter=1000, verbose=False, implementation='dnn_conv3d',
                 dropout=False):
        super().__init__()
        
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_hiddens_conv = n_hiddens_conv
        self.filter_shapes = filter_shapes
        self.pool_shapes = pool_shapes
        self.n_hiddens_full = n_hiddens_full
        self.n_output = n_output
        self.hidden_transfers_conv = hidden_transfers_conv
        self.hidden_transfers_full = hidden_transfers_full
        self.out_transfer = out_transfer
        self.loss_ident = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.implementation = implementation

        self.dropout = dropout

        self.border_modes = 'valid'

        self._init_exprs()

    def _init_exprs(self):
        inpt = torch.randn(1, self.image_depth, self.n_channel, self.image_height, self.image_width)

        target = torch.randn(1, self.n_output)

        parameters = ParameterSet()

        self.conv_net = nn.Sequential(
            nn.Conv3d(self.n_channel, self.n_hiddens_conv[0], kernel_size=self.filter_shapes[0], padding=self.border_modes),
            nn.ReLU(),
            nn.MaxPool3d(self.pool_shapes[0]),
            nn.Conv3d(self.n_hiddens_conv[0], self.n_hiddens_conv[1], kernel_size=self.filter_shapes[1], padding=self.border_modes),
            nn.ReLU(),
            nn.MaxPool3d(self.pool_shapes[1]),
            nn.Flatten(),
            nn.Linear(self._get_conv_output_size(), self.n_hiddens_full[0]),
            nn.ReLU(),
            nn.Linear(self.n_hiddens_full[0], self.n_output)
        )

        output = self.conv_net(inpt)

        if self.imp_weight:
            imp_weight = torch.randn(1, self.n_output)
        else:
            imp_weight = None

        self.loss_layer = nn.CrossEntropyLoss(weight=imp_weight)

        SupervisedModel.__init__(self)
        self.exprs['imp_weight'] = imp_weight

    def _get_conv_output_size(self):
        """Calculate the size of the flattened convolution output"""
        x = torch.randn(1, self.n_channel, self.image_depth, 
                       self.image_height, self.image_width)
        for layer in self.conv_net:
            if isinstance(layer, (nn.Conv3d, nn.MaxPool3d)):
                x = layer(x)
        return int(np.prod(x.shape))

    def forward(self, x):
        """Forward pass through the network"""
        for layer in self.conv_net:
            x = layer(x)
        return x

class Lenet3d(SupervisedModel):
    
    def __init__(self, image_height, image_width, image_depth,
                 n_channel, n_hiddens_conv, filter_shapes, pool_shapes,
                 n_hiddens_full, n_output, hidden_transfers_conv,
                 hidden_transfers_full, out_transfer, loss, optimizer='adam',
                 batch_size=1, max_iter=1000, verbose=False, implementation='dnn_conv3d',
                 pool=True):
        super().__init__()
        
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_hiddens_conv = n_hiddens_conv
        self.n_hiddens_full = n_hiddens_full
        self.filter_shapes = filter_shapes
        self.pool_shapes = pool_shapes
        self.n_output = n_output
        self.hidden_transfers_conv = hidden_transfers_conv
        self.hidden_transfers_full = hidden_transfers_full
        self.out_transfer = out_transfer
        self.loss_ident = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.implementation=implementation
        self.pool = pool

        self._init_exprs()

    def _init_exprs(self):
        inpt = torch.randn(1, self.image_depth, self.n_channel, self.image_height, self.image_width)

        target = torch.randn(1, self.n_output)

        parameters = ParameterSet()
       
        self.lenet = nn.Sequential(
            nn.Conv3d(self.n_channel, self.n_hiddens_conv[0], kernel_size=self.filter_shapes[0], padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(self.pool_shapes[0]),
            nn.Conv3d(self.n_hiddens_conv[0], self.n_hiddens_conv[1], kernel_size=self.filter_shapes[1], padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(self.pool_shapes[1]),
            nn.Flatten(),
            nn.Linear(self._get_conv_output_size(), self.n_hiddens_full[0]),
            nn.ReLU(),
            nn.Linear(self.n_hiddens_full[0], self.n_output)
        )

        output = self.lenet(inpt)

        if self.imp_weight:
            imp_weight = torch.randn(1, self.n_output)
        else:
            imp_weight = None

        self.loss_layer = nn.CrossEntropyLoss(weight=imp_weight)

        SupervisedModel.__init__(self)
        self.exprs['imp_weight'] = imp_weight

    def _get_conv_output_size(self):
        """Calculate the size of the flattened convolution output"""
        x = torch.randn(1, self.n_channel, self.image_depth, 
                       self.image_height, self.image_width)
        for layer in self.lenet:
            if isinstance(layer, (nn.Conv3d, nn.MaxPool3d)):
                x = layer(x)
        return int(np.prod(x.shape))

    def forward(self, x):
        """Forward pass through the network"""
        for layer in self.lenet:
            x = layer(x)
        return x