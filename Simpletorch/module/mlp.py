from .module import Network
from . import Activation
from .Activation import ReLU,Sigmoid
from .Linear import Linear

class MLP(Network):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fn='ReLU', initialization ='normal'):
        if activation_fn not in ['ReLU','Sigmoid']:
            raise ValueError('Activation Function not supported. Expected: ReLU or Sigmoid')
    
        if initialization not in ['normal','zeros']:
            raise ValueError('Initialization not supported.  Expected: normal or zeros')
        
        super().__init__()
        self.layers = []
        for i,size in enumerate(hidden_sizes):
            self.layers.append(Linear(input_size, size))
            self.register_modules("layer"+str(i), self.layers[-1])
            self.layers.append(getattr(Activation,activation_fn)())
            self.register_modules("activation"+str(i), self.layers[-1])
            input_size = size

        self.layers.append(Linear(hidden_sizes[-1], output_size))
        self.register_modules("layer"+str(i+1), self.layers[-1])
        self.initialize(initialization=initialization)
    
    def initialize(self, initialization):
        for layer in self.layers:
            if initialization == 'normal':
                if isinstance(layer, Linear):
                    layer.init_normal()
            
            if initialization == 'zeros':
                if isinstance(layer, Linear):
                    layer.init_zeros()


    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
   