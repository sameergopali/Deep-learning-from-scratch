from ..simplegrad.function import LinearFunction
from ..tensor import Tensor
import numpy as np
from .module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.out_features =  out_features
        self.in_features =  in_features
        self.weight = Tensor(np.ones((out_features, in_features)), requires_grad=True,is_parameter_=True)
        self.bias = Tensor(np.ones(out_features), requires_grad=True,is_parameter_=True)
        self.linear = LinearFunction()
      
    
    def init_zeros(self):
        self.bias.data  = np.zeros_like(self.bias.data)
        self.weight.data = np.zeros_like(self.weight.data)
    
    def init_normal(self):
        self.weight.data = np.random.normal(loc=0, scale=1, size=(self.out_features, self.in_features))
        self.bias.data = np.zeros(self.out_features)
    
    def forward(self, input):
        output = self.linear.apply(input, self.weight, self.bias)
        return output
    

    def __str__(self) -> str:
        return f'Linear({self.in_features},{self.out_features})'

    def __repr__(self) -> str:
        return f'Linear({self.in_features},{self.out_features})'