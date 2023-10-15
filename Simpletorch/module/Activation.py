from ..simplegrad import function
from .module import Module

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.activation = function.relu()

    def forward(self,input):
        out  = self.activation.apply(input)
        return out
    
    def __call__(self, input):
        return self.forward(input)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.activation = function.sigmoid()

    def forward(self,input):
        out = self.activation.apply(input)
        return out
    
    def __call__(self,input):
        return self.forward(input)
