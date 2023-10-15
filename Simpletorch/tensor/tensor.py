import numpy as np
from Simpletorch.simplegrad import AutoGradMeta,AccumulatorBox

class Tensor:
    def __init__(self,value, requires_grad=False, is_parameter_=False) -> None:   
        self.data =  np.array(value)
        self.requires_grad=requires_grad
        self.is_parameter = is_parameter_
    
    def __repr__(self) -> str:
        return f'Tensor(data={self.data}, requires_grad={self.requires_grad})'
    
    @property
    def is_parameter(self):
        return self.is_parameter_
    
    @is_parameter.setter
    def is_parameter(self,is_parameter_):
        self.is_parameter_ = is_parameter_

    @property
    def grad(self):
        if self.requires_grad:
            return self.autogradMeta.grad
        else:
            raise RuntimeError('Trying to get grad for Data with requires_grad=False')    
    
    
    @property
    def autogradMeta(self):
        return self.autoGrad_
    

    @autogradMeta.setter
    def autogradMeta(self,autograd_):
        self.autoGrad_ = autograd_
    
   
    @property
    def  requires_grad(self):
        return self.requires_grad_
    

    @requires_grad.setter
    def requires_grad(self,requires_grad):
        if requires_grad:
            self.autogradMeta = AccumulatorBox(np.zeros_like(self.data) )
        else:
            self.autogradMeta = None
        self.requires_grad_ = requires_grad
    
    
    def backward(self, grad=None):
        if grad is None:
            grad = np.array([1])
        if self.requires_grad:
            if self.autogradMeta is None:
                raise RuntimeError("Doesn't have box attribute")
            self.autogradMeta.backward(grad)
        else:
            raise RuntimeError('Running backward() on Data with requires_grad=False')

    def zero_(self):
        self.data = np.zeros_like(self.data) 



