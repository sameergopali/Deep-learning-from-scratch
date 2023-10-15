from ..tensor import Tensor
import numpy as np



class Function:
    def forward(self,*args):
        pass

    def backward(self, out_grad):
        '''
        return Tuple of gradient correspondings to input in forward pass
        '''
        pass
   
    def apply(self,*variables):
        out = self.forward(*variables)
        if any(v.requires_grad for v in variables):
            out.requires_grad = True
            out.autogradMeta.grad_fn=self.backward
            out.autogradMeta.grad_fname=self.__str__()
            for v in variables:
                if v.requires_grad:
                    out.autogradMeta.next_fn.append(v.autogradMeta)
        else:
            out.autogradMeta = None
        return out
    
    def __str__(self):
        return self.__class__.__qualname__
    
class AddFunction(Function):
    def forward(self, input1, input2):
        self.input1= input1
        self.input2 = input2
        output =  Tensor(input1.data + input2.data)
        return output
        
    def backward(self, grad_output):
        output = []
        if self.input1.requires_grad:
            grad_input1 = grad_output 
            output.append(grad_input1)
        if self.input2.requires_grad:
            grad_input2 = grad_output 
            output.append(grad_input2)
        return output

class ConstantMultiplication(Function):
    def forward(self, input1):
        self.input1= input1
        output =  Tensor(input1.data*2)
        return output
        
    def backward(self, grad_output):
        output=[]
        if self.input.requires_grad:
            grad_input1 = 2 * grad_output
            output.append(grad_input1)
        return output

class MulFunction(Function):
    def forward(self, input1, input2):
        self.input1= input1
        self.input2 = input2
        output =  Tensor(input1.data * input2.data)
        return output
        
    def backward(self, grad_output):
        output = []
        if self.input1.requires_grad:
            grad_input1 = grad_output * self.input2
            output.append(grad_input1)
        if self.input2.requires_grad:    
            grad_input2 = grad_output* self.input1
            output.append(grad_input2)
        return output

class LinearFunction(Function):
    def forward(self,input,weight,bias):
        '''
        output : Tensor
        self.input : tensor
        self.weight : tensor
        self.bias : tensor
    
        '''
        self.input = input
        self.weight = weight
        self.bias  = bias
        output = Tensor(np.matmul(input.data, weight.data.T) + bias.data[np.newaxis,:],requires_grad=True)
        return output
    
    def backward(self, grad_output):
        '''
        Returns
            grad_output : numpy_array
            grad_input : numpy_array
            grad_weight : numpy_array
            grad_bias : numpy_array
        '''
        output = []
        if self.input.requires_grad:
            grad_input= np.matmul(grad_output,self.weight.data)
            output.append(grad_input)
        if self.weight.requires_grad:
            grad_weight= np.matmul(grad_output.T,self.input.data)
            output.append(grad_weight)
        if self.bias.requires_grad:
            grad_bias=  grad_output.sum(0)
            output.append(grad_bias)
        return  output

class sigmoid(Function):
    def forward(self, input):
        self.input =  input
        self.y =   Tensor(1.0 / (1.0 + np.exp(-input.data)))
        return self.y
    
    def backward(self, grad_output):
        output = []
        if self.input.requires_grad:
            grad_input =  self.y.data* (1-self.y.data) * grad_output
            output.append(grad_input)
        return output

class relu(Function):
    def forward(self, input):
        self.input =  input
        output = Tensor(np.where(input.data > 0, input.data,0),requires_grad=True)
        return output

    def backward(self, grad_output):
        output = []
        if self.input.requires_grad:
            grad_input = np.where(self.input.data >0,grad_output,0) 
            output.append(grad_input)
        return output

class softmax_crossentropy_logits(Function):
    def softmax(self, input):
        exps = np.exp(input.data - np.max(input.data, axis=1).reshape(-1, 1))
        output = exps/np.sum(exps,axis=1).reshape(-1,1)
        return output 
    
    def logsoftmax(self,input):
        '''
        The logsoftmax function computes the log of the softmax activation function on the input Tensor.
         It first computes the maximum value of each row of the input Tensor and subtracts it from the input Tensor, 
         then exponentiates each element of the resulting Tensor, and finally computes the logarithm of the sum 
         of the exponentiated values of the corresponding row. The output is the input Tensor with each element
        replaced by the corresponding log-softmax value
        '''
        c = input.data.max(axis=1).reshape(-1,1)
        logsumexp = np.log(np.exp(input.data - c).sum(axis=1)).reshape(-1,1)
        return input.data - c - logsumexp

    
    def forward(self, input, target):
        self.input = input
        self.target = target
        logsoftmax_ = self.logsoftmax(self.input)
        out = Tensor(np.mean((-target.data * logsoftmax_).sum(axis=1)))
        return out
    
    def backward(self, outgrad):
        output=[]
        if self.input.requires_grad:
            input_grad = (self.softmax(self.input) - self.target.data) *outgrad /self.input.data.shape[0]
            output.append(input_grad)
        return output

class mean_squared_error(Function):
    def forward(self, input, target):
        self.input = input
        self.target = target 
        output =  Tensor(np.mean((self.input.data - self.target.data)**2))
        return output

    def backward(self, outgrad):
        output = []
        if self.input.requires_grad:
            input_grad = outgrad *2*(self.input.data-self.target.data)/self.input.data.size
            output.append(input_grad)
        return output    
    
class binarycrossentropy(Function):
    def forward(self, input, target):
        self.input = input 
        self.target = target
        out = Tensor(np.mean(-target.data *np.log(input.data) - (1 -target.data)*np.log(1-input.data)))
        return out
        
    def backward(self, out_grad):
        output = []
        if self.input.requires_grad:
            input_grad = (-(self.target.data/self.input.data) + (1-self.target.data)/(1-self.input.data) )/self.input.data.size*out_grad
            output.append(input_grad)
        return output

class sigmoid_binarycrossentropy_logits(Function):

    def forward(self, input,target):
        self.input = input
        self.target  = target 
        clamp = np.where(self.input.data <0, 0 ,self.input.data)
        out = Tensor(np.mean(clamp - input.data * target.data + np.log(1 + np.exp(-np.abs(input.data)))))
        return out 

    def backward(self,outgrad):
        output=[]
        if self.input.requires_grad:
            inp_sigmoid = sigmoid().forward(self.input)
            input_grad = outgrad *(inp_sigmoid.data - self.target.data)/ inp_sigmoid.data.size
            output.append(input_grad)
        return output

class hingeloss(Function):
    def forward(self, input, actual) :
        # get prediction value for true class
        self.input =  input
        self.actual =  actual
        true_pred = (input.data * actual.data).sum(axis=-1).reshape(-1,1)
        #get predictions value for other classes
        other_pred = (1. - actual.data) * input.data
        #calculate margins for all predictions by broadcasting
        self.margin = other_pred  - true_pred +1 
        #  max(margin,0)
        out = np.where(self.margin>0, self.margin,0)  
        # We don't sum for true class, so remove it
        out = np.where(actual.data==1, 0, out)
        # sum over values
        out = out.sum(axis=1)
        # Reduce with mean
        out = out.mean()
        return Tensor(out)



    def backward(self, outgrad):
        output = []
        if self.input.requires_grad:
            grad_ = np.where(self.margin>0,1,0)
            grad_ = np.where(self.actual.data==1,0,grad_)
            grad_[self.actual.data==1] = (-1) *np.sum(grad_, axis=1)
            input_grad = grad_/len(self.input.data) * outgrad
            output.append(input_grad)
        return output

class softmax:
    def __call__(self, input):
        exps = np.exp(input.data - np.max(input.data, axis=1).reshape(-1, 1))
        output = exps/np.sum(exps,axis=1).reshape(-1,1)
        return output 