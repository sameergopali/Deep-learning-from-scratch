from ..simplegrad import function
class Loss:
    def forward(self, input, target):
        pass
    def __call__(self,input,target):
        return self.forward(input,target)

class CrossEntropyWithLogits(Loss):
    def __init__(self):
        self.loss = function.softmax_crossentropy_logits()

    def forward(self,input,target):
        out =  self.loss.apply(input,target)
        return out
    
   


class HingeLoss(Loss):
    def __init__(self):
        self.loss = function.hingeloss()

    def forward(self,input,target):
        out =  self.loss.apply(input,target)
        return out
    
   

class MSEloss(Loss):
    def __init__(self):
        self.loss = function.mean_squared_error()

    def forward(self, input, target):
        out = self.loss.apply(input, target)
        return out
    


    
class BinaryCrossEntropy(Loss):
    def __init__(self):
        self.loss = function.binarycrossentropy()

    def forward(self, input,target):
        out =  self.loss.apply(input,target)
        return out
    

class BinaryCrossEntorpyWithLogits(Loss):
    def __init__(self):
        self.loss = function.sigmoid_binarycrossentropy_logits()

    def forward(self,input,target):
        out = self.loss.apply(input,target)