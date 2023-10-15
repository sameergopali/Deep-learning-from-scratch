from .optimizer import Optimizer
class SGD(Optimizer):
    def __init__(self, params, lr=1) -> None:
        super().__init__(params)
        self.lr = lr 
    
    def step(self):
        for p in self.params:
            #print(p.grad.shape, p.data.shape)
            # print(p.grad)
            p.data -= p.grad*self.lr
    