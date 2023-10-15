class Optimizer:
    def __init__(self, params) -> None:
        self.params =  list(params)
    
    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0)

    def step():
        pass
