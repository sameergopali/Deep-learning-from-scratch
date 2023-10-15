class AutoGradMeta:
    def __init__(self, grad_fn,grad_fname) -> None:
        '''
        Args
        grad_fn: Function
        grad_fname :str
        '''
        self.grad_buffer = 0
        self.grad_fn = grad_fn
        self.dep_nr = 0
        self.grad_fname = grad_fname
        # List of next input gradient function
        self.next_fn = []
        

    @property
    def grad_fn(self):
        return self.grad_fn_
    
    @grad_fn.setter
    def grad_fn(self, grad_fn):
        self.grad_fn_ = grad_fn
    
    
    def apply(self,outgrad):
        return self.grad_fn(outgrad)
    

    def __repr__(self) -> str:
        return f'AutoGradMeta({self.grad_fname})'
    
     
    def backward(self,grad):
        # The compute_dependencies_nr function is used to recursively traverse the computational graph of the current node
        #  and its dependent nodes, and count the number of dependencies of each node.

        def compute_dependencies_nr(root):
            for child in root.next_fn:
                if child is  None:
                        continue
                child.dep_nr += 1
                compute_dependencies_nr(child)
        compute_dependencies_nr(self)

        # The traverse function is used to perform backpropagation
        # and compute the gradients for all the dependencies of the current node.
        def traverse(root):
            # If the current node has no more dependencies, 
            # compute the gradient for the node by calling its grad_fn 
            # and passing it the accumulated gradient stored in its grad_buffer attribute.
            if root.dep_nr == 0:
                grad = root.grad_fn(root.grad_buffer)
                root.grad_buffer = 0
            for i,child in enumerate(root.next_fn):
                if child is  None:
                    continue
                # Decrement the dependency count for the current node's dependent node.
                child.dep_nr -= 1
                # Add the current node's gradient to the dependent node's grad_buffer attribute
                child.grad_buffer += grad[i]
                if child.dep_nr == 0:
                     # If the dependent node has no more dependencies, recursively call traverse on the dependent node to continue the backpropagation process.
                    traverse(child)
        # Add the current gradient to the grad_buffer attribute of the current node to accumulate the gradients from all the dependent nodes.
        self.grad_buffer += grad    
         # Start the backpropagation process by calling traverse on the current node 
        traverse(self)

class AccumulatorBox(AutoGradMeta):
    def __init__(self, grad):
        super().__init__(grad_fn=self.accumulator, grad_fname='accumulator')
        self.grad = grad 

    def accumulator(self,grad):
        self.grad += grad    
