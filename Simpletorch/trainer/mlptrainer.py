from .basetrainer import BaseTrainer
from ..tensor import Tensor
from ..metrics import Metric, calc_accuracy
import numpy as np

class MLPTrainer(BaseTrainer):
    def __init__(self, train_dataloader, val_dataloader,model, criterion, optimizer, epochs):
        super().__init__(model=model, optimizer=optimizer, criterion=criterion, epochs=epochs)
        self.train_loader =  train_dataloader
        self.val_dataloader = val_dataloader

    def train_one_epoch(self, epoch):
        sloss = 0 
        correct = 0 
        for x,y in self.train_loader:
            x_ = Tensor(x)
            y_ = Tensor(y)
            out =  self.model(x_)
            loss = self.criterion(out,y_)
            correct += np.sum(np.argmax(out.data, axis=1) == np.argmax(y, axis=1)) 
            sloss += loss.data
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        

        
        return Metric(sloss/len(self.train_loader), correct/len(self.train_loader.x))
        
    
    def val_one_epoch(self,epoch):
        sloss= 0 
        correct = 0
        for x,y in self.val_dataloader:
            x_ = Tensor(x)
            y_ = Tensor(y)
            out =  self.model(x_)
            loss = self.criterion(out,y_)
            correct += np.sum(np.argmax(out.data, axis=1) == np.argmax(y, axis=1))
            sloss += loss.data

        
        return  Metric(sloss/len(self.val_dataloader),  correct/len(self.val_dataloader.x))
        
    