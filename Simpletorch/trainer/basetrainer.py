class BaseTrainer:
    def __init__(self, model, optimizer, criterion,epochs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_results =[]
        self.validation_results = []
        self.epochs = epochs


    def train_one_epoch(self):
        pass

    def val_one_epoch(self):
        pass
    
    def train(self):
        for epoch in range(self.epochs):
            training_result = self.train_one_epoch(epoch)
            validation_result = self.val_one_epoch(epoch)
            self.training_results.append(training_result)
            self.validation_results.append(validation_result)
            print(f'Epoch#:{epoch}|| training_results:[{training_result} ]|| validation_results:[{validation_result}]')
        