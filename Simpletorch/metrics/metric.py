import numpy as np
from ..simplegrad import function
from ..tensor import Tensor


class Metric:
    def __init__(self, loss, accuracy):
        self.loss = loss
        self.accuracy = accuracy
    
    def __repr__(self) -> str:
        return f'loss:{self.loss:.4f}, accuracy:{self.accuracy:.4f}'
    


def calc_accuracy(model, test_loader):
    total = 0
    correct = 0
    
    for images, labels in test_loader:
        images_ = Tensor(images)
        # calculate outputs by running images through the network
        outputs = model(images_)
        # the class with the highest energy is what we choose as prediction
        predicted = np.argmax(outputs.data, 1)
        labels = np.argmax(labels,1)
        total += labels.shape[0]
        correct += (predicted == labels).sum().item()
      
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')


def per_class_accuracy( model, test_loader):
    correct_pred = {classname: 0 for classname in range(0,10)}
    total_pred = {classname: 0 for classname in range(0,10)}
    for images,labels in test_loader:
        images_ =Tensor(images)
        outputs = model(images_)
        predictions = np.argmax(outputs.data, 1)
        labels_ = np.argmax(labels,1)
        for label, prediction in zip(labels_, predictions):
            if label == prediction:
                correct_pred[prediction] += 1
            total_pred[prediction] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:} is {accuracy:.1f} %')
