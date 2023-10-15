import numpy as np

def split_train_val(data_x, data_y, split_ratio,seed):
    assert len(data_x) == len(data_y), 'Training set and Training label mismatch'
    np.random.seed(seed)
    indices = [i for i in range(len(data_x))]
    np.random.shuffle(indices)
    split_index = int(split_ratio * len(data_x))
    train_x = data_x[indices[:split_index]]
    train_y = data_y[indices[:split_index]]
    test_x =  data_x[indices[split_index:]]
    test_y =  data_y[indices[split_index:]]
    return train_x, test_x, train_y, test_y

class Dataloader:
    def __init__(self, x,y , shuffle=False, batch_size=1,seed =1 ):
        if batch_size > len(x):
            batch_size = len(x)
        if batch_size < 1 :
            batch_size =1
        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        global_state = np.random.get_state()
        np.random.seed(seed)
        self.state = np.random.get_state()
        np.random.set_state(global_state)

    def __iter__(self):
     

        indicies = [i for i in range(len(self.x))]
        if self.shuffle:
            global_state = np.random.get_state()
            np.random.set_state(self.state)
            np.random.shuffle(indicies)
            self.state = np.random.get_state()
            np.random.set_state(global_state)
        for i in range(0,len(indicies),self.batch_size):
            x = self.x[indicies[i:i+self.batch_size]]
            y = self.y[indicies[i:i+self.batch_size]]
            yield x,y
            
            

    def __len__(self):
        return (len(self.x) + self.batch_size -1)// self.batch_size






