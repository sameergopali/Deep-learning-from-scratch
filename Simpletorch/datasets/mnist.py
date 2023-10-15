
import gzip
import pickle
import urllib.request
from urllib.error import HTTPError
import os
import numpy as np



def download_mnist(filenames,savepath):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for filename in filenames:
        filepath = os.path.join(savepath, filename)
        if not os.path.isfile(filepath):
            fileurl = base_url + filename
            print(fileurl)
            print(f"Downloading {fileurl}...")
            try:
                urllib.request.urlretrieve(fileurl, filepath)
                print(f"Download complete.")
            except HTTPError as e:
                print(
                    "Something went wrong.Download failed."
                )
    
def load_mnist_dataset(savepath='data'):
    ''''
    Downloads and returns MNIST train and test images and labels.
    Returns training images, training labels, test images, test labels
    '''
    datasets = {
        "training_images": "train-images-idx3-ubyte.gz",
        "test_images" : "t10k-images-idx3-ubyte.gz"}
    labels ={
        "training_labels": "train-labels-idx1-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    os.makedirs(savepath,exist_ok=True)
    download_mnist(filenames=datasets.values(), savepath=savepath)
    download_mnist(filenames=labels.values(), savepath=savepath)
    mnist= {}
    try:
        for name,filename in datasets.items():
            filepath = os.path.join(savepath, filename)
            with gzip.open(filepath, 'rb') as f:
                mnist[name] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
        for name,filename in labels.items():
            filepath = os.path.join(savepath, filename)
            with gzip.open(filepath, 'rb') as f:
                mnist[name] = np.frombuffer(f.read(), np.uint8, offset=8)
    except IOError:
        print("IOError, Something went wrong")
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
    

    
