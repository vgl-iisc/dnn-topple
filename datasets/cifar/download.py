import tarfile
import os

os.system("curl -L -o ./datasets/cifar10/cifar10.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")

with tarfile.open("./datasets/cifar10/cifar10.tar.gz", "r:gz") as tar:
    tar.extractall(path="./datasets/cifar10/data")
    
os.remove("./datasets/cifar10/cifar10.tar.gz")