import tarfile
import os

os.system("curl -L -o ./datasets/cifar/cifar.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")

with tarfile.open("./datasets/cifar/cifar.tar.gz", "r:gz") as tar:
    tar.extractall(path="./datasets/cifar/data")
    
os.remove("./datasets/cifar/cifar.tar.gz")