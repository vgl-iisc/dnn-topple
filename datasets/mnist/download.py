import zipfile
import os

os.system("curl -L -o ./datasets/mnist/mnist.zip https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset")
with zipfile.ZipFile("./datasets/mnist/mnist.zip", 'r') as zip_ref:
    zip_ref.extractall("./datasets/mnist/data")
    
os.remove("./datasets/mnist/mnist.zip")