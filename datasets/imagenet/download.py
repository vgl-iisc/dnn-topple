import tarfile
import os

os.system("curl -L -o /media/solis/DATA/imagenet/imagenet.tar.gz https://image-net.org/data/ILSVRC/2017/ILSVRC2017_DET.tar.gz")

with tarfile.open("/media/solis/DATA/imagenet/imagenet.tar.gz", "r:gz") as tar:
    tar.extractall(path="/media/solis/DATA/imagenet/data")
    
os.remove("/media/solis/DATA/imagenet/imagenet.tar.gz")