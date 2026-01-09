import tarfile
import os

os.system("curl -L -o /media/solis/DATA/imagenet/ILSVRC2012_devkit_t12.tar.gz https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz")
os.system("curl -L -o /media/solis/DATA/imagenet/ILSVRC2012_img_train.tar https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar")
os.system("curl -L -o /media/solis/DATA/imagenet/ILSVRC2012_img_val.tar https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar")