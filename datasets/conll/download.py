import zipfile
import os

os.system("curl -L -o ./datasets/conll/conll2003-dataset.zip https://www.kaggle.com/api/v1/datasets/download/juliangarratt/conll2003-dataset")

with zipfile.ZipFile("./datasets/conll/conll2003-dataset.zip", "r") as zip_ref:
	zip_ref.extractall("./datasets/conll/data")

os.remove("./datasets/conll/conll2003-dataset.zip")