import zipfile
import os
import gzip
import shutil

# Download EMNIST ByClass dataset
os.system("curl -L -o ./datasets/emnist/emnist-byclass.zip https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip")

# Extract the zip file
with zipfile.ZipFile("./datasets/emnist/emnist-byclass.zip", 'r') as zip_ref:
    zip_ref.extractall("./datasets/emnist/temp")

# Move the ByClass files to the data directory
os.makedirs("./datasets/emnist/data", exist_ok=True)

# Map files to expected names (similar to MNIST structure)
file_mapping = {
    "emnist-byclass-train-images-idx3-ubyte.gz": "train-images.idx3-ubyte",
    "emnist-byclass-train-labels-idx1-ubyte.gz": "train-labels.idx1-ubyte",
    "emnist-byclass-test-images-idx3-ubyte.gz": "t10k-images.idx3-ubyte",
    "emnist-byclass-test-labels-idx1-ubyte.gz": "t10k-labels.idx1-ubyte"
}

for gz_name, target_name in file_mapping.items():
    gz_path = os.path.join("./datasets/emnist/temp/gzip", gz_name)
    if os.path.exists(gz_path):
        with gzip.open(gz_path, 'rb') as f_in:
            with open(os.path.join("./datasets/emnist/data", target_name), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

# Clean up
shutil.rmtree("./datasets/emnist/temp")
os.remove("./datasets/emnist/emnist-byclass.zip")
