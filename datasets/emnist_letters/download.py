import zipfile
import os
import gzip
import shutil

# Download EMNIST Letters dataset
os.system("curl -L -o ./datasets/emnist_letters/emnist-letters.zip https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip")

# Extract the zip file
with zipfile.ZipFile("./datasets/emnist_letters/emnist-letters.zip", 'r') as zip_ref:
    zip_ref.extractall("./datasets/emnist_letters/temp")

# Move the Letters files to the data directory
os.makedirs("./datasets/emnist_letters/data", exist_ok=True)

# Map files to expected names (similar to MNIST structure)
file_mapping = {
    "emnist-letters-train-images-idx3-ubyte.gz": "train-images.idx3-ubyte",
    "emnist-letters-train-labels-idx1-ubyte.gz": "train-labels.idx1-ubyte",
    "emnist-letters-test-images-idx3-ubyte.gz": "t10k-images.idx3-ubyte",
    "emnist-letters-test-labels-idx1-ubyte.gz": "t10k-labels.idx1-ubyte"
}

for gz_name, target_name in file_mapping.items():
    gz_path = os.path.join("./datasets/emnist_letters/temp/gzip", gz_name)
    if os.path.exists(gz_path):
        with gzip.open(gz_path, 'rb') as f_in:
            with open(os.path.join("./datasets/emnist_letters/data", target_name), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

# Clean up
shutil.rmtree("./datasets/emnist_letters/temp")
os.remove("./datasets/emnist_letters/emnist-letters.zip")
