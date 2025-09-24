import os
from glob import glob

class Dataset:
    def __init__(self, name: str, path: str, classes_path: str, splits: list[str]) -> None:
        self.name = name
        self.path = path
        self.classes_path = classes_path
        self.splits = splits
        
        self.size_by_split = {split: 0 for split in splits}
                
        with open(classes_path, 'r') as f:
            self.classes = f.readline().split(",")
            
        self.labels_by_split = {split: [] for split in splits}
        self.class_size_by_split = {split: {clss: 0 for clss in self.classes} for split in splits}
        self.largest_class_size_by_split = {split: 0 for split in splits}
        
        for split in splits:
            split_path = self.get_split_path(split)
            with open(split_path, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                self.size_by_split[split] = len(lines)
                for line in lines:
                    label = int(line.strip().split(",")[-1])
                    self.labels_by_split[split].append(label)
                    self.class_size_by_split[split][self.classes[label]] += 1
                    
                    self.largest_class_size_by_split[split] = max(self.largest_class_size_by_split[split], self.class_size_by_split[split][self.classes[label]])
                    
    def get_split_path(self, split: str) -> str:
        return os.path.join(self.path, f"{split}.csv")
    
# class Model:
#     def __init__(self, name: str) -> None:
#         self.name = name

class LossLandscapeExperiment:
    def __init__(self, dataset: Dataset, split: str, model: str, k: int, epoch: int, layer: int = -1) -> None:
        self.dataset = dataset
        self.split = split
        self.model = model
        self.layer = layer
        self.epoch = epoch
        self.k = k

        self.layer_tag = f"a{layer * -1}"
        self.epoch_tag = f"e{epoch}"
        self.model_data = f"{self.model}_{self.dataset.name}"
        
    def __hash__(self) -> int:
        return hash((self.dataset.name, self.split, self.model, self.k, self.epoch, self.layer))

    def __repr__(self) -> str:
        return f"{self.model} ({self.dataset.name}), k={self.k}, layer={self.layer}, epoch={self.epoch}"
    
    def get_paths(self, data_dir, ct_dir) -> dict:
        return {
            "ctree": os.path.join(ct_dir, f"{self.model_data}", self.split, f"ctree_{self.layer_tag}_{self.epoch_tag}_{self.k}"),
            "losses": os.path.join(data_dir, f"{self.model_data}", "Losses", self.split, f"loss_{self.layer_tag}_{self.epoch_tag}.txt"),
            "tensors": os.path.join(data_dir, f"{self.model_data}", "Tensors", self.split, f"vectors_{self.layer_tag}_{self.epoch_tag}.txt"),
            "predictions": os.path.join(data_dir, f"{self.model_data}", "Predictions", self.split, f"predictions_{self.layer_tag}_{self.epoch_tag}.txt"),
        }
    
    def validate_paths(self, data_dir, ct_dir) -> bool:
        paths = self.get_paths(data_dir, ct_dir)

        for key, path in paths.items():
            if key == "ctree":
                paths_should_exist = [f"{path}.{ext}" for ext in ["order.dat", "order.bin", "part.raw", "rg.bin", "rg.dat"]]

                for p in paths_should_exist:
                    if not os.path.exists(p):
                        print(f"Path for {key} does not exist: {p}")
                        return False
                    
                continue

            if not os.path.exists(path):
                print(f"Path for {key} does not exist: {path}")
                return False
            
        return True
    
def find_all_datasets(datasets_dir: str) -> dict[str, Dataset]:
    datasets = {}
    for dataset_name in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, dataset_name)
        if os.path.isdir(dataset_path):
            classes_path = os.path.join(dataset_path, "classes.txt")

            splits = glob("*.csv", root_dir=dataset_path)
            splits = [os.path.splitext(os.path.basename(split))[0] for split in splits]

            if not os.path.exists(classes_path):
                print(f"Warning: classes.txt not found for dataset {dataset_name}")
                continue

            if len(splits) == 0:
                print(f"Warning: No splits found for dataset {dataset_name}")
                continue

            dataset = Dataset(dataset_name, dataset_path, classes_path, splits)
            datasets[dataset_name] = dataset

    return datasets

def find_all_experiments(datasets: dict[str, Dataset], data_dir: str, ct_dir: str) -> list[LossLandscapeExperiment]:
    experiments: list[LossLandscapeExperiment] = []

    for (root, dirs, files) in os.walk(ct_dir):
        tree_files = glob("*.order.dat", root_dir=root)

        if len(tree_files) == 0:
            if len(dirs) == 0:
                print(f"Warning: No contour tree files found in {root}")
            continue

        rel_path = os.path.relpath(root, ct_dir)
        parts = rel_path.split(os.sep)

        assert len(parts) == 2

        model_data, split = parts
        model, dataset_name = model_data.split("_")

        if dataset_name not in datasets:
            print(f"Warning: Dataset {dataset_name} not found for model {model}")
            continue

        dataset = datasets[dataset_name]
        if split not in dataset.splits:
            print(f"Warning: Split {split} not found for dataset {dataset_name}")
            continue

        for tree_file in tree_files:
            _, layer, epoch, k = tree_file.split("_")

            layer = int(layer[1:]) * -1
            epoch = int(epoch[1:])
            k = int(k.split(os.path.extsep)[0])
            
            experiment = LossLandscapeExperiment(dataset, split, model, k, epoch, layer)

            if experiment.validate_paths(data_dir, ct_dir):
                experiments.append(experiment)

    return experiments