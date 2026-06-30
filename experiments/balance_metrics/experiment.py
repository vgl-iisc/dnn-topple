import os
import re
from glob import glob
import pandas as pd

NER_LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

_BERT_CTREE_RE = re.compile(r'^ctree_a(.+)_e(\d+)_(\d+)\.order\.dat$')


class BertDataset:
    """Minimal Dataset-compatible representation for BERT NER experiments."""

    def __init__(self, name: str):
        self.name = name
        self.path = ""
        self.classes = NER_LABELS
        self.size_by_split: dict[str, int] = {}
        self.labels_by_split: dict[str, list[int]] = {}
        self.class_size_by_split: dict[str, dict[str, int]] = {}
        self.largest_class_size_by_split: dict[str, int] = {}

    def get_split_path(self, split: str) -> str:
        return ""


class BertExperiment:
    """Parallel to LossLandscapeExperiment for BERT NER data (standalone, no streamlit)."""

    is_bert = True

    def __init__(self, dataset: BertDataset, split: str, tag: str, k: int, epoch: int,
                 landscape_dir: str, ct_dir: str, complexes_dir: str = ""):
        self.dataset = dataset
        self.split = split
        self.tag = tag
        self.k = k
        self.epoch = epoch
        self.landscape_dir = landscape_dir
        self.ct_dir = ct_dir
        self.complexes_dir = complexes_dir
        self.paths = None

        self.layer = f"a{tag}"
        self.layer_tag = f"a{tag}"
        self.pretty_layer = tag
        self.epoch_tag = f"e{epoch}"
        self.model = "bert"
        self.model_data = dataset.name

    def __hash__(self) -> int:
        return hash(("bert", self.dataset.name, self.split, self.tag, self.k, self.epoch))

    def __repr__(self) -> str:
        return (f"BERT NER {self.dataset.name} ({self.split}),"
                f" k={self.k}, layer={self.pretty_layer}, epoch={self.epoch}")

    def get_paths(self) -> dict:
        self.paths = {
            "ctree": os.path.join(
                self.ct_dir, self.split,
                f"ctree_a{self.tag}_{self.epoch_tag}_{self.k}"),
            "losses": os.path.join(
                self.landscape_dir, "Losses", self.split, f"losses_{self.epoch_tag}.pt"),
            "labels": os.path.join(
                self.landscape_dir, "Labels", self.split, f"labels_{self.epoch_tag}.pt"),
            "predictions": os.path.join(
                self.landscape_dir, "Predictions", self.split, f"predictions_{self.epoch_tag}.pt"),
        }
        return self.paths

    def validate_paths(self) -> bool:
        paths = self.get_paths()
        for key, path in paths.items():
            if key == "ctree":
                for ext in ["order.dat", "order.bin", "part.raw", "rg.bin", "rg.dat"]:
                    if not os.path.exists(f"{path}.{ext}"):
                        print(f"BERT: Missing ctree file: {path}.{ext}")
                        return False
            else:
                if not os.path.exists(path):
                    print(f"BERT: Missing {key}: {path}")
                    return False
        return True


def find_all_bert_experiments(data_dir: str, ct_dir: str, complexes_dir: str = "") -> list[BertExperiment]:
    """Scan ct_dir for BERT NER experiments (ctree files live in {ct_dir}/{split}/)."""
    experiments: list[BertExperiment] = []
    bert_datasets: dict[str, BertDataset] = {}

    if not os.path.isdir(ct_dir):
        return experiments

    ds_name = os.path.basename(data_dir.rstrip("/\\"))

    for split_name in sorted(os.listdir(ct_dir)):
        split_dir = os.path.join(ct_dir, split_name)
        if not os.path.isdir(split_dir):
            continue

        for fname in sorted(os.listdir(split_dir)):
            m = _BERT_CTREE_RE.match(fname)
            if m is None:
                continue

            tag, epoch_str, k_str = m.group(1), m.group(2), m.group(3)

            if ds_name not in bert_datasets:
                bert_datasets[ds_name] = BertDataset(ds_name)

            exp = BertExperiment(
                bert_datasets[ds_name], split_name, tag,
                int(k_str), int(epoch_str),
                data_dir, ct_dir, complexes_dir,
            )
            if exp.validate_paths():
                experiments.append(exp)
                print(f"Found BERT experiment: {exp}")

    return experiments


class Dataset:
    def __init__(self, name: str, path: str, classes_path: str, splits: list[str]) -> None:
        self.name = name
        self.path = path
        self.classes_path = classes_path
        self.splits = splits
        
        self.size_by_split = {split: 0 for split in splits}  
        self.classes = pd.read_csv(classes_path, header=None).iloc[0].to_list()
            
        self.labels_by_split = {split: [] for split in splits}
        self.class_size_by_split = {split: {clss: 0 for clss in self.classes} for split in splits}
        self.largest_class_size_by_split = {split: 0 for split in splits}
        
        for split in splits:
            split_path = self.get_split_path(split)
            split_df = pd.read_csv(split_path)
            self.size_by_split[split] = len(split_df)
            
            
            labels_arr = split_df.iloc[:, -1].to_numpy(dtype=int)
            adjusted = False
            counts = split_df.iloc[:, -1].value_counts()
            if 0 not in counts.index:
                labels_arr -= 1  # Adjust labels to be zero-indexed if necessary (EMNIST case)
                adjusted = True
                # print(f"Adjusted labels for dataset {self.name}, split {split} to be zero-indexed.")
            
            self.labels_by_split[split] = labels_arr.astype(int).tolist()
            max_size = 0
            for label, count in counts.items():
                lab = int(label) # type: ignore
                if adjusted:
                    lab -= 1
                self.class_size_by_split[split][self.classes[lab]] = count
                max_size = max(max_size, count)
            self.largest_class_size_by_split[split] = max_size
                    
    def get_split_path(self, split: str) -> str:
        return os.path.join(self.path, f"{split}.csv")
    
# class Model:
#     def __init__(self, name: str) -> None:
#         self.name = name

class LossLandscapeExperiment:
    is_bert = False

    def __init__(self, dataset: Dataset, split: str, model: str, k: int, epoch: int, layer: str, landscape_dir: str, ct_dir: str) -> None:
        self.dataset = dataset
        self.split = split
        self.model = model
        self.layer = layer
        self.epoch = epoch
        self.k = k
        self.paths = None

        self.layer_tag = f"{layer}"
        self.pretty_layer = self.layer_tag[1:]
        self.epoch_tag = f"e{epoch}"
        self.model_data = f"{self.model}_{self.dataset.name}"

        if self.model.startswith("resnet") and (self.model.endswith("_a") or self.model.endswith("_b") or self.model.endswith("_c")):
            tag = self.model.split("_")[-1]
            self.model_data = f"resnet_{self.dataset.name}_{tag}"
        else:
            self.model_data = f"{self.model}_{self.dataset.name}" if not st.session_state.model_eq_dataset else self.dataset.name
        
        self.landscape_dir = landscape_dir
        self.ct_dir = ct_dir
        
    def __hash__(self) -> int:
        return hash((self.dataset.name, self.split, self.model, self.k, self.epoch, self.layer))

    def __repr__(self) -> str:
        return f"{self.model} ({self.dataset.name}-{self.split}), k={self.k}, layer={self.pretty_layer}, epoch={self.epoch}"
    
    def get_paths(self) -> dict:
        data_dir = self.landscape_dir
        ct_dir = self.ct_dir
        
        self.paths = {
            "ctree": os.path.join(ct_dir, f"{self.model_data}", self.split, f"ctree_{self.layer_tag}_{self.epoch_tag}_{self.k}"),
            "losses": os.path.join(data_dir, f"{self.model_data}", "Losses", self.split, f"losses_{self.epoch_tag}.pt"),
            "tensors": os.path.join(data_dir, f"{self.model_data}", "Tensors", self.split, f"{self.layer_tag}_{self.epoch_tag}.pt"),
            "predictions": os.path.join(data_dir, f"{self.model_data}", "Predictions", self.split, f"predictions_{self.epoch_tag}.pt"),
            "compiled_res": os.path.join(data_dir, f"{self.model_data}", "compiled_results.csv"),
        }
        
        return self.paths 
    
    def validate_paths(self) -> bool:
        paths = self.get_paths()

        for key, path in paths.items():
            if key in ["tensors", "predictions"]:
                continue
            
            if key == "ctree":
                paths_should_exist = [f"{path}.{ext}" for ext in ["order.dat", "order.bin", "part.raw", "rg.bin", "rg.dat"]]

                for p in paths_should_exist:
                    if not os.path.exists(p):
                        print(f"Path for {key} does not exist: {p}")
                        return False
                    
                continue

            if not os.path.exists(path):
                print(f"Path for {key} does not exist: {path}")
                            
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

            # print(f"Found dataset: {dataset_name} with splits: {splits}")
            dataset = Dataset(dataset_name, dataset_path, classes_path, splits)
            datasets[dataset_name] = dataset

            # for r in ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0"]:
            #     random_name = f"{dataset_name}-r{r}"
            #     ds = Dataset(random_name, dataset_path, classes_path, splits)
            #     datasets[random_name] = ds  # Alias for random splits

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
        parts = model_data.split("_")
        model = parts[0]
        dataset_name = "_".join(parts[1:])
        
        if dataset_name.startswith("mnist") and len(parts) > 2:
            dataset_name = "mnist"
            model = model + "_" + "_".join(parts[2:])
        
        if dataset_name not in datasets:
            print(f"Warning: Dataset {dataset_name} not found for model {model}")
            continue

        dataset = datasets[dataset_name]
        if split not in dataset.splits:
            print(f"Warning: Split {split} not found for dataset {dataset_name}")
            continue

        for tree_file in tree_files:
            _, layer, epoch, k = tree_file.split("_")

            epoch = int(epoch[1:])
            k = int(k.split(os.path.extsep)[0])
            
            experiment = LossLandscapeExperiment(dataset, split, model, k, epoch, layer, data_dir, ct_dir)

            if experiment.validate_paths():
                experiments.append(experiment)

    return experiments