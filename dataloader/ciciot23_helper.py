import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ── Remap nhan cho bo data 100-client ─────────────────────────────────────────
# Bo 100-client giu NGUYEN label ID goc cua CIC-IoT23 (preserve_original_label_ids)
# voi thu tu task phi tuan tu, mo ta trong `task_mapping_label_ids.json`:
#     Task 1: [1, 0, 11, 12, 27, 26]   Task 2: [2, 14, 25, 24, 20, 28] ...
# Trong khi code CIL gia dinh label tuan tu: task 1 = [0..5], task 2 = [6..11], ...
# Bo data cu (da tuan tu san) khong co file json nay -> khong doi gi, tuong thich nguoc.
_LABEL_LUT = None
_LABEL_LUT_READY = False


def _get_label_lut(data_root=None):
    global _LABEL_LUT, _LABEL_LUT_READY
    if _LABEL_LUT_READY:
        return _LABEL_LUT
    _LABEL_LUT_READY = True

    candidates = []
    if data_root:
        candidates += [
            os.path.join(data_root, "task_mapping_label_ids.json"),
            os.path.join(os.path.dirname(data_root), "task_mapping_label_ids.json"),
        ]
    if os.path.exists("/kaggle/input"):
        candidates += sorted(glob.glob("/kaggle/input/**/task_mapping_label_ids.json",
                                       recursive=True))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "task_mapping_label_ids.json"))

    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r") as f:
                task_orders = json.load(f)
            flat = [int(c) for task in task_orders for c in task]
            if sorted(flat) != list(range(len(flat))):
                print(f"[Ciciot23_helper] CANH BAO: {path} khong phu kin 0..N-1, bo qua remap.")
                continue
            lut = torch.full((max(flat) + 1,), -1, dtype=torch.long)
            for seq_id, orig_id in enumerate(flat):
                lut[orig_id] = seq_id
            _LABEL_LUT = lut
            print(f"[Ciciot23_helper] Remap label goc -> tuan tu theo: {path}")
            print(f"[Ciciot23_helper] Thu tu task (label goc): {task_orders}")
            return _LABEL_LUT

    print("[Ciciot23_helper] Khong thay task_mapping_label_ids.json -> gia dinh label da tuan tu.")
    return None


def _remap_labels(y, data_root=None):
    """Ap LUT remap cho tensor label y. No-op neu data da tuan tu."""
    lut = _get_label_lut(data_root)
    if lut is None or y is None or len(y) == 0:
        return y
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    out = lut[y.long()]
    if (out < 0).any():
        bad = torch.unique(y[out < 0]).tolist()
        raise ValueError(f"[Ciciot23_helper] Label {bad} khong co trong task_mapping_label_ids.json")
    return out

class CICIoT23Dataset(Dataset):
    def __init__(self, x, y):
        if isinstance(x, np.ndarray):
            self.x = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            self.x = x.float()
        else:
            self.x = torch.tensor(x, dtype=torch.float32)

        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).long()
        elif isinstance(y, torch.Tensor):
            self.y = y.long()
        else:
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return tuple to match F2SCIL expectations
        return self.x[idx], self.y[idx]
    
    @property
    def targets(self):
        return self.y.numpy()

class Ciciot23_helper:
    def __init__(self, args, data_root="C:/FederatedLearning/FL/core/data_split"):
        self.args = args
        self.data_root = data_root

        # Ho tro ca 2 layout:
        #   <root>/federated_data/client_*.pt   (tren may)
        #   <root>/client_*.pt                  (layout PHANG cua Kaggle dataset)
        self.federated_dir = os.path.join(data_root, "federated_data")
        if not os.path.isdir(self.federated_dir):
            self.federated_dir = data_root
            print(f"[Ciciot23_helper] Layout phang -> doc client tu: {self.federated_dir}")

        self.global_test_file = os.path.join(data_root, "global_test_data.pt")
        if not os.path.exists(self.global_test_file) and os.path.exists("/kaggle/input"):
            hits = glob.glob("/kaggle/input/**/global_test_data.pt", recursive=True)
            if hits:
                self.global_test_file = hits[0]
                print(f"[Ciciot23_helper] Auto-detect test file: {self.global_test_file}")

        print("[Ciciot23_helper] Loading global test data...")
        test_dict = torch.load(self.global_test_file, map_location="cpu", weights_only=False)
        self.test_x = test_dict["x"].float()
        self.test_y = _remap_labels(test_dict["y"].long(), data_root)
        print(f"[Ciciot23_helper] Loaded global test set: {self.test_x.shape[0]} samples")

    def get_test_dataset(self, seen_classes, max_samples_per_class=None):
        """Filters global test data for only the seen classes up to the current task"""
        x_filtered_list = []
        y_filtered_list = []
        
        for c in seen_classes:
            class_mask = (self.test_y == c)
            x_c = self.test_x[class_mask]
            y_c = self.test_y[class_mask]
            
            if len(x_c) > 0:
                if max_samples_per_class is not None and len(x_c) > max_samples_per_class:
                    indices = np.random.choice(len(x_c), max_samples_per_class, replace=False)
                    x_filtered_list.append(x_c[indices])
                    y_filtered_list.append(y_c[indices])
                else:
                    x_filtered_list.append(x_c)
                    y_filtered_list.append(y_c)
                    
        if len(x_filtered_list) == 0:
            return CICIoT23Dataset(torch.empty(0, self.test_x.shape[1]), torch.empty(0, dtype=torch.long))
            
        x_filtered = torch.cat(x_filtered_list, dim=0)
        y_filtered = torch.cat(y_filtered_list, dim=0)
        return CICIoT23Dataset(x_filtered, y_filtered)

    def get_client_train_dataset(self, task, client_idx):
        """Loads data from federated_data folder. Task is 0-indexed in F2SCIL, but 1-indexed in federated_data files"""
        task_id_file = task + 1
        path = os.path.join(self.federated_dir, f"client_{client_idx}_task_{task_id_file}.pt")
        if not os.path.exists(path):
            # Client might not have data for this task
            return None
        
        data = torch.load(path, map_location="cpu", weights_only=False)
        return CICIoT23Dataset(data["x"], _remap_labels(data["y"].long(), self.data_root))

    def get_global_train_dataset(self, task):
        """Combines all clients data for the current task to form a global trainset for the server"""
        task_id_file = task + 1
        x_all, y_all = [], []
        for client_idx in range(self.args["num_users"]):
            path = os.path.join(self.federated_dir, f"client_{client_idx}_task_{task_id_file}.pt")
            if os.path.exists(path):
                data = torch.load(path, map_location="cpu", weights_only=False)
                x_all.append(data["x"])
                y_all.append(_remap_labels(data["y"].long(), self.data_root))
        if len(x_all) == 0:
            return None
        
        x_combined = torch.cat(x_all, dim=0)
        y_combined = torch.cat(y_all, dim=0)
        return CICIoT23Dataset(x_combined, y_combined)
