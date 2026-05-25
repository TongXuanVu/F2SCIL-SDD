import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

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
        self.federated_dir = os.path.join(data_root, "federated_data")
        self.global_test_file = os.path.join(data_root, "global_test_data.pt")
        
        print("[Ciciot23_helper] Loading global test data...")
        test_dict = torch.load(self.global_test_file, map_location="cpu", weights_only=False)
        self.test_x = test_dict["x"].float()
        self.test_y = test_dict["y"].long()
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
        return CICIoT23Dataset(data["x"], data["y"])

    def get_global_train_dataset(self, task):
        """Combines all clients data for the current task to form a global trainset for the server"""
        task_id_file = task + 1
        x_all, y_all = [], []
        for client_idx in range(self.args["num_users"]):
            path = os.path.join(self.federated_dir, f"client_{client_idx}_task_{task_id_file}.pt")
            if os.path.exists(path):
                data = torch.load(path, map_location="cpu", weights_only=False)
                x_all.append(data["x"])
                y_all.append(data["y"])
        if len(x_all) == 0:
            return None
        
        x_combined = torch.cat(x_all, dim=0)
        y_combined = torch.cat(y_all, dim=0)
        return CICIoT23Dataset(x_combined, y_combined)
