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

# ── Bo du lieu nao dang chay ─────────────────────────────────────────────────
# CIC-IoT23 100-client giu label ID GOC voi thu tu task phi tuan tu -> phai remap.
# CAN-bus/IoV 100-client da tuan tu 0..12 san -> KHONG duoc remap.
#
# Phai co cong tac nay vi _get_label_lut() tim file json o hai cho nguy hiem:
#   1. glob "/kaggle/input/**" — vo phai bang cua IoT neu dataset IoT con gan kem;
#   2. ban du phong nhung san trong repo (dataloader/task_mapping_label_ids.json)
#      — luon ton tai, nen KHONG gan kem IoT cung van vo phai.
# Ca hai deu se anh xa nham nhan IoV ma khong bao loi gi.
CAU_HINH_BO = {
    "ciciot23": {"remap_nhan": True,  "num_class": 34, "tasks": 6, "base_class": 6},
    "can_iov":  {"remap_nhan": False, "num_class": 13, "tasks": 5, "base_class": 3},
}
BO_HIEN_TAI = "ciciot23"          # mac dinh, giu tuong thich nguoc


def set_dataset(ten):
    """Chon bo du lieu. Goi TRUOC khi tao Ciciot23_helper."""
    global BO_HIEN_TAI, _LABEL_LUT, _LABEL_LUT_READY
    if ten not in CAU_HINH_BO:
        raise SystemExit(f"[Ciciot23_helper] Dataset khong ho tro: {ten}. "
                         f"Chi co: {list(CAU_HINH_BO)}")
    BO_HIEN_TAI = ten
    _LABEL_LUT, _LABEL_LUT_READY = None, False    # xoa cache LUT cu
    cfg = CAU_HINH_BO[ten]
    print(f"[Ciciot23_helper] set_dataset({ten}): {cfg['num_class']} lop, "
          f"{cfg['tasks']} task, base {cfg['base_class']}, "
          f"remap nhan: {'CO' if cfg['remap_nhan'] else 'KHONG'}")


def _get_label_lut(data_root=None):
    global _LABEL_LUT, _LABEL_LUT_READY
    if _LABEL_LUT_READY:
        return _LABEL_LUT
    _LABEL_LUT_READY = True

    # Bo khong can remap thi dung han o day — KHONG duoc roi xuong phan tim file
    # ben duoi, vi no se vo phai bang cua bo khac (xem ghi chu o CAU_HINH_BO).
    if not CAU_HINH_BO[BO_HIEN_TAI]["remap_nhan"]:
        print(f"[Ciciot23_helper] Bo '{BO_HIEN_TAI}' co nhan tuan tu san "
              f"-> KHONG remap (bo qua moi task_mapping_label_ids.json).")
        return None

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
        if not os.path.isdir(data_root):
            raise SystemExit(
                f"[Ciciot23_helper] --data_dir khong ton tai: {data_root}\n"
                f"  Chua gan dataset, hoac go sai duong dan. Dung de code tu di dò\n"
                f"  roi vo phai du lieu cua bo khac.")

        self.federated_dir = os.path.join(data_root, "federated_data")
        if not os.path.isdir(self.federated_dir):
            self.federated_dir = data_root
            print(f"[Ciciot23_helper] Layout phang -> doc client tu: {self.federated_dir}")
            _n_shard = len(glob.glob(os.path.join(data_root, "client_*_task_*.pt")))
            if _n_shard == 0:
                raise SystemExit(
                    f"[Ciciot23_helper] {data_root} khong co thu muc federated_data/ "
                    f"lan khong co file client_*_task_*.pt nao.\n"
                    f"  Kiem tra lai --data_dir.")
            print(f"[Ciciot23_helper] Tim thay {_n_shard} shard o layout phang.")

        # ── Kich ban FEW-SHOT ────────────────────────────────────────────────
        # Neu dat --fewshot_dir, moi task >= 1 doc tu thu muc do thay vi
        # federated_dir. Task 0 (base) LUON dung full data — hai bo few-shot
        # co tinh khong co file task_1, dung quy uoc voi HFIN va AFSIC-IDS.
        self.fewshot_dir = (args.get("fewshot_dir") or "").strip() or None
        if self.fewshot_dir:
            if not os.path.isdir(self.fewshot_dir):
                raise FileNotFoundError(
                    f"[Ciciot23_helper] Khong thay thu muc few-shot: {self.fewshot_dir}")
            n = len(glob.glob(os.path.join(self.fewshot_dir, "*.pt")))
            print(f"[Ciciot23_helper] FEW-SHOT tu task 1 tro di: "
                  f"{self.fewshot_dir} ({n} file .pt)")
        else:
            print("[Ciciot23_helper] FULL data cho moi task")

        self.global_test_file = os.path.join(data_root, "global_test_data.pt")
        if not os.path.exists(self.global_test_file) and os.path.exists("/kaggle/input"):
            # Ban cu lay hits[0] — file DAU TIEN tim thay, bat ke thuoc bo nao.
            # Da tung vo phai global_test_data.pt cua IoT (34 lop) khi chay IoV.
            # Nay chi nhan file co SO LOP khop bo dang chay.
            _ky_vong = CAU_HINH_BO[BO_HIEN_TAI]["num_class"]
            hits = sorted(glob.glob("/kaggle/input/**/global_test_data.pt", recursive=True))
            hop_le, bo_qua = None, []
            for h in hits:
                try:
                    _y = torch.load(h, map_location="cpu", weights_only=False)["y"]
                    _n = int(_y.max()) + 1
                except Exception as e:
                    bo_qua.append(f"{h} (khong doc duoc: {e})")
                    continue
                if _n == _ky_vong:
                    hop_le = h
                    break
                bo_qua.append(f"{h} ({_n} lop)")
            for b in bo_qua:
                print(f"[Ciciot23_helper] Bo qua {b}")
            if hop_le is None:
                raise SystemExit(
                    f"[Ciciot23_helper] Khong thay global_test_data.pt nao co "
                    f"{_ky_vong} lop cho bo '{BO_HIEN_TAI}'.\n"
                    f"  --data_dir tro toi: {data_root} (ton tai: {os.path.isdir(data_root)})\n"
                    f"  Da quet {len(hits)} file trong /kaggle/input.\n"
                    f"  Kiem tra da gan dung dataset chua.")
            self.global_test_file = hop_le
            print(f"[Ciciot23_helper] Auto-detect test file: {self.global_test_file}")

        print("[Ciciot23_helper] Loading global test data...")
        test_dict = torch.load(self.global_test_file, map_location="cpu", weights_only=False)
        self.test_x = test_dict["x"].float()
        self.test_y = _remap_labels(test_dict["y"].long(), data_root)
        print(f"[Ciciot23_helper] Loaded global test set: {self.test_x.shape[0]} samples")

        # Doi chieu voi cau hinh bo dang chay. Can thiet vi dong glob o tren lay
        # global_test_data.pt DAU TIEN tim thay trong /kaggle/input — neu ca hai
        # dataset IoT va IoV cung duoc gan, no co the vo phai file cua bo kia.
        _n_thuc = int(self.test_y.max()) + 1
        _n_ky_vong = CAU_HINH_BO[BO_HIEN_TAI]["num_class"]
        if _n_thuc != _n_ky_vong:
            raise SystemExit(
                f"[Ciciot23_helper] {self.global_test_file} co {_n_thuc} lop nhung bo "
                f"'{BO_HIEN_TAI}' can {_n_ky_vong}. Gan nham dataset, hoac --data_dir sai.")
        print(f"[Ciciot23_helper] Da doi chieu: {_n_thuc} lop, "
              f"{self.test_x.shape[1]} dac trung — khop bo '{BO_HIEN_TAI}'.")

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

    def _task_dir(self, task):
        """Thu muc chua du lieu cua task nay.

        Task 0 luon lay tu federated_dir (full data); tu task 1 tro di, neu bat
        few-shot thi lay tu fewshot_dir. KHONG fallback ve full khi thieu file —
        fallback se am tham dung full data va lam hong ca thi nghiem.
        """
        if self.fewshot_dir and task > 0:
            return self.fewshot_dir
        return self.federated_dir

    def get_client_train_dataset(self, task, client_idx):
        """Loads data from federated_data folder. Task is 0-indexed in F2SCIL, but 1-indexed in federated_data files"""
        task_id_file = task + 1
        path = os.path.join(self._task_dir(task), f"client_{client_idx}_task_{task_id_file}.pt")
        if not os.path.exists(path):
            # Client might not have data for this task
            return None

        data = torch.load(path, map_location="cpu", weights_only=False)
        return CICIoT23Dataset(data["x"], _remap_labels(data["y"].long(), self.data_root))

    def get_global_train_dataset(self, task):
        """Combines all clients data for the current task to form a global trainset for the server"""
        task_id_file = task + 1
        task_dir = self._task_dir(task)
        x_all, y_all = [], []
        for client_idx in range(self.args["num_users"]):
            path = os.path.join(task_dir, f"client_{client_idx}_task_{task_id_file}.pt")
            if os.path.exists(path):
                data = torch.load(path, map_location="cpu", weights_only=False)
                x_all.append(data["x"])
                y_all.append(_remap_labels(data["y"].long(), self.data_root))
        if len(x_all) == 0:
            return None
        
        x_combined = torch.cat(x_all, dim=0)
        y_combined = torch.cat(y_all, dim=0)
        return CICIoT23Dataset(x_combined, y_combined)
