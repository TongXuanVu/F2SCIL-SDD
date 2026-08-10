# Báo Cáo Phân Tích & Cấu Hình Resume F2SCIL (1% và 10shot)

Tài liệu này tổng hợp toàn bộ các bước phân tích, dọn dẹp hệ thống và cấu hình script để resume tiếp tục hai tiến trình Federated Learning `1%` và `10shot` trên nền tảng Kaggle.

---

## 1. Phân Tích Tình Trạng Log (Round 90)
Dựa vào các file log (`.txt` và `metrics_f2scil.csv`) của 2 tiến trình, chúng tôi đã rút ra được tình trạng chính xác như sau:

* **Tiến trình 1% (`100client_fewshot1`)**:
  * Đã chạy **hoàn tất Task 2** (30 rounds, từ round 61 đến round 90).
  * Đã thực hiện xong các bước hậu kỳ của Task 2: sinh dữ liệu tổng hợp (synthesis) và fine-tuning mô hình.
  * Đã tạo thành công mô hình Teacher `ciciot23_session_2.pth`.
  * Sẵn sàng chuyển sang Task 3 (bắt đầu từ round 91).
  
* **Tiến trình 10shot (`100client_10shot`)**:
  * Đã huấn luyện xong **Round 90** (Round cuối cùng của Task 2) và tính toán xong loss/accuracy.
  * Đã lưu `checkpoint_round_90.pth`.
  * **Sự cố**: Tiến trình bị ngắt đột ngột trong giai đoạn *Synthesis* (sinh dữ liệu tổng hợp) của Task 2. Do đó, bước fine-tuning chưa được diễn ra và mô hình `ciciot23_session_2.pth` **chưa** được tạo.
  * **Giải pháp**: Resume lại từ **Round 90** để hệ thống chạy lướt lại round này và tính toán nốt bước Synthesis/Fine-tuning đang bị dở dang.

---

## 2. Dọn Dẹp và Tổ Chức Lại GitHub
Để chuẩn bị cho việc resume gọn gàng và không gây nặng Git, các thao tác sau đã được thực hiện bằng lệnh tự động hóa:

1. **Xóa các Checkpoint/Session rác của quá trình cũ (Full Data)**:
   - Đã xóa `checkpoint_round_150.pth`, `checkpoint_round_180.pth`.
   - Đã xóa các Teacher model thừa từ `ciciot23_session_1.pth` đến `ciciot23_session_5.pth` (thuộc thư mục `100client`).
   - Đã xóa hàng ngàn file dữ liệu tổng hợp từ Task 1 đến Task 5 của Full Data để giảm tải repo.

2. **Bảo tồn File Gốc Cần Thiết**:
   - Đặc biệt giữ lại `ciciot23_session_0.pth` (Teacher model khởi điểm).
   - Khôi phục thư mục `synthesis/task_0` (Dữ liệu tổng hợp từ gốc) do chúng là nền tảng bắt buộc để các tiến trình Fewshot kế thừa.

3. **Phân Luồng Thư Mục Resume Cụ Thể**:
   - Chuyển và thêm các checkpoint cần thiết vào từng thư mục riêng rẽ:
     - Thư mục `resume_checkpoints/100client_fewshot1/` chứa `session_2.pth`, `checkpoint_round_90.pth` cùng synthesis của task 1 & 2.
     - Thư mục `resume_checkpoints/100client_10shot/` chứa `session_1.pth`, `checkpoint_round_89.pth`, `checkpoint_round_90.pth` cùng synthesis task 1.
   - Commit và Push thẳng lên nhánh `main` của repository GitHub.

---

## 3. Kaggle Script (Template Code)

### A. Code Resume Cho Tiến Trình 10shot (Từ Round 90)
Sử dụng script sau để resume tiến trình 10shot. Script sẽ load checkpoint của round 89 để bắt đầu chạy lại round 90, hoàn tất nốt Task 2.

```python
!git clone -q https://github.com/TongXuanVu/F2SCIL-SDD.git /kaggle/working/f2scil
%cd /kaggle/working/f2scil
import os, glob, shutil

ROOT = "/kaggle/input/datasets/tongxuanvu/iot100client"
DATA = f"{ROOT}/100client"
FS1  = f"{ROOT}/iot100client_fewshot/federated_data_10shot"
SRC  = "resume_checkpoints/100client"
NEW  = "resume_checkpoints/100client_10shot"

os.makedirs("run/model", exist_ok=True)

# 1. Copy Teacher models 
shutil.copy(f"{SRC}/ciciot23_session_0.pth", "run/model/")
shutil.copy(f"{NEW}/ciciot23_session_1.pth", "run/model/")

# 2. Copy Checkpoint (Round 89, 90)
shutil.copy(f"{NEW}/checkpoint_round_89.pth", "run/model/")
shutil.copy(f"{NEW}/checkpoint_round_90.pth", "run/model/")

# 3. Copy Synthesis
shutil.copytree(f"{SRC}/synthesis/task_0", "run/synthesis/task_0", dirs_exist_ok=True)
shutil.copytree(f"{NEW}/synthesis/task_1", "run/synthesis/task_1", dirs_exist_ok=True)

assert len(glob.glob(f"{DATA}/client_*_task_*.pt")) == 430, "Sai thu muc full"
assert len(glob.glob(f"{FS1}/client_*_task_*.pt"))  == 383, "Sai thu muc few-shot"
print("OK — session 0,1 + checkpoint 89,90 + synthesis task 0,1 da san sang")

# Resume round 90
!python main.py --mode resume --resume_round 90 \
    --data_dir "{DATA}" --fewshot_dir "{FS1}" \
    --dataset ciciot23 --tasks 6 --num_class 34 --base_class 6 --incremental_class 10 \
    --num_users 100 --net resnet20 --seed 1008 --loss CE \
    --local_ep 1 --base_ep 100 --com_round 1 --inc_ep 30 \
    --syn_round 100 --syn_round2 100 \
    --local_bs 128 --syn_bs 25 --synthesis_batch_size 256 \
    --model_save_dir run/model
```

### B. Code Resume Cho Tiến Trình 1% (Từ Round 91)
Do 1% đã xử lý dứt điểm Task 2, script này sẽ load `session_2.pth` và checkpoint 90 để tiến thẳng vào Task 3 (Round 91).

```python
!git clone -q https://github.com/TongXuanVu/F2SCIL-SDD.git /kaggle/working/f2scil
%cd /kaggle/working/f2scil
import os, glob, shutil

ROOT = "/kaggle/input/datasets/tongxuanvu/iot100client"
DATA = f"{ROOT}/100client"
FS1  = f"{ROOT}/iot100client_fewshot/federated_data_fewshot"
SRC  = "resume_checkpoints/100client"
NEW  = "resume_checkpoints/100client_fewshot1"

os.makedirs("run/model", exist_ok=True)

# 1. Copy Teacher models 
shutil.copy(f"{SRC}/ciciot23_session_0.pth", "run/model/")
shutil.copy(f"{NEW}/ciciot23_session_1.pth", "run/model/")
shutil.copy(f"{NEW}/ciciot23_session_2.pth", "run/model/")

# 2. Copy Checkpoint (Round 90)
shutil.copy(f"{NEW}/checkpoint_round_90.pth", "run/model/")

# 3. Copy Synthesis
shutil.copytree(f"{SRC}/synthesis/task_0", "run/synthesis/task_0", dirs_exist_ok=True)
shutil.copytree(f"{NEW}/synthesis/task_1", "run/synthesis/task_1", dirs_exist_ok=True)
shutil.copytree(f"{NEW}/synthesis/task_2", "run/synthesis/task_2", dirs_exist_ok=True)

assert len(glob.glob(f"{DATA}/client_*_task_*.pt")) == 430, "Sai thu muc full"
assert len(glob.glob(f"{FS1}/client_*_task_*.pt"))  == 383, "Sai thu muc few-shot"
print("OK — session 0,1,2 + checkpoint 90 + synthesis task 0,1,2 da san sang")

# Resume round 91
!python main.py --mode resume --resume_round 91 \
    --data_dir "{DATA}" --fewshot_dir "{FS1}" \
    --dataset ciciot23 --tasks 6 --num_class 34 --base_class 6 --incremental_class 10 \
    --num_users 100 --net resnet20 --seed 1008 --loss CE \
    --local_ep 1 --base_ep 100 --com_round 1 --inc_ep 30 \
    --syn_round 100 --syn_round2 100 \
    --local_bs 128 --syn_bs 25 --synthesis_batch_size 256 \
    --model_save_dir run/model
```
