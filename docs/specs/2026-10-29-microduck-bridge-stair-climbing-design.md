# Thiết Kế Kiến Trúc: Hệ Thống Leo Bậc Thang Cầu Vồng Không Lan Can (MicroDuck 8-DOF)
**Ngày lập:** 2026-10-29  
**Trạng thái:** Chờ duyệt đặc tả kỹ thuật (Pending Spec Approval)  
**Tác giả:** Antigravity Pair Programming Architecture  
**Phạm vi:** Địa hình 3D MuJoCo, Bộ quản lý Curriculum tham số, Hàm phần thưởng chống rơi mép & vấp bậc, Huấn luyện PPO và Xuất mô hình Sim-to-Real.

---

## 1. Bối Cảnh & Mục Tiêu Kỹ Thuật (Background & Technical Goals)

### 1.1. Thách Thức Địa Hình Nâng Cao
Sau khi robot MicroDuck 8-DOF đã làm chủ địa hình mặt phẳng và vượt chướng ngại vật rào chắn thấp ở Stage 3 (tại `microduck_ppo.pt`), bài toán nâng cao đòi hỏi robot phải chinh phục **địa hình cầu thang hai chiều không có lan can (Two-Way Bridge Staircase without Railings)**:
1. **Leo lên (Ascent):** Vượt 3 bậc thang nâng dần độ cao, yêu cầu khớp gối gập sâu và đưa đùi dứt khoát để tránh vấp mũi chân vào thành đứng (toe stubbing).
2. **Đi qua đỉnh (Top Platform):** Di chuyển trên bục phẳng trên cao, ổn định thăng bằng trọng tâm CoM.
3. **Bước xuống (Descent):** Bước xuống 3 bậc thang dốc mà không bị chúi đầu hay mất kiểm soát quán tính ngã chúi về phía trước.
4. **Không có lan can (No Railings):** Toàn bộ nhịp cầu thang không có thành chắn hai bên. Bề ngang bị thu hẹp dần theo từng cấp độ ($500\text{ mm} \to 300\text{ mm}$). Bước chệch tim đường sẽ rơi tự do khỏi mép bậc thang xuống sàn đáy.

### 1.2. Mục Tiêu Cốt Lõi
- **Bảo toàn giao thức Sim-to-Real:** Giữ nguyên không gian quan sát 42 chiều (`OBS_DIM = 42`) và không gian hành động 8 chiều (`ACTION_DIM = 8`) để mô hình sau huấn luyện có thể xuất trực tiếp sang Float32 ONNX (Raspberry Pi 5) và INT8 C Header (Dual ESP32-S3) mà không cần chỉnh sửa firmware vi điều khiển.
- **Tối ưu tốc độ huấn luyện:** Tích hợp trực tiếp vào `ParallelSimRunner` qua cơ chế cập nhật tham số in-place trên MuJoCo CPU (`apply_stage()`), duy trì tốc độ mô phỏng $\ge 2.000\text{ FPS}$.

---

## 2. Kiến Trúc Địa Hình Cầu Thang Cầu Vồng Trong MuJoCo

### 2.1. Cấu Trúc Hình Học Trục Dọc ($X$-axis Profile)
Robot xuất phát tại $X = -0.40\text{ m}, Y = 0.0\text{ m}, Z = 0.170\text{ m}$.
Nhịp cầu thang gồm 7 khối geom hộp chữ nhật (`geom type="box"`):

```text
       Z (Độ cao)
       ▲
  3h ──┼                      ┌───────────────┐ (Bục đỉnh)
       │                      │  top_platform │
  2h ──┼              ┌───────┘               └───────┐
       │              │ stair_up_2         stair_dn_1 │
   h ──┼      ┌───────┘                               └───────┐
       │      │ stair_up_1                         stair_dn_2 │
 0.0 ──┴──────┴───────────────────────────────────────────────┴────────► X
     Spawn   [0.08m  0.20m]  [0.20m  0.32m]  [0.32m  0.44m]  [0.44m 0.74m]  [0.74m 0.86m]  [0.86m 0.98m]  [>0.98m]
             (Bậc lên 1)     (Bậc lên 2)     (Bậc lên 3)     (Bục đỉnh)      (Bậc xuống 1)   (Bậc xuống 2)  (Sàn hạ cánh)
```

| Tên Geom | Kiểu | Tọa độ $X$ | Chiều sâu $\Delta X$ | Độ cao bề mặt $Z_{top}$ | Chiều rộng $Y$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `stair_up_1` | `box` | $[0.08, 0.20]\text{ m}$ | $120\text{ mm}$ | $h$ | $W$ (Không lan can) |
| `stair_up_2` | `box` | $[0.20, 0.32]\text{ m}$ | $120\text{ mm}$ | $2h$ | $W$ (Không lan can) |
| `stair_up_3` | `box` | $[0.32, 0.44]\text{ m}$ | $120\text{ mm}$ | $3h$ | $W$ (Không lan can) |
| `top_platform` | `box` | $[0.44, 0.74]\text{ m}$ | $300\text{ mm}$ | $3h$ | $W$ (Không lan can) |
| `stair_dn_1` | `box` | $[0.74, 0.86]\text{ m}$ | $120\text{ mm}$ | $2h$ | $W$ (Không lan can) |
| `stair_dn_2` | `box` | $[0.86, 0.98]\text{ m}$ | $120\text{ mm}$ | $h$ | $W$ (Không lan can) |
| `floor_exit` | `plane` | $[0.98, 3.00]\text{ m}$ | Sàn phẳng | $0.0\text{ m}$ | Sàn tiếp đất |

### 2.2. Cơ Chế Phát Hiện Rơi Mép Vực (Edge-Fall Termination)
Vì hai bên mép cầu thang hoàn toàn không có lan can:
- **Biên độ an toàn trục ngang:** $Y \in [-W/2, +W/2]$.
- **Điều kiện rơi mép (Edge Fall Trigger):**
  Khi robot đang trong vùng cầu thang ($X \in [0.08\text{ m}, 1.00\text{ m}]$):
  $$\text{Fall Condition} = (|Y_{base}| > W/2 + 0.04) \lor (Z_{base} < Z_{expected\_stair}(X) - 0.08)$$
- Khi thỏa mãn điều kiện rơi mép:
  1. Đánh dấu `terminated = True` (ngắt ngay episode).
  2. Phạt điểm cực nặng: $R_{fall} = -5.0$.

---

## 3. Lộ Trình Curriculum 4 Giai Đoạn (4-Stage Progression)

Chuyển đổi tham số $h$ (chiều cao bậc) và $W$ (chiều rộng mép) linh hoạt qua hàm `apply_stage(model, stage)`:

| Giai đoạn | Chiều cao bậc $h$ | Chiều rộng $W$ | Tổng độ cao đỉnh $3h$ | Điều kiện thăng cấp (Giữ vững 10 vòng liên tiếp) |
| :---: | :---: | :---: | :---: | :--- |
| **Stage 0: Khởi động** | $10\text{ mm}$ | $500\text{ mm}$ | $30\text{ mm}$ | $\bar{L}_{ep} \ge 380$ bước & Tỷ lệ $V_x \ge 70\%$ mục tiêu |
| **Stage 1: Nâng cao** | $18\text{ mm}$ | $420\text{ mm}$ | $54\text{ mm}$ | $\bar{L}_{ep} \ge 400$ bước & Vượt trọn vẹn nhịp lên |
| **Stage 2: Chuẩn 2 chiều** | $25\text{ mm}$ | $360\text{ mm}$ | $75\text{ mm}$ | $\bar{L}_{ep} \ge 420$ bước & Leo đỉnh $\to$ bước xuống an toàn |
| **Stage 3: Cực hạn không lan can**| **$32\text{ mm}$** | **$300\text{ mm}$** | **$96\text{ mm}$** | Duy trì hội tụ, hoàn tất toàn bộ chu trình cầu vượt |

---

## 4. Thiết Kế Không Gian Nhận Thức & Hàm Phần Thưởng

### 4.1. Không Gian Quan Sát (Observation Vector 42 Chiều)
Giữ nguyên kiến trúc chuẩn hóa của MicroDuck:
1. **IMU Vận tốc góc:** `base_ang_vel` (3 chiều) $\to$ Phát hiện độ chúi/nghiêng khi lên/xuống dốc.
2. **Hình chiếu trọng lực:** `proj_grav` (3 chiều) $\to$ Vector trọng trường trong hệ quy chiếu thân.
3. **Lệnh vận tốc:** `cmd_vel = [0.25, 0.0, 0.0]` (3 chiều).
4. **Sai số góc khớp:** `joint_pos - default_pos` (8 chiều).
5. **Vận tốc khớp:** `joint_vel` (8 chiều).
6. **Hành động vòng trước:** `last_actions` (8 chiều).
7. **Tiếp xúc bàn chân:** `foot_contacts` (4 chiều: mũi/gót trái, mũi/gót phải) $\to$ Xác định trạng thái đáp bậc.
8. **LiDAR ToF 5 tia:** `lidar_ranges` (5 chiều, góc quét $[-30^\circ, -15^\circ, 0^\circ, +15^\circ, +30^\circ]$) $\to$ Đo khoảng cách tới thành đứng bậc thang kế tiếp.

### 4.2. Cấu Trúc Hàm Phần Thưởng Đa Mục Tiêu (Multi-Objective Reward)

$$R_{total} = R_{tracking\_vx} + R_{centering} + R_{clearance} + R_{foot\_flat} + R_{posture} - P_{fall} - P_{stub} - P_{torque}$$

1. **Thưởng tiến độ trục $X$:**
   $$R_{tracking\_vx} = 1.5 \cdot \exp\left(-\frac{(V_x - V_{cmd})^2}{0.25}\right)$$
2. **Phạt lệch tâm cầu thang (Centering Penalty - Cực kỳ quan trọng để không rơi mép):**
   $$R_{centering} = -2.5 \cdot |Y_{base}| - 1.0 \cdot |V_y|$$
3. **Thưởng nhấc cao chân (Foot Clearance):**
   Khi chân ở pha vung (swing phase), thưởng độ cao nhấc bàn chân để mũi chân không quét vào góc bậc:
   $$R_{clearance} = 0.5 \cdot \min(Z_{swing\_foot} - Z_{step\_edge}, 0.04)$$
4. **Phạt vấp mũi chân (Anti-Stubbing):**
   Phạt phản lực tiếp xúc theo phương ngang $F_{contact, x}$ khi mũi chân đâm vào vách bậc thang:
   $$P_{stub} = 0.05 \cdot \|F_{horizontal}\|$$
5. **Thưởng tiếp xúc phẳng mặt bậc (Flat Landing):**
   Thưởng khi cả 2 điểm tiếp xúc trước/sau của bàn chân cùng chạm mặt phẳng bậc thang ($contact_{front} \land contact_{back}$).
6. **Phạt rơi mép vực:**
   $$P_{fall} = 5.0 \quad (\text{khi kích hoạt Fall Condition})$$

---

## 5. Kế Hoạch Triển Khai & Kiểm Chứng (Testing & Verification)

### 5.1. Bộ Kiểm Thử Tự Động (Automated Test Suite)
1. `tests/test_stair_geometry.py`: Kiểm tra định dạng XML, các geom bậc thang, độ khít bề mặt tiếp xúc, không trùng lấn va chạm ảo.
2. `tests/test_stair_curriculum.py`: Kiểm tra hàm `apply_stage()` thay đổi đúng chiều cao $h$ và chiều rộng $W$ qua 4 Stage; kiểm tra logic thăng cấp `next_stage()`.
3. `tests/test_edge_fall_detection.py`: Mô phỏng robot bị đẩy sang ngang mép bậc $\to$ kiểm tra kích hoạt `terminated=True` và áp dụng đúng điểm phạt $R_{fall}$.
4. `tests/test_stair_ppo_training.py`: Chạy thử nghiệm PPO 1 iteration kiểm tra tính toán tensor, không bị lỗi shape mismatch.

### 5.2. Công Cụ Trực Quan Hóa 3D
Mở rộng [scripts/interactive_viewer.py](file:///E:/duckV1/scripts/interactive_viewer.py) hỗ trợ cờ `--stairs` và `--stage [0, 1, 2, 3]` để người dùng trực tiếp quan sát robot leo nhịp cầu thang trong không gian 3D MuJoCo.

---

## 6. Tiêu Chí Nghiệm Thu (Acceptance Criteria)
- [ ] Mô hình MuJoCo chứa đầy đủ 7 khối địa hình bậc thang cầu vồng không lan can.
- [ ] Hàm `apply_stage` thay đổi linh hoạt kích thước bậc thang theo đúng 4 Stage.
- [ ] Cơ chế rơi mép vực hoạt động chính xác khi robot bước chệch $|Y| > W/2$.
- [ ] 100% các file unit test mới và cũ đều PASS.
- [ ] Script huấn luyện `scripts/train_ppo.py` tương thích hoàn toàn với địa hình cầu thang mới.

