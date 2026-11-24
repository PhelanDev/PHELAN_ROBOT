# Đặc Tả Thiết Kế CyberDuck V2: Ưu Tiên Raspberry Pi 4B 4GB, Đầu OLED 2.42 inch, Cụm Trán Brow Pod và Ba Lô Kín Thân

**Ngày:** 2026-11-20  
**Nhánh làm việc:** `wip/repo-reorg`  
**Phạm vi:** Kiến trúc tổng thể gồm 2 đợt triển khai liên tiếp: Đợt 1 cơ khí CAD và Đợt 2 phần mềm, mô phỏng và firmware.

---

## 1. Bối cảnh và mục tiêu thiết kế

CyberDuck V2 hiện dùng dải 5 cảm biến ToF `VL53L0X` xếp quạt ±30 độ nối qua mạch chia I2C `TCA9548A` nằm sau mặt kính `visor_lens`, kết hợp hợp đồng `42` observation (`37` nội tại + `5` tia LiDAR) và ba lô lưng hở khe `8 mm` phía sau thân `trunk_front`. Kiến trúc mới giải quyết đồng thời bốn mục tiêu:

1. **Ưu tiên Raspberry Pi 4B 4GB làm bo mạch chính** để giảm chi phí phần cứng và giảm yêu cầu dòng nguồn từ `5V / 5A` xuống `5V / 3A`, vẫn giữ khay gá `85 x 56 mm` tương thích cơ khí thả vừa Pi 5 và duy trì 3 biến thể kiểm chứng CAD `rpi5`, `pizero2w`, `dual_esp32s3` để không làm vỡ hệ thống test đa biến thể.
2. **Thay cụm 5 ToF sau kính mặt bằng 1 màn hình OLED 2.42 inch `SSD1309`** (`75.0 x 43.0 mm`, `128 x 64`) để hiển thị mắt biểu cảm sắc nét xuyên kính khói.
3. **Gom 1 Camera và 1 cảm biến ToF đơn tia `VL53L0X` hoặc `VL53L1X` lên cụm trán `brow pod`** có cửa quang học hở riêng để đầu xoay `head_yaw` ±90 độ và gật `head_pitch` -30 đến +45 độ quét chủ động, đồng bộ hướng nhìn với mắt hiển thị.
4. **Đóng kín ba lô lưng ăn khớp phẳng với thân `trunk_front`** nhằm bỏ khe hở `8 mm` ở mặt hông và giấu hoàn toàn bo mạch xanh phía trong dưới trần khối lượng `1.50 kg`.

### Phạm vi không thay đổi

- Giữ nguyên động học 8-DOF và các giới hạn khớp tại `src/robot/robot_spec.py` và `src/cad/kinematics.py` (`head_yaw`, `head_pitch`, 2 chân 3-DOF mỗi bên; chưa thêm khớp `head_roll`).
- Giữ nguyên toàn bộ hình học đã nghiệm thu của cụm chân gồm `hip_clevis`, `thigh_shell`, `shin_shell`, `integrated_foot`, `sole_tread`, các nắp chụp `horn_hub_cap`, `idler_hub_cap` và hộp pin bụng kín `torso_lower`.
- Giữ nguyên vòng `reactor` trên mũi ngực `trunk_front`.

---

## 2. Cấu hình phần cứng ưu tiên Raspberry Pi 4B 4GB

### 2.1 Bo mạch chính và phân phối nguồn

| Thành phần | Quy cách chọn cho bản chính Pi 4B 4GB | Ghi chú tương thích |
|---|---|---|
| Bo xử lý chính | Raspberry Pi 4B 4GB (`85.0 x 56.0 x 1.6 mm`, lỗ ốc `58 x 49 mm`, M2.5, `46 g`) | Thêm `raspberry_pi_4b` vào `BOARDS` và gán vào `TRAY_LAYOUTS['rpi5']`; vẫn giữ `raspberry_pi_5` trong `BOARDS` |
| Tản nhiệt | Heatsink nhôm + quạt 5V kích thước gọn `30 x 30 x 12 mm` | Giữ tên linh kiện lắp ráp `active_cooler` để tương thích test ba lô |
| Nguồn logic | Buck DC-DC `5V / 4A đến 5A` (`buck_5v_5a`) tách khỏi UBEC servo | Dư tải cho Pi 4B (`~3A`) + USB camera + OLED 2.42 inch + ToF + mic |
| Nguồn động lực | Pin LiPo 2S + UBEC `6V / 15A` + tụ đệm `4700 uF` | Giữ nguyên kiến trúc hiện có |
| Hiển thị mắt | OLED 2.42 inch `SSD1309` (`128 x 64`, SPI 4-wire, `75.0 x 43.0 x 5.0 mm`, `~12 g`) | Chạy `30 đến 60 FPS` qua `SPI0` của Pi 4B |
| Camera trên đầu | Module USB camera `25 x 25 mm` dây silicon 4 lõi mềm (chuẩn UVC) | Khớp vật lý hốc `25 x 24 mm`, chịu xoắn cổ ±90 độ tốt hơn cáp dẹt CSI; `dual_esp32s3` giữ `ov2640` nối thẳng `head_mcu1` |
| Cảm biến khoảng cách | `1 x VL53L0X` hoặc `VL53L1X` (`13.5 x 10.5 mm`, I2C) | Bỏ hẳn 4 mắt phụ và mạch chia `TCA9548A` |

### 2.2 Cổng giao tiếp Raspberry Pi 4B trên vỏ ba lô

Trong hệ tọa độ tâm bo PCB của `src/cad/purchased/electronics.py` (cạnh ngắn `+X = +42.5` chứa Ethernet/USB, cạnh dài `-Y = -28.0` chứa nguồn/HDMI/Audio, cạnh `+Y = +28.0` là GPIO 40-pin):

- **Cạnh `+X`:** RJ45 Ethernet ở `(+35.5, +17.75)`, USB 3.0 ở `(+35.5, +1.0)`, USB 2.0 ở `(+35.5, -17.0)`. Cửa I/O trên ba lô mở đối xứng ôm trọn vùng `Y_board in [-25.5, +25.5]` nên cắm gọn cho Pi 4B và vẫn thả vừa Pi 5 vốn đảo chiều Ethernet và USB 2.0.
- **Cạnh `-Y`:** USB-C nguồn `5V/3A` ở `X_board = -31.3`, hai cổng micro-HDMI ở `-16.5` và `-3.0`, jack audio `3.5 mm` ở `+11.0`. Cửa cạnh bên ba lô mở hẹp và gọn trải dài `X_board in [-37.0, +17.0]` cao `9.0 mm` để tiếp cận đủ USB-C, micro-HDMI và jack 3.5 mm.

---

## 3. Thiết kế cơ khí và CAD — Đợt 1

### 3.1 Khoang đầu mới `head_front` và mặt kính `visor_lens` cho OLED 2.42 inch

- **Linh kiện tham chiếu mới:** Bổ sung `oled_242` vào `src/cad/purchased/electronics.py` với kích thước `(75.0, 43.0, 5.0)`, khối lượng `0.012 kg`, tâm 4 lỗ ốc `(68.0, 36.0)` đường kính `2.7 mm`.
- **Thay đổi trong `src/cad/assembly.py` và `src/cad/layout.py`:**
  - Xóa vòng lặp gắn 5 `tof_sensor_*` và xóa `tof_mux` (`tca9548a`) ở mọi biến thể.
  - Gắn `oled_display` (`oled_242`) ngay sau mặt kính trước đầu tại `head` frame, hướng mặt hiển thị ra `+X`.
  - Di chuyển `camera` (`pi_camera_3` / UVC `25 x 24 mm` trên `rpi5`/`pizero2w`, `ov2640` trên `dual_esp32s3`) và gắn đúng 1 `tof_sensor_0` (`vl53l0x`) lên khoang `brow pod` trên đỉnh trán đầu.
- **Hình học `head_front` và `visor_lens` trong `src/cad/monocoque.py`:**
  - Giữ bề ngang tổng `98 mm` (`hy = 49.0 mm`, đủ bọc bo OLED rộng `75 mm` kèm hai thành bên và gờ ốc).
  - Nâng chiều cao khoang mặt trước thêm `~14 mm` (`hz` từ `31.0` lên `~38.0 mm`, `Z_max` mái đầu từ `~54` lên `~68 mm`) để mở cửa sổ hiển thị `76.0 x 43.5 mm`.
  - Đổi `visor_lens` từ mặt trụ cong gắt `R = 69.6 mm` sang tấm polycarbonate khói phẳng hoặc cong nhẹ dày `>= 1.85 mm` áp sát mặt trước OLED `0.5 đến 1.0 mm`, có gờ định vị chống lọt sáng và dễ gia công cắt thực tế.
  - Giữ nguyên phần hàm vát bọc kín `idler_screw_head_pitch`, nắp sau `head_rear_cover` và rãnh thoát dây `head_harness` ở mặt sau không cọ khi gật `-30 đến +45 độ`.

### 3.2 Cụm trán `brow pod` chứa Camera và 1 ToF

- Đúc liền khối lên đỉnh trán phía trước của `head_front` bằng PLA dày tối thiểu `1.82 mm`, vát cạnh 45 độ đồng bộ với mặt vát mũ đầu.
- Kích thước bao: rộng `54 mm` (`Y in [-27, +27]`), cao tổng khoang trong `25.5 mm` ăn âm một phần xuống trên mép bo OLED để phần nhô trên mái đầu chỉ khoảng `10 đến 12 mm` và nhô về trước mặt kính khoảng `8 đến 10 mm`.
- Bố trí bên trong: `camera` đặt gần tâm lệch nhẹ về một bên (`Y = -8.0 mm`), `tof_sensor_0` đặt song song ngay cạnh (`Y = +16.0 mm`), mỗi cảm biến có 1 cửa sổ quang học hở riêng trên mặt trước `brow pod` nên ống kính camera và tia hồng ngoại ToF không đi qua kính khói.

### 3.3 Đường dây cổ `head_harness`

- Giữ lộ trình S-loop phía sau trục `head_yaw` với mọi điểm neo trên `neck` thỏa `X <= -30 mm` trong `src/cad/wiring/routes.py`.
- Đường kính mô phỏng bó dây `head_harness` là `6.5 mm` cho bản Pi 4B/Pi 5 và Pi Zero 2 W (đủ cho bó dây silicon mảnh gồm SPI OLED 7 sợi + I2C ToF 4 sợi + I2S Mic 5 sợi + USB camera 4 sợi) và `4.5 mm` cho `dual_esp32s3`; mở rộng nhẹ rãnh thoát dây trên `head_rear_cover` để không cọ trên toàn dải `-30 đến +45 độ` pitch và `±90 độ` yaw.

### 3.4 Ba lô lưng kín khít thân `backpack`

- **Xóa khe hở `8 mm` nối thân:** Trong `src/cad/armor/backpacks.py`, giữ nguyên `sled` và hệ tọa độ `module_frames` để không làm xô lệch vị trí bo mạch bên trong, đồng thời kéo vành vỏ trước của `backpack` tiến về `+X` thêm `7.8 mm` (từ `X = -60.0 mm` đến `X = -52.2 mm`, chừa khe lắp `0.2 mm` với mặt lưng `trunk_front` ở `X = -52.0 mm`).
- **Đóng kín vách ngoài chống lộ PCB xanh:**
  - Bỏ các ô khoét hở rộng ở hai bên hông, mặt trên và mặt dưới (`bay_w/bay_h`, `side_w/side_h`).
  - Mặt lưng thay dải rãnh rộng `88 mm` bằng cụm khe tản nhiệt mảnh (`3.0 mm` mỗi khe, thanh ngăn `>= 2.0 mm` PLA) bố trí gọn đúng vùng quạt và tản nhiệt của Pi 4B.
  - Giữ đúng 2 cửa I/O tối thiểu ở mép dưới/hông khớp cổng Pi 4B (và tương thích Pi 5).
- **Khe luồn dây và khoang kỹ thuật bên trong vành nối `X in [-60.0, -52.2]`:**
  - Chừa hốc rỗng phía trong cho `xt30_connector` (`Pos(-53, 0, -34)`), `main_power` và phần đuôi `torso_lower`.
  - Chừa đúng 3 cửa luồn dây nhỏ gọn trên vành nối: 1 khe đỉnh giữa cho `head_harness` (`Y in [-6.5, +6.5]`, `Z >= +28`) và 2 khe nhỏ bên dưới hai bên cho `leg_harness_left/right` (`Y in ±[22, 34]`, `Z in [-14, -2]`).

### 3.5 Ngân sách khối lượng và DfAM

- Trần khối lượng dự án: `MAX_MASS_KG = 1.50 kg`.
- Khối lượng dự kiến bản chính Pi 4B (`rpi5` slot): khoảng `1.40 đến 1.43 kg` (dư `70 đến 100 g`).
- Chuẩn thành tối thiểu: `PLA/CLEAR >= 1.82 mm`, `PETG >= 2.55 mm`, `TPU >= 2.25 mm`.
- Tổng danh mục in giữ nguyên `26` chi tiết (`25` STL xuất nháp, nằm gọn trong ràng buộc `25 <= len(stls) <= 28` của `tests/cad/test_v2_monocoque.py`).

---

## 4. Thiết kế phần mềm, mô phỏng và firmware — Đợt 2

### 4.1 Hợp đồng 37 observation cho tầng đi bộ RL 50 Hz

Cập nhật `NUM_OBSERVATIONS = 37` tại `src/robot/robot_spec.py` làm hợp đồng chuẩn V2. Cấu trúc chính xác của vector `37` chiều trong `compute_observations()` (`src/sim/env/env_config.py`):

| Chỉ số | Tên trường | Số chiều | Nguồn thực tế trên Pi 4B |
|---|---|---|---|
| `0..2` | `base_ang_vel` | 3 | Gyro `MPU6050` |
| `3..5` | `proj_grav` | 3 | Mahony AHRS từ `MPU6050` |
| `6..8` | `commands` | 3 | Lệnh vận tốc `[vx, vy, wz]` từ tầng dẫn đường hoặc điều khiển |
| `9..16` | `joint_pos` | 8 | Góc 8 khớp theo `JOINT_ORDER` |
| `17..24` | `joint_vel` | 8 | Vận tốc 8 khớp |
| `25..32` | `last_actions` | 8 | Hành động bước trước |
| `33..34` | `foot_contact` | 2 | Trạng thái chạm sàn chân trái, phải |
| `35..36` | `phase_clock` | 2 | `[sin, cos]` nhịp bước khi `embed_phase_clock=True` (hoặc 2 số 0 đệm khi `False`) |

- `src/sim/env/policy_contract.py` giữ hỗ trợ `obs_dim in (37, 42)` để không làm gãy kiểm tra schema tổng quát, trong khi `MicroDuck8DoFConfig.OBS_DIM`, `ParallelSimRunner`, `export_onnx`, `BaseRPiInferenceEngine`, `pack_rpi_observation`, và header C `src/hardware/dual_esp32s3/deployment/v2_policy_contract.h` chuyển mặc định sang `37` chiều (`lidar_ranges=None`).
- Tầng thưởng RL đi bộ trong `env_config.py` / `run_parallel_sim.py` vận hành ở chế độ locomotion thuần `37` obs; khi không truyền `lidar_ranges`, thành phần thưởng né vật cản trực tiếp trả về `0.0` gọn gàng.

### 4.2 Đồng bộ mô hình MJCF

- Cập nhật `src/robot/cyberduck_v2.xml` và `src/robot/cyberduck_v2_robot.xml`:
  - Xóa 5 site `site_lidar_m30..p30` và 5 `<rangefinder name="lidar_ray_...">`.
  - Giữ `<camera name="head_camera">` và bổ sung đúng 1 `<site name="site_head_tof">` cùng `<rangefinder name="head_tof" site="site_head_tof" cutoff="2.0"/>` gắn trên body `head` theo đúng vị trí `brow pod`.
  - Cập nhật `ParallelSimRunner` cung cấp hàm đọc 1 tia `get_head_tof_range(env_idx)` thay cho mảng 5 tia cũ (và giữ `get_observations()` trả về ma trận `(num_envs, 37)`).

### 4.3 Bộ quét cổ chủ động `ActiveHeadScanner`

Tạo module độc lập `src/sim/behaviors/active_head_scanner.py` (thuần NumPy, không phụ thuộc Torch hay driver phần cứng, dùng chung giữa MuJoCo và runtime Pi 4B):

- **Trạng thái nội bộ:** Bộ đệm 5 bin khoảng cách tương ứng 5 góc ngang quanh thân `RAY_ANGLES = [-30, -15, 0, +15, +30]` độ (khớp trực tiếp với `LidarWaypointPlanner` trong `src/sim/behaviors/nav_planner.py`), kèm mảng thời điểm cập nhật cuối cùng cho từng bin và tham số `ttl_s = 2.0`, `max_range_m = 2.0`, `floor_pitch_cutoff_rad = 0.35`.
- **Giao diện chính:**
  - `update(head_yaw_rad, head_pitch_rad, tof_range_m, now_s)`:
    - Nếu `head_pitch_rad > floor_pitch_cutoff_rad` (đầu đang cúi sâu nhìn sàn) thì bỏ qua mẫu để tránh nhận nhầm mặt sàn là tường.
    - Nếu `tof_range_m < 0` hoặc `NaN` hoặc `>= max_range_m` thì coi là thoáng (`max_range_m`); ngược lại tính khoảng cách chiếu ngang `d_horiz = tof_range_m * cos(head_pitch_rad)` và gán vào bin góc gần `head_yaw_rad` nhất (trong phạm vi `±37.5 độ`, lệch ngoài phạm vi đó không ghi đè vào vùng trước mặt).
  - `get_virtual_rays(now_s) -> np.ndarray(5,)`: trả về mảng 5 khoảng cách; bin nào quá hạn `now_s - last_ts > ttl_s` tự hồi về `max_range_m`.
  - `desired_head_angles(mode, now_s, target_norm_xy=None) -> tuple[float, float]`:
    - Chế độ `SCAN_NAV`: tạo quỹ đạo lắc đầu hình sin `head_yaw = 0.52 * sin(2 * pi * 0.75 * now_s)` (tức ±30 độ tại 0.75 Hz) và `head_pitch = 0.0` để 1 mắt ToF liên tục cập nhật đủ 5 góc trước mặt cho `LidarWaypointPlanner`.
    - Chế độ `INTERACT`: tính góc bám tỉ lệ từ lệch tâm mục tiêu `target_norm_xy` trên ảnh camera, kẹp chặt trong biên `head_yaw in [-1.57, 1.57]` và `head_pitch in [-0.52, 0.78]`.

### 4.4 Bộ hiển thị mắt OLED 2.42 inch `128 x 64`

Tạo module `src/hardware/oled_eyes.py` tách sạch phần vẽ và phần xuất thiết bị:

- **`EyeRenderer` (thuần NumPy, chạy test offline không cần phần cứng):**
  - Xuất mảng nhị phân `(64, 128)` kiểu `uint8` / `bool` cho hai mắt cân đối trên màn ngang.
  - Hỗ trợ 5 trạng thái biểu cảm: `neutral`, `happy`, `curious`, `alert` (khi có vật cản gần hơn `0.35 m`), `sleepy`.
  - Hỗ trợ chớp mắt theo chu kỳ (`blink_ratio` từ `0.0` đến `1.0`) và liếc đồng tử `(gaze_x, gaze_y)` trong `[-1.0, 1.0]` đồng bộ trực tiếp từ góc `head_yaw`, `head_pitch` hoặc tọa độ mục tiêu camera.
- **`SSD1309Display` (lớp driver phần cứng có `mock=True`):**
  - Khi `mock=False` trên Raspberry Pi 4B: đẩy khung hình `128 x 64` qua SPI (`luma.oled` hoặc `spidev`), tự bắt lỗi thiếu thư viện hoặc lỗi thiết bị và chuyển an toàn sang log cảnh báo mà không làm sập vòng điều khiển `50 Hz`.
  - Khi `mock=True`: lưu khung hình cuối trong RAM để chạy kiểm thử tự động.

### 4.5 Đồng bộ runtime trên Pi 4B và các profile còn lại

- `src/hardware/rpi_infer_core.py`:
  - `pack_rpi_observation()` đặt `lidar_ranges: Optional[...] = None` và mặc định trả vector `37` chiều khớp `NUM_OBSERVATIONS = 37`.
  - Tích hợp tuỳ chọn vòng `ActiveHeadScanner` và `SSD1309Display` (mặc định `mock=True` trong test, chạy ở luồng hoặc chu kỳ tách biệt để không gây trễ vòng suy luận ONNX `50 Hz`).
- `src/hardware/rpi5/deployment/infer_rpi5.py` & `src/hardware/rpi5/docs/rpi5_hardware_spec.md`: bổ sung cấu hình và tài liệu pinout ưu tiên cho **Raspberry Pi 4B 4GB** (`SPI0` cho OLED 2.42 inch, `I2C1` cho `PCA9685` + `MPU6050` + `1 x VL53L0X/VL53L1X`, `USB` cho UVC camera, giữ tương thích Pi 5).
- `src/hardware/pizero2w/` và `src/hardware/dual_esp32s3/`: cập nhật hợp đồng `37` observation đồng bộ để toàn bộ test hợp đồng không lệch phiên bản.

---

## 5. Chiến lược kiểm chứng và tiêu chí nghiệm thu

### Đợt 1 — Kiểm chứng CAD

1. `pytest tests/cad/ -q` đạt 100% xanh (cập nhật bài test `tof_sensor_` từ 5 về 1 và kiểm tra sự hiện diện của `oled_display`, `raspberry_pi_4b`, không còn `tof_mux`).
2. `26 / 26` chi tiết in là B-rep hợp lệ một khối liền; đối xứng gương chân lệch `0.000000 mm3`; `chest_loft_shell().is_inside(Vector(40, 53, -6)) == True`.
3. `cad.validation --samples 2` trên cả 3 biến thể đạt `63 / 63` tư thế với `0` va chạm (đặc biệt kiểm tra `head_pitch` -30/+45 độ, `head_yaw` ±90 độ với `brow pod`, `head_harness` và khe nối ba lô kín), dây nối đạt và khối lượng tất cả tư thế `<= 1.500 kg`.
4. `cad.export_pipeline --draft` xuất `25` tệp STL kín nước và `cad.artifacts` đạt 100% khớp SHA-256 (`release_ready` tiếp tục giữ `false` đúng giai đoạn).
5. Dựng và kiểm tra trực quan các góc cận cảnh mới: cận cảnh đầu OLED + `brow pod` mặt trước/nghiêng, góc hông và sau lưng ba lô kín khít thân, mặt cắt bên trong đầu và ba lô, và bộ 4 góc toàn thân.

### Đợt 2 — Kiểm chứng phần mềm, mô phỏng và firmware

1. Viết TDD unit test mới cho `ActiveHeadScanner` (chiếu `cos(head_pitch)`, lọc cúi sàn, hết hạn `TTL`, góc `SCAN_NAV` và `INTERACT`, ghép nối với `LidarWaypointPlanner`) và `EyeRenderer` / `SSD1309Display` (kích thước `128 x 64`, 5 biểu cảm, chớp mắt, liếc mắt, fallback an toàn khi thiếu driver).
2. Cập nhật và chạy xanh toàn bộ bộ test dự án: `tests/robot/`, `tests/sim/`, `tests/hardware/` và `tests/cad/` dưới hợp đồng `37` observation và `1 x head_tof`.
3. Xác nhận `ParallelSimRunner` bước mô phỏng ổn định với `cyberduck_v2.xml` mới và khớp 100% vector `37` chiều giữa mô phỏng với `pack_rpi_observation()` của `BaseRPiInferenceEngine`.
4. Ghi nhận sổ bàn giao 3–5 dòng vào `README.md` và `docs/his.md` sau khi hoàn thành; giữ nguyên nguyên tắc Zero-Trust Git (không tự động `git push`).
