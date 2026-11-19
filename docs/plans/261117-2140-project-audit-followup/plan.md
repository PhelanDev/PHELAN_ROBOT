# Kế hoạch rà soát và sửa lỗi dự án MicroDuck

> Phạm vi: rà soát source tại HEAD `950173a539a523d2170bb15946546923dbbfc235`.
> GitNexus chưa đăng ký repository; `.gitnexus/meta.json` đang trỏ tới `E:\duckV1` và commit cũ `bd25e347310640fc6edcc8306d0db4503ed7486e`. Findings trong tài liệu này là **source-derived**, đã đối chiếu với mã hiện tại và probe được ghi rõ bên dưới.
> Worktree có hai thay đổi có sẵn trong `pipelines/rpi5_stack/deployment/infer_rpi5.py` và `pipelines/pizero2w_stack/deployment/infer_pizero2w.py`. Kế hoạch không sửa hoặc hoàn nguyên hai thay đổi đó.
> Xác minh hiện trạng: `python -B -m pytest tests/ -q -p no:cacheprovider --basetemp <thư-mục-trong-workspace>` đạt `521 passed, 26 warnings` trong `239.45s`; `python -B duck.py verify --skip-tests` đạt xác minh mô hình vật lý. Lệnh pytest mặc định ban đầu bị Windows `PermissionError` khi quét `C:\Users\LEGION\AppData\Local\Temp\pytest-of-LEGION`, nên dùng `--basetemp` dưới workspace.

## 1. Mục tiêu

Sửa các lỗi đã tái hiện ở đọc IMU Pi, an toàn lệnh UART ESP32, đồng bộ observation train/inference, navigation, tự đứng dậy, và round-trip checkpoint PPO/SAC. Giữ riêng từng thay đổi để có thể kiểm tra bằng regression test mà không ghi đè checkpoint, policy, header triển khai, hoặc anchor hiện có.

## 2. Kết quả rà soát

Mức **P1**: có thể làm robot giữ lệnh quá hạn, bỏ dữ liệu cảm biến, đánh giá/sử dụng policy với observation sai, hoặc làm chức năng an toàn/phục hồi không dùng được. Mức **P2**: phá tương thích checkpoint hoặc quy tắc tuần tự hóa trong cấu hình cụ thể.

| ID | Mức | Tình huống lỗi và bằng chứng | Khu vực cần sửa |
| --- | --- | --- | --- |
| F1 | P1 | Giữ vận tốc và LiDAR cũ khi UART vẫn nhận heartbeat/packet loại khác; timestamp chung được cập nhật tại MCU2 `mcu2_main.ino:397-424`, còn timeout LiDAR và watchdog cùng dùng timestamp đó tại `:476-484`, `:614-619`. Probe C++: gửi `vx=.20`, LiDAR `.10m` lúc `t=1000`, rồi chỉ heartbeat đến `t=5000`; actual vẫn `vx=.20`, range `.10m`, tuổi liên lạc bằng 0. | Tách độ mới của command, LiDAR và link heartbeat trong MCU2. |
| F2 | P1 | Trên NumPy 2.5.4, `np.int16(unsigned_word)` ném `OverflowError` với mẫu MPU6050 âm; lỗi bị nuốt và trạng thái IMU cũ tiếp tục được dùng. Đường chuyển đổi ở `pipelines/rpi_infer_core.py:204-206`, `:314-319`; nhánh im lặng ở `:210-211`, `:335-336`. Probe bus giả: sau mẫu `gx=+131` rồi 10 mẫu `gx=-131`, gyro vẫn là `+0.00523599`; với `ax=-1`, bộ lọc bị reset về 0 thì sample cũng không cập nhật. | Giải mã word 16-bit signed nhất quán ở calibration và read; làm rõ tình trạng đọc lỗi. |
| F3 | P1 | Observation clock trên ESP32 không theo contract huấn luyện. Runner mặc định tắt clock cho stack phần cứng ở `training/run_parallel_sim.py:249-252`, ghi `[35:37]=[0,0]` ở reset (`:755-761`); firmware MCU2 luôn ghi clock và tăng trước quan sát đầu tại `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino:533-545`. `export_esp32_headers` ở `scripts/export_policy.py:317-357` xuất policy 42 chiều mà không kiểm/đưa observation schema vào header. Probe C++ trích từ `assemble_observation`: lần moving đầu cho `[.1564345,.9876884]`, còn runner trả `[0,0]`; firmware còn reset clock khi đứng, ngược contract `zero_command=advance`. | Nối schema qua train → export → firmware, hoặc từ chối schema chưa được firmware hỗ trợ. |
| F4 | P1 | Viewer luôn nhúng `[sin,cos]` vào observation Pi, kể cả khi policy được train với `embed_phase_clock=False`. `extract_viewer_observation` tại `scripts/interactive_viewer.py:505-523` luôn truyền `phase_clock`; runner Pi tắt clock theo stack name. Probe runner Pi5 thật sau reset: `embed=False`, train clock `[0,0]`, viewer clock `[0,1]`, max chênh observation bằng `1.0`. Kết quả đánh giá/điều khiển trong viewer không đại diện policy được train. | Đọc `observation_schema` từ checkpoint/ONNX và dùng đúng layout/clock của policy ở mọi viewer loop. |
| F5 | P1 | Navigation nhận waypoint tùy ý nhưng reward và điều kiện kết thúc episode giả định heading theo world-X. `training/run_parallel_sim.py:1190` khóa `target_yaw=0`; `:1242-1246` đánh dấu yaw lệch trên 1 rad là ngã khi lệnh quay nhỏ. Probe robot đứng thẳng yaw `π/2`, mục tiêu `(0,2)`: planner phát lệnh đi thẳng đến waypoint nhưng `step()` trả `done=True`, `yaw_fallen=True`. | Tính heading mục tiêu theo bearing waypoint; không coi heading có chủ đích là ngã. Đồng bộ reward và termination. |
| F6 | P1 | Controller khóa thành công đứng ở `root_z >= 0.18m` (`training/self_righting.py:193-197`, `:231-250`), trong khi MJCF V1/V2 đặt thân ở `z=.128m`; probe giữ upright tại chiều cao danh định vẫn vào `FAILED_LOCKED` sau 192 bước. Robot có thể đã đứng ổn định nhưng recovery không bao giờ kết thúc. | Lấy ngưỡng từ hình học/model hoặc tư thế đứng đã hiệu chỉnh theo model; giữ giới hạn để phân biệt robot mắc kẹt thật. |
| F7 | P1 | `AdvancedBehaviorController.step` chỉ kiểm tra `is_recovering`; sau `FAILED_LOCKED` nó phân loại robot vẫn ngã và trigger lại ở `training/advanced_gaits.py:325-333`. Probe với `phase_timeout=10` vào lock tại bước 90, rồi bước kế tiếp trở lại `PHASE1`, `attempts=4`, `recovering=True`. Safety lock không tồn tại qua bước điều khiển tiếp theo. | Coi `FAILED_LOCKED` là trạng thái dừng tường minh đến khi reset được yêu cầu. |
| F8 | P1 | PPO mặc định khởi tạo `squash_actions=True` tại `scripts/train_ppo.py:668-676`, nhưng chỉ đọc lại cờ trong nhánh `resume` (`:717-740`). Transfer checkpoint không chép cờ nguồn trước khi load weight. Probe policy nguồn `squash_actions=False`, bias actor `.8`: action trước optimizer là `.8`, transfer phát `.66403675` (`tanh(.8)`). Checkpoint từ BC cũng không khai báo transform nên có cùng rủi ro. | Preserve transform nguồn đã xác thực khi warm-start; nếu legacy thiếu metadata thì yêu cầu semantics rõ ràng hoặc bước chuyển đổi tường minh. |
| F9 | P2 | SAC lưu actor với keys `net.*`, `mean_head.*`, `log_std_head.*` (`scripts/train_sac.py:116-126`), nhưng generic loader dựng actor PPO dạng Sequential rồi nạp `actor_state_dict` (`scripts/export_policy.py:210-229`). Nạp checkpoint SAC qua viewer, benchmark hoặc exporter vì vậy lỗi keys/architecture; smoke test SAC chỉ kiểm việc ghi checkpoint. | Nhận diện kiến trúc checkpoint; thêm actor adapter deterministic SAC và chỉ hỗ trợ hardware export khi semantics/dimension được xác minh. |
| F10 | P2 | `RunningMeanStd` dùng `clip=0` như tắt clipping trong `normalize` (`scripts/train_ppo.py:90-105`) và cho phép tạo state đó, nhưng `load_state_dict` từ chối mọi scalar `<=0` (`:128-130`). Do đó state hợp lệ theo producer không thể round-trip để resume. | Cho `clip=0` hợp lệ theo cùng contract, còn `count`, `epsilon`, `min_std` phải dương; hoặc từ chối sớm ở constructor nếu tắt clipping không được hỗ trợ. |

## 3. Kiến trúc và ảnh hưởng

- [verified] MCU2 nhận packet velocity, LiDAR, heartbeat, voice và color chung trong `process_uart_packets`; hiện dùng một timestamp cho nhiều nghĩa tuổi dữ liệu. Cổng sửa là decoder/loop trong `mcu2_main.ino`, không đổi UART format nếu không cần.
- [verified] Observation Pi đi từ runner và schema checkpoint qua `training.policy_contract`, exporter `scripts/export_policy.py`, runtime `pipelines/rpi_infer_core.py`, rồi viewer `scripts/interactive_viewer.py`. F3 và F4 là hai consumer lệch nhau; sửa phải giữ schema theo từng stack, không áp quy tắc của model V1 chung cho mọi target.
- [verified] `ParallelSimRunner.step` dùng `target_yaw` trong reward rồi tính `yaw_fallen`; sửa navigation cần cùng frame/world bearing ở hai nơi.
- [verified] Recovery gồm `SelfRightingController` và wrapper `AdvancedBehaviorController`; F6 liên quan điều kiện hoàn tất, F7 liên quan quyền sở hữu trạng thái FAILED_LOCKED.
- [verified] PPO, BC và SAC có checkpoint layouts khác nhau. Không nạp SAC bằng cách đổi tên state-dict keys hoặc giả định nó là actor PPO.

## 4. Ràng buộc implementation

- [verified] Branch hiện tại là `wip/fix-project-audit`. Git status chỉ có hai thay đổi đường dẫn model trong wrapper RPi; giữ nguyên chúng và chạy lại `git status` trước/sau từng nhóm.
- [verified] Có anchor tag `v1-flat-walk-stable` cùng `models/checkpoints/v1-flat-walk-stable.pt`; không thay tag/checkpoint này và không ghi artifact test vào `models/` hoặc `pipelines/`.
- [verified] Full suite hiện xanh khi tách `--basetemp` vào workspace: `521 passed, 26 warnings`. Warnings ONNX deprecation không phải findings correctness của audit này.
- [inferred] C++ host probe xác minh logic trích xuất bằng peripheral stubs; vẫn cần build firmware Arduino hoàn chỉnh và smoke test trên board để kết luận thời gian thực/I2C/UART.

## 5. Thay đổi đề xuất

### F1 — tuổi riêng cho command và LiDAR

- [verified] Thay `g_last_mcu1_pkt_time` chung bằng timestamp theo loại dữ liệu: command hợp lệ, LiDAR hợp lệ, và heartbeat/link nếu cần cho telemetry.
- [verified] Cập nhật timestamp command chỉ sau decode CRC/type hợp lệ của velocity hoặc voice; timestamp LiDAR chỉ sau packet LiDAR hợp lệ. Color và heartbeat không được làm mới lệnh vận tốc hoặc khoảng cách.
- [inferred] Khi command quá hạn, dừng vận tốc ngay cả khi heartbeat vẫn đến. Khi LiDAR quá hạn, dùng fallback clear-range hiện có hoặc policy fail-safe được thống nhất; không sử dụng range cũ như phép đo mới.

### F2 — giải mã MPU6050 signed

- [verified] Dùng một helper chuyển hai byte thành signed int16 bằng quy tắc two's complement, dùng chung cho sáu trục accelerometer/gyro trong `read()` và ba trục gyro calibration.
- [verified] Không nuốt lỗi parse/overflow như một sample hợp lệ. Giữ lần đọc cuối nếu I2C lỗi chỉ khi API cũng thể hiện rõ sample stale; nếu điều khiển safety cần freshness, truyền trạng thái đọc lỗi tới loop.

### F3/F4 — schema clock train, firmware và viewer

- [verified] Định nghĩa một hàm/contract đóng gói clock duy nhất theo `observation_schema`: `obs_dim`, `embed_phase_clock`, `period`, `control_dt`, reset và hành vi zero-command.
- [verified] ESP32 exporter kiểm schema trước khi ghi bất kỳ header nào. Header mang constants được MCU2 dùng; firmware từ chối schema unsupported (hiện tại chỉ có clock 50 Hz nếu không thể cấu hình dt/period).
- [verified] Firmware chỉ tăng clock sau action theo đúng runner; reset theo episode contract, không reset theo command zero nếu schema quy định `zero_command=advance`.
- [verified] Viewer chọn schema từ checkpoint/ONNX; dimension 37 không nhận clock/LiDAR của layout 42, và layout 42 chỉ điền các trường theo schema đã xác thực.
- [inferred] Không hỗ trợ legacy bằng cách đoán schema từ `obs_dim`; giữ đường override explicit hiện có.

### F5 — frame heading của navigation

- [verified] Tính bearing từ vị trí hiện tại đến mục tiêu world `(goal_x, goal_y)` và chuẩn hóa sai số góc theo wraparound.
- [verified] Truyền target yaw nhất quán vào reward và dùng termination dựa trên fall thật hoặc lỗi hướng so với target, không phải world yaw tuyệt đối.
- [assumed] Không thay thuật toán waypoint/LiDAR hoặc reward obstacle trong cùng thay đổi; xác minh phạm vi bằng tests trước khi mở rộng.

### F6/F7 — hoàn tất và khóa recovery

- [verified] Chuyển standing height/orientation threshold thành giá trị cấu hình/model-derived; hiệu chỉnh trên cả V1 và V2 bằng trạng thái upright ở hình học danh định.
- [verified] Tách tiêu chí “đã đứng” khỏi “đang ở đúng joint target” chỉ khi cần để tránh khai báo thành công khi thân vẫn ngã.
- [verified] `FAILED_LOCKED` phải trả trạng thái failure và không gọi `trigger()` tự động ở tick sau. Reset chỉ xảy ra qua lệnh/operator rõ ràng hoặc lifecycle reset đã xác định.
- [inferred] Không làm controller tiếp tục lặp recovery vô hạn sau khi hết retry; nếu robot còn bất ổn, caller báo failure và giữ hành vi dừng an toàn hiện có.

### F8 — action transform khi transfer

- [verified] Nạp `squash_actions` từ checkpoint có metadata sau khi validate kiểu, trước khi actor được dùng/lấy action; giữ reset optimizer và selection state hiện có của transfer.
- [verified] Với checkpoint PPO/BC legacy thiếu transform, xác định semantics theo provenance rõ ràng hoặc yêu cầu cờ explicit; không đổi policy bằng `tanh` âm thầm.
- [inferred] Giữ behavior của resume hợp lệ và đường conversion explicit hiện có.

### F9/F10 — contract checkpoint

- [verified] Thêm kiến trúc/phiên bản actor SAC vào checkpoint; loader phải dựng đúng module, output deterministic đúng một lần tanh và giữ hidden_dim lưu trong tensor metadata.
- [assumed] Chỉ mở hardware export cho SAC nếu đã xác định được parity với quantized firmware; nếu chưa, nạp được vào viewer/benchmark vẫn phải hoạt động, còn export phải báo unsupported rõ.
- [verified] Tách validation scalar RMS theo semantics; test round-trip clip=0 cho NumPy/Torch normalize và PPO checkpoint, đồng thời giữ từ chối NaN/Inf/variance âm/shape sai.

## 6. Trình tự thực hiện

1. **Khóa baseline và regression đỏ.** Ghi branch, HEAD, status và giữ nguyên hai wrapper Pi. Thêm tests tái hiện F1-F10 bằng stub, runner nhỏ và checkpoint tạm; chứng minh mỗi test thất bại vì hành vi hiện tại trước khi sửa. Không chạy train dài.
2. **Sửa F1 và F2: freshness/signed sensor input.** Sửa timestamp packet MCU2 và parser MPU6050. Chạy host C++ tests với clock giả và Python tests trên biên int16; xác minh dữ liệu cũ hết hạn độc lập.
3. **Sửa F3 và F4: observation schema.** Thêm train/export/firmware contract rồi viewer consumer. Chạy train→header→assembly và checkpoint/ONNX→viewer parity ở reset, bước đầu, nhiều bước, zero-command và resume.
4. **Sửa F5: waypoint frame.** Cập nhật target yaw/reward/termination và tests waypoint trục ±Y cùng forward hiện tại.
5. **Sửa F7 rồi F6: lifecycle recovery.** Trước tiên giữ FAILED_LOCKED qua các tick; tiếp theo hiệu chỉnh tiêu chí thành công trên geometry V1/V2; chạy tests fall thật và stuck thật.
6. **Sửa F8: PPO/BC warm-start.** Khóa action parity trước optimizer cho `squash_actions` true/false, checkpoint mới/legacy và actor BC; xác minh transfer vẫn reset optimizer.
7. **Sửa F9/F10: checkpoint SAC/RMS.** Round-trip SAC state qua benchmark/viewer; kiểm RMS no-clip và malformed checkpoint bị từ chối không làm đổi state.
8. **Xác minh tổng thể và bàn giao.** Chạy focused tests theo nhóm, sau đó `python -B -m pytest tests/ -q -p no:cacheprovider --basetemp <thư-mục-tạm-trong-workspace>` và `python -B duck.py verify`. Kiểm `git diff/status`, artifacts tạm và hai wrapper Pi; cập nhật handover của repo theo quy ước khi implementation hoàn tất.

## 7. Chiến lược kiểm thử

| Test file | Regression cần thêm |
| --- | --- |
| `tests/test_deployment_artifacts.py` | MCU2 fake-time: heartbeat/color không refresh command; command không refresh LiDAR; CRC sai không refresh timestamp; stale command dừng lệnh; clock encode parities và reject schema unsupported trước ghi header. |
| `tests/test_policy_consumer_contract.py` | Runner Pi5/ESP32 ↔ viewer/schema cùng observation lúc reset và sau action; 37/42 chiều; zero-command; ONNX/checkpoint metadata. |
| `tests/test_nav_planner.py` | Goal `(0, ±2)`, robot yaw ±π/2, command thẳng theo bearing không trả `yaw_fallen`; forward goal `(2,0)` giữ behavior. |
| `tests/test_self_righting.py` | Chiều cao đứng theo geometry V1/V2 đạt DONE; robot nằm/nghiêng không được đánh dấu thành công; stuck vẫn hết retry. |
| `tests/test_advanced_gaits.py` | Sau `FAILED_LOCKED`, nhiều tick tiếp theo không trigger lại, không tăng attempts và không chạy gait/recovery; reset tường minh mới mở khóa. |
| `tests/test_ppo_training.py` | Transfer actor bias `.8` giữ action `.8` với transform false và đúng tanh một lần với true; BC/legacy semantics; RMS `clip=0` round-trip; lỗi metadata không publish checkpoint. |
| `tests/test_sac_smoke.py` | Lưu rồi nạp SAC actor `hidden_dim=16` và mặc định; output khớp actor gốc ở input âm/dương; viewer/benchmark loader nhận artifact hoặc export từ chối rõ trước tạo file. |

Focused command dự kiến sau khi thêm tests:

```powershell
python -B -m pytest tests/test_deployment_artifacts.py tests/test_policy_consumer_contract.py tests/test_nav_planner.py tests/test_self_righting.py tests/test_advanced_gaits.py tests/test_ppo_training.py tests/test_sac_smoke.py -q -p no:cacheprovider --basetemp .pytest_cache/audit-focused
python -B -m pytest tests/ -q -p no:cacheprovider --basetemp .pytest_cache/audit-full
python -B duck.py verify
```

## 8. Rủi ro và giới hạn

- [inferred] F3/F4 thay đổi observation distribution; parity test cần so sánh từng field và policy output, không chỉ shape. Checkpoint legacy vẫn cần provenance đã xác minh.
- [inferred] F5 đổi heading reward/termination có thể ảnh hưởng policy đi thẳng; regression giữ target world-X và benchmark so sánh trước/sau trên scenario cố định.
- [inferred] F6/F7 ảnh hưởng state machine an toàn; không tăng ngưỡng hoặc retry tùy ý để làm test pass. Cần xác minh dừng an toàn khi thất bại.
- [verified] Firmware probe là host compile/trích hàm với stubs; chưa có bằng chứng hardware cho watchdog latency, I2C, servo hoặc timing loop.
- [verified] Toàn bộ suite đang xanh nhưng không có assertion khóa các tình huống đã tìm ra; cảnh báo ONNX exporter hiện có là deprecation.

## 9. Việc ngoài phạm vi

- Không train policy dài hoặc đánh giá chất lượng locomotion mới trong bước sửa correctness; chạy benchmark xác định sau khi hợp đồng observation được sửa.
- Không thay đổi/xóa checkpoint hoặc policy/header đang có, không cập nhật anchor tag/checkpoint stable.
- Không sửa đường dẫn model trong hai wrapper Pi đang dirty.
- Không nâng dependency NumPy/ONNX như cách né lỗi; parser phải đúng với kiểu dữ liệu và phiên bản hiện tại.

## 10. Tiêu chí hoàn tất

- [ ] F1-F10 có regression test chứng minh lỗi trước sửa và hành vi mong đợi sau sửa.
- [ ] Focused suites, full suite và `duck.py verify` qua trong môi trường test cô lập; không còn artifact phát sinh trong `models/` hoặc `pipelines/`.
- [ ] Train/export/runtime/firmware có parity observation/action theo schema được hỗ trợ; schema không hỗ trợ bị từ chối trước side effect.
- [ ] Watchdog dùng tuổi lệnh đúng; packet telemetry không hồi sinh command/LiDAR cũ.
- [ ] Recovery lock không tự mất; đứng theo geometry hợp lệ được nhận diện, tình huống stuck vẫn thất bại có kiểm soát.
- [ ] Checkpoint PPO/BC/SAC round-trip giữ architecture và action transform; trạng thái RMS hợp lệ round-trip, trạng thái sai lỗi rõ.
- [ ] Git status vẫn giữ hai wrapper Pi ban đầu; anchor code/checkpoint stable không thay đổi.
