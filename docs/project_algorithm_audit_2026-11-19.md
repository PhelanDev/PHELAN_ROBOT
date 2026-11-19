# Rà soát thuật toán CyberDuck V2 — 2026-11-19

Kết luận: **chưa đủ điều kiện triển khai robot thật hoặc nghiệm thu CAD**. Đã xác nhận 13 vấn đề: 4 mức P1 cần ưu tiên trước khi chạy phần cứng, 9 mức P2 cần khắc phục trong thuật toán, đánh giá và kiểm thử. Đây là rà soát hiện trạng toàn dự án; các phát hiện không đồng nghĩa đều do thay đổi mới gây ra.

## Phạm vi và bằng chứng

- Kho: `E:\Phelan_Duck`, nhánh `wip/fix-project-audit`, HEAD `6ac4c124a8aac9e3e706ff9245fa5e9e979306af`, bao gồm nguồn sửa đổi và nguồn mới chưa commit. Không sửa thuật toán, firmware hoặc asset trong lượt rà soát này.
- Đọc các luồng PPO/SAC/BC, observation/action/reward, reset/curriculum, benchmark, planner, vision, self-righting, exporter, Pi/ESP32 và các phép tính/điều kiện nghiệm thu CAD liên quan.
- GitNexus: `DUCK_MINI_TRAINING`, index ngày 2026-11-19, cùng HEAD; đối chiếu SHA-256 xác nhận các nguồn thuật toán chính trùng index. Truy vấn `compute_rewards`, `reset_env`, luồng quan sát và caller của reset. Nguồn CAD có thay đổi đồng thời, nên kết luận CAD dựa vào nguồn đọc trực tiếp và phép thử, không coi graph CAD là cập nhật đầy đủ.
- Python mặc định: `python -m pytest tests/ -q -p no:cacheprovider --disable-warnings --tb=short` dừng ở collection vì thiếu `build123d` tại `test_v2_monocoque.py`.
- Bỏ riêng module gây lỗi collection: `python -m pytest tests/ -q -p no:cacheprovider --ignore=tests/test_v2_monocoque.py --disable-warnings --tb=short` đạt **579 passed, 8 skipped, 38 warnings**, 228,07 giây. Các module bỏ qua có phụ thuộc CAD.
- `python scripts/verify_model.py` đạt: mass MJCF 0,811 kg, joint/frame/URDF/inertia parity và 100 bước vật lý PD hữu hạn. Kết quả này không chứng nhận dáng đi, phần cứng hoặc khối lượng CAD.
- Reviewer CAD chạy thêm 43 test tập trung và 3 test asset sinh tự động, đều đạt. Reviewer hành vi chạy 33 test tập trung, đều đạt. Không cộng các lượt này vào số test độc lập của toàn dự án.
- Các phép thử firmware dùng host compiler và I2C/UART giả lập; chưa thử trên Pi hoặc ESP32 thật. Không huấn luyện dài, không chạy sweep CAD đầy đủ, không rà soát giao diện trình duyệt trong lượt này.

## Các vấn đề đã xác nhận

### F01 — P1: Pi vẫn điều khiển servo khi IMU không khởi tạo được

**Nguồn:** `pipelines/rpi_infer_core.py:197`, `:328`, `:409`, `:556`.

MPU6050 bắt lỗi I2C lúc khởi tạo rồi tự đặt `mock=True`. Nhánh `read()` của IMU giả lại báo `is_stale=False`; engine phần cứng không kiểm tra sự chuyển đổi này. Vòng lặp vì vậy vẫn dùng lệnh tiến 0,2 m/s cùng dữ liệu tư thế giả.

**Bằng chứng:** cho PCA9685 hoạt động và địa chỉ IMU `0x68` lỗi; cả hai wrapper Pi, với ONNX V2 hợp lệ và `mock=False`, vẫn ghi đủ 8 servo trong một bước. Engine giữ `mock=False`, IMU là `mock=True`. Phép thử độc lập trên reader cũng xác nhận IMU lỗi khởi tạo trả tư thế đứng và `stale=False`.

**Khắc phục:** phần cứng phải từ chối chạy khi IMU khởi tạo thất bại; chế độ giả cần được chọn rõ ràng. Bổ sung test lỗi khởi tạo, vì test lỗi đọc sau khởi tạo không bao phủ nhánh này.

### F02 — P1: MCU1 phát LiDAR cố định như dữ liệu cảm biến mới

**Nguồn:** `pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino:59`, `:113`, `:150`.

`g_lidar_distances` khởi tạo bằng năm giá trị 1,5 m. `sample_lidar_sensors()` chỉ so sánh mảng này với ngưỡng, không đọc ToF. Loop vẫn đóng gói UART đều đặn; MCU2 nhận packet mới nên watchdog không phát hiện thiếu dữ liệu thật, trong khi MCU1 có lệnh tiến mặc định.

**Bằng chứng:** chạy các hàm firmware gốc với thời gian giả lập 20–1000 ms tạo 50 packet LiDAR; cả 250 giá trị đều bằng 1,5 m.

**Khắc phục:** triển khai driver đọc ToF, truyền trạng thái hợp lệ và thời điểm lấy mẫu; thiếu cảm biến hoặc mẫu hợp lệ phải chặn đi. Test cần nối sự kiện đọc cảm biến với payload, không chỉ kiểm tra mã hóa UART.

### F03 — P1: ESP32 đảo dấu trọng lực ngang khi đọc IMU lỗi

**Nguồn:** `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino:347`, `:372`.

Nhánh đọc thành công dùng dấu âm cho hai thành phần trọng lực ngang; nhánh fallback bỏ hai dấu âm đó. Cùng quaternion nhưng policy thấy robot nghiêng sang phía đối diện ngay khi I2C không trả đủ dữ liệu. Nhánh lỗi cũng trả gyro bằng không mà không truyền trạng thái stale vào điều khiển.

**Bằng chứng:** hàm `read_mpu6050()` thật trong harness C++ với pitch +20° trả xấp xỉ `[+0.342017, 0, -0.939685]` khi đọc thành công và `[-0.342017, 0, -0.939685]` khi không có byte.

**Khắc phục:** dùng chung một hàm chiếu quaternion cho hai nhánh và xử lý mẫu stale. Test phải đi qua cả nhánh thành công lẫn nhánh lỗi; tìm công thức đầu tiên trong source không đủ.

### F04 — P1: Planner vẫn ra lệnh tiến khi cả năm hướng bị chặn

**Nguồn:** `training/nav_planner.py:78`.

`clearance_scale` bị kẹp tối thiểu ở 0,2. Khi goal ở phía trước và cả năm khoảng cách bằng không, planner chọn hướng giữa, không quay và vẫn yêu cầu tiến.

**Bằng chứng:** gọi hàm thật với vị trí `[0,0]`, yaw `0`, goal `[2,0]`, LiDAR `[0,0,0,0,0]` trả `[0.06,0,0]` m/s, rad/s. Test hiện kiểm tra chọn hướng khi còn khoảng trống, chưa yêu cầu dừng khi mọi hướng đều đóng.

**Khắc phục:** cho vận tốc tiến về không khi thiếu khoảng hở an toàn; quyết định quay hoặc tìm đường riêng sau đó. Bổ sung tình huống bị chặn toàn bộ và khoảng cách sát không.

### F05 — P2: Self-righting trong mô phỏng được trợ giúp bằng sửa trực tiếp tư thế

**Nguồn:** `training/run_parallel_sim.py:1041`, `:1044`; liên quan `training/self_righting.py:236`.

Khi bật `enable_auto_recovery`, sau 15 bước ở pha push-to-stand, runner trực tiếp nâng `qpos[2]` lên chiều cao gốc, đặt quaternion đứng thẳng và xóa vận tốc. Thành công sau đó không chứng minh robot tự đứng bằng mô-men servo. Tính năng mặc định tắt nên phát hiện này không có nghĩa mọi lượt PPO mặc định đều nhận trợ giúp.

Controller còn lưu `fall_type` nhưng không dùng hướng ngã để chọn quỹ đạo định hướng. Phép thử riêng controller với MuJoCo và PD thật kết thúc `FAILED_LOCKED` ở hai tư thế ngã nghiêng. Test integration hiện cho phép runner sửa tư thế nên không chứng minh được phục hồi thuần vật lý.

**Khắc phục:** tách trợ giúp/reset khỏi đánh giá recovery; đánh dấu rõ episode được trợ giúp. Nghiệm thu controller bằng lực servo và điều kiện hướng thân thực, với nhiều hướng ngã.

### F06 — P2: Hợp đồng action/observation chưa hỗ trợ đầy đủ ngồi và nằm

**Nguồn:** `training/run_parallel_sim.py:453`, `:743`, `:1023`; `training/env_config.py:45`, `:182`, `:275`.

Runner luôn scale action quanh tư thế đứng với biên độ dành cho residual đi bộ, kể cả khi `cmd_modes` chuyển sang SIT/LIE. Với action chuẩn hóa trong `[-1,1]`, mục tiêu hip pitch tối đa chỉ khoảng 0,30 rad và knee 0,65 rad; tư thế SIT yêu cầu 0,70 và 1,35 rad. Đây là giới hạn mục tiêu điều khiển; va chạm vẫn có thể đẩy góc khớp vượt mục tiêu, nhưng không thay thế khả năng ra lệnh đúng tư thế.

**Bằng chứng:** round-trip `SIT_NOMINAL_QPOS -> unscale_actions -> scale_actions`, cùng `nominal_centered=True`, sai tới 0,70 rad. Tư thế LIE sai 0,05 rad ở hip. Đồng thời đổi WALK/SIT/LIE trên cùng trạng thái không làm thay đổi observation: chênh lệch tối đa bằng 0. Một policy dùng chung không nhận được chế độ cần thực hiện.

**Khắc phục:** thiết kế lại vùng action theo các hành vi cần hỗ trợ; thêm tín hiệu chế độ hoặc cơ chế policy riêng. Mọi thay đổi observation cần version schema và đồng bộ exporter/runtime. Test reward từng mode không kiểm tra được tính đạt tới và khả năng phân biệt mode.

### F07 — P2: Domain randomization bị cố định trong huấn luyện song song

**Nguồn:** `training/run_parallel_sim.py:654`, `:726`, `:1354`; `scripts/train_ppo.py:890`.

`reset_env()` chỉ randomize động lực học khi `num_envs == 1`. PPO gọi reset toàn bộ lúc đầu, sau đó auto-reset các episode bằng `reset_env()`. Với nhiều môi trường dùng chung model, friction và torso mass không được lấy mẫu lại trong vòng huấn luyện đó. Randomize vị trí cột mỗi iteration không cập nhật động lực học.

**Bằng chứng:** 12 lần `reset_env(0)` với hai môi trường giữ nguyên mass `0.3453903845 kg` và friction `1.4486840145`; chạy một môi trường thì hai giá trị thay đổi. Test hiện xác nhận friction biến thiên bằng `num_envs=1`, nên bỏ sót trường hợp huấn luyện phổ biến.

**Khắc phục:** có lịch cập nhật model dùng chung tại ranh giới đồng bộ, hoặc model động lực học riêng cho từng môi trường. Không sửa model dùng chung khi worker còn bước vật lý. Thêm test nhiều môi trường qua các episode/rollout.

### F08 — P2: Benchmark có thể ghi lần ngã thành thành công và bỏ mất góc nghiêng cuối

**Nguồn:** `scripts/evaluate_models_sim.py:169`, `:176`, `:205`.

Success chỉ dựa trên `alive_steps >= max_steps`, nên ngã đúng bước cuối vẫn được tính thành công. Đồng thời runner đã auto-reset khi trả `done`; benchmark lại lấy tilt từ `next_obs` và effort từ `runner.datas`, làm số liệu cuối episode trở thành dữ liệu sau reset.

**Bằng chứng:** dùng runner MuJoCo V2 thật với hai robot nghiêng 90°, benchmark một bước: cả hai đều `fallen=True`, tilt terminal khoảng `90.03°`, nhưng kết quả là `success_rate=1.0`, `mean_tilt_deg=0.0`, `mean_torque_sq=0.0`.

**Khắc phục:** phân biệt timeout với failure khi tính success; dùng `terminal_obs` và snapshot torque/vận tốc trước reset. Bổ sung test ngã đúng bước cuối và kiểm tra metric terminal, không chỉ kiểm tra metric tồn tại hoặc nằm trong khoảng.

### F09 — P2: Đưa đầu về giữa khi mất mục tiêu không phụ thuộc thời gian

**Nguồn:** `training/vision_tracker.py:200`; caller `scripts/interactive_viewer.py:861`, `:1259`.

Nhánh mất mục tiêu không sử dụng `dt`; mỗi lần gọi giảm tối đa 0,04 rad với tham số mặc định. Tốc độ thực vì vậy tăng theo tần số gọi; viewer có thể gọi ở bước vật lý 0,002 s.

**Bằng chứng:** bắt đầu yaw 0,8 rad, một lần `update(..., detected=False)` giảm 0,04 rad cho cả `dt=0.001`, `0.002`, `0.033`, tương ứng 40, 20 và khoảng 1,21 rad/s.

**Khắc phục:** tính tốc độ và giới hạn thay đổi góc theo `dt` ở mọi nhánh. Test cùng thời lượng thực ở nhiều chu kỳ gọi; test một tick chỉ xác nhận góc giảm sẽ không bắt được lỗi.

### F10 — P2: Fallback NumPy gộp các điểm nhiễu thành một mục tiêu

**Nguồn:** `training/vision_tracker.py:133`.

Nhánh OpenCV chọn blob lớn nhất; nhánh NumPy cộng tất cả pixel đúng màu rồi lấy trung bình tọa độ. Nhiều điểm rời rạc nhỏ hơn `min_area` vẫn tạo một mục tiêu giả ở khoảng trống giữa ảnh.

**Bằng chứng:** ảnh có 40 pixel vàng rời rạc, `min_area=30`: OpenCV trả `detected=False`; NumPy trả `True`, tâm `(77,40)`, diện tích 40. Test hiện chủ yếu chạy với OpenCV có sẵn.

**Khắc phục:** tìm thành phần liên thông và áp dụng cùng quy tắc chọn blob ở fallback. Test bắt buộc chạy cả hai backend với nhiễu rời rạc và nhiều vật thể.

### F11 — P2: Exporter sinh literal C++ sai cho period nguyên

**Nguồn:** `scripts/export_policy.py:530`, `:531`.

Định dạng `.9g` rồi nối `f` biến giá trị hợp lệ `period=1.0` thành `#define POLICY_GAIT_PERIOD 1f`. Consumer C++ không biên dịch được. Period mặc định 0,8 không lộ vấn đề này.

**Bằng chứng:** tạo schema `observation_schema(42, True, 0.02, period=1.0)`, xuất header rồi biên dịch consumer với `g++ -std=c++17`: lỗi `unable to find numeric literal operator 'operator""f'`.

**Khắc phục:** dùng định dạng float C++ luôn có dấu thập phân hoặc số mũ, áp dụng cho cả period và control_dt. Bổ sung compile test với giá trị nguyên hợp lệ.

### F12 — P2: Khối lượng dây CAD thiếu phần đuôi và thay đổi theo tư thế

**Nguồn:** `duck_v2/cad_src/assembly.py:113`; `duck_v2/cad_src/wiring/routes.py:18`, `:77`.

Assembly lấy mass từ `path.length * 0.00002`, trong khi phép kiểm chiều dài dây cộng thêm `tail_allowance=20 mm` mỗi bó. Bốn bó thiếu tổng 1,6 g trong mass. Độ dài đường spline thay đổi theo pose cũng làm khối lượng robot thay đổi, dù cùng một lượng dây lắp đặt.

**Bằng chứng:** manifest nháp đã xuất ghi bản Pi 5 `1.318426876 kg`; chỉ cộng đuôi dây thành `1.320026876 kg`, đổi kết quả kiểm ngân sách 1,32 kg từ đạt sang trượt. Phép tính dây trên nguồn đọc lúc rà soát đổi từ `9.389262 g` ở pose không sang `10.511915 g` tại `head_yaw=1.57`. Manifest này đã cũ so với nguồn cuối lượt rà soát; ví dụ ngân sách minh họa lỗi tính trên snapshot đã xuất, không phải chứng nhận mass toàn assembly hiện tại.

**Khắc phục:** tính mass theo chiều dài dây lắp đặt cố định, gồm đuôi và phần dự phòng; đánh giá hình học dây theo pose riêng. Thêm test bảo toàn mass theo pose và ngưỡng ngân sách. Các gate nghiệm thu khác hiện vẫn chặn release, nên lỗi này chưa tạo kết quả release tổng thể dương tính giả.

### F13 — P2: Test CAD làm hỏng collection trong môi trường không cài CAD

**Nguồn:** `tests/test_v2_monocoque.py:7`, `:9`.

`from build123d import Vector` xuất hiện trước `pytest.importorskip('build123d')`. Python mặc định thiếu thư viện này nên pytest dừng ngay, không thể áp dụng cơ chế skip. Lệnh full test được README hướng dẫn, và `duck.py verify` khi không bỏ test, đều gặp cùng đường lỗi collection.

**Bằng chứng:** lệnh full suite phía trên trả exit 1, `ModuleNotFoundError: No module named 'build123d'`, một lỗi collection. Bỏ riêng module này cho phép chạy phần còn lại đạt 579 test.

**Khắc phục:** thực hiện `importorskip` trước mọi import phụ thuộc; tách rõ môi trường CAD và bộ kiểm tra cơ bản. Kiểm chứng collection trên cả hai môi trường.

## Trạng thái CAD và policy đã biết

- Kiểm trực tiếp `verify_artifacts('duck_v2/cad/draft/manifest.json')` cuối lượt rà soát trả `passed=false`, `source_current=false`, 78 artifact, không sai nội dung file. Các file đúng hash của manifest nhưng manifest không còn đại diện nguồn CAD hiện tại.
- `verification.json` và `monocoque_nominal.json` cũng không khớp source receipt hiện tại. Các số giao cắt/mass trong đó là kết quả lịch sử, không dùng để kết luận hình học mới đã đạt hoặc còn đúng số lỗi cũ. File `artifact_integrity.json` lưu trước đó có `source_current=true` cũng cần được chạy lại để đánh giá nguồn mới.
- Manifest nháp đang lưu mass danh định Pi 5/Pi Zero/dual ESP lần lượt khoảng 1,318427/1,240767/1,237200 kg; canonical MJCF vẫn 0,811 kg. Cần tái tạo báo cáo từ cùng snapshot trước khi đồng bộ vật lý.
- `release.py` vẫn chủ động chặn các gate reference, mating, cổ/dây, DfAM, hướng in/support, CAD–simulation parity và torque. Đây là công việc chưa nghiệm thu đã được ghi nhận, không tính thêm vào 13 lỗi thuật toán.
- Checkpoint/header cũ cần train/export lại với metadata V2 theo tài liệu hiện hành. Không tạo checkpoint hoặc điểm neo V2 trong lượt audit này.

## Thứ tự xử lý đề xuất

1. F01–F04: sửa dữ liệu cảm biến và hành vi dừng; kiểm tra nhánh lỗi trên harness rồi trên phần cứng.
2. F05–F08: làm rõ recovery thuần vật lý, vùng action/chế độ hành vi, lịch domain randomization và benchmark trước khi dùng kết quả để chọn policy.
3. F09–F13: sửa vision, exporter, mass dây và collection; thêm kiểm tra hồi quy theo tình huống đã tái hiện.
4. Chốt snapshot CAD, tái tạo receipt/collision/sweep/mass, hoàn tất các gate; sau đó mới đồng bộ mô hình, huấn luyện dài và nghiệm thu triển khai.
