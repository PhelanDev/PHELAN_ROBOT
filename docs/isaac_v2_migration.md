# Đưa CyberDuck V2 vào Isaac

## Kết luận hiện tại

V2 có đủ thành phần để bắt đầu **nhập asset và kiểm tra vật lý**: mesh hiển thị,
collision đơn giản, cây khớp, inertial từng link và floating base. Chưa có kết quả
import USD hoặc chạy Isaac trong repository. Không coi việc MuJoCo chạy được là
bằng chứng solver của Isaac đã khớp.

Phần cứng chốt **8 servo MG996R**. Khung skeleton hiện dùng kích thước concept
micro-servo 23 × 12 × 22 mm; chưa phải khung gá MG996R để chế tạo. Vì thay servo
có thể buộc đổi chiều dài chân, vị trí khớp và phân bố khối lượng, cần chốt cơ khí
trước đợt training phục vụ robot thật. Có thể thử nghiệm pipeline Isaac bằng V2
hiện tại, nhưng policy đó chỉ ứng với hình học và inertial đang mô phỏng.

## Nguồn dữ liệu duy nhất

| File | Vai trò |
| --- | --- |
| `duck_v2/cyberduck_v2.xml` | Nguồn MJCF chính, có robot, sàn và địa hình |
| `duck_v2/robot_spec.py` | Danh tính, thứ tự khớp và thông số V2 dùng chung |
| `duck_v2/cyberduck_v2_robot.xml` | MJCF robot riêng, bỏ sàn/địa hình/đèn |
| `duck_v2/cyberduck_v2.urdf` | Asset dẫn xuất để thử URDF importer |
| `duck_v2/meshes/` | Mesh hiển thị theo từng link |
| `archive/v1/` | Tài sản lịch sử; không dùng cho training hoặc làm fallback |

Giữ nguyên vị trí hai file dẫn xuất cạnh thư mục `meshes/` để đường dẫn mesh
tiếp tục đúng. OBJ assembly dùng để xem hình; nhập toàn bộ OBJ thành một rigid
body sẽ làm mất cấu trúc 8 khớp.

Tạo lại asset sau mỗi lần sửa MJCF:

```powershell
python duck_v2/scripts/build_robot_only_mjcf.py
python duck_v2/scripts/export_v2_urdf.py
python scripts/verify_model.py --steps 1000
```

URDF giữ inertial, frame, giới hạn khớp và collision. MJCF capsule được tách thành
cylinder và hai sphere trong URDF. URDF không mang trọn bộ sensor, actuator,
contact filtering, solver và armature của MJCF; phải cấu hình lại trong Isaac.

## Thứ tự thực hiện trong Isaac

1. **Chốt phiên bản và môi trường riêng.** Theo hướng dẫn cài Isaac Lab và backend
   tương ứng. Kiểm tra GPU, driver, VRAM và ví dụ có sẵn trước khi nhập robot.
   Không cài đè môi trường MuJoCo đang chạy của dự án. Python được kiểm tra trong
   phiên chuẩn hóa này chưa có `isaaclab`, `isaacsim`, `pxr`; Torch là bản CPU.
2. **Nhập một robot.** Với Isaac Sim, thử URDF importer bằng
   `duck_v2/cyberduck_v2.urdf`; cũng có thể thử MJCF importer với bản robot riêng
   nếu phiên bản đã chọn hỗ trợ. Xuất USD vào `duck_v2/usd/` sau khi kiểm tra.
   Đặt đơn vị mét, kg, giây; Z hướng lên; robot có floating base, không cố định
   thân vào world. Kiểm tra importer xử lý đúng floating joint và link `world`.
3. **Kiểm tra hình học và inertial sau import.** Dùng mesh cho hiển thị và
   primitive cho collision; không tạo collider mới từ giáp ngoài rồi cộng thêm
   collider dẫn xuất. Giữ khối lượng/inertia tường minh; không để importer tự
   suy chúng từ mật độ mesh. So sánh kích thước, CoM và joint frames với bảng dưới.
4. **Chạy điều khiển một robot.** Tạo sàn trong scene Isaac, kiểm tra dấu và
   thứ tự từng khớp, đứng tại nominal pose, rồi thử target nhỏ và đẩy nhẹ.
   Đọc state/contact bằng sensor hoặc API của backend; không dùng ground-truth
   base velocity trong actor nếu deployment không có tín hiệu tương đương.
5. **Port task sang Isaac Lab.** Viết environment/config riêng cho V2; ánh xạ
   observation, action, reward, termination, reset và terrain từ code `training/`.
   Runner hiện tại gọi MuJoCo trực tiếp; đổi `--device cuda` không chuyển mô
   phỏng sang Isaac hoặc làm physics chạy GPU.
6. **Train tăng dần.** Bắt đầu 1 robot và ít environment không render, sau đó
   tăng theo VRAM/throughput đo thực tế. Học đứng → đi trên sàn → đổi command →
   nhiễu/tham số actuator → chướng ngại/cầu thang. Chỉ chạy dài sau khi các phép
   đối chiếu đạt yêu cầu. Giữ log riêng cho từng backend và thông số solver.

Nếu chọn backend Newton/MuJoCo Warp trong một bản Isaac Lab có hỗ trợ, dùng đúng
tài liệu của bản đó và kiểm tra khả năng nhập asset/sensor; route này vẫn cần
task Isaac Lab riêng và kiểm chứng tương tự. Chưa có backend Isaac nào được
cài hoặc benchmark trong đợt chuẩn hóa này.

## Hợp đồng phải giữ khi port

| Đại lượng | Giá trị hiện tại của mô hình |
| --- | --- |
| Danh tính checkpoint | `cyberduck_v2.xml` |
| Actor / privileged critic / action | 42 / 53 / 8 |
| Physics timestep / decimation | 0.002 s / 10 |
| Control timestep | 0.02 s, 50 Hz |
| Khối lượng toàn robot | 0.811 kg; chưa phải khối lượng cân robot thật |
| Cao độ gốc thân ở zero pose | 0.128 m; nominal pose có thể làm cao độ thay đổi |
| Offset đùi / cẳng chân | 0.044 / 0.036 m |
| Sole collision, mỗi chân | 0.050 × 0.026 × 0.008 m |
| Giới hạn torque / tốc độ | ±1 Nm / 6 rad/s; giả định mô phỏng |
| PD | kp 13.5 Nm/rad, kd 0.9 Nm·s/rad |
| Damping / frictionloss / armature MJCF | 0.25 / 0.008 / 0.005 |
| Clock | period 0.8 s; reset step về 0; tăng sau action kể cả command bằng 0 |

Thứ tự khớp: `head_yaw`, `head_pitch`, `left_hip_roll`, `left_hip_pitch`,
`left_knee_pitch`, `right_hip_roll`, `right_hip_pitch`, `right_knee_pitch`.

Actor nhận: gyro thân 3, projected gravity 3, command 3, góc khớp 8, vận tốc
khớp 8, action trước 8, contact hai chân 2, phase sin/cos 2, range 5.
Đối chiếu cách scale góc/vận tốc và xử lý range bằng `compute_observations`
trong `training/env_config.py`; giữ normalizer và clip khi export. Clock nằm ở
hai vị trí cuối của block trước LiDAR. Quaternion API giữa các backend có thể
khác thứ tự; kiểm tra rõ trước khi tính projected gravity.

Action hiện là residual quanh nominal pose và có rate limit, giới hạn góc,
torque/tốc độ. Port `scale_actions`, `clip_action_rates` và vòng điều khiển từ
`training/run_parallel_sim.py`; không đưa trực tiếp [-1, 1] vào khớp như radian.
Runner MuJoCo cập nhật PD ở physics step; nếu dùng PD tự viết trong Isaac thì
không bật thêm một bộ drive PD khác. Bộ drive tích hợp cần được so sánh đáp ứng,
không chỉ điền cùng kp/kd rồi giả định hai solver tương đương.

Các phép đối chiếu trước train dài: số khớp/axis/range đúng; tổng mass và CoM
khớp nguồn; chân không xuyên sàn; động học từng khớp đúng dấu; nominal stance
không reset để che ngã; state/action/reward hữu hạn; torque được chặn; observation
cùng state cho cùng kết quả; reset và time-limit bootstrap giữ nguyên quy tắc.
Đo sai khác quỹ đạo thay vì yêu cầu hai solver có kết quả bit-for-bit.

## Chốt cơ khí MG996R trước sim-to-real

Đo đúng biến thể servo sẽ mua: thân, tai bắt vít, tâm lỗ, tâm/trục spline, horn,
chiều cao horn và vùng dây. Dựng gối đỡ/horn theo các mặt chuẩn đó, rồi sweep
toàn bộ giới hạn 8 khớp để kiểm tra giáp, đùi, cẳng chân và dây không va chạm.
Không phóng to khối micro-servo để coi là hoàn tất thiết kế MG996R.

Cân từng cụm gồm servo, horn, vít, pin, dây và bo mạch; cập nhật inertial/CoM
trong MJCF rồi sinh lại hai asset dẫn xuất. Đo tốc độ, giới hạn torque hữu dụng,
deadband, backlash, độ trễ và nguồn cấp khi nhiều servo chạy đồng thời. Chính
các số đo này quyết định khoảng randomization và chính sách triển khai.

Runtime Pi hiện dùng góc lệnh làm ước lượng góc khớp, cùng fallback cho một số
tín hiệu contact/range. Cần bổ sung phản hồi hoặc mô phỏng đúng thiếu hụt cảm
biến đó trước khi coi policy có thể chuyển sang robot thật.

Các checkpoint cũ trong `models/checkpoints/cyberduck_v2_*.pt` thiếu identity/schema;
chúng không phải mốc V2 đã nghiệm thu. Các header ESP đang có cần được export
lại từ policy V2 hợp lệ; compiler guard sẽ từ chối header cũ. Train mới sau khi
chốt hình học, giữ đường dẫn output riêng cho smoke test và benchmark.

## Tài liệu chính thức

- [Isaac Lab: cài đặt và workflow](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim: yêu cầu hệ thống và import robot](https://docs.isaacsim.omniverse.nvidia.com/)

Chọn đúng nhánh tài liệu theo phiên bản đã cài; file này mô tả hợp đồng V2 và
trình tự chuyển đổi, không xác nhận kết quả chạy Isaac.
