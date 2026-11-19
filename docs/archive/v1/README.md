# V1 — chỉ lưu lịch sử

Thư mục này lưu nguyên trạng tài sản MicroDuck V1 đã ngừng sử dụng: mesh, CAD,
ảnh, MJCF/URDF, bản mô hình theo nền tảng, checkpoint và ONNX cũ. Cấu trúc đường
dẫn gốc được giữ lại để có thể đối chiếu hoặc khôi phục thủ công; ban đầu có 111 tệp
được kiểm tra SHA-256 ngày 2026-11-18, hiện còn 110 tệp sau khi xóa
`assets/media/microduck_concept_render.jpg` theo yêu cầu dọn kho ảnh.

Luồng đang dùng chỉ nạp CyberDuck V2 từ `duck_v2/cyberduck_v2.xml`. Không dùng
các tệp trong kho này để train, export hoặc chạy robot. Tên nền tảng `rpi5`,
`pizero2w`, `esp32` chỉ chọn cách triển khai cùng một V2.

Các checkpoint `cyberduck_v2_*.pt` cũ vẫn nằm ở `models/checkpoints/` để bảo toàn
dữ liệu. Chúng chưa được xác nhận lại sau khi chuẩn hóa collider bàn chân V2;
cần train/evaluate lại trước khi xuất policy dùng trên phần cứng hoặc Isaac.

Firmware ESP32 giữ các header policy cũ tại đường dẫn triển khai và chủ động
báo lỗi biên dịch cho đến khi được thay bằng header xuất từ checkpoint V2 hợp lệ.
