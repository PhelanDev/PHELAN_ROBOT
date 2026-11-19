# Thư viện linh kiện CyberDuck V2 MG996R

Đơn vị nguồn CAD và STEP là **mm**. Model servo lấy tâm spline làm gốc XY;
mặt đáy hộp servo là Z=0. Tâm bốn lỗ tai là
`(-14.25, ±5.0)` và `(34.35, ±5.0)`; đỉnh spline Z=42.9.

Các solid được dựng tham số để kiểm tra fit, không phải model do nhà sản xuất
chứng nhận. Thân PCB, lỗ định vị và bao hình đầu nối được tách khỏi dữ liệu
điện. Hình răng spline, ren vít, vòng bi bên trong và mạch đồng không được
mô phỏng. Không dùng STEP này để chế tạo lại linh kiện mua ngoài.

Nguồn kích thước ưu tiên: cấu hình đã khóa trong mục 2 của plan ngày
18-11-2026; datasheet TowerPro MG996R và bản vẽ Akizuki được plan dẫn;
bản vẽ cơ khí Raspberry Pi; tài liệu phần cứng của ba stack trong repo.
Lỗ PCB khác M3 giữ nguyên kích thước gốc. Các cổng trên DevKit và module
không tiêu chuẩn chỉ là bao hình danh định, phải so với đúng bo mua thực tế.

`step/` chứa 32 fit reference do dự án tự dựng. Manifest ghi rõ
`geometry_kind=parametric_fit_reference` và `vendor_certified=false`,
kèm SHA-256 của mỗi file đã xuất.

`sources/` chứa **9 STEP tải thật từ step.parts**, cố định tại commit
`c6113328a5695b976a010a203a90fe86191769bf`. Script kiểm tra kích thước byte
và SHA-256 so với Git LFS pointer trước khi lưu. `sources.json` lưu metadata,
đường dẫn tải, commit và giới hạn của mỗi model; giấy phép upstream nằm trong
`LICENSE-step.parts.txt`. Không tự thay model đang lắp bằng file vừa tải.

Các sai khác phải giữ lại để review:

- `bearing_flanged_f623_simple.step` có bbox `11.5 × 11.5 × 5.0 mm`, dù metadata
  khai width 4 mm. Model này không đạt bao hình F623 `3 × 10 × 4 mm` của plan.
- Model Pi 5 gồm nhiều solid, trục khác hệ CAD nội bộ và bao cả đầu nối;
  cần xác minh datum/keepout thay vì chỉ đồng nhất kích thước PCB `85 × 56`.
- `raspberry_pi_zero_mounting_holes_cutter_reference` chỉ là mẫu cutter lỗ,
  không phải model Pi Zero 2 W; `esp32_s3_wroom_1` là module RF, không phải DevKit.
- Hình ren được giản lược. Lỗ insert Ø4.05 trong fit model là bao hình lắp
  danh định; OD insert thật theo plan lớn hơn và cần xét biến dạng cấy nhiệt.

```powershell
.\.venv_cad\Scripts\python.exe -m duck_v2.cad_src.reference_export
.\.venv_cad\Scripts\python.exe -m duck_v2.cad_src.purchased.fetch_sources
.\.venv_cad\Scripts\python.exe -m duck_v2.cad_src.purchased.source_audit
.\.venv_cad\Scripts\python.exe -m pytest tests/test_mg996r_cad_reference.py -q
```

Khối lượng trong manifest là giá trị danh định để lập dự toán, chưa phải cân
linh kiện. Chưa dùng các giá trị này thay inertial của mô hình chuẩn trước
khi assembly và sweep hình học được nghiệm thu.
