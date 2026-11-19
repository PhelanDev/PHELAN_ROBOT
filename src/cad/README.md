# CyberDuck V2 MG996R: nguồn CAD tham số đang phát triển

Nguồn dựng B-rep sử dụng build123d/OpenCASCADE. Đơn vị CAD, STEP, STL và 3MF
là **mm**; tensor quán tính xuất JSON dùng **kg·m²**, tâm khối lượng dùng **m**.
`params.py` chứa thông số giao diện khóa trong plan; `kinematics.py` là cây
tọa độ dành cho bản CAD mới. Mô hình mô phỏng chuẩn hiện chưa đổi hình học.

**Trạng thái: bản nháp chưa đạt kiểm tra giao cắt và dự toán khối lượng.**
Một solid hợp lệ hoặc STL watertight không đủ để kết luận robot lắp được.
Chưa cho phép dùng output này để chế tạo, đưa robot lên nguồn hoặc train policy
cho hình học mới. Không có chứng nhận sweep liên tục, độ dày toàn bộ thành,
tiếp cận dụng cụ hay tải trọng liên tục MG996R.

## Môi trường riêng

```powershell
python -m venv .venv_cad
.\.venv_cad\Scripts\python.exe -m pip install -r src/cad/requirements-lock.txt
.\.venv_cad\Scripts\python.exe -m pytest tests/cad -q -p no:cacheprovider
```

`requirements.txt` khai báo phụ thuộc trực tiếp; `requirements-lock.txt` ghi lại
phiên bản thực tế đã cài. Python training không cài build123d. Bộ test CAD sẽ
báo skipped nếu chạy bằng Python thiếu CAD; vì thế phải chạy lệnh CAD riêng.

## Các bước kiểm tra và xuất

```powershell
$env:PYTHONPATH = "src"

# Thư viện linh kiện: 32 STEP tham chiếu và manifest có hash
.\.venv_cad\Scripts\python.exe -m cad.reference_export

# Tải STEP nguồn từ commit đã khóa; kiểm hash LFS trước khi lưu
.\.venv_cad\Scripts\python.exe -m cad.purchased.fetch_sources
.\.venv_cad\Scripts\python.exe -m cad.purchased.source_audit

# Kiểm tra B-rep giao cắt thật: exit 1 nếu chưa đủ điều kiện phát hành
.\.venv_cad\Scripts\python.exe -m cad.validation --nominal-only
.\.venv_cad\Scripts\python.exe -m cad.validation --samples 9

# Đo mẫu thành theo normal B-rep; không chứng minh minimum toàn cục
.\.venv_cad\Scripts\python.exe -m cad.dfam --max-samples 128

# Chỉ xuất bộ nháp để review, không xuất sang thư mục sản xuất chính
.\.venv_cad\Scripts\python.exe -m cad.export_pipeline --draft
.\.venv_cad\Scripts\python.exe -m cad.artifacts
.\.venv_cad\Scripts\python.exe -m cad.render_drafts
```

`artifacts/cad/verification.json` chứa các cặp giao cắt và thể tích đo được bằng
OpenCASCADE. Validator dừng trước các tư thế tiếp theo nếu nominal còn cấn;
`sweep_completed=false` là chưa quét xong, không phải quét thành công.
Các tư thế đứng/ngồi/nằm lấy trực tiếp từ `src/sim/env/env_config.py`.
Lấy mẫu từng trục không chứng minh an toàn trên toàn bộ không gian 8 khớp.

`artifacts/cad/draft/` chứa STEP từng part/lắp ráp, STL đặt lên bàn in theo heuristic,
3MF xếp bản sao trên bàn 256×256 mm, ảnh review, manifest và inventory nháp.
STL được kiểm tra watertight, thể tích dương và một solid. Hướng đặt chưa có
phân tích support; độ dày thành tối thiểu toàn bộ chưa được đo xác nhận.
Manifest luôn ghi rõ `DRAFT_NOT_FOR_FABRICATION` và `release_ready=false`.
Các report và manifest có receipt SHA-256 của nguồn CAD, cấu hình tư thế,
tham số chuẩn, model tải về và phiên bản thư viện chính. Manifest liệt kê hash
từng STEP, STL, 3MF và BOM được sinh trong lần xuất. `artifacts` chỉ xác minh
tính đồng bộ file, không cấp phép chế tạo, kể cả khi ai đó sửa cờ trong manifest.

## Những giao diện đã kiểm tra riêng

- Chi tiết chịu lực chân phải là phép phản chiếu thực qua mặt phẳng XZ.
- Càng rời, cassette và cốc đáy không cấn tại 9 góc mẫu từ −90° tới +90°.
  Bích hai F623 đặt quay ra hai mặt ngoài, không chồng vào nhau hoặc lấp lỗ bởi gân.
- Ba cụm khay/PCB/kẹp M3/cooler/vỏ/nắp balô đạt kiểm tra B-rep riêng.
  Peg giữ nguyên lỗ PCB; ESP32 có gối đỡ bên dưới, không chỉ có thanh chặn cạnh.
- Khoang pin, EVA, nắp trượt, giắc và tuyến nguồn chính được kiểm tra cùng nhau.

Đây là kiểm tra từng cụm ở vị trí danh định, không phải chứng nhận 8 khớp
trên robot hoàn chỉnh. Hình ren và biến dạng khi cấy nhiệt chưa được mô hình hóa;
reference insert Ø4.05 là bao hình lắp danh định, không đại diện OD mua thực tế.

## Những cổng chưa qua của plan

- Lắp ráp cả ba biến thể phải hết giao cắt trước khi công nhận Phase 1.
- Cổ yaw cần cơ cấu truyền động chừa đường dây qua lõi Ø17; horn hiện tại
  chưa cung cấp đường đi xuyên tâm. Chưa duyệt giải pháp belt/gear lệch tâm.
- Hai hông cùng nghiêng vào trong phải kiểm tra riêng; khoảng cách chân
  130–136 mm không tự bảo đảm đủ khoảng hở ở mọi tổ hợp góc.
- Khối lượng đang tích phân với polymer đặc và khối lượng linh kiện danh định.
  Chưa mô hình hóa infill/skin hoặc cân thực tế; không ép kết quả về 1,23 kg.
- Có **7 nắp idler** cho 6 khớp chân + head_pitch. Số 6 trong danh sách chi tiết
  của plan không bao gồm đủ bảy khớp gối F623; tổng ổ bi vẫn là 11.
- Chưa thay canonical MJCF/URDF, xuất visual sim mới, hay công nhận torque smoke
  của hình học mới. Bộ 42 observations/8 actions/50 Hz hiện tại được giữ.
- Phép đo normal hiện phát hiện nhiều mẫu mỏng, kể cả tại peg, khe, lỗ và tab.
  Cần xác định vùng chịu lực và sửa hoặc duyệt từng ngoại lệ; không tự coi
  `design_wall_mm` là kết quả đo, không miễn trừ các mẫu mỏng để làm xanh report.

Xem `docs/plans/261118-1705-cyberduck-v2-mg996r-3d-rebuild/execution.md` để đối chiếu
từng pha với kết quả chạy thực tế.
