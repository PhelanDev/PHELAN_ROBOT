# Đặc tả quy hoạch kiến trúc thư mục CyberDuck V2

Ngày lập: 2026-11-19  
Nhánh nền hiện tại: `wip/fix-project-audit`  
Nhánh thực thi dự kiến: `wip/repo-reorg`

## 1. Bối cảnh và mục tiêu

Kho mã nguồn CyberDuck V2 hiện có 152 file Python hoạt động, 59 bộ kiểm thử `pytest`, hai môi trường Python độc lập gồm môi trường RL/MuJoCo chính và môi trường CAD `.venv_cad` chạy `build123d`, ba profile phần cứng cho cùng một robot V2 và lộ trình port sang Isaac Lab trong `docs/isaac_v2_migration.md`.

Cấu trúc hiện tại phát sinh năm điểm rối chính:
- `duck_v2/` đang gộp chung tài sản mô phỏng chuẩn, mã nguồn CAD tham số `cad_src/`, ảnh render tài liệu và 150 MB sản phẩm xuất CAD nháp/review trong `duck_v2/cad/`.
- Nhóm script bị tách hai nơi là `scripts/` và `duck_v2/scripts/`, phát sinh ba file wrapper trùng lặp chỉ chuyển tiếp sang nhau.
- `training/` chứa lẫn cả lõi môi trường RL, hành vi vận động và cảm biến thị giác/giọng nói.
- `tests/` đặt phẳng toàn bộ 59 file test chung một thư mục dù có 9 file test CAD chạy bằng `.venv_cad`.
- Dữ liệu nháp cục bộ gồm `cad_parallel/` và `codex_backup/` trong `.superpowers/` chiếm 64.9 MB dù không còn code hay test nào tham chiếu, còn `docs/plans/` đang trống.

Mục tiêu của đợt quy hoạch theo **Phương án A**:
- Chia repo thành 5 trụ cột chức năng rõ ràng tại gốc: `robot/`, `cad/`, `sim/`, `hardware/`, `tools/`.
- Tách hoàn toàn đầu ra sinh ra nặng khoảng 150 MB sang `artifacts/` và đưa vào `.gitignore`, giữ Git chỉ theo dõi mã nguồn và tài sản chuẩn nhẹ.
- Phân 59 file kiểm thử vào 4 miền khớp trực tiếp với hai môi trường Python.
- Chạy trực tiếp bằng Python như hiện tại không cần cài gói bằng pip, giữ ổn định mặt tiền CLI `duck.py` và hợp đồng `viewer_3d.html`.
- Sẵn sàng mở rộng sang Isaac Lab bằng cách chừa ranh giới tự nhiên cho `robot/usd/` và `sim/isaac/` khi triển khai về sau.

## 2. Cấu trúc thư mục đích

```text
e:\Phelan_Duck/
|-- robot/                      # Tài sản chuẩn duy nhất của CyberDuck V2
|   |-- __init__.py
|   |-- robot_spec.py           # từ duck_v2/robot_spec.py
|   |-- cyberduck_v2.xml        # giữ cùng cấp meshes/ để giữ nguyên meshdir="meshes"
|   |-- cyberduck_v2_robot.xml  # MJCF robot-only cho Isaac Lab
|   |-- cyberduck_v2.urdf       # URDF dẫn xuất dùng đường dẫn tương đối meshes/*.stl
|   |-- meshes/                 # 17 file STL hình học chuẩn
|   |-- media/                  # 16 ảnh render dùng cho README.md và viewer_3d.html
|   |-- cad/                    # Bản lắp ráp visual chuẩn phục vụ duck status và test
|   |   |-- cyberduck_v2_assembly.obj
|   |   |-- cyberduck_v2_assembly.mtl
|   |   |-- cyberduck_v2_exploded.obj
|   |   |-- cyberduck_v2_skeleton.obj
|   |   |-- cyberduck_v2_skeleton.mtl
|   |   |-- cyberduck_v2_stl_parts.zip
|   |   |-- cyberduck_v2_skeleton_parts.zip
|   |   +-- skeleton_stl_parts/ # 17 STL khung concept
|   +-- README.md               # từ duck_v2/README.md
|-- cad/                        # Mã nguồn CAD tham số MG996R dùng .venv_cad (từ duck_v2/cad_src/)
|   |-- __init__.py
|   |-- armor/
|   |-- purchased/              # gồm sources/ và step/ tham chiếu đã khóa hash
|   |-- skeleton/
|   |-- wiring/
|   |-- artifacts.py, assembly.py, catalogue.py, collision.py, dfam.py,
|   |   export_pipeline.py, geometry.py, kinematics.py, layout.py,
|   |   mass_properties.py, monocoque.py, params.py, placements.py,
|   |   print_plates.py, provenance.py, reference_export.py, release.py,
|   |   render_drafts.py, render_mujoco.py, validation.py
|   |-- requirements.txt
|   |-- requirements-lock.txt
|   +-- README.md
|-- sim/                        # Lõi mô phỏng, RL và hành vi tự hành (từ training/)
|   |-- __init__.py
|   |-- env/
|   |   |-- __init__.py
|   |   |-- env_config.py
|   |   |-- run_parallel_sim.py
|   |   |-- terrain_curriculum.py
|   |   +-- policy_contract.py
|   |-- behaviors/
|   |   |-- __init__.py
|   |   |-- advanced_gaits.py
|   |   |-- self_righting.py
|   |   +-- nav_planner.py
|   +-- perception/
|       |-- __init__.py
|       |-- vision_tracker.py
|       +-- voice_commander.py
|-- hardware/                   # Ba profile phần cứng dùng chung robot V2 (từ pipelines/)
|   |-- __init__.py
|   |-- rpi_infer_core.py
|   |-- rpi5/                   # từ pipelines/rpi5_stack/
|   |   |-- __init__.py
|   |   |-- deployment/
|   |   |-- docs/
|   |   +-- training/
|   |-- pizero2w/               # từ pipelines/pizero2w_stack/
|   |   |-- __init__.py
|   |   |-- deployment/
|   |   |-- docs/
|   |   +-- training/
|   +-- dual_esp32s3/           # từ pipelines/dual_esp32s3_stack/
|       |-- __init__.py
|       |-- deployment/         # giữ nguyên #include tương đối giữa .ino và .h
|       |-- docs/
|       +-- training/
|-- tools/                      # Hợp nhất scripts/ và duck_v2/scripts/
|   |-- __init__.py
|   |-- train/
|   |   |-- __init__.py
|   |   |-- train_ppo.py
|   |   |-- train_sac.py
|   |   |-- pretrain_bc.py
|   |   |-- evaluate_models_sim.py
|   |   +-- audit_anti_cheating_loop.py
|   |-- robot/
|   |   |-- __init__.py
|   |   |-- verify_model.py
|   |   |-- interactive_viewer.py
|   |   |-- export_policy.py
|   |   |-- build_cyberduck_v2.py
|   |   |-- build_robot_only_mjcf.py
|   |   +-- export_v2_urdf.py
|   +-- cad/
|       |-- __init__.py
|       |-- export_cyberduck_v2_cad.py
|       |-- build_cyberduck_v2_skeleton.py
|       |-- generate_bom.py
|       |-- organize_3d_print_stls.py
|       |-- inspect_parts.py
|       |-- render_cyberduck_v2.py
|       |-- render_skeleton.py
|       |-- render_v2_hd.py     # giữ bản thực thi từ scripts/render_v2_hd.py
|       |-- render_exploded_view.py
|       +-- render_leg_joints_visual.py
|-- tests/
|   |-- conftest.py             # nạp REPO_ROOT + tests/ vào sys.path, khóa bảo vệ hardware/ + models/
|   |-- model_test_helpers.py
|   |-- cad/                    # 9 file test CAD chạy bằng .venv_cad
|   |-- robot/                  # 9 file test MJCF/URDF/asset/viewer/CLI
|   |-- hardware/               # 6 file test RPi 5, Pi Zero 2 W, ESP32-S3, policy export/consumer
|   +-- sim/                    # 35 file test RL, PPO/SAC/BC, reward, curriculum, gaits, perception
|-- models/
|   +-- checkpoints/
|-- artifacts/                  # GITIGNORED: toàn bộ đầu ra sinh ra nặng
|   +-- cad/                    # draft/, monocoque_draft/, monocoque_inspection/, review_views/,
|                               # sculpt_review/, artifact_integrity.json, verification.json,
|                               # wall_thickness_sampled.json, source_reference_review.json,
|                               # current_nominal_debug.json, monocoque_nominal.json,
|                               # monocoque_structural_debug.json
|-- docs/
|   |-- isaac_v2_migration.md
|   +-- specs/
|-- plans/                      # GITIGNORED: nhật ký kế hoạch
|-- archive/v1/                 # Lưu trữ tài sản V1 đã ngừng dùng
|-- duck.py
|-- viewer_3d.html
|-- README.md, GEMINI.md, CLAUDE.md, his.md, LICENSE, .gitignore
```

## 3. Bảng ánh xạ chi tiết và các tệp loại bỏ

### 3.1 Loại bỏ wrapper trùng và thư mục thừa
- Xóa `scripts/export_cad_assembly.py` vì chỉ là wrapper 16 dòng gọi `export_cyberduck_v2_cad.py`.
- Xóa `scripts/render_model_3d.py` vì chỉ là wrapper 15 dòng gọi `render_cyberduck_v2.py`.
- Xóa `duck_v2/scripts/render_v2_hd.py` vì chỉ là wrapper 14 dòng gọi `scripts/render_v2_hd.py`; bản thực thi chính chuyển về `tools/cad/render_v2_hd.py`.
- Xóa thư mục trống `docs/plans/`.
- Xóa hai thư mục bản sao cục bộ trong `.superpowers/` là `.superpowers/cad_parallel/` và `.superpowers/codex_backup/`, giữ nguyên `.superpowers/sdd/` và `.superpowers/executing-plans/`.

### 3.2 Ánh xạ 59 tệp kiểm thử vào 4 miền

- `tests/cad/` gồm 9 tệp:
  `test_mg996r_cad_reference.py`, `test_v2_cad_artifact_receipts.py`, `test_v2_cad_pipeline.py`, `test_v2_cad_validation_contract.py`, `test_v2_dfam_sampling.py`, `test_v2_mechanical_interfaces.py`, `test_v2_monocoque.py`, `test_v2_pinned_cad_sources.py`, `test_v2_polish_validation.py`.
- `tests/robot/` gồm 9 tệp:
  `test_cyberduck.py`, `test_cyberduck_touch_and_gait_fixes.py`, `test_duck_cli.py`, `test_interactive_viewer.py`, `test_mjcf_model.py`, `test_urdf_model.py`, `test_v2_generated_assets.py`, `test_v2_geometry_and_dynamics_contract.py`, `test_verify_script.py`.
- `tests/hardware/` gồm 6 tệp:
  `test_deployment_artifacts.py`, `test_esp32_model.py`, `test_pipeline_training.py`, `test_pizero2w_model.py`, `test_policy_consumer_contract.py`, `test_rpi5_model.py`.
- `tests/sim/` gồm 35 tệp:
  `test_5pass_anti_cheating_audit.py`, `test_advanced_gaits.py`, `test_anti_backward_gait_lock.py`, `test_anti_cheating_pass1.py`, `test_anti_cheating_pass2.py`, `test_bc_warmup.py`, `test_corridor_clearance_reward.py`, `test_curriculum_ep_len.py`, `test_emotion_config.py`, `test_env_config.py`, `test_eplen_suicide_fix.py`, `test_eval_obstacle_metrics.py`, `test_foot_rewards.py`, `test_four_golden_hoops.py`, `test_locomotion_state_contract.py`, `test_lying_capability.py`, `test_multilayer_fixes.py`, `test_nav_planner.py`, `test_obstacle_rewards.py`, `test_parallel_runner.py`, `test_pillar_randomization.py`, `test_ppo_locomotion_diagnosis_fixes.py`, `test_ppo_std_and_curriculum_fixes.py`, `test_ppo_training.py`, `test_sac_smoke.py`, `test_self_righting.py`, `test_sitting_capability.py`, `test_six_part_locomotion_overhaul.py`, `test_stage1_slalom_and_speed_fixes.py`, `test_stair_climbing.py`, `test_terrain_curriculum.py`, `test_time_limit_bootstrap.py`, `test_v2_training_defaults.py`, `test_vision_tracker.py`, `test_voice_commander.py`.

## 4. Hợp đồng kỹ thuật và điểm ghép nối bắt buộc

1. **Hợp đồng robot và policy không đổi**:
   - Tên mô hình chuẩn `MODEL_NAME = "cyberduck_v2.xml"`, `MJCF_MODEL_NAME = "cyberduck_v2_pro"`, `CHECKPOINT_PREFIX = "cyberduck_v2"`, 42 observations, 8 actions, tần số 50 Hz và thứ tự 8 khớp giữ nguyên hoàn toàn trong `robot/robot_spec.py`.
   - `robot/cyberduck_v2.xml`, `robot/cyberduck_v2_robot.xml` và `robot/cyberduck_v2.urdf` nằm cùng cấp với `robot/meshes/` nên không thay đổi nội dung hình học hay đường dẫn mesh bên trong XML/URDF.
2. **Hợp đồng xuất và băm nguồn CAD**:
   - `cad/provenance.py` đổi `REPO_ROOT = Path(__file__).resolve().parents[1]` vì giảm 1 tầng so với `duck_v2/cad_src/provenance.py`.
   - Danh sách tệp nguồn băm trong `source_receipt` chuyển sang `cad/**/*.py`, `cad/requirements*.txt`, `cad/purchased/sources/*`, `robot/robot_spec.py` và `sim/env/env_config.py`.
   - Toàn bộ tham số `--output` hoặc đường dẫn manifest mặc định trong `cad/artifacts.py`, `cad/dfam.py`, `cad/export_pipeline.py`, `cad/validation.py`, `cad/render_drafts.py` và `cad/purchased/source_audit.py` chuyển sang `artifacts/cad/...`.
   - Trong `cad/render_drafts.py`, đổi `ROOT = Path(__file__).resolve().parents[2]` sang `parents[1]` và đổi đường dẫn helper từ `duck_v2/cad_src/render_mujoco.py` sang `cad/render_mujoco.py`.
   - Trong `cad/validation.py`, import chuyển sang `from robot.robot_spec import EXPECTED_JOINTS` và `from sim.env.env_config import DEFAULT_NOMINAL_QPOS, LIE_NOMINAL_QPOS, SIT_NOMINAL_QPOS`.
3. **Hợp đồng CLI và trình xem web**:
   - `duck.py` giữ nguyên toàn bộ subcommand và cờ lệnh (`status`, `verify`, `train`, `view`, `export`), cập nhật đường dẫn gọi sang `tools/robot/verify_model.py`, `tools/robot/interactive_viewer.py`, `tools/train/train_ppo.py`, `tools/robot/export_policy.py`, `robot/cad/cyberduck_v2_assembly.obj`, `robot/cad/cyberduck_v2_stl_parts.zip`, `hardware/dual_esp32s3/deployment/...`.
   - `duck.py verify --fast` cập nhật danh sách tệp test nhanh sang đường dẫn mới dưới `tests/sim/` và `tests/robot/`.
   - `viewer_3d.html` giữ tại gốc repo và đổi 5 đường dẫn ảnh từ `duck_v2/media/*.png` sang `robot/media/*.png`.
4. **Độ sâu `REPO_ROOT` và bảo vệ môi trường kiểm thử**:
   - `tests/conftest.py` thêm cả `REPO_ROOT` và `REPO_ROOT / "tests"` vào `sys.path` một lần duy nhất, đồng thời đổi biến bảo vệ ghi đè trong `make_isolated_env` từ `pipelines` sang `hardware`.
   - Tất cả tệp test trong `tests/{cad,robot,hardware,sim}/` cập nhật `Path(__file__).resolve().parents[1]` hoặc `.parent.parent` thành `parents[2]` hoặc nhập trực tiếp `REPO_ROOT` từ `robot.robot_spec`.
   - Trong `sim/env/run_parallel_sim.py`, cập nhật dòng `sys.path.insert` từ `parent.parent` sang `parents[2]` vì di chuyển sâu thêm một tầng so với `training/`.
   - Tất cả script chạy trực tiếp trong `tools/{train,robot,cad}/` cập nhật phép tính `REPO_ROOT` thành `Path(__file__).resolve().parents[2]` vì nằm sâu hơn `scripts/` một tầng.
5. **Quy tắc `.gitignore`**:
   - Bổ sung `artifacts/` vào `.gitignore`.
   - Bỏ dòng `assets/media/*.mp4` đã lỗi thời.

## 5. Quy trình Git an toàn và thứ tự triển khai

1. **Tạo điểm mốc trước khi dời tệp**:
   - Tách toàn bộ artifact nháp CAD khỏi phạm vi track, chốt phần công việc kiểm toán/CAD đang dở trên `wip/fix-project-audit` thành commit riêng trước khi tổ chức lại.
   - Mở nhánh `wip/repo-reorg` từ commit mốc đó để toàn bộ diff di chuyển thư mục nằm tách biệt hoàn toàn.
   - Ưu tiên `git mv` cho mọi tệp đã track để bảo toàn lịch sử Git; không chạy `git push` khi chưa có xác nhận rõ ràng từ người dùng.
2. **Thứ tự thực thi theo lớp phụ thuộc**:
   - Bước 1: Tạo `artifacts/cad/`, di chuyển các thư mục nháp và JSON chẩn đoán từ `duck_v2/cad/`, cập nhật `.gitignore`, dọn `.superpowers/cad_parallel/`, `.superpowers/codex_backup/` và `docs/plans/`.
   - Bước 2: Chuyển tài sản chuẩn sang `robot/` và cập nhật `robot/robot_spec.py`.
   - Bước 3: Chuyển `duck_v2/cad_src/` sang `cad/` và cập nhật `provenance.py`, `validation.py`, `render_drafts.py`, `export_pipeline.py`, `artifacts.py`, `dfam.py`, `purchased/source_audit.py`.
   - Bước 4: Chuyển `training/` sang `sim/{env,behaviors,perception}/` và cập nhật import nội bộ.
   - Bước 5: Chuyển `pipelines/` sang `hardware/{rpi5,pizero2w,dual_esp32s3}/` cùng `hardware/rpi_infer_core.py`.
   - Bước 6: Hợp nhất `scripts/` và `duck_v2/scripts/` về `tools/{train,robot,cad}/`, xóa 3 wrapper thừa, sửa `REPO_ROOT` và import.
   - Bước 7: Cập nhật `duck.py` và `viewer_3d.html`.
   - Bước 8: Cập nhật `tests/conftest.py`, chuyển 59 tệp test sang `tests/{cad,robot,hardware,sim}/`, sửa import và độ sâu đường dẫn.
   - Bước 9: Đồng bộ tài liệu gồm `README.md`, `GEMINI.md`, `his.md`, `docs/isaac_v2_migration.md`, `robot/README.md`, `cad/README.md` và tài liệu dưới `hardware/*/docs/`.

## 6. Tiêu chí nghiệm thu

Đợt quy hoạch chỉ được xem là hoàn tất khi toàn bộ các kiểm tra dưới đây đều đạt:
1. Quét toàn bộ mã nguồn hoạt động trong `duck.py`, `robot/`, `cad/`, `sim/`, `hardware/`, `tools/`, `tests/` xác nhận 0 import cũ trỏ tới `duck_v2`, `training`, `pipelines`, `scripts`.
2. Bộ kiểm thử môi trường chính chạy xanh toàn bộ:
   ```powershell
   python -m pytest tests/robot tests/sim tests/hardware -q -p no:cacheprovider
   ```
3. Bộ kiểm thử môi trường CAD chạy xanh toàn bộ:
   ```powershell
   .\.venv_cad\Scripts\python.exe -m pytest tests/cad -q -p no:cacheprovider
   ```
4. Lệnh CLI chuẩn chạy thành công không lỗi:
   ```powershell
   python duck.py status
   python duck.py verify
   python tools/robot/verify_model.py --steps 1000
   ```
5. Kiểm tra `git status` xác nhận cây thư mục sạch, không còn thư mục cũ `duck_v2/`, `training/`, `pipelines/`, `scripts/` và không có tệp trong `artifacts/` bị theo dõi bởi Git.
