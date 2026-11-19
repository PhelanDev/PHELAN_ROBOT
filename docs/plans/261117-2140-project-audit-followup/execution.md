# Kết quả thực hiện audit follow-up F1–F10 (2026-11-17)

Hoàn tất toàn bộ kế hoạch [plans/261117-2140-project-audit-followup/plan.md](file:///e:/Phelan_Duck/plans/261117-2140-project-audit-followup/plan.md) trên nhánh `wip/fix-project-audit`, xuất phát từ commit `950173a539a523d2170bb15946546923dbbfc235`. Toàn bộ sửa đổi triển khai theo kỷ luật RED → GREEN, giữ nguyên tag `v1-flat-walk-stable` (`734e3c10c8c7cc86984173f608a5e6091d45998e`), checkpoint `models/checkpoints/v1-flat-walk-stable.pt`, hai chỉnh sửa đường dẫn có sẵn ở `pipelines/pizero2w_stack/deployment/infer_pizero2w.py` và `pipelines/rpi5_stack/deployment/infer_rpi5.py`, cùng toàn bộ `.pt`/`.onnx`/header triển khai sẵn trong `models/` và `pipelines/`.

## Tổng hợp thay đổi theo finding

| Finding | Thay đổi chính | Tệp ảnh hưởng |
| --- | --- | --- |
| F1 | Khởi tạo zero cho `g_cmd_pkt`, tách `g_last_cmd_pkt_time` / `g_has_cmd_pkt` và `g_last_lidar_pkt_time` / `g_has_lidar_pkt` khỏi `g_last_mcu1_pkt_time`, bổ sung `update_mcu1_watchdogs` để heartbeat, gói màu và gói lỗi CRC không làm mới tuổi lệnh hay LiDAR. | `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino`, `tests/test_deployment_artifacts.py` |
| F2 | Chuẩn hóa helper `_decode_int16_be` dùng chung giữa `_calibrate_gyro` và `read` trên NumPy 2.x, ngừng nuốt lỗi kiểu/biên giá trị I2C, bổ sung trạng thái `is_stale`, `consecutive_read_errors`, `last_read_error` và dừng lệnh `[0.0, 0.0, 0.0]` trong `BaseRPiInferenceEngine.run_loop`. | `pipelines/rpi_infer_core.py`, `tests/test_deployment_artifacts.py` |
| F3 | Bổ sung `phase_clock_for_schema` dùng chung; `_resolve_esp32_schema` kiểm tra 42 chiều và `control_dt=0.02` trước khi tạo thư mục/header, xuất `POLICY_OBS_SCHEMA_VERSION`, `POLICY_EMBED_PHASE_CLOCK`, `POLICY_CONTROL_DT`, `POLICY_GAIT_PERIOD`, `POLICY_CLOCK_*` sang `mcu2_main.ino` kèm tăng `g_episode_steps` sau action và `reset_policy_state()`. | `training/policy_contract.py`, `pipelines/rpi_infer_core.py`, `scripts/export_policy.py`, `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino`, `tests/test_deployment_artifacts.py`, `tests/test_policy_consumer_contract.py`, `tests/test_ppo_locomotion_diagnosis_fixes.py`, `tests/test_ppo_training.py` |
| F4 | Đồng bộ schema observation clock trên `interactive_viewer.py` (`_attach_viewer_policy_schema`, `_step_viewer_policy_control`, `--legacy-observation-schema`, reset `[r]` và từ chối hot-reload đổi schema). | `scripts/interactive_viewer.py`, `tests/test_policy_consumer_contract.py`, `tests/test_interactive_viewer.py` |
| F5 | Tính `target_yaw` theo bearing `np.arctan2(dy_g, dx_g)` ngoài `goal_tol` (giữ `current_yaw` khi vào vùng đích) cho cả `heading_penalty` lẫn điều kiện dừng `yaw_fallen` ở Stage 0 khi `use_nav_planner=True`. | `training/run_parallel_sim.py`, `tests/test_nav_planner.py` |
| F6 | Thay ngưỡng đứng cứng `0.18m` bằng `SelfRightingController.from_model` (`stand_min_z = 0.85 * nominal_z`, mặc định `0.1088m`) và `_is_standing`; gắn qua `ParallelSimRunner`. | `training/self_righting.py`, `training/run_parallel_sim.py`, `tests/test_self_righting.py` |
| F7 | Chặn `trigger()` khi đã ở `FAILED_LOCKED`; `AdvancedBehaviorController` giữ trạng thái khóa qua các tick, trả `recovering=False` ngay từ tick chuyển khóa, chỉ mở khi gọi `reset_recovery()`. | `training/self_righting.py`, `training/advanced_gaits.py`, `tests/test_advanced_gaits.py` |
| F8 | Bảo toàn `squash_actions` khi PPO transfer, hỗ trợ cờ `--squash-actions` / `--no-squash-actions` từ CLI, từ chối checkpoint legacy thiếu metadata hoặc sai kiểu bool trước khi ghi file; `save_bc_checkpoint` ghi rõ `"squash_actions": bool(model.squash_actions)`. | `scripts/train_ppo.py`, `scripts/pretrain_bc.py`, `tests/test_ppo_training.py`, `tests/test_ppo_locomotion_diagnosis_fixes.py` |
| F9 | Lưu `policy_arch`, `algorithm`, `squash_actions`, `model_path`, `model_name`, `observation_schema` trong `train_sac.py`; chuẩn hóa `_build_sac_sequential`, `_validate_ckpt_dims`, `_validate_esp32_actor_arch` để nạp SAC ReLU qua viewer/benchmark/ONNX đúng 1 lần `tanh` và từ chối ESP32 export cả ở `prepare_target_exports` lẫn `export_esp32_headers` trước khi tạo file. | `scripts/train_sac.py`, `scripts/export_policy.py`, `tests/test_sac_smoke.py` |
| F10 | Cho phép `RunningMeanStd` dùng `clip >= 0` (tắt clip tại `clip == 0`) đồng thời giữ `count/epsilon/min_std > 0`, hữu hạn ở cả `train_ppo.py` và `apply_obs_norm_to_actor`. | `scripts/train_ppo.py`, `scripts/export_policy.py`, `tests/test_ppo_training.py` |

## Rà soát code và kiểm chứng cuối

Subagent review độc lập (`7e3c2148-4c4c-40b8-a4ee-cdd3139c1204`) kiểm tra toàn bộ diff và chỉ ra 1 vấn đề Important (`prepare_target_exports` bỏ sót từ chối kiến trúc SAC cho `esp32`) cùng 3 điểm Minor (`load_actor_from_checkpoint` ép kiểu `bool`/`int` chặt, `recovering` ngay tại tick chuyển `FAILED_LOCKED`, và cờ CLI `--squash-actions` cho `train_ppo.py`). Cả 4 điểm đã được sửa kèm regression test trước đợt chạy nghiệm thu cuối.

| Kiểm tra | Kết quả quan sát |
| --- | --- |
| Baseline 7 file trọng tâm trước sửa | `132 passed, 16 warnings in 152.42s` |
| Focused 7 file sau khi vá toàn bộ F1–F10 và review fixes | `154 passed, 28 warnings in 172.19s` (`audit-focused`) |
| Full suite `python -B -m pytest tests/ -q -p no:cacheprovider --basetemp .pytest_cache/audit-full` | `543 passed, 38 warnings in 262.52s` |
| Full `python -B duck.py verify` (kèm `PYTEST_ADDOPTS="-p no:cacheprovider --basetemp .pytest_cache/duck-verify"`) | Exit `0`; xác thực vật lý MJCF/URDF thành công (`Verification SUCCESS`) và `543 passed, 38 warnings in 265.64s` |

## Bảo toàn và giới hạn hiện hành

- Cảnh báo `DeprecationWarning` từ `torch.onnx.export(..., dynamo=False)` vẫn xuất hiện như trước đây trong Python 3.12 / PyTorch 2.15.0+cpu / ONNX 1.23.2.
- Kiểm thử firmware ESP32 tiếp tục dựa trên trích hàm + host `g++` C++17 stubs; bài đo watchdog latency, bus I2C MPU6050 thật, độ trễ vòng lặp 50 Hz và smoke test chính sách mới cần kiểm tra thêm trên phần cứng Pi và ESP32 trước khi triển khai hoặc gắn điểm neo mới.
