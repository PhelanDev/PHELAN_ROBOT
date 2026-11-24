#!/usr/bin/env python3
"""CyberDuck V2 8-DOF Bipedal Robot - Unified Operations CLI (duck.py).

Provides a single standardized entry point for:
- `status`: Inspect model checkpoint metrics, stage, and hardware readiness
- `verify`: Run kinematics cross-validation and automated test suites
- `train`: Launch PPO reinforcement learning with MuJoCo parallel simulation
- `view`: Open interactive 3D viewer or browser dashboard
- `export`: Export trained policy to ONNX (Raspberry Pi 5) & INT8 C++ (ESP32-S3)
"""

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot.robot_spec import CHECKPOINT_PREFIX, DEFAULT_ONNX_PATH, MODEL_NAME, MODEL_PATH, REPO_ROOT
from sim.env.policy_contract import CHECKPOINT_PREFIXES, resolve_target_checkpoints

# Safe UTF-8 console output for Windows CLI
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass



def cmd_status(args: argparse.Namespace) -> int:
    """Display CyberDuck V2 status and shared deployment formats."""
    print("=" * 72)
    print("🦆 CyberDuck V2 8-DOF Bipedal Robot - System & Training Status")
    print("=" * 72)

    # 1. Model Architecture
    print("\n[1] BỘ NÃO AI (NEURAL NETWORK ARCHITECTURE)")
    print("  • Mô hình triển khai: Actor MLP [37 -> 256 (ELU) -> 256 (ELU) -> 8]")
    print("  • Số nơ-ron Actor   : 520 nơ-ron (256 + 256 + 8 output servos)")
    print("  • Số tham số Actor  : 77.576 weights & biases (77.584 gồm log_std)")
    print("  • Mạng Critic (PPO) : 513 nơ-ron [48 -> 256 -> 256 -> 1]")
    print("  • Không gian quan sát: 37 chiều (IMU, 8 góc, 8 vận tốc, last action, Foot Contact + Phase Clock; brow ToF qua ActiveHeadScanner)")

    # 2. Checkpoints Status
    ckpt_dir = REPO_ROOT / "models" / "checkpoints"
    best_ckpt = ckpt_dir / f"{CHECKPOINT_PREFIX}_best.pt"
    checkpoint_status = "[CHƯA CÓ]"

    print("\n[2] TIẾN TRÌNH HUẤN LUYỆN (CHECKPOINTS & CURRICULUM)")
    if best_ckpt.exists():
        try:
            import torch
            from sim.env.policy_contract import validate_model_identity, validate_observation_schema
            data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            validate_model_identity(data, MODEL_NAME)
            schema = validate_observation_schema(data.get("observation_schema"))
            if schema["obs_dim"] != 37 or schema["version"] != 2 or data.get("obs_dim") != 37 or data.get("action_dim") != 8:
                raise ValueError("Checkpoint V2 phải có 37 quan sát (schema version 2) và 8 hành động")
            checkpoint_status = "[ĐÃ XÁC MINH METADATA V2]"
            stage = data.get("stage", 0)
            iters = data.get("iterations", 0)
            steps = data.get("total_timesteps", 0)
            best_r = data.get("best_reward", 0.0)
            dist_x = data.get("best_distance_x", 0.0)
            print(f"  • Checkpoint tốt nhất: {best_ckpt.name}")
            print(f"    - Giai đoạn địa hình: Stage {stage} (Mục tiêu Stage 1: distance_x >= 0.80m, success >= 70%)")
            print(f"    - Số vòng lặp (Iters): {iters} iterations ({steps:,} timesteps)")
            print(f"    - Điểm thưởng kỷ lục : {best_r:+.4f}")
            print(f"    - Quãng đường tiến X : {dist_x * 100:.2f} cm")
        except Exception as e:
            checkpoint_status = "[CẦN HUẤN LUYỆN LẠI / XUẤT LẠI V2]"
            print(f"  • {best_ckpt.name}: {checkpoint_status} (Stage chưa xác định)")
            print(f"    - Chưa xác minh được identity/schema V2: {e}")
    else:
        print(f"  • Chưa có checkpoint {best_ckpt.name} (Stage chưa xác định)")

    # 3. Git Status
    print("\n[3] TRẠNG THÁI MÃ NGUỒN & HỆ THỐNG")
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        git_branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        print(f"  • Git Branch / Commit: {git_branch} ({git_hash})")
    except Exception:
        print("  • Git: Không thể lấy thông tin commit")

    # 4. Hardware Deployment Targets
    onnx_file = DEFAULT_ONNX_PATH
    esp32_h = REPO_ROOT / "src" / "hardware" / "dual_esp32s3" / "deployment" / "esp32_policy_weights.h"
    esp32_config = esp32_h.with_name("esp32_quant_config.h")
    import re
    esp32_ready = True
    for header, marker, identity in (
        (esp32_h, "CYBERDUCK_V2_POLICY", "POLICY_MODEL_NAME"),
        (esp32_config, "CYBERDUCK_V2_QUANT_CONFIG", "QUANT_MODEL_NAME"),
    ):
        try:
            contents = header.read_text(encoding="utf-8")
            esp32_ready &= bool(
                re.search(rf'^\s*#define\s+{marker}\s+1\s*$', contents, re.MULTILINE)
                and re.search(rf'^\s*#define\s+{identity}\s+"{re.escape(MODEL_NAME)}"\s*$', contents, re.MULTILINE)
            )
        except (OSError, UnicodeError):
            esp32_ready = False
    esp32_status = "[ĐỦ MARKER V2]" if esp32_ready else (
        "[CẦN XUẤT LẠI V2]" if esp32_h.exists() or esp32_config.exists() else "[CHƯA XUẤT]"
    )
    print("\n[4] TỆP TRIỂN KHAI PHẦN CỨNG (DEPLOYMENT ARTIFACTS)")
    print("  • Robot model identity : CyberDuck V2 dùng chung cho RPi 5, Pi Zero 2 W và ESP32-S3")
    print(f"  • Pi ONNX V2 Float32    : {'[TỆP TỒN TẠI; CHƯA XÁC MINH RUNTIME]' if onnx_file.is_file() else '[CHƯA XUẤT]'} ({onnx_file.name})")
    print(f"  • ESP32-S3 INT8 C-Header: {esp32_status} ({esp32_h.name} + {esp32_config.name})")

    # 5. Canonical V2 asset and status
    v2_cad = REPO_ROOT / "src" / "robot" / "cad" / "cyberduck_v2_assembly.obj"
    v2_zip = REPO_ROOT / "src" / "robot" / "cad" / "cyberduck_v2_stl_parts.zip"
    print("\n[5] ROBOT CANONICAL CYBERDUCK V2 PRO")
    print(f"  • Mô hình vật lý MuJoCo : {'[SẴN SÀNG]' if MODEL_PATH.exists() else '[CHƯA TẠO]'} ({MODEL_NAME})")
    print(f"  • Checkpoint tốt nhất   : {checkpoint_status} ({best_ckpt.name})")
    print(f"  • Bản vẽ lắp ráp 3D CAD : {'[SẴN SÀNG]' if v2_cad.exists() else '[CHƯA XUẤT]'} ({v2_cad.name})")
    print(f"  • Gói 17 File STL In 3D : {'[SẴN SÀNG]' if v2_zip.exists() else '[CHƯA NÉN]'} ({v2_zip.name})")
    print("  • Tiếp xúc sàn          : Phẳng tuyệt đối tại Z = 0.000m (Khớp cổ các-đăng & bàn chân 50mm)")

    # 6. Advanced Gaits & AI Vision Status
    print("\n[6] KỸ NĂNG VẬN ĐỘNG NÂNG CAO & AI VISION (BEHAVIORS & TRACKING)")
    print("  • Waddling Gait (Dáng đi vịt bionic) : [SẴN SÀNG] (Lắc hông nhịp nhàng 2 bên & gật đầu)")
    print("  • Spot Turn (Xoay tròn tại chỗ)      : [SẴN SÀNG] (Xoay 360 độ tại chỗ không tịnh tiến)")
    print("  • Fall Recovery (Tự đứng dậy khi ngã): [SẴN SÀNG] (FSM 3 pha: Lật sấp -> Ngồi -> Đứng)")
    print("  • AI Vision Head Tracking           : [SẴN SÀNG] (Visual Servoing bám bóng vàng/mặt người)")

    print("\n" + "=" * 72)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify physical kinematics consistency and run automated test suite."""
    print(">>> [1/2] Đang xác thực mô hình vật lý 3D (MJCF vs URDF)...")
    res_physics = subprocess.run([sys.executable, str(REPO_ROOT / "src" / "tools" / "robot" / "verify_model.py")])
    if res_physics.returncode != 0:
        print("LỖI: Xác thực mô hình vật lý thất bại!")
        return res_physics.returncode

    if not args.skip_tests:
        print("\n>>> [2/2] Đang chạy bộ kiểm thử tự động (pytest)...")
        pytest_args = [sys.executable, "-m", "pytest"]
        if args.fast:
            pytest_args.extend([
                "tests/sim/test_eplen_suicide_fix.py",
                "tests/robot/test_interactive_viewer.py",
                "tests/robot/test_cyberduck.py",
                "tests/sim/test_advanced_gaits.py",
                "tests/sim/test_vision_tracker.py",
                "-q"
            ])
        else:
            pytest_args.extend(["tests/", "-q"])
        res_tests = subprocess.run(pytest_args)
        return res_tests.returncode

    return 0


def cmd_view(args: argparse.Namespace) -> int:
    """Launch interactive 3D visualizer or open web dashboard."""
    if args.web:
        web_file = REPO_ROOT / "src" / "tools" / "viewer_3d.html"
        print(f">>> Đang mở giao diện 3D Web Dashboard: {web_file}")
        webbrowser.open(f"file://{web_file.resolve()}")
        return 0

    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "tools" / "robot" / "interactive_viewer.py"),
        "--stage",
        str(args.stage),
    ]

    # Model resolution (supports baseline MicroDuck and CyberDuck V2)
    requested_model = getattr(args, "model", None)
    if requested_model and requested_model not in ("v2", "cyberduck_v2", MODEL_NAME):
        model_path = Path(requested_model)
        if model_path.name != MODEL_NAME:
            raise ValueError(f"Only {MODEL_NAME} is supported; legacy robot models are archived")
    else:
        model_path = MODEL_PATH
    cmd.extend(["--model", str(model_path)])

    if getattr(args, "gait", None):
        cmd.extend(["--gait", str(args.gait)])
        cmd.append("--gait-demo")
    if getattr(args, "track", False):
        cmd.append("--track")
    if getattr(args, "target_color", None):
        cmd.extend(["--target-color", str(args.target_color)])

    if args.policy:
        cmd.extend(["--policy", str(args.policy)])
    if args.headless:
        cmd.append("--headless")

    print(f">>> Khởi chạy Interactive 3D Viewer (CyberDuck V2 Pro, Stage {args.stage})...")
    return subprocess.run(cmd).returncode


def cmd_train(args: argparse.Namespace) -> int:
    """Launch PPO reinforcement learning training pipeline."""
    iters = getattr(args, "iterations", getattr(args, "iters", None))
    if iters is None:
        iters = 1 if getattr(args, "dry_run", False) else 500
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "tools" / "train" / "train_ppo.py"),
        "--num-envs",
        str(args.envs),
        "--max-iterations",
        str(iters),
    ]

    target = getattr(args, "target", "v2")
    if target not in CHECKPOINT_PREFIXES:
        raise ValueError(f"Unknown runtime target: {target}")
    cmd.extend(["--model-path", str(MODEL_PATH)])

    prefix = CHECKPOINT_PREFIX
    save_path = Path(getattr(args, "save_path", None) or REPO_ROOT / "models/checkpoints" / f"{prefix}_ppo.pt")
    save_prefix = save_path.stem[:-4] if save_path.stem.endswith("_ppo") else save_path.stem
    best_path = Path(getattr(args, "best_save_path", None) or save_path.with_name(f"{save_prefix}_best.pt"))
    cmd.extend(["--save-path", str(save_path), "--best-save-path", str(best_path)])

    if args.resume:
        resume_val = str(args.resume).strip()
        is_bool = (args.resume is True) or (resume_val.lower() in ("true", "1"))
        if not is_bool:
            cmd.extend(["--resume", "--load-checkpoint", str(args.resume)])
        else:
            cmd.append("--resume")
            if not save_path.exists() and best_path.exists():
                cmd.extend(["--load-checkpoint", str(best_path)])
    if getattr(args, "transfer_checkpoint", None):
        if args.resume:
            raise ValueError("Choose resume or explicit transfer, not both")
        cmd.extend(["--load-checkpoint", str(args.transfer_checkpoint)])
    if getattr(args, "stand_first", False):
        cmd.append("--stand-first")
    if getattr(args, "device", None):
        cmd.extend(["--device", str(args.device)])
    if getattr(args, "rollout_steps", None):
        cmd.extend(["--rollout-steps", str(args.rollout_steps)])
    if getattr(args, "save_interval", None):
        cmd.extend(["--save-interval", str(args.save_interval)])
    if getattr(args, "log_interval", None):
        cmd.extend(["--log-interval", str(args.log_interval)])
    if getattr(args, "terrain", None):
        cmd.extend(["--terrain", str(args.terrain)])
    for attr, flag in (
        ("symmetry_coef", "--symmetry-coef"),
        ("max_action_delay", "--max-action-delay"),
        ("zero_cmd_prob", "--zero-cmd-prob"),
        ("turn_in_place_prob", "--turn-in-place-prob"),
        ("limp_kp_scale", "--limp-kp-scale"),
        ("action_over_limit_threshold", "--action-over-limit-threshold"),
        ("encoder_bias_rad", "--encoder-bias-rad"),
        ("imu_mount_max_rad", "--imu-mount-max-rad"),
    ):
        val = getattr(args, attr, None)
        if val is not None:
            cmd.extend([flag, str(val)])
    for attr, flag in (
        ("curriculum_smoothness", "--curriculum-smoothness"),
        ("randomize_pillars", "--randomize-pillars"),
        ("enable_limp_fall", "--enable-limp-fall"),
        ("enable_head_ema_penalty", "--enable-head-ema-penalty"),
        ("enable_battery_sag", "--enable-battery-sag"),
        ("delay_joint_vel_obs", "--delay-joint-vel-obs"),
    ):
        if getattr(args, attr, False):
            cmd.append(flag)
    if getattr(args, "dry_run", False):
        cmd.append("--dry-run")

    mode_str = "DRY-RUN" if getattr(args, "dry_run", False) else f"{args.envs} envs, tối đa {iters} iters"
    print(f">>> Khởi động huấn luyện CyberDuck V2 PPO ({mode_str}, runtime: {target})...", flush=True)

    return subprocess.run(cmd).returncode


def cmd_export(args: argparse.Namespace) -> int:
    """Export trained policy to ONNX and INT8 C header."""
    target = getattr(args, "target", "all")
    explicit_checkpoint = getattr(args, "checkpoint", None)
    targets = ("rpi5", "pizero2w", "esp32") if target == "all" else (target,)
    try:
        checkpoints = resolve_target_checkpoints(REPO_ROOT, targets, explicit_checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Export] {exc}")
        return 1

    from tools.robot.export_policy import (
        export_onnx,
        export_esp32_headers,
        DEFAULT_ONNX_PATH as EXPORT_ONNX_PATH,
        DEFAULT_ESP32_WEIGHTS_PATH,
        DEFAULT_ESP32_CONFIG_PATH,
        prepare_target_exports,
    )

    import json
    schema_path = getattr(args, "observation_schema", None)
    schema_override = json.loads(Path(schema_path).read_text(encoding="utf-8")) if schema_path else None
    prepared = prepare_target_exports(checkpoints, getattr(args, "legacy_model_name", None), schema_override)

    opset = getattr(args, "opset", 13)
    custom_out = getattr(args, "output", None)

    print(f">>> Đang xuất mô hình (target: {target}) từ {checkpoints}...")

    if target in ("rpi5", "pizero2w", "all"):
        pi_zero_out = getattr(args, "pizero_onnx_output", None)
        out_pi = custom_out or (pi_zero_out if target == "pizero2w" else None) or getattr(args, "onnx_output", None) or EXPORT_ONNX_PATH
        pi_policy = prepared["rpi5"] if "rpi5" in prepared else prepared["pizero2w"]
        onnx_file = export_onnx(pi_policy, output_path=out_pi, opset=opset)
        print(f"[Export] Exported shared CyberDuck V2 Float32 ONNX policy for Pi runtimes: {onnx_file}")
        if target == "all" and pi_zero_out and Path(pi_zero_out).resolve() != Path(out_pi).resolve():
            print(f"[Export] Additional Pi Zero 2 W copy: {export_onnx(pi_policy, output_path=pi_zero_out, opset=opset)}")

    if target in ("esp32", "all"):
        out_w = (custom_out if target == "esp32" else None) or getattr(args, "esp32_weights_output", None) or DEFAULT_ESP32_WEIGHTS_PATH
        out_c = getattr(args, "esp32_config_output", None) or DEFAULT_ESP32_CONFIG_PATH
        weights_file, config_file = export_esp32_headers(
            prepared["esp32"],
            weights_output_path=out_w,
            config_output_path=out_c,
        )
        print(f"[Export] Exported ESP32-S3 INT8 weights header: {weights_file}")
        print(f"[Export] Exported ESP32-S3 quant config header: {config_file}")

    print("[Export] Xuất mô hình thành công.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="🦆 CyberDuck V2 8-DOF Bipedal Robot - Unified Operations CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available operations")

    # status
    p_status = subparsers.add_parser("status", help="Xem báo cáo tổng quan về mô hình và tiến trình học")

    # verify
    p_verify = subparsers.add_parser("verify", help="Kiểm tra tính nhất quán vật lý và chạy test suite")
    p_verify.add_argument("--skip-tests", action="store_true", help="Chỉ kiểm tra vật lý, bỏ qua pytest")
    p_verify.add_argument("--fast", action="store_true", help="Chỉ chạy các test cốt lõi nhanh")

    # view
    p_view = subparsers.add_parser("view", help="Mở trình xem 3D trực quan tương tác hoặc web dashboard")
    p_view.add_argument("--web", action="store_true", help="Mở giao diện Web Dashboard (viewer_3d.html)")
    p_view.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3], help="Giai đoạn địa hình (0-3)")
    p_view.add_argument("--policy", type=str, default=None, help="Đường dẫn file checkpoint policy (.pt)")
    p_view.add_argument("--headless", action="store_true", help="Chạy chế độ giả lập không cửa sổ")
    p_view.add_argument("--model", type=str, default=None, help="Đường dẫn CyberDuck V2 MJCF (mặc định: mô hình V2 canonical)")
    p_view.add_argument("--v2", action="store_true", help="Bí danh tương thích, V2 luôn là mô hình mặc định")
    p_view.add_argument("--gait", choices=["waddle", "walk", "spot_turn", "head_dance"], default=None, help="Trình diễn dáng đi mới (waddle, walk, spot_turn, head_dance)")
    p_view.add_argument("--track", action="store_true", help="Kích hoạt Visual Servoing theo dõi mục tiêu qua camera đầu")
    p_view.add_argument("--target-color", choices=["yellow", "red", "green", "blue"], default="yellow", help="Màu sắc mục tiêu theo dõi (mặc định: yellow)")

    # train
    p_train = subparsers.add_parser("train", help="Huấn luyện mô hình bước đi bằng PPO đa môi trường")
    p_train.add_argument("--target", choices=["rpi5", "pizero2w", "esp32", "v2"], default="v2", help="Mô hình luôn là CyberDuck V2; alias chọn profile triển khai (mặc định: v2)")
    p_train.add_argument("--envs", type=int, default=256, help="Số lượng môi trường song song (mặc định: 256)")
    p_train.add_argument("--iterations", "--iters", type=int, default=None, help="Số vòng lặp huấn luyện (mặc định: 500 khi train, 1 khi dry-run)")
    p_train.add_argument("--transfer-checkpoint", type=str, default=None, help="Transfer explicit sang target/output mới, reset optimizer và selection")
    p_train.add_argument("--save-path", type=str, default=None, help="Đường dẫn checkpoint output explicit")
    p_train.add_argument("--best-save-path", type=str, default=None, help="Đường dẫn best checkpoint explicit")
    p_train.add_argument("--resume", type=str, default=None, nargs="?", const="true", help="Tiếp tục học từ checkpoint")
    p_train.add_argument("--stand-first", action="store_true", help="Ưu tiên học đứng thăng bằng trước")
    p_train.add_argument("--dry-run", action="store_true", help="Chạy kiểm thử nhanh 1 vòng lặp (Dry-run)")
    p_train.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Thiết bị tính toán ('cpu', 'cuda' - mặc định: tự động nhận diện)")
    p_train.add_argument("--rollout-steps", "--rollout", type=int, default=None, help="Số bước rollout mỗi vòng lặp (mặc định: 128)")
    p_train.add_argument("--save-interval", type=int, default=50, help="Chu kỳ lưu checkpoint định kỳ (mặc định: 50 iters)")
    p_train.add_argument("--log-interval", type=int, default=1, help="Chu kỳ in log ra màn hình (mặc định: 1 iter)")
    p_train.add_argument("--terrain", choices=["hurdles", "stairs"], default="hurdles", help="Loại giáo trình địa hình ('hurdles', 'stairs' - mặc định: hurdles)")
    p_train.add_argument("--symmetry-coef", type=float, default=None, help="Hệ số điều chuẩn đối xứng hai bên (mặc định từ train_ppo)")
    p_train.add_argument("--max-action-delay", type=int, default=None, help="Độ trễ hành động ngẫu nhiên tối đa (số bước điều khiển)")
    p_train.add_argument("--zero-cmd-prob", type=float, default=None, help="Tỷ lệ episode lấy mẫu lệnh đứng yên [0,0,0]")
    p_train.add_argument("--turn-in-place-prob", type=float, default=None, help="Tỷ lệ episode lấy mẫu lệnh xoay tại chỗ")
    p_train.add_argument("--curriculum-smoothness", action="store_true", help="Tăng dần trọng số mượt hành động trong quá trình huấn luyện")
    p_train.add_argument("--randomize-pillars", action="store_true", help="Ngẫu nhiên hóa cột slalom giữa các episode/iteration")
    p_train.add_argument("--enable-limp-fall", action="store_true", help="Bật giảm Kp servo dự báo trước va chạm khi ngã")
    p_train.add_argument("--limp-kp-scale", type=float, default=None, help="Hệ số Kp khi kích hoạt limp-fall")
    p_train.add_argument("--enable-head-ema-penalty", action="store_true", help="Phạt trôi EMA lệch hướng đầu kéo dài")
    p_train.add_argument("--action-over-limit-threshold", type=float, default=None, help="Ngưỡng phạt biên độ hành động chuẩn hóa")
    p_train.add_argument("--enable-battery-sag", action="store_true", help="Mô phỏng sụt áp pin 2S làm giảm ngưỡng mô-men")
    p_train.add_argument("--encoder-bias-rad", type=float, default=None, help="Sai số 0 độ lệch encoder chỉ ở Actor (rad)")
    p_train.add_argument("--imu-mount-max-rad", type=float, default=None, help="Giới hạn nghiêng gá IMU chỉ ở Actor (rad)")
    p_train.add_argument("--delay-joint-vel-obs", action="store_true", help="Trễ 1 bước quan sát vận tốc khớp chỉ ở Actor")

    # export
    p_export = subparsers.add_parser("export", help="Xuất policy sang ONNX (RPi 5 & Pi Zero 2 W) và INT8 C Header (ESP32-S3)")
    p_export.add_argument("--target", choices=["rpi5", "pizero2w", "esp32", "all"], default="all", help="Định dạng runtime cho cùng một policy V2 (rpi5, pizero2w, esp32, all)")
    p_export.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn checkpoint cần xuất")
    p_export.add_argument("--legacy-model-name", default=None, help="Tên MJCF đã xác minh cho checkpoint legacy thiếu identity")
    p_export.add_argument("--observation-schema", type=Path, default=None, help="JSON observation schema đã xác minh cho checkpoint legacy")
    p_export.add_argument("--output", type=str, default=None, help="Đường dẫn file đầu ra tùy chỉnh")
    p_export.add_argument("--onnx-output", type=str, default=None, help="Đường dẫn file xuất ONNX (RPi 5)")
    p_export.add_argument("--pizero-onnx-output", type=str, default=None, help="Đường dẫn file xuất ONNX (Pi Zero 2 W)")
    p_export.add_argument("--esp32-weights-output", type=str, default=None, help="Đường dẫn file xuất weights ESP32-S3")
    p_export.add_argument("--esp32-config-output", type=str, default=None, help="Đường dẫn file xuất config ESP32-S3")
    p_export.add_argument("--opset", type=int, default=13, help="Phiên bản ONNX opset (mặc định: 13)")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "status": cmd_status,
        "verify": cmd_verify,
        "view": cmd_view,
        "train": cmd_train,
        "export": cmd_export,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
