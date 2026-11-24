"""TDD tests for SSD1309 128x64 OLED expressive eye renderer and SPI hardware fallback."""

import logging
import numpy as np
import pytest


def test_eye_renderer_expressions_blink_gaze_and_rotate_180():
    from hardware.oled_eyes import EXPRESSIONS, EyeRenderer

    assert EXPRESSIONS == ("neutral", "happy", "curious", "alert", "sleepy")
    renderer = EyeRenderer()

    frames = {}
    for expr in EXPRESSIONS:
        frame = renderer.render(expression=expr)
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (64, 128)
        assert set(np.unique(frame)).issubset({0, 1})
        assert int(frame.sum()) > 50
        frames[expr] = frame

    for i, a in enumerate(EXPRESSIONS):
        for b in EXPRESSIONS[i + 1 :]:
            assert not np.array_equal(frames[a], frames[b]), f"{a} and {b} frames must differ"

    with pytest.raises(ValueError, match="expression"):
        renderer.render(expression="unknown_mood")

    # blink_ratio=1.0 closes the eyes sharply
    closed = renderer.render(expression="neutral", blink_ratio=1.0)
    assert int(closed.sum()) < int(frames["neutral"].sum()) * 0.4

    # rotate_180=True is exact 180-degree pixel flip of rotate_180=False
    unrot = renderer.render(expression="neutral", gaze_x=0.6, rotate_180=False)
    rot = renderer.render(expression="neutral", gaze_x=0.6, rotate_180=True)
    np.testing.assert_array_equal(rot, unrot[::-1, ::-1])

    # gaze_x shifts illuminated horizontal centroid consistently
    left_gaze = renderer.render(expression="neutral", gaze_x=-0.8, rotate_180=False)
    right_gaze = renderer.render(expression="neutral", gaze_x=0.8, rotate_180=False)
    xs = np.arange(128, dtype=np.float32)[None, :]
    cx_left = float((left_gaze * xs).sum() / max(int(left_gaze.sum()), 1))
    cx_right = float((right_gaze * xs).sum() / max(int(right_gaze.sum()), 1))
    assert cx_right > cx_left + 4.0


def test_ssd1309_display_mock_and_graceful_spi_fallback(caplog, monkeypatch):
    import builtins
    from hardware.oled_eyes import EyeRenderer, SSD1309Display

    renderer = EyeRenderer()
    frame = renderer.render("happy")

    mock_disp = SSD1309Display(mock=True)
    assert mock_disp.rotate_180 is True
    assert mock_disp.show(frame) is True
    np.testing.assert_array_equal(mock_disp.last_frame, frame)

    # Simulate missing spidev/luma drivers without crashing 50 Hz loop; warn at most once
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name.startswith(("luma", "spidev")):
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    hw_disp = SSD1309Display(mock=False)
    with caplog.at_level(logging.WARNING):
        assert hw_disp.show(frame) is False
        assert hw_disp.show(frame) is False
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
