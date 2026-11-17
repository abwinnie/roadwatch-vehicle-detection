"""Detect road vehicles in a live stream with the official YOLOX-S ONNX model."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Sequence
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import onnxruntime as ort
import requests
from dotenv import load_dotenv


DEFAULT_THB_STREAM_URL = "http://cctv-ss04.thb.gov.tw/T1-92K+450"
DEFAULT_TAINAN_STREAM_URL = "https://trafficvideo3.tainan.gov.tw/aed99d50"
DEFAULT_MODEL = Path("weights/yolox_s.onnx")
YOLOX_S_ONNX_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
    "0.1.1rc0/yolox_s.onnx"
)
MODEL_INPUT_SIZE = (640, 640)
PANEL_SIZE = (320, 240)
COCO_NAMES = {
    1: "BIKE",
    2: "CAR",
    3: "MOTO",
    5: "BUS",
    7: "TRUCK",
}
CAR_CLASS_IDS = frozenset({2, 5, 7})
MOTO_CLASS_IDS = frozenset({1, 3})
TARGET_CLASS_IDS = tuple(sorted(CAR_CLASS_IDS | MOTO_CLASS_IDS))
SCREENSHOT_DIR = Path("01_監視器")


def parse_confidence(value: str) -> float:
    try:
        confidence = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("信心閾值必須是 0 到 1 之間的數字。") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise argparse.ArgumentTypeError("信心閾值必須介於 0 到 1。")
    return confidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 YOLOX-S 連續辨識公路監視器串流中的車輛。"
    )
    parser.add_argument(
        "--left-source",
        default=os.getenv("STREAM_URL_THB") or os.getenv("STREAM_URL") or DEFAULT_THB_STREAM_URL,
        help="左側監視器的 RTSP/HTTP 串流網址、影片路徑或攝影機編號。",
    )
    parser.add_argument(
        "--right-source",
        default=os.getenv("STREAM_URL_TAINAN") or DEFAULT_TAINAN_STREAM_URL,
        help="右側監視器的 RTSP/HTTP 串流網址、影片路徑或攝影機編號。",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="YOLOX ONNX 模型路徑（預設自動下載官方 YOLOX-S）。",
    )
    parser.add_argument(
        "--confidence",
        type=parse_confidence,
        default=0.3,
        help="偵測信心閾值，介於 0 到 1（預設：0.3）。",
    )
    return parser


class MJPEGStreamSource:
    """Read JPEG frames continuously from a multipart MJPEG response."""

    def __init__(self, response: requests.Response) -> None:
        self._response = response
        self._chunks = response.iter_content(chunk_size=4096)
        self._buffer = bytearray()
        self._closed = False

    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        if self._closed:
            return False, None

        while True:
            start = self._buffer.find(b"\xff\xd8")
            end = self._buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                jpeg = bytes(self._buffer[start : end + 2])
                del self._buffer[: end + 2]
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
                continue

            try:
                chunk = next(self._chunks)
            except StopIteration:
                self.release()
                return False, None
            except requests.RequestException:
                self.release()
                return False, None

            if chunk:
                self._buffer.extend(chunk)
                if len(self._buffer) > 8 * 1024 * 1024:
                    last_start = self._buffer.rfind(b"\xff\xd8")
                    if last_start >= 0:
                        del self._buffer[:last_start]
                    else:
                        self._buffer.clear()

    def release(self) -> None:
        if not self._closed:
            self._closed = True
            self._response.close()


def open_capture(source: str) -> cv2.VideoCapture | MJPEGStreamSource:
    if source.lower().startswith(("http://", "https://")):
        headers = {
            "Accept": "*/*",
            "Referer": source,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
        }
        try:
            response = requests.get(source, headers=headers, stream=True, timeout=(10, 10))
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError("無法連接 HTTP 影像來源。") from exc

        if "multipart/x-mixed-replace" in response.headers.get("Content-Type", "").lower():
            return MJPEGStreamSource(response)
        response.close()

    capture_source: str | int = int(source) if source.isdecimal() else source
    capture = cv2.VideoCapture(capture_source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("無法開啟影像來源；請確認網址、網路連線與串流格式。")
    return capture


def is_network_stream(source: str) -> bool:
    return source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://"))


def ensure_model(model_path: Path) -> None:
    if model_path.is_file():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = model_path.with_suffix(model_path.suffix + ".part")
    print("下載官方 YOLOX-S ONNX 模型（約 36 MB）…")
    try:
        with requests.get(YOLOX_S_ONNX_URL, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with partial_path.open("wb") as model_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        model_file.write(chunk)
        if partial_path.stat().st_size < 1_000_000:
            raise RuntimeError("下載的 YOLOX 模型檔案不完整。")
        partial_path.replace(model_path)
    except (OSError, requests.RequestException, RuntimeError) as exc:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"無法下載 YOLOX-S 模型：{exc}") from exc


def preprocess(frame: cv2.typing.MatLike) -> tuple[np.ndarray, float]:
    input_height, input_width = MODEL_INPUT_SIZE
    height, width = frame.shape[:2]
    ratio = min(input_height / height, input_width / width)
    resized_width = int(width * ratio)
    resized_height = int(height * ratio)
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    padded = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    padded[:resized_height, :resized_width] = resized
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)[None, ...]
    return tensor, ratio


def decode_yolox_output(output: np.ndarray) -> np.ndarray:
    input_height, input_width = MODEL_INPUT_SIZE
    strides = (8, 16, 32)
    grids: list[np.ndarray] = []
    expanded_strides: list[np.ndarray] = []

    for stride in strides:
        grid_height = input_height // stride
        grid_width = input_width // stride
        yv, xv = np.meshgrid(np.arange(grid_height), np.arange(grid_width), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((1, grid.shape[1], 1), stride, dtype=np.float32))

    grid_values = np.concatenate(grids, axis=1)
    stride_values = np.concatenate(expanded_strides, axis=1)
    predictions = np.asarray(output, dtype=np.float32).copy()
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.shape[0] != grid_values.shape[1]:
        raise RuntimeError(
            f"YOLOX 輸出格數 {predictions.shape[0]} 與 640×640 模型設定不相符。"
        )

    predictions[:, :2] = (predictions[:, :2] + grid_values[0]) * stride_values[0]
    predictions[:, 2:4] = np.exp(np.clip(predictions[:, 2:4], -20, 20)) * stride_values[0]
    return predictions


def nms_indices(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    kept: list[int] = []

    while order.size:
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        if not remaining.size:
            break

        intersection_x1 = np.maximum(x1[current], x1[remaining])
        intersection_y1 = np.maximum(y1[current], y1[remaining])
        intersection_x2 = np.minimum(x2[current], x2[remaining])
        intersection_y2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
            0.0, intersection_y2 - intersection_y1
        )
        union = areas[current] + areas[remaining] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[overlap <= threshold]

    return kept


def detect(
    session: ort.InferenceSession,
    frame: cv2.typing.MatLike,
    confidence_threshold: float,
) -> list[tuple[np.ndarray, float, int]]:
    input_tensor, ratio = preprocess(frame)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]
    predictions = decode_yolox_output(output)

    boxes = predictions[:, :4].copy()
    boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2
    boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2
    boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2
    boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2
    boxes /= ratio

    class_scores = predictions[:, 4:5] * predictions[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    eligible = (scores >= confidence_threshold) & np.isin(class_ids, TARGET_CLASS_IDS)

    height, width = frame.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)

    detections: list[tuple[np.ndarray, float, int]] = []
    for class_id in TARGET_CLASS_IDS:
        class_indices = np.flatnonzero(eligible & (class_ids == class_id))
        if not len(class_indices):
            continue
        kept = nms_indices(boxes[class_indices], scores[class_indices], threshold=0.45)
        for local_index in kept:
            prediction_index = class_indices[local_index]
            detections.append(
                (boxes[prediction_index].copy(), float(scores[prediction_index]), class_id)
            )
    return detections


def save_screenshot(frame: cv2.typing.MatLike) -> Path:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d%H%M%S")
    path = SCREENSHOT_DIR / f"{timestamp}.jpg"
    suffix = 1
    while path.exists():
        path = SCREENSHOT_DIR / f"{timestamp}_{suffix:02d}.jpg"
        suffix += 1

    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError("無法將目前畫面編碼為 JPEG。")
    path.write_bytes(encoded.tobytes())
    return path


def annotate_frame(
    frame: cv2.typing.MatLike,
    detections: list[tuple[np.ndarray, float, int]],
    camera_name: str,
) -> cv2.typing.MatLike:
    current_cars = sum(class_id in CAR_CLASS_IDS for _, _, class_id in detections)
    current_motos = sum(class_id in MOTO_CLASS_IDS for _, _, class_id in detections)
    frame_height, frame_width = frame.shape[:2]
    label_scale = max(0.28, min(0.32, min(frame_width, frame_height) / 750))
    label_thickness = 1

    for box, score, class_id in detections:
        if class_id in CAR_CLASS_IDS:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        x1, y1, x2, y2 = np.rint(box).astype(int)
        label = f"{COCO_NAMES[class_id]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            color,
            label_thickness,
            cv2.LINE_AA,
        )

    text_lines = (f"Cars: {current_cars}", f"Moto/Bike: {current_motos}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    padding = max(4, round(min(frame_width, frame_height) * 0.02))
    max_panel_width = max(80, frame_width // 2)
    max_panel_height = max(48, frame_height // 2)
    max_text_width = max_panel_width - 2 * padding
    max_text_height = (max_panel_height - 2 * padding) // len(text_lines)

    base_width = max(cv2.getTextSize(text, font, 1.0, thickness)[0][0] for text in text_lines)
    base_height = max(cv2.getTextSize(text, font, 1.0, thickness)[0][1] for text in text_lines)
    font_scale = min(0.52, max_text_width / max(base_width, 1), max_text_height / max(base_height, 1))
    font_scale = max(0.28, font_scale)
    line_sizes = [cv2.getTextSize(text, font, font_scale, thickness) for text in text_lines]
    line_gap = max(2, round(font_scale * 6))
    panel_width = max(size[0][0] for size in line_sizes) + 2 * padding
    panel_height = sum(size[0][1] + size[1] for size in line_sizes) + line_gap + 2 * padding

    panel = frame.copy()
    cv2.rectangle(panel, (8, 8), (8 + panel_width, 8 + panel_height), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.68, frame, 0.32, 0, frame)
    y = 8 + padding
    for text, (size, baseline), color in zip(
        text_lines,
        line_sizes,
        ((70, 70, 255), (90, 255, 90)),
    ):
        y += size[1]
        cv2.putText(frame, text, (8 + padding, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += baseline + line_gap

    title_scale = 0.34
    title_size = cv2.getTextSize(camera_name, font, title_scale, 1)[0]
    title_x = max(4, frame_width - title_size[0] - 7)
    cv2.putText(
        frame,
        camera_name,
        (title_x, frame_height - 7),
        font,
        title_scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return frame


def fit_panel(
    frame: cv2.typing.MatLike,
    detections: list[tuple[np.ndarray, float, int]],
    camera_name: str,
) -> cv2.typing.MatLike:
    panel_width, panel_height = PANEL_SIZE
    frame_height, frame_width = frame.shape[:2]
    scale = min(panel_width / frame_width, panel_height / frame_height)
    resized_width = max(1, round(frame_width * scale))
    resized_height = max(1, round(frame_height * scale))
    offset_x = (panel_width - resized_width) // 2
    offset_y = (panel_height - resized_height) // 2
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized

    panel_detections = []
    for box, score, class_id in detections:
        scaled_box = box.copy() * scale
        scaled_box[[0, 2]] += offset_x
        scaled_box[[1, 3]] += offset_y
        panel_detections.append((scaled_box, score, class_id))
    return annotate_frame(panel, panel_detections, camera_name)


def make_status_panel(camera_name: str, status: str) -> cv2.typing.MatLike:
    panel = np.zeros((PANEL_SIZE[1], PANEL_SIZE[0], 3), dtype=np.uint8)
    cv2.putText(
        panel,
        camera_name,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        status,
        (12, PANEL_SIZE[1] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


class StreamWorker:
    """Read one source independently so a stalled camera cannot freeze the other."""

    def __init__(self, name: str, source: str) -> None:
        self.name = name
        self.source = source
        self._lock = Lock()
        self._stop_event = Event()
        self._thread = Thread(target=self._read_loop, name=f"stream-{name}", daemon=True)
        self._frame: cv2.typing.MatLike | None = None
        self._sequence = 0
        self._frame_time = 0.0
        self._status = "CONNECTING"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)

    def snapshot(self) -> tuple[cv2.typing.MatLike | None, int, float, str]:
        with self._lock:
            return self._frame, self._sequence, self._frame_time, self._status

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def _read_loop(self) -> None:
        network_source = is_network_stream(self.source)
        while not self._stop_event.is_set():
            capture = None
            try:
                capture = open_capture(self.source)
                self._set_status("LIVE")
                while not self._stop_event.is_set():
                    success, frame = capture.read()
                    if not success or frame is None:
                        raise RuntimeError("stream ended")
                    with self._lock:
                        self._frame = frame
                        self._sequence += 1
                        self._frame_time = time.monotonic()
                        self._status = "LIVE"
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._set_status("RECONNECTING")
                print(f"{self.name} 串流中斷：{exc}", file=sys.stderr)
            finally:
                if capture is not None:
                    capture.release()

            if not network_source:
                self._set_status("SOURCE ENDED")
                return
            self._stop_event.wait(3)


def run(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    ensure_model(model_path)
    print("載入 YOLOX-S ONNX 模型…")
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    workers = [
        StreamWorker("THB 92K+450", args.left_source),
        StreamWorker("TAINAN AED", args.right_source),
    ]
    panels = [make_status_panel(worker.name, "CONNECTING") for worker in workers]
    last_sequences = [-1, -1]
    divider = np.full((PANEL_SIZE[1], 3, 3), (80, 80, 80), dtype=np.uint8)
    # OpenCV's Windows title bar can garble non-ASCII text on some builds.
    window_name = "YOLOX - Dual Camera Vehicle Detection"
    try:
        for worker in workers:
            worker.start()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, PANEL_SIZE[0] * 2 + divider.shape[1], PANEL_SIZE[1])
        print("YOLOX 雙監視器串流辨識已啟動（左：公路局；右：臺南）：按 Enter 截圖，按 Esc 離開。")
        while True:
            now = time.monotonic()
            for index, worker in enumerate(workers):
                frame, sequence, frame_time, status = worker.snapshot()
                if frame is not None and sequence != last_sequences[index]:
                    try:
                        detections = detect(session, frame, args.confidence)
                        panels[index] = fit_panel(frame, detections, worker.name)
                        last_sequences[index] = sequence
                    except Exception as exc:
                        print(f"{worker.name} 影格辨識失敗：{exc}", file=sys.stderr)
                if frame is None:
                    panels[index] = make_status_panel(worker.name, status)
                elif now - frame_time > 10:
                    stale_panel = panels[index].copy()
                    cv2.putText(
                        stale_panel,
                        "RECONNECTING",
                        (8, PANEL_SIZE[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        (0, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    panels[index] = stale_panel

            combined = np.hstack((panels[0], divider, panels[1]))
            cv2.imshow(window_name, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (10, 13):
                try:
                    screenshot_path = save_screenshot(combined)
                    print(f"截圖已存檔：{screenshot_path}")
                except Exception as exc:
                    print(f"截圖處理失敗，雙串流會繼續：{exc}", file=sys.stderr)
            elif key == 27:
                break
    finally:
        for worker in workers:
            worker.stop()
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n已收到中斷訊號，結束串流。")
    except Exception as exc:
        print(f"執行失敗：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
