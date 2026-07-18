"""Desktop GUI for YOLO object detection and ByteTrack object tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
from collections import Counter, defaultdict, deque
from time import perf_counter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


class DetectionTrackerApp:
    """Tkinter application that displays tracked YOLO detections in real time."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("YOLO Object Detection and Tracking")
        self.root.geometry("1000x760")
        self.root.minsize(720, 560)

        self.model_path = Path("yolo11n.pt")
        self.model: Optional[YOLO] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.running = False
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.frame_count = 0
        self.last_frame_time: Optional[float] = None
        self.track_history: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=30))

        self.confidence = tk.DoubleVar(value=0.35)
        self.camera_index = tk.StringVar(value="0")
        self.status = tk.StringVar(value="Choose a source, then start detection.")
        self.object_count = tk.StringVar(value="Detected objects: 0")
        self.class_counts = tk.StringVar(value="No detections yet.")
        self.fps = tk.StringVar(value="FPS: --")
        self.save_path: Optional[str] = None

        self._build_interface()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_interface(self) -> None:
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Camera:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            controls, textvariable=self.camera_index, values=("0", "1", "2", "3"), width=3, state="readonly"
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Start Webcam", command=self.start_webcam).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(controls, text="Open Video File", command=self.open_video).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Button(controls, text="Save Output As…", command=self.choose_save_path).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Button(controls, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=8)

        ttk.Label(controls, text="Confidence:").pack(side=tk.LEFT, padx=(25, 5))
        ttk.Scale(
            controls,
            from_=0.1,
            to=0.9,
            variable=self.confidence,
            orient=tk.HORIZONTAL,
            length=160,
        ).pack(side=tk.LEFT)

        content = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        video_frame = ttk.Frame(content)
        content.add(video_frame, weight=4)
        self.video_label = ttk.Label(
            video_frame,
            text="Video output will appear here",
            anchor=tk.CENTER,
            background="#202020",
            foreground="white",
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        count_frame = ttk.LabelFrame(content, text="Detected Objects", padding=12)
        content.add(count_frame, weight=1)
        ttk.Label(count_frame, textvariable=self.object_count, font=("Segoe UI", 12, "bold")).pack(
            anchor=tk.W, pady=(0, 10)
        )
        ttk.Label(count_frame, textvariable=self.fps).pack(anchor=tk.W, pady=(0, 10))
        ttk.Separator(count_frame).pack(fill=tk.X, pady=(0, 10))
        ttk.Label(count_frame, text="Current frame", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(count_frame, textvariable=self.class_counts, justify=tk.LEFT, wraplength=190).pack(
            anchor=tk.W, pady=(6, 0)
        )

        ttk.Label(self.root, textvariable=self.status, padding=(12, 6)).pack(fill=tk.X)

    def load_model(self) -> bool:
        if self.model is not None:
            return True
        if not self.model_path.exists():
            messagebox.showerror("Model missing", f"Could not find {self.model_path.resolve()}")
            return False
        try:
            self.status.set("Loading YOLO model…")
            self.root.update_idletasks()
            self.model = YOLO(str(self.model_path))
            return True
        except Exception as error:
            messagebox.showerror("Model error", str(error))
            self.status.set("Could not load the YOLO model.")
            return False

    def start_webcam(self) -> None:
        index = int(self.camera_index.get())
        self.start_source(index, f"webcam {index}")

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.start_source(path, Path(path).name)

    def choose_save_path(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save annotated video",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if path:
            self.save_path = path
            self.status.set(f"Output will be saved to {Path(path).name} when detection starts.")

    def start_source(self, source: Union[int, str], description: str) -> None:
        self.stop()
        if not self.load_model():
            return
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            messagebox.showerror("Source error", f"Could not open the {description}.")
            return
        self.frame_count = 0
        self.last_frame_time = None
        self.track_history.clear()
        self.object_count.set("Detected objects: 0")
        self.class_counts.set("Loading video...")
        self.running = True
        self.status.set(f"Tracking objects from {description}. Click Stop to finish.")
        self.process_frame()

    def process_frame(self) -> None:
        if not self.running or self.capture is None or self.model is None:
            return

        ok, frame = self.capture.read()
        if not ok:
            self.status.set("Video finished.")
            self.stop(keep_status=True)
            return

        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence.get(),
            verbose=False,
        )[0]
        annotated = result.plot(labels=False, boxes=True)
        self._update_counts(result)

        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id, class_id in zip(
                result.boxes.xyxy.cpu().numpy().astype(int),
                result.boxes.id.int().cpu().tolist(),
                result.boxes.cls.int().cpu().tolist(),
            ):
                x1, y1, _, _ = box
                label = f"{self.model.names[class_id]}  ID {track_id}"
                center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                self.track_history[track_id].append(center)
                cv2.putText(
                    annotated, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
                )
                points = list(self.track_history[track_id])
                for start, end in zip(points, points[1:]):
                    cv2.line(annotated, start, end, (255, 255, 0), 2, cv2.LINE_AA)

        self._draw_fps(annotated)

        self._save_frame(annotated)
        self._show_frame(annotated)
        self.frame_count += 1
        self.root.after(1, self.process_frame)

    def _draw_fps(self, frame) -> None:
        """Calculate and draw processing FPS for the most recently rendered frame."""
        now = perf_counter()
        if self.last_frame_time is not None:
            fps_value = 1 / max(now - self.last_frame_time, 0.0001)
            self.fps.set(f"FPS: {fps_value:.1f}")
            cv2.putText(
                frame, self.fps.get(), (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2, cv2.LINE_AA,
            )
        self.last_frame_time = now

    def _update_counts(self, result) -> None:
        """Show total and per-class detections in the current video frame."""
        if result.boxes is None or len(result.boxes) == 0:
            self.object_count.set("Detected objects: 0")
            self.class_counts.set("No objects detected.")
            return

        class_ids = result.boxes.cls.int().cpu().tolist()
        counts = Counter(self.model.names[class_id] for class_id in class_ids)
        self.object_count.set(f"Detected objects: {len(class_ids)}")
        self.class_counts.set("\n".join(f"{name}: {count}" for name, count in counts.most_common()))

    def _show_frame(self, frame) -> None:
        maximum_width = max(self.video_label.winfo_width(), 640)
        maximum_height = max(self.video_label.winfo_height(), 480)
        height, width = frame.shape[:2]
        scale = min(maximum_width / width, maximum_height / height)
        display = cv2.resize(frame, (int(width * scale), int(height * scale)))
        image = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo, text="")

    def _save_frame(self, frame) -> None:
        if not self.save_path:
            return
        if self.writer is None:
            height, width = frame.shape[:2]
            fps = self.capture.get(cv2.CAP_PROP_FPS) if self.capture else 0
            self.writer = cv2.VideoWriter(
                self.save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (width, height)
            )
        self.writer.write(frame)

    def stop(self, keep_status: bool = False) -> None:
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if not keep_status:
            self.status.set("Stopped.")

    def close(self) -> None:
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    DetectionTrackerApp(window)
    window.mainloop()
