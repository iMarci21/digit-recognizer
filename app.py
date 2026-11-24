import tkinter as tk

from PIL import Image, ImageDraw

from predict import load_trained_model, predict

INITIAL_CANVAS_SIZE = 280
LIVE_PREDICT_DELAY_MS = 100


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.resizable(True, True)
        self.root.minsize(320, 420)

        # Current canvas dimensions (updated dynamically when window resizes)
        self.canvas_w = INITIAL_CANVAS_SIZE
        self.canvas_h = INITIAL_CANVAS_SIZE

        # Track previous mouse coordinates for smooth continuous lines
        self.last_x = None
        self.last_y = None

        # Live prediction state and scheduled Tkinter timer ID
        self.live_predict = True
        self._predict_job = None

        # In-memory PIL Image matching canvas dimensions (black background)
        self.pil_image = Image.new("L", (self.canvas_w, self.canvas_h), color=0)
        self.draw = ImageDraw.Draw(self.pil_image)

        # Load PyTorch model
        try:
            self.model, self.device = load_trained_model("model.pt")
            self.model_loaded = True
        except FileNotFoundError:
            self.model, self.device = None, None
            self.model_loaded = False

        self._build_ui()

    def _build_ui(self):
        # Header (pinned to top)
        self.title_label = tk.Label(
            self.root, text="Draw a digit (0-9) below", font=("Helvetica", 14, "bold")
        )
        self.title_label.pack(side="top", pady=(12, 6))

        # Buttons (pinned to bottom so they never get clipped when shrinking)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side="bottom", pady=(5, 15))

        self.btn_clear = tk.Button(
            btn_frame,
            text="Clear Canvas",
            font=("Helvetica", 11),
            width=13,
            command=self.clear_canvas,
        )
        self.btn_clear.grid(row=0, column=0, padx=6)

        self.btn_live = tk.Button(
            btn_frame,
            text="Live Predict: ON",
            font=("Helvetica", 11, "bold"),
            width=16,
            bg="#007acc",
            fg="white",
            command=self.toggle_live_predict,
        )
        self.btn_live.grid(row=0, column=1, padx=6)

        # Result display (pinned above buttons)
        self.result_frame = tk.Frame(self.root)
        self.result_frame.pack(side="bottom", pady=10)

        self.prediction_label = tk.Label(
            self.result_frame,
            text="Prediction: -",
            font=("Helvetica", 18, "bold"),
            fg="#2b579a",
        )
        self.prediction_label.pack()

        self.confidence_label = tk.Label(
            self.result_frame, text="Confidence: -", font=("Helvetica", 11)
        )
        self.confidence_label.pack()

        if not self.model_loaded:
            self.confidence_label.config(
                text="!! No model.pt. Run train.py first !!", fg="red"
            )

        # Drawing canvas (expands to fill all remaining window space)
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_w,
            height=self.canvas_h,
            bg="black",
            cursor="crosshair",
            highlightthickness=2,
            highlightbackground="#444444",
        )
        self.canvas.pack(side="top", fill="both", expand=True, padx=20, pady=5)

        # Mouse drawing & resize bindings
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<Button-1>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._on_stroke_end)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _brush_size(self):
        # Scale brush thickness with canvas size so strokes stay proportional for the model
        return max(8, int(min(self.canvas_w, self.canvas_h) * 0.06))

    def _on_canvas_resize(self, event):
        new_w, new_h = event.width, event.height
        if new_w <= 1 or new_h <= 1:
            return
        if new_w == self.canvas_w and new_h == self.canvas_h:
            return

        # Keep the in-memory PIL image matched to the canvas size
        new_image = Image.new("L", (new_w, new_h), color=0)
        new_image.paste(self.pil_image, (0, 0))
        self.pil_image = new_image
        self.draw = ImageDraw.Draw(self.pil_image)
        self.canvas_w = new_w
        self.canvas_h = new_h

    def _paint(self, event):
        x, y = event.x, event.y
        brush = self._brush_size()
        r = brush // 2

        if self.last_x is not None and self.last_y is not None:
            # Draw a smooth connected line between points
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                x,
                y,
                width=brush,
                fill="white",
                capstyle=tk.ROUND,
                smooth=True,
            )
            self.draw.line([(self.last_x, self.last_y), (x, y)], fill=255, width=brush)
            # Add round end cap to PIL line
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        else:
            # Starting dot of a new stroke
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r, fill="white", outline="white"
            )
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

        self.last_x, self.last_y = x, y

        # Schedule a throttled live prediction while actively drawing
        if self.live_predict and self.model_loaded and self._predict_job is None:
            self._predict_job = self.root.after(
                LIVE_PREDICT_DELAY_MS, self._run_scheduled_prediction
            )

    def _run_scheduled_prediction(self):
        self._predict_job = None
        self.run_prediction()

    def _cancel_scheduled_prediction(self):
        if self._predict_job is not None:
            self.root.after_cancel(self._predict_job)
            self._predict_job = None

    def _on_stroke_end(self, event):
        # Reset mouse tracking so the next stroke doesn't connect to this one
        self.last_x = None
        self.last_y = None

        # Cancel any queued timer and immediately run final prediction for the stroke
        self._cancel_scheduled_prediction()
        if self.model_loaded:
            self.run_prediction()

    def toggle_live_predict(self):
        self.live_predict = not self.live_predict
        if self.live_predict:
            self.btn_live.config(text="Live Predict: ON", bg="#007acc", fg="white")
        else:
            self._cancel_scheduled_prediction()
            self.btn_live.config(text="Live Predict: OFF", bg="#d9d9d9", fg="#333333")

    def clear_canvas(self):
        self._cancel_scheduled_prediction()
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
        self.pil_image = Image.new("L", (self.canvas_w, self.canvas_h), color=0)
        self.draw = ImageDraw.Draw(self.pil_image)
        self.prediction_label.config(text="Prediction: -")
        self.confidence_label.config(
            text="Confidence: -"
            if self.model_loaded
            else "!! No model.pt. Run train.py first !!"
        )

    def run_prediction(self):
        if not self.model_loaded:
            # Try reloading if trained in another process
            try:
                self.model, self.device = load_trained_model("model.pt")
                self.model_loaded = True
            except FileNotFoundError:
                return

        # Check if canvas is completely blank
        if self.pil_image.getbbox() is None:
            self.prediction_label.config(text="Prediction: -")
            self.confidence_label.config(text="Draw something first!")
            return

        digit, probs = predict(self.pil_image, model=self.model, device=self.device)
        confidence = probs[digit] * 100

        self.prediction_label.config(text=f"Prediction: {digit}")
        self.confidence_label.config(
            text=f"Confidence: {confidence:.1f}%",
            fg="green" if confidence > 70 else "orange",
        )


def main():
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
