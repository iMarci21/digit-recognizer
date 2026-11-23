import tkinter as tk

from PIL import Image, ImageDraw

from predict import load_trained_model, predict

CANVAS_SIZE = 280
BRUSH_SIZE = 16


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.resizable(False, False)

        # Track previous mouse coordinates for smooth continuous lines
        self.last_x = None
        self.last_y = None

        # In-memory PIL Image matching canvas dimensions (black background)
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
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
        # Header / Status
        self.title_label = tk.Label(
            self.root, text="Draw a digit (0-9) below", font=("Helvetica", 14, "bold")
        )
        self.title_label.pack(pady=(12, 6))

        # Canvas container
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="crosshair",
            highlightthickness=2,
            highlightbackground="#444444",
        )
        self.canvas.pack(padx=20, pady=5)

        # Mouse drawing bindings
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<Button-1>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._on_stroke_end)

        # Result display
        self.result_frame = tk.Frame(self.root)
        self.result_frame.pack(pady=10)

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

        # Buttons (Clear & Predict)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=(5, 15))

        self.btn_clear = tk.Button(
            btn_frame,
            text="Clear Canvas",
            font=("Helvetica", 11),
            width=12,
            command=self.clear_canvas,
        )
        self.btn_clear.grid(row=0, column=0, padx=6)

        self.btn_predict = tk.Button(
            btn_frame,
            text="Predict",
            font=("Helvetica", 11, "bold"),
            width=12,
            bg="#007acc",
            fg="white",
            command=self.run_prediction,
        )
        self.btn_predict.grid(row=0, column=1, padx=6)

    def _paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_SIZE // 2

        if self.last_x is not None and self.last_y is not None:
            # Draw a smooth connected line between points
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=BRUSH_SIZE, fill="white", capstyle=tk.ROUND, smooth=True
            )
            self.draw.line([(self.last_x, self.last_y), (x, y)], fill=255, width=BRUSH_SIZE)
            # Add round end cap to PIL line
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        else:
            # Starting dot of a new stroke
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

        self.last_x, self.last_y = x, y

    def _on_stroke_end(self, event):
        # Reset mouse tracking so the next stroke doesn't connect to this one
        self.last_x = None
        self.last_y = None

        # Auto-predict after every mouse stroke
        if self.model_loaded:
            self.run_prediction()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.pil_image)
        self.prediction_label.config(text="Prediction: -")
        self.confidence_label.config(
            text="Confidence: -" if self.model_loaded else "⚠️ Run train.py first."
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
