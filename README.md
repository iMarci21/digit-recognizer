## Project Structure

- `model.py` — Defines the `DigitClassifierCNN` architecture (Convolution, ReLU, MaxPool, Dropout, Dense layers).
- `train.py` — Downloads the MNIST dataset for digits, trains the model for 5 epochs (gives roughly ~99% test accuracy) and saves the weights to `model.pt`.
- `predict.py` — Preprocessing pipeline (resizing, centering bounding box, normalizing) and inference helper.
- `app.py` — Pretty bad Tkinter desktop GUI for drawing digits. Includes real-time predictions


## How To Use

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Training the Model
```bash
python train.py
```
Automatically downloads MNIST into `./data/` and saves the trained weights into `model.pt`.

### 3. Launch the Drawing App
```bash
python app.py
```
Draw any single digit from 0 to 9 on the black canvas. The model will automatically predict the digit as you draw!
Disclaimer: Only recognizes white ink on black canvas for now.
(Yes the GUI may be very small and not adjustable. Might fix soon)