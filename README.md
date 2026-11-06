# Persian Alphabet Character Recognition System

A desktop app that reads a photo of **one** Persian letter and shows the letter, its name, its
English equivalent, how confident the model is, and its top three guesses. It recognizes all 33
letters using a small convolutional neural network (CNN) trained on printed letters from many fonts.

## What the app does

- **Choose photo:** load a JPG, PNG, BMP or WebP picture of one letter (dark ink on light paper).
- **Size and Rotate sliders:** zoom the picture in or out and turn it sideways or upside down before
  recognizing, to see how the model copes. Double-click a slider to reset it.
- **Recognize:** shows the predicted letter, its name, the English equivalent (for example
  ژ is Zhe, ZH), a confidence bar, and the top three guesses.
- **What the model sees:** the cleaned-up 64x64 picture the model actually received, which makes
  bad photos easy to diagnose.

## Setup

1. Install Python 3.10 or newer (tkinter comes with the standard Windows and macOS installers).
2. Install the packages:

```
pip install -r requirements.txt
```

## Run

```
python src/app.py
```

For the best results use one letter, dark ink on plain light paper, even light without strong shadows
or glare, and keep the whole letter inside the picture.

## How it works

1. `src/locate.py` finds the letter in the photo and crops around it. Shadowed page edges and big
   dark areas are ignored.
2. `src/preprocess.py` evens out the lighting, turns the crop into black and white, thickens thin
   pen strokes to a printed-like width, centers the letter and resizes it to 64x64.
3. `src/predict.py` runs the CNN in `models/persian_cnn.keras` and returns the best match, the top
   three guesses and a confidence value.
4. `src/app.py` is the window around all of this.

## Training (only needed to rebuild the model)

```
python src/generate_printed.py   # draws every letter from the fonts in /fonts
python src/build_dataset.py      # preprocesses the images into dataset.npz
python src/train.py              # trains the CNN and saves it to /models
```

The training images include letters that are zoomed in and out, sideways, upside down and tilted.
Fonts are split by family, so the model is tested on fonts it has never seen: **Almarai** is the test
font, **MarkaziText** is the validation font, and every other family is used for training.

## Results

Accuracy on the unseen test font: **97.7%** (full report in `models/test_report.txt`).

`stress_test.py` redraws all 33 letters in the unseen test font, distorts them and measures how
often the app still gets them right:

| Test | Accuracy |
|---|---|
| Upright | 100% |
| Sideways (90 and 270 degrees) | 100% and 100% |
| Upside down (180 degrees) | 97% |
| Tilted 15 / 30 / 45 degrees | 100% / 94% / 67% |
| Zoomed out 0.5x / 0.25x | 100% / 97% |
| Picture resized 0.25x / 4x | 94% / 100% |
| Zoomed in 1.3x / 1.6x (letter still mostly in frame) | 91% / 55% |

Simulated phone photos (notebook grid, shadowed edges, glare, thin pen strokes) read correctly
85 to 94% of the time. Run the stress test yourself with `python stress_test.py`.

## Known limits

- Letters cut off by the edge of the picture are often misread.
- Tilts near 45 degrees are weaker.
- Letters smaller than about 10 pixels tall are unreliable.
- The model was trained on printed fonts, so handwriting is the weakest case. A thick dark marker on
  plain paper reads best.

## Project structure

| Path | Purpose |
|---|---|
| `src/app.py` | the desktop app |
| `src/locate.py`, `src/preprocess.py` | find the letter and clean the picture |
| `src/predict.py` | loads the model and predicts |
| `src/generate_printed.py`, `src/build_dataset.py`, `src/train.py` | build the data and train the model |
| `src/labels.py` | the 33 letters with their names and English equivalents |
| `models/` | trained model, label list, test report and training charts |
| `fonts/` | the fonts used to generate the training images |
| `stress_test.py` | zoom and rotation test |
| `CHANGES.md` | notes on the zoom and rotation fix |

## Author

Ernest Jairus Escobar, Computer Science
Ryndle Carl Obar, Computer Science