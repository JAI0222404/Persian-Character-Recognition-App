import json
from pathlib import Path

import numpy as np

from src.locate import find_character
from src.preprocess import preprocess_image

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "persian_cnn.keras"
LABELS_PATH = ROOT / "models" / "labels.json"
MIN_CONFIDENCE = 0.60 
MIN_MARGIN = 0.20    


class ModelNotFoundError(Exception):
    pass


class Recognizer:
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        for p in (model_path, labels_path):
            if not Path(p).exists():
                raise ModelNotFoundError(
                    f"Missing file:\n{p}\n\nRun train.py and copy its output into the models folder.")
        import tensorflow as tf         
        self.model = tf.keras.models.load_model(model_path)
        self.labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))

    def predict_array(self, image_bgr):
        """Returns None if no usable character was found."""
        processed = preprocess_image(find_character(image_bgr))
        if processed is None:
            return None
        probs = self.model(processed[None, ..., None], training=False).numpy()[0]
        order = np.argsort(probs)[::-1][:3]
        top3 = [dict(self.labels[i], confidence=float(probs[i])) for i in order]
        gap = top3[0]["confidence"] - top3[1]["confidence"]
        reliable = top3[0]["confidence"] >= MIN_CONFIDENCE and gap >= MIN_MARGIN
        return {"best": top3[0], "top3": top3, "reliable": reliable, "processed": processed}