import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import load_image
from src.predict import ModelNotFoundError, Recognizer

PREVIEW = 300
FONT = "Segoe UI"
BG      = "#f5f3ff"   # window background (soft lavender)
CARD    = "#ffffff"   # panels
ACCENT  = "#6d28d9"   # buttons, highlights
TEXT    = "#1f2937"
MUTED   = "#6b7280"

class App:
    def __init__(self, root):
        self.root = root
        root.title("Persian Alphabet Character Recognition System")
        root.geometry("860x640")
        root.resizable(False, False)
        root.columnconfigure((0, 1), weight=1)
        self.recognizer = None
        self.image = None
        self.photo = None
        self.proc_photo = None

        header = tk.Frame(root, bg=ACCENT)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(header, text="Persian Alphabet Character Recognition System",
                 font=(FONT, 18, "bold"), bg=ACCENT, fg="white").pack(pady=12)
        tk.Label(root, text="Upload a photo of ONE isolated Persian letter (dark ink on light paper).",
                 font=(FONT, 10), fg="#555").grid(row=1, column=0, columnspan=2, pady=(0, 8))

        left = tk.Frame(root, padx=16)
        left.grid(row=2, column=0, sticky="n")
        right = tk.Frame(root, padx=16)
        right.grid(row=2, column=1, sticky="n")

        self.canvas = tk.Canvas(left, width=PREVIEW, height=PREVIEW, bg="#f2f2f2",
                                highlightthickness=1, highlightbackground="#888")
        self.canvas.pack()
        self._draw_hint()
        buttons = tk.Frame(left)
        buttons.pack(pady=10)
        btn_style = dict(relief="flat", font=(FONT, 10, "bold"), cursor="hand2",
                         fg="white", activeforeground="white", bd=0, pady=4)
        self.upload_btn = tk.Button(buttons, text="Upload Image", width=13, command=self.upload,
                                    bg=ACCENT, activebackground="#5b21b6", **btn_style)
        self.recog_btn = tk.Button(buttons, text="Recognize Character", width=18,
                                   command=self.recognize, state="disabled",
                                   bg=ACCENT, activebackground="#5b21b6", **btn_style)
        self.clear_btn = tk.Button(buttons, text="Clear", width=8, command=self.clear,
                                   bg="#9ca3af", activebackground="#6b7280", **btn_style)
        for b in (self.upload_btn, self.recog_btn, self.clear_btn):
            b.pack(side="left", padx=3)
        self.zoom_var = tk.DoubleVar(value=1.0)
        tk.Scale(left, from_=0.25, to=4.0, resolution=0.05, orient="horizontal",
                 label="Zoom (image size)", variable=self.zoom_var, length=PREVIEW,
                 command=self.show_preview).pack()
        self.angle_var = tk.DoubleVar(value=0)
        tk.Scale(left, from_=-180, to=180, resolution=1, orient="horizontal",
         label="Rotate (degrees)", variable=self.angle_var, length=PREVIEW,     
         command=self.show_preview).pack()

        tk.Label(right, text="Predicted character", font=(FONT, 11, "bold")).pack(anchor="w")
        self.char_label = tk.Label(right, text="-", font=(FONT, 96), width=3)
        self.char_label.pack()
        self.name_label = tk.Label(right, text="", font=(FONT, 16))
        self.name_label.pack()
        self.english_label = tk.Label(right, text="", font=(FONT, 14, "bold"), fg="#1a5fb4")
        self.english_label.pack()
        self.conf_label = tk.Label(right, text="", font=(FONT, 11))
        self.conf_label.pack()
        self.status_label = tk.Label(right, text="Loading model...", font=(FONT, 10),
                                     fg="#555", wraplength=380, justify="left")
        self.status_label.pack(pady=6)
        self.top3_label = tk.Label(right, text="", font=(FONT, 10), justify="left", fg="#333")
        self.top3_label.pack(anchor="w")
        tk.Label(right, text="What the model sees (64x64):", font=(FONT, 9),
                 fg="#555").pack(anchor="w", pady=(8, 0))
        self.proc_label = tk.Label(right)
        self.proc_label.pack(anchor="w")

        root.update_idletasks()
        root.after(200, self.load_model)

    def _draw_hint(self):
        self.canvas.delete("all")
        self.canvas.create_text(PREVIEW // 2, PREVIEW // 2, text="No image selected", fill="#777")

    def load_model(self):
        try:
            self.recognizer = Recognizer()
            self.status_label.config(text="Model loaded. Upload an image to begin.")
        except ModelNotFoundError as err:
            self.status_label.config(text="Model not loaded.")
            messagebox.showerror("Model file missing", str(err))
        except Exception as err:
            self.status_label.config(text="Model not loaded.")
            messagebox.showerror("Could not load model", str(err))

    def zoomed_image(self):
        if self.image is None:
            return None
        img = self.image
        z = self.zoom_var.get()
        if abs(z - 1.0) > 1e-6:
            interp = cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA
            img = cv2.resize(img, None, fx=z, fy=z, interpolation=interp)
        a = self.angle_var.get()
        if abs(a) > 1e-6:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        return img

    def show_preview(self, *_):
        if self.image is None:
            return
        img = self.zoomed_image()
        h, w = img.shape[:2]
        rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        rgb.thumbnail((PREVIEW, PREVIEW))
        self.photo = ImageTk.PhotoImage(rgb)
        self.canvas.delete("all")
        self.canvas.create_image(PREVIEW // 2, PREVIEW // 2, image=self.photo)
        self.status_label.config(text=f"Zoom {self.zoom_var.get():.2f}x: image is now {w}x{h} px")

    def upload(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            image = load_image(path)
        except Exception as err:
            messagebox.showerror("Cannot open file", f"{err}")
            return
        if image is None:
            messagebox.showerror("Unsupported image",
                                 "This file is empty, corrupted, or not a supported image.")
            return
        self.clear_result()
        self.image = image
        self.zoom_var.set(1.0)
        self.show_preview()
        self.recog_btn.config(state="normal")
        self.status_label.config(text="Image loaded. Click Recognize Character.")

    def recognize(self):
        if self.image is None:
            messagebox.showinfo("No image", "Upload an image first.")
            return
        if self.recognizer is None:
            messagebox.showerror("Model not loaded", "The model file is missing or failed to load.")
            return
        try:
            result = self.recognizer.predict_array(self.zoomed_image())
        except Exception as err:
            messagebox.showerror("Recognition failed", str(err))
            return
        self.clear_result()
        if result is None:
            self.char_label.config(text="?")
            self.status_label.config(
                text="No clear character found. Use dark ink on light paper, good light, "
                     "and keep the letter centered and in focus.")
            return
        best = result["best"]
        self.char_label.config(text=best["char"])
        self.name_label.config(text=f'{best["name"]}  ({best["translit"]})')
        t = best["translit"]
        if t == "'":
            english = "No direct English letter (a glottal sound, written ')"
        else:
            english = f"English equivalent: {t.upper()}  ({t})"
        self.english_label.config(text=english)
        self.conf_label.config(
            text=f'Model confidence: {best["confidence"] * 100:.1f}%  (not a guarantee)')
        if result["reliable"]:
            self.status_label.config(text="Done.", fg="#555")
        else:
            self.status_label.config(
                text="Low confidence or ambiguous. This result may be wrong; try a clearer photo.",
                fg="#b00020")
        self.top3_label.config(text="Top 3:\n" + "\n".join(
            f'  {t["char"]}  {t["name"]}  {t["confidence"] * 100:.1f}%' for t in result["top3"]))
        small = Image.fromarray((result["processed"] * 255).astype(np.uint8)).resize(
            (96, 96), Image.NEAREST)
        self.proc_photo = ImageTk.PhotoImage(small)
        self.proc_label.config(image=self.proc_photo)

    def clear_result(self):
        self.char_label.config(text="-")
        for lbl in (self.name_label, self.english_label, self.conf_label, self.top3_label):
            lbl.config(text="")
        self.proc_label.config(image="")
        self.status_label.config(fg="#555")

    def clear(self):
        self.zoom_var.set(1.0)
        self.image = None
        self.photo = None
        self._draw_hint()
        self.clear_result()
        self.recog_btn.config(state="disabled")
        self.status_label.config(text="Cleared. Upload an image to begin.")


if __name__ == "__main__":
    window = tk.Tk()
    App(window)
    window.mainloop()