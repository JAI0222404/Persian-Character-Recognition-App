import sys
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import load_image
from src.predict import ModelNotFoundError, Recognizer

MAX_SIDE = 2400       

PAPER = "#f3ede1"    
CARD = "#fbf8f1"     
LINE = "#d6ccb8"     
INK = "#221f1c"
MUTED = "#7a7164"
FAINT = "#d9d0bf"
TEAL = "#1d6b74"
TEAL_DARK = "#14505a"
CLAY = "#b4532a"


def pick_font(root, *names, default):
    have = set(tkfont.families(root))
    for name in names:
        if name in have:
            return name
    return default


def mix(a, b, t):
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(ca, cb))


def ease_out(t):
    return 1 - (1 - t) ** 3


class App:
    def __init__(self, root):
        self.root = root
        root.title("Persian Alphabet Character Recognition System")
        root.configure(bg=PAPER)
        root.resizable(False, False)

        self.k = root.tk.call("tk", "scaling") / (96 / 72)
        self.serif = pick_font(root, "Georgia", "Palatino Linotype", "Cambria", "DejaVu Serif",
                               default="Times")
        self.sans = pick_font(root, "Segoe UI", "Helvetica Neue", "Helvetica", "DejaVu Sans",
                              default="Arial")
        self.preview = self.S(304)

        self.recognizer = None
        self.image = None
        self.filename = ""
        self.photo = None
        self.proc_photo = None
        self._job = None
        self._shown = None  

        self._style()
        self._build()

        root.bind("<Return>", lambda e: self.recognize() if self.recog_btn["state"] == "normal" else None)
        root.bind("<Control-o>", lambda e: self.choose_file())
        root.update_idletasks()
        root.after(200, self.load_model)

    def S(self, n):
        return int(round(n * self.k))

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Pcr.Horizontal.TScale", background=TEAL, troughcolor=LINE,
                        bordercolor=PAPER, lightcolor=TEAL, darkcolor=TEAL, gripcount=0,
                        sliderlength=self.S(16), sliderthickness=self.S(14))
        style.map("Pcr.Horizontal.TScale", background=[("active", TEAL_DARK)])

    def _build(self):
        sans, serif, S = self.sans, self.serif, self.S

        self.strip = tk.Canvas(self.root, height=S(14), bg=PAPER, highlightthickness=0)
        self.strip.pack(fill="x")
        self.strip.bind("<Configure>", self._draw_strip)

        header = tk.Frame(self.root, bg=PAPER)
        header.pack(fill="x", padx=S(32), pady=(S(10), 0))
        tk.Label(header, text="Persian Alphabet", font=(serif, 26, "bold"),
                 bg=PAPER, fg=INK).pack(anchor="w")
        tk.Label(header, text="Character Recognition System", font=(serif, 13, "italic"),
                 bg=PAPER, fg=MUTED).pack(anchor="w")
        tk.Frame(self.root, bg=LINE, height=1).pack(fill="x", padx=S(32), pady=(S(12), 0))

        body = tk.Frame(self.root, bg=PAPER)
        body.pack(padx=S(32), pady=(S(18), S(28)))
        left = tk.Frame(body, bg=PAPER)
        left.grid(row=0, column=0, sticky="n")
        right = tk.Frame(body, bg=PAPER)
        right.grid(row=0, column=1, sticky="n", padx=(S(30), 0))

        self.canvas = tk.Canvas(left, width=self.preview, height=self.preview, bg=CARD,
                                highlightthickness=1, highlightbackground=LINE, cursor="hand2")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._canvas_click)
        self._draw_hint()

        self.caption = tk.Label(left, text="", font=(sans, 9), bg=PAPER, fg=MUTED, anchor="w")
        self.caption.pack(fill="x", pady=(S(6), S(10)))

        buttons = tk.Frame(left, bg=PAPER)
        buttons.pack(fill="x")
        self.upload_btn = self._button(buttons, "Choose photo", self.choose_file)
        self.recog_btn = self._button(buttons, "Recognize", self.recognize, primary=True)
        self.clear_btn = tk.Label(buttons, text="Clear", font=(sans, 10, "underline"),
                                  bg=PAPER, fg=MUTED, cursor="hand2")
        self.clear_btn.bind("<Button-1>", lambda e: self.clear())
        self.clear_btn.bind("<Enter>", lambda e: self.clear_btn.config(fg=CLAY))
        self.clear_btn.bind("<Leave>", lambda e: self.clear_btn.config(fg=MUTED))
        self.upload_btn.pack(side="left")
        self.recog_btn.pack(side="left", padx=(S(8), 0))
        self.clear_btn.pack(side="right", padx=(0, S(2)))
        self._enable(self.recog_btn, False)

        self.zoom_var = tk.DoubleVar(value=1.0)
        self.angle_var = tk.DoubleVar(value=0.0)
        self.zoom_value = self._slider(left, "Size", self.zoom_var, 0.25, 4.0,
                                       lambda: f"{self._zoom():.2f}×", 1.0)
        self.angle_value = self._slider(left, "Rotate", self.angle_var, -180, 180,
                                        lambda: f"{self._angle():d}°", 0.0)

        card = tk.Frame(right, bg=CARD, highlightthickness=1, highlightbackground=LINE)
        card.pack()
        inner = tk.Frame(card, bg=CARD)
        inner.pack(padx=S(22), pady=S(18))

        self._small(inner, "RESULT").grid(row=0, column=0, columnspan=2, sticky="w")

        self.gw, self.gh = S(160), S(172)
        self.glyph = tk.Canvas(inner, width=self.gw, height=self.gh, bg=CARD, highlightthickness=0)
        self.glyph.grid(row=1, column=0, rowspan=2, sticky="n", pady=(S(4), 0))

        details = tk.Frame(inner, bg=CARD)
        details.grid(row=1, column=1, sticky="nw", padx=(S(14), 0), pady=(S(14), 0))
        self.name_label = tk.Label(details, text="", font=(serif, 24), bg=CARD, fg=INK,
                                   anchor="w", width=12)
        self.name_label.pack(anchor="w")
        self.english_title = self._small(details, "")
        self.english_title.pack(anchor="w", pady=(S(10), 0))
        self.english_label = tk.Label(details, text="", font=(serif, 26, "bold"), bg=CARD,
                                      fg=TEAL, anchor="w")
        self.english_label.pack(anchor="w")
        self.english_note = tk.Label(details, text="", font=(sans, 9), bg=CARD, fg=MUTED,
                                     anchor="w", wraplength=S(220), justify="left")
        self.english_note.pack(anchor="w")

        conf = tk.Frame(inner, bg=CARD)
        conf.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(S(8), 0))
        head = tk.Frame(conf, bg=CARD)
        head.pack(fill="x")
        self._small(head, "CONFIDENCE").pack(side="left")
        self.conf_label = tk.Label(head, text="", font=(sans, 10, "bold"), bg=CARD, fg=INK)
        self.conf_label.pack(side="right")
        self.bar_w, self.bar_h = S(440), S(8)
        self.bar = tk.Canvas(conf, width=self.bar_w, height=self.bar_h, bg=CARD, highlightthickness=0)
        self.bar.pack(anchor="w", pady=(S(4), 0))
        self.status_label = tk.Label(conf, text="Loading model...", font=(sans, 10), bg=CARD,
                                     fg=MUTED, anchor="w", justify="left", wraplength=S(440))
        self.status_label.pack(fill="x", pady=(S(8), 0))

        tk.Frame(inner, bg=LINE, height=1).grid(row=4, column=0, columnspan=2, sticky="ew",
                                                pady=(S(12), S(10)))

        lower = tk.Frame(inner, bg=CARD)
        lower.grid(row=5, column=0, columnspan=2, sticky="ew")
        guesses = tk.Frame(lower, bg=CARD)
        guesses.pack(side="left", anchor="n")
        self._small(guesses, "TOP THREE GUESSES").pack(anchor="w")
        self.rows_w, self.rows_h = S(250), S(90)
        self.rows = tk.Canvas(guesses, width=self.rows_w, height=self.rows_h, bg=CARD,
                              highlightthickness=0)
        self.rows.pack(anchor="w", pady=(S(4), 0))
        seen = tk.Frame(lower, bg=CARD)
        seen.pack(side="right", anchor="n")
        self._small(seen, "WHAT THE MODEL SEES").pack(anchor="w")
        self.proc_label = tk.Label(seen, bg="black", width=S(96), height=S(96), bd=0,
                                   highlightthickness=1, highlightbackground=LINE)
        self.proc_blank = ImageTk.PhotoImage(Image.new("RGB", (S(96), S(96)), "#15130f"))
        self.proc_label.config(image=self.proc_blank, width=S(96), height=S(96))
        self.proc_label.pack(anchor="w", pady=(S(4), 0))

        self._reset_view()

    def _small(self, parent, text):
        return tk.Label(parent, text=text, font=(self.sans, 8, "bold"), bg=parent["bg"], fg=MUTED)

    def _button(self, parent, text, command, primary=False):
        rest, fg = (TEAL, "#ffffff") if primary else (CARD, INK)
        hover = TEAL_DARK if primary else "#ece4d2"
        b = tk.Button(parent, text=text, command=command, relief="flat", bd=0, cursor="hand2",
                      font=(self.sans, 10, "bold" if primary else "normal"),
                      padx=self.S(16), pady=self.S(7), bg=rest, fg=fg,
                      activebackground=hover, activeforeground=fg,
                      highlightthickness=1, highlightbackground=TEAL if primary else LINE)
        b.look = (rest, fg, hover, primary)
        b.bind("<Enter>", lambda e: b["state"] == "normal" and b.config(bg=hover))
        b.bind("<Leave>", lambda e: b["state"] == "normal" and b.config(bg=rest))
        return b

    def _enable(self, b, on):
        rest, fg, hover, primary = b.look
        if on:
            b.config(state="normal", bg=rest, fg=fg, cursor="hand2")
        else:
            b.config(state="disabled", bg=mix(rest, PAPER, 0.6) if primary else PAPER,
                     disabledforeground="#ffffff" if primary else "#a79e8e", cursor="arrow")

    def _slider(self, parent, title, var, lo, hi, text, rest):
        row = tk.Frame(parent, bg=PAPER)
        row.pack(fill="x", pady=(self.S(12), 0))
        head = tk.Frame(row, bg=PAPER)
        head.pack(fill="x")
        tk.Label(head, text=title, font=(self.sans, 10), bg=PAPER, fg=INK).pack(side="left")
        value = tk.Label(head, text=text(), font=(self.sans, 10, "bold"), bg=PAPER, fg=TEAL)
        value.pack(side="right")

        def moved(_=None):
            value.config(text=text())
            self.show_preview()

        scale = ttk.Scale(row, from_=lo, to=hi, variable=var, command=moved,
                          style="Pcr.Horizontal.TScale")
        scale.pack(fill="x", pady=(self.S(2), 0))
        # double-click puts the slider back to its resting value
        scale.bind("<Double-Button-1>",
                   lambda e: self.root.after(5, lambda: (var.set(rest), moved())))
        value.text = text
        return value

    def _draw_strip(self, event=None):
        c, S = self.strip, self.S
        c.delete("all")
        w = c.winfo_width()
        step, r = S(22), S(4)
        y = S(7)
        c.create_line(0, y, w, y, fill=LINE)
        for i, x in enumerate(range(S(10), w, step)):
            colour = CLAY if i % 6 == 3 else TEAL
            c.create_polygon(x, y - r, x + r, y, x, y + r, x - r, y, fill=colour, outline=PAPER)

    def _draw_hint(self):
        c, p, S = self.canvas, self.preview, self.S
        c.delete("all")
        m, n = S(18), S(26)                       # viewfinder corners
        for x, y, dx, dy in ((m, m, 1, 1), (p - m, m, -1, 1), (m, p - m, 1, -1), (p - m, p - m, -1, -1)):
            c.create_line(x, y + dy * n, x, y, x + dx * n, y, fill=TEAL, width=2)
        c.create_text(p // 2, p // 2 - S(8), text="Click to choose a photo",
                      font=(self.sans, 12), fill=INK)
        c.create_text(p // 2, p // 2 + S(14), text="one letter, dark ink on light paper",
                      font=(self.sans, 9), fill=MUTED)

    def _canvas_click(self, _):
        if self.image is None:
            self.choose_file()

    def _glyph(self, char, colour, dy=0):
        self.glyph.delete("all")
        self.glyph.create_text(self.gw // 2, int(self.gh * 0.50) + dy, text=char,
                               font=(self.sans, 70), fill=colour)

    def _bar(self, fraction, colour):
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, self.bar_w, self.bar_h, fill=LINE, outline="")
        if fraction > 0:
            self.bar.create_rectangle(0, 0, max(2, self.bar_w * fraction), self.bar_h,
                                      fill=colour, outline="")

    def _rows(self, top3, t, reliable=True):
        c, S = self.rows, self.S
        c.delete("all")
        bar_x, bar_w = S(132), S(70)
        for i, item in enumerate(top3):
            y = S(15) + i * S(30)
            first = i == 0
            c.create_text(S(10), y, text=item["char"], font=(self.sans, 15),
                          fill=INK if first else MUTED)
            c.create_text(S(30), y, text=item["name"], anchor="w",
                          font=(self.sans, 10, "bold" if first else "normal"),
                          fill=INK if first else MUTED)
            c.create_rectangle(bar_x, y - S(3), bar_x + bar_w, y + S(3), fill=LINE, outline="")
            fill = (TEAL if reliable else CLAY) if first else "#b9ae98"
            c.create_rectangle(bar_x, y - S(3), bar_x + max(1, bar_w * item["confidence"] * t),
                               y + S(3), fill=fill, outline="")
            c.create_text(S(248), y, text=f'{item["confidence"] * 100 * t:.1f}%', anchor="e",
                          font=(self.sans, 9), fill=INK if first else MUTED)

    def _reset_view(self):
        self._stop()
        self._shown = None
        self._glyph("؟", FAINT)               
        self.name_label.config(text="")
        self.english_title.config(text="")
        self.english_label.config(text="")
        self.english_note.config(text="")
        self.conf_label.config(text="")
        self._bar(0, TEAL)
        self.rows.delete("all")
        self.proc_label.config(image=self.proc_blank)

    def set_status(self, text, colour=MUTED):
        self.status_label.config(text=text, fg=colour)

    def _stop(self):
        if self._job is not None:
            try:
                self.root.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None

    def _animate(self, frames, draw, done=None):
        """Call draw(t) with t easing from 0 to 1 over `frames` frames (about 16 ms each)."""
        self._stop()

        def step(i):
            try:
                draw(ease_out(min(1.0, i / frames)))
                if i < frames:
                    self._job = self.root.after(16, step, i + 1)
                elif done:
                    done()
            except tk.TclError:           
                pass

        step(0)

    def load_model(self):
        try:
            self.recognizer = Recognizer()
            self.set_status("Model loaded. Choose a photo to begin.")
        except ModelNotFoundError as err:
            self.set_status("Model not loaded.", CLAY)
            messagebox.showerror("Model file missing", str(err))
        except Exception as err:
            self.set_status("Model not loaded.", CLAY)
            messagebox.showerror("Could not load model", str(err))

    def _zoom(self):
        return round(self.zoom_var.get() / 0.05) * 0.05

    def _angle(self):
        return int(round(self.angle_var.get()))

    def zoomed_image(self, base=1.0):
        """The photo after the Size and Rotate sliders (this is what the model receives).

        `base` only shrinks the result, so the preview can be built from a small copy.
        """
        if self.image is None:
            return None
        img = self.image
        z = self._zoom() * base
        if abs(z - 1.0) > 1e-6:
            interp = cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA
            img = cv2.resize(img, None, fx=z, fy=z, interpolation=interp)
        a = self._angle()
        if a:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return img

    def _output_size(self):
        h, w = self.image.shape[:2]
        z = self._zoom()
        w, h = w * z, h * z
        a = np.radians(self._angle())
        return int(h * abs(np.sin(a)) + w * abs(np.cos(a))), int(h * abs(np.cos(a)) + w * abs(np.sin(a)))

    def show_preview(self, *_):
        if self.image is None:
            return
        p = self.preview
        h0, w0 = self.image.shape[:2]
        base = min(1.0, (p - 6) / max(h0, w0))
        img = self.zoomed_image(base)
        ih, iw = img.shape[:2]
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        c = self.canvas
        c.delete("all")
        c.create_image(p // 2, p // 2, image=self.photo)
        clipped = iw > p or ih > p
        if clipped:
            tag = c.create_text(p - self.S(8), p - self.S(8), text="showing the centre",
                                anchor="se", font=(self.sans, 8), fill=MUTED)
            x1, y1, x2, y2 = c.bbox(tag)
            box = c.create_rectangle(x1 - 4, y1 - 2, x2 + 4, y2 + 2, fill=CARD, outline=LINE)
            c.tag_raise(tag, box)
        else:
            x0, y0 = (p - iw) // 2, (p - ih) // 2
            c.create_rectangle(x0 - 1, y0 - 1, x0 + iw, y0 + ih, outline=LINE)
        w, h = self._output_size()
        name = self.filename if len(self.filename) <= 26 else self.filename[:12] + "..." + self.filename[-11:]
        self.caption.config(text=f"{name}   {w} x {h} px")

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        if path:
            self.load_path(path)

    def load_path(self, path):
        try:
            image = load_image(path)
        except Exception as err:
            messagebox.showerror("Cannot open file", f"{err}")
            return
        if image is None:
            messagebox.showerror("Unsupported image",
                                 "This file is empty, corrupted, or not a supported image.")
            return
        big = max(image.shape[:2])
        if big > MAX_SIDE:
            s = MAX_SIDE / big
            image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self._reset_view()
        self.image = image
        self.filename = Path(path).name
        self.zoom_var.set(1.0)
        self.angle_var.set(0.0)
        self.zoom_value.config(text=self.zoom_value.text())
        self.angle_value.config(text=self.angle_value.text())
        self.show_preview()
        self._enable(self.recog_btn, True)
        self.set_status("Photo loaded. Press Recognize.")

    def recognize(self):
        if self.image is None:
            messagebox.showinfo("No image", "Upload an image first.")
            return
        if self.recognizer is None:
            messagebox.showerror("Model not loaded", "The model file is missing or failed to load.")
            return
        self._reset_view()
        self._enable(self.recog_btn, False)
        self.set_status("Reading the letter...")
        self.root.after(60, self._read)        

    def _read(self):
        try:
            result = self.recognizer.predict_array(self.zoomed_image())
        except Exception as err:
            self._enable(self.recog_btn, True)
            self.set_status("Recognition failed.", CLAY)
            messagebox.showerror("Recognition failed", str(err))
            return
        self._enable(self.recog_btn, True)
        if result is None:
            self._glyph("?", CLAY)
            self.set_status("No clear character found. Use dark ink on light paper, good light, "
                            "and keep the letter centered and in focus.", CLAY)
            return
        self._show(result)

    def _show(self, result):
        best = result["best"]
        reliable = result["reliable"]
        t = best["translit"]
        pct = best["confidence"] * 100
        self.name_label.config(text=best["name"])
        self.english_title.config(text="ENGLISH EQUIVALENT")
        if t == "'":
            self.english_label.config(text="none")
            self.english_note.config(text="A glottal sound, written '")
        else:
            self.english_label.config(text=t.upper())
            self.english_note.config(text=f"written “{t}”")
        if reliable:
            self.set_status("Model confidence, not a guarantee.")
        else:
            self.set_status("Low confidence or ambiguous. This result may be wrong; "
                            "try a clearer photo.", CLAY)

        small = Image.fromarray((result["processed"] * 255).astype(np.uint8)).resize(
            (self.S(96), self.S(96)), Image.NEAREST)
        self.proc_photo = ImageTk.PhotoImage(small)
        self.proc_label.config(image=self.proc_photo)

        bar_colour = TEAL if reliable else CLAY
        top3 = result["top3"]

        def draw(f):
            self._glyph(best["char"], mix(CARD, INK, f), dy=int((1 - f) * self.S(14)))
            self._bar(best["confidence"] * f, bar_colour)
            self.conf_label.config(text=f"{pct * f:.1f}%")
            self._rows(top3, f, reliable)

        self._animate(22, draw)

    def clear(self):
        self.image = None
        self.photo = None
        self.filename = ""
        self.zoom_var.set(1.0)
        self.angle_var.set(0.0)
        self.zoom_value.config(text=self.zoom_value.text())
        self.angle_value.config(text=self.angle_value.text())
        self._draw_hint()
        self.caption.config(text="")
        self._reset_view()
        self._enable(self.recog_btn, False)
        self.set_status("Cleared. Choose a photo to begin.")


if __name__ == "__main__":
    window = tk.Tk()
    App(window)
    window.mainloop()