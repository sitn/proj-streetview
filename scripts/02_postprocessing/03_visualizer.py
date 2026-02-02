import cv2, json, os, sys, time, numpy as np, pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

sys.path.insert(1, 'scripts')
from utils.logging import get_logger
from utils.config import get_config

logger = get_logger()
cfg = get_config("This script prepares COCO datasets.")

COLORS = {"TP": (0, 255, 0), "FP": (247, 195, 79), "FN": (37, 168, 249), "oth": (0, 0, 0)}

def load_coco():
    images, anns = [], []
    for d in cfg['tagged_coco_files']:
        with open(cfg['tagged_coco_files'][d]) as f:
            data = json.load(f)
        if isinstance(data, dict):
            images += data['images']
            anns += data['annotations']
        else:
            anns += data
            for k in cfg['coco_file_for_images']:
                with open(cfg['coco_file_for_images'][k]) as f:
                    images += json.load(f)['images']
    return (pd.DataFrame(images).drop_duplicates('file_name'), pd.DataFrame(anns))

def draw_boxes(img, anns):
    has_tag = 'tag' in anns.columns
    for a in anns.itertuples():
        tag = a.tag if has_tag else 'oth'
        x, y, w, h = map(int, a.bbox)
        c = COLORS.get(tag, (255, 0, 0))
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
        txt = f"{a.dataset} {a.id} {round(a.score, 2)} {tag if has_tag else ''}"
        cv2.putText(img, txt, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return img

class CocoViewer(tk.Tk):
    def __init__(self, images, anns):
        super().__init__()
        self.title("COCO Visualizer")
        self.geometry("1000x800")
        self.images = {os.path.basename(r.file_name): r for r in images.itertuples()}
        self.anns = anns
        self.img = None
        self.zoom = 1.0
        self.pan = [0, 0]
        self._ui()

    def _ui(self):
        bar = ttk.Frame(self); bar.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(bar, text="Image:").pack(side=tk.LEFT)
        self.entry = ttk.Entry(bar, width=35); self.entry.pack(side=tk.LEFT, padx=5)
        self.entry.bind('<Return>', lambda e: self.load())
        ttk.Button(bar, text="Load", command=self.load).pack(side=tk.LEFT)
        ttk.Button(bar, text="Zoom +", command=lambda: self.zoom_by(1.25)).pack(side=tk.LEFT)
        ttk.Button(bar, text="Zoom -", command=lambda: self.zoom_by(0.8)).pack(side=tk.LEFT)
        ttk.Button(bar, text="Reset", command=self.reset).pack(side=tk.LEFT)
        self.canvas = tk.Canvas(self, bg='black'); self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.canvas.bind('<ButtonPress-1>', self.pan_start)
        self.canvas.bind('<B1-Motion>', self.pan_move)
        self.bind('<Configure>', lambda e: self.render())

    def load(self):
        name = self.entry.get().strip()
        if name not in self.images: return
        r = self.images[name]
        path = os.path.join(cfg['image_dir'][r.AOI], name)
        img = cv2.imread(path)
        if img is None: return
        a = self.anns[self.anns.image_id == r.id]
        if not a.empty: img = draw_boxes(img, a)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.zoom, self.pan = 1.0, [0, 0]
        self.render()

    def render(self):
        if self.img is None: return
        h, w = self.img.shape[:2]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 2 or ch < 2: return
        s = min(cw / w, ch / h) * self.zoom
        img = cv2.resize(self.img, (int(w * s), int(h * s)), cv2.INTER_AREA)
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.delete('all')
        self.canvas.create_image(cw//2 + self.pan[0], ch//2 + self.pan[1], image=self.tkimg)

    def zoom_by(self, f): 
        self.zoom *= f; self.render()

    def reset(self): 
        self.zoom, self.pan = 1.0, [0, 0]; self.render()

    def wheel(self, e): 
        self.zoom_by(1.1 if e.delta > 0 else 0.9)

    def pan_start(self, e): 
        self._p = (e.x, e.y)

    def pan_move(self, e):
        dx, dy = e.x - self._p[0], e.y - self._p[1]
        self.pan[0] += dx; self.pan[1] += dy
        self._p = (e.x, e.y); self.render()

def main():
    images, anns = load_coco()
    if 'id' not in anns.columns:
        anns['id'] = [d if d == np.nan else l for d, l in zip(anns.det_id, anns.label_id)]
    if 'tag' in anns.columns and anns.tag.isna().any():
        t = anns.loc[anns.tag.notna(), 'score'].min()
        anns.loc[anns.tag.isna(), 'tag'] = 'oth'
        anns = anns[anns.score >= t]
    CocoViewer(images, anns).mainloop()

if __name__ == '__main__':
    t0 = time.time()
    main()
    logger.info(f"Finished in {time.time() - t0:.2f}s")