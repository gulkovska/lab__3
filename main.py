import os
import sys
import subprocess

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox


BLOCK = 8
DWT_LEVELS = 4


def dct_matrix(n=BLOCK):
    u = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    C = np.cos((2 * x + 1) * u * np.pi / (2 * n))
    alpha = np.full(n, np.sqrt(2.0 / n))
    alpha[0] = np.sqrt(1.0 / n)
    return alpha[:, None] * C


DCT_C = dct_matrix()


def pad_image(img, multiple):
    h, w = img.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="edge")
    return img, h, w


def keep_largest(coeffs, keep_ratio):
    out = coeffs.copy()
    k = max(1, int(round(keep_ratio * out.size)))
    if k >= out.size:
        return out, 1.0
    flat = np.abs(out).ravel()
    thresh = np.partition(flat, -k)[-k]
    out[np.abs(out) < thresh] = 0.0
    kept = np.count_nonzero(out)
    return out, kept / out.size


def psnr(original, restored):
    a = original.astype(np.float64)
    b = np.clip(restored, 0, 255).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def dct_forward(img):
    x, h0, w0 = pad_image(img.astype(np.float64) - 128.0, BLOCK)
    h, w = x.shape
    blocks = x.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).transpose(0, 2, 1, 3)
    y = np.matmul(np.matmul(DCT_C, blocks), DCT_C.T)
    return y, h0, w0, h, w


def dct_inverse(y, h0, w0, h, w):
    blocks = np.matmul(np.matmul(DCT_C.T, y), DCT_C)
    rec = blocks.transpose(0, 2, 1, 3).reshape(h, w) + 128.0
    return rec[:h0, :w0]


def compress_dct(img, keep_ratio):
    y, h0, w0, h, w = dct_forward(img)
    y_q, kept = keep_largest(y, keep_ratio)
    rec = dct_inverse(y_q, h0, w0, h, w)
    return rec, kept


def haar_1d(x):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    s = (even + odd) / np.sqrt(2.0)
    d = (even - odd) / np.sqrt(2.0)
    return np.concatenate([s, d], axis=-1)


def ihaar_1d(y):
    n = y.shape[-1] // 2
    s, d = y[..., :n], y[..., n:]
    even = (s + d) / np.sqrt(2.0)
    odd = (s - d) / np.sqrt(2.0)
    out = np.empty_like(y)
    out[..., 0::2] = even
    out[..., 1::2] = odd
    return out


def haar_2d(img):
    tmp = haar_1d(img)
    return haar_1d(tmp.T).T


def ihaar_2d(img):
    tmp = ihaar_1d(img.T).T
    return ihaar_1d(tmp)


def dwt_forward(img, levels=DWT_LEVELS):
    step = 2 ** levels
    x, h0, w0 = pad_image(img.astype(np.float64), step)
    coeffs = x.copy()
    h, w = coeffs.shape
    sizes = []
    for _ in range(levels):
        sizes.append((h, w))
        coeffs[:h, :w] = haar_2d(coeffs[:h, :w])
        h //= 2
        w //= 2
    return coeffs, h0, w0, sizes


def dwt_inverse(coeffs, h0, w0, sizes):
    rec = coeffs.copy()
    for h, w in reversed(sizes):
        rec[:h, :w] = ihaar_2d(rec[:h, :w])
    return rec[:h0, :w0]


def compress_dwt(img, keep_ratio, levels=DWT_LEVELS):
    coeffs, h0, w0, sizes = dwt_forward(img, levels)
    coeffs_q, kept = keep_largest(coeffs, keep_ratio)
    rec = dwt_inverse(coeffs_q, h0, w0, sizes)
    return rec, kept


def make_test_image(n=512):
    y, x = np.mgrid[0:n, 0:n]
    img = 30 + 40 * (x / n) + 25 * (y / n)
    img += 35 * np.sin(2 * np.pi * x / 32) * np.sin(2 * np.pi * y / 48)
    img += 20 * ((x // 32 + y // 32) % 2)
    cx, cy, r = n * 0.35, n * 0.4, n * 0.12
    img[(x - cx) ** 2 + (y - cy) ** 2 < r ** 2] = 220
    img[n // 2 : n // 2 + 40, n // 5 : 4 * n // 5] = 40
    img[n // 6 : n // 6 + 8, :] = 255
    return np.clip(img, 0, 255).astype(np.float64)


def load_gray(path, size=512):
    img = Image.open(path).convert("L")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float64)


def choose_image_path():
    if sys.platform != "darwin":
        return ""
    script = (
        "try\n"
        'POSIX path of (choose file with prompt "Оберіть зображення")\n'
        "on error\n"
        'return ""\n'
        "end try"
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_experiments(img, ratios=(0.20, 0.10, 0.05)):
    rows = []
    for r in ratios:
        dct_img, dct_kept = compress_dct(img, r)
        dwt_img, dwt_kept = compress_dwt(img, r)
        rows.append({
            "keep": r,
            "dct_img": dct_img,
            "dwt_img": dwt_img,
            "dct_kept": dct_kept,
            "dct_psnr": psnr(img, dct_img),
            "dct_cr": 1.0 / dct_kept,
            "dwt_kept": dwt_kept,
            "dwt_psnr": psnr(img, dwt_img),
            "dwt_cr": 1.0 / dwt_kept,
        })
    return rows


def format_table(rows):
    lines = [
        "Keep%   DCT kept   DCT PSNR   DCT CR   DWT kept   DWT PSNR   DWT CR",
        "------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['keep'] * 100:5.1f}   {row['dct_kept'] * 100:7.2f}%   {row['dct_psnr']:8.2f}   "
            f"{row['dct_cr']:6.1f}   {row['dwt_kept'] * 100:7.2f}%   {row['dwt_psnr']:8.2f}   "
            f"{row['dwt_cr']:6.1f}"
        )
    lines.append("CR = усі коефіцієнти / ненульові (не розмір файлу)")
    lines.append(make_conclusions(rows))
    return "\n".join(lines)


def make_conclusions(rows):
    dct_better = sum(r["dct_psnr"] > r["dwt_psnr"] for r in rows)
    winner = "DCT" if dct_better >= 2 else "DWT"
    return (
        f"Висновок: менше Keep% → нижчий PSNR. На цьому фото {winner} має вищий PSNR. "
        "DCT дає сітку 8x8, Haar — сходинки на границях."
    )


class App:
    def __init__(self):
        self.img = make_test_image(512)
        self.fig, self.axes = plt.subplots(1, 3, figsize=(14, 8.2))
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.42, wspace=0.18)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("ЛР3: DCT vs DWT")

        self.im_orig = self.axes[0].imshow(self.img, cmap="gray", vmin=0, vmax=255)
        self.im_dct = self.axes[1].imshow(self.img, cmap="gray", vmin=0, vmax=255)
        self.im_dwt = self.axes[2].imshow(self.img, cmap="gray", vmin=0, vmax=255)
        for ax, title in zip(self.axes, ["Original", "DCT", "DWT (Haar)"]):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, pad=8)

        ax_s = self.fig.add_axes([0.12, 0.30, 0.78, 0.035])
        self.slider = Slider(ax_s, "Keep %", 1.0, 40.0, valinit=10.0, valstep=1.0)

        ax_path = self.fig.add_axes([0.12, 0.23, 0.78, 0.04])
        self.path_box = TextBox(ax_path, "Файл ", initial="")
        ax_load = self.fig.add_axes([0.12, 0.155, 0.22, 0.05])
        ax_exp = self.fig.add_axes([0.37, 0.155, 0.22, 0.05])
        self.btn_load = Button(ax_load, "Відкрити зображення")
        self.btn_exp = Button(ax_exp, "3 рівні стиснення")

        ax_info = self.fig.add_axes([0.12, 0.02, 0.78, 0.12])
        ax_info.axis("off")
        self.info = ax_info.text(
            0, 1, "", va="top", ha="left", family="monospace", fontsize=9
        )

        self.slider.on_changed(self.on_slider)
        self.btn_load.on_clicked(self.on_load)
        self.btn_exp.on_clicked(self.on_experiments)
        self.path_box.on_submit(self.load_from_path)
        self.exp_fig = None
        self.update()
        self.info.set_text(format_table(run_experiments(self.img)))

    def on_slider(self, _):
        self.update()

    def load_from_path(self, path):
        path = path.strip().strip('"').strip("'")
        if not path:
            return
        if not os.path.isfile(path):
            self.info.set_text(f"Файл не знайдено:\n{path}")
            self.fig.canvas.draw_idle()
            return
        try:
            self.img = load_gray(path, 512)
        except Exception as exc:
            self.info.set_text(f"Не вдалося відкрити файл:\n{path}\n{exc}")
            self.fig.canvas.draw_idle()
            return
        self.path_box.eventson = False
        self.path_box.set_val(path)
        self.path_box.eventson = True
        self.update()
        self.info.set_text(format_table(run_experiments(self.img)))

    def on_load(self, _):
        path = choose_image_path()
        if path:
            self.load_from_path(path)
            return
        typed = self.path_box.text.strip()
        if typed:
            self.load_from_path(typed)

    def on_experiments(self, _):
        rows = run_experiments(self.img)
        table = format_table(rows)
        print("\n" + table + "\n")
        self.info.set_text(table)
        self.fig.canvas.draw_idle()
        self.show_levels(rows)

    def show_levels(self, rows):
        if self.exp_fig is not None:
            plt.close(self.exp_fig)
        self.exp_fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        self.exp_fig.suptitle("Три рівні стиснення: 20%, 10%, 5% коефіцієнтів")
        if self.exp_fig.canvas.manager is not None:
            self.exp_fig.canvas.manager.set_window_title("Експерименти DCT / DWT")

        axes[0, 0].imshow(self.img, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title("Original")
        axes[1, 0].imshow(self.img, cmap="gray", vmin=0, vmax=255)
        axes[1, 0].set_title("Original")
        for i, row in enumerate(rows):
            axes[0, i + 1].imshow(np.clip(row["dct_img"], 0, 255), cmap="gray", vmin=0, vmax=255)
            axes[0, i + 1].set_title(
                f"DCT  kept {row['dct_kept'] * 100:.1f}%\n"
                f"PSNR {row['dct_psnr']:.2f} dB   CR {row['dct_cr']:.1f}"
            )
            axes[1, i + 1].imshow(np.clip(row["dwt_img"], 0, 255), cmap="gray", vmin=0, vmax=255)
            axes[1, i + 1].set_title(
                f"DWT  kept {row['dwt_kept'] * 100:.1f}%\n"
                f"PSNR {row['dwt_psnr']:.2f} dB   CR {row['dwt_cr']:.1f}"
            )
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        self.exp_fig.tight_layout()
        self.exp_fig.subplots_adjust(bottom=0.12)
        self.exp_fig.text(0.5, 0.03, make_conclusions(rows), ha="center", fontsize=10)
        self.exp_fig.show()

    def update(self):
        keep = self.slider.val / 100.0
        dct_img, dct_kept = compress_dct(self.img, keep)
        dwt_img, dwt_kept = compress_dwt(self.img, keep)
        dct_p = psnr(self.img, dct_img)
        dwt_p = psnr(self.img, dwt_img)

        self.im_orig.set_data(self.img)
        self.im_dct.set_data(np.clip(dct_img, 0, 255))
        self.im_dwt.set_data(np.clip(dwt_img, 0, 255))
        self.axes[0].set_title("Original")
        self.axes[1].set_title("DCT")
        self.axes[2].set_title("DWT (Haar)")
        self.axes[0].set_xlabel("")
        self.axes[1].set_xlabel(
            f"kept {dct_kept * 100:.1f}%   PSNR {dct_p:.2f} dB   CR {1 / dct_kept:.1f}"
        )
        self.axes[2].set_xlabel(
            f"kept {dwt_kept * 100:.1f}%   PSNR {dwt_p:.2f} dB   CR {1 / dwt_kept:.1f}"
        )
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    app = App()
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            app.img = load_gray(path, 512)
            app.update()
            app.info.set_text(format_table(run_experiments(app.img)))
        else:
            print(f"Файл не знайдено: {path}")
    app.show()
