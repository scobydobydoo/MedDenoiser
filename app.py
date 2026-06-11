"""
MedDenoiser — Streamlit Frontend
Run with:  streamlit run app.py
Place alongside: generator.py  dataset.py  noise.py  best_generator.pth
"""

import io
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ── Project imports ───────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from generator import UNet
    from dataset   import from_tensor, to_tensor, generate_images
    from noise     import add_mixed_noise
    IMPORTS_OK   = True
    IMPORT_ERROR = ""
except Exception as e:
    IMPORTS_OK   = False
    IMPORT_ERROR = str(e)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CKPT      = os.path.join(PROJECT_DIR, "best_generator.pth")
IMG_SIZE  = 256
NOISE_STD = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedDenoiser",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #080c10;
    color: #dde3ec;
}
.topbar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px 0 28px;
    border-bottom: 1px solid #1a2030;
    margin-bottom: 32px;
}
.topbar-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #1a6bff, #0d3fa6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.topbar h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.45rem;
    font-weight: 600;
    color: #fff;
    margin: 0;
    letter-spacing: -0.3px;
}
.topbar p {
    font-size: 0.78rem;
    color: #4a5568;
    margin: 3px 0 0;
    font-family: 'JetBrains Mono', monospace;
}
[data-testid="stFileUploader"] {
    border: 1.5px dashed #1e3050 !important;
    border-radius: 12px !important;
    background: #0d1420 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1a6bff 0%, #0d3fa6 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
    width: 100%;
    letter-spacing: 0.3px;
    margin-top: 6px;
}
.stButton > button:hover  { opacity: 0.88; }
.stButton > button:disabled { opacity: 0.35; }
.chips {
    display: flex;
    gap: 10px;
    margin: 18px 0 6px;
    flex-wrap: wrap;
}
.chip {
    background: #0d1420;
    border: 1px solid #1a2030;
    border-radius: 8px;
    padding: 10px 16px;
    flex: 1;
    min-width: 110px;
    text-align: center;
}
.chip .clabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #3a4a60;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 4px;
}
.chip .cval {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #1a6bff;
    line-height: 1;
}
.chip .cunit { font-size: 0.65rem; color: #3a4a60; }
.chip.up    .cval { color: #22c55e; }
.chip.down  .cval { color: #ef4444; }
.chip.muted .cval { color: #4a5568; font-size: 0.95rem; }
.img-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #3a4a60;
    text-align: center;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.sec {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    color: #1a6bff;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #1a2030;
    padding-bottom: 5px;
    margin: 28px 0 14px;
}
[data-testid="stDownloadButton"] > button {
    background: #0d1420 !important;
    border: 1px solid #1a2030 !important;
    color: #dde3ec !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    width: 100% !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 780px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not IMPORTS_OK:
        return None, IMPORT_ERROR
    if not os.path.exists(CKPT):
        return None, f"Checkpoint not found: {CKPT}"
    try:
        G = UNet(base=32).to(DEVICE)
        G.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        G.eval()
        return G, "ok"
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def read_image(upload) -> np.ndarray:
    pil = Image.open(upload).convert("L")
    arr = np.array(pil, dtype=np.float32) / 255.0
    return cv2.resize(arr, (IMG_SIZE, IMG_SIZE))


def run_pipeline(G, clean: np.ndarray, noise_std: float):
    noisy = add_mixed_noise(clean, std=noise_std)
    with torch.no_grad():
        t        = to_tensor(noisy).unsqueeze(0).to(DEVICE)
        denoised = from_tensor(G(t).squeeze(0))
    return noisy, denoised


def compute_metrics(clean, noisy, denoised):
    pn = psnr(clean, noisy,    data_range=1.0)
    pd = psnr(clean, denoised, data_range=1.0)
    sn = ssim(clean, noisy,    data_range=1.0)
    sd = ssim(clean, denoised, data_range=1.0)
    return pn, pd, sn, sd


def make_figure(clean, noisy, denoised, pn, pd):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#080c10")
    panels = [
        (clean,    "Original"),
        (noisy,    f"Noisy — {pn:.1f} dB"),
        (denoised, f"Denoised — {pd:.1f} dB"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, color="#dde3ec", fontsize=10,
                     fontweight="bold", fontfamily="monospace")
        ax.axis("off")
    plt.suptitle("MedDenoiser Results", color="#1a6bff",
                 fontsize=12, fontweight="bold", fontfamily="monospace")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def arr_to_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-icon">🫁</div>
  <div>
    <h1>MedDenoiser</h1>
    <p>GAN · UNet generator · PatchGAN discriminator</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EARLY-EXIT GUARDS
# ─────────────────────────────────────────────────────────────────────────────
if not IMPORTS_OK:
    st.error(f"**Import error** — place `generator.py`, `dataset.py`, `noise.py` "
             f"next to `app.py`.\n\n`{IMPORT_ERROR}`")
    st.stop()

G, msg = load_model()
if G is None:
    st.error(f"**Could not load model:** {msg}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a medical image (JPEG / PNG / TIFF)",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

noise_std = NOISE_STD

run_btn = st.button("⚡  Run Denoising", disabled=(uploaded is None))

# ─────────────────────────────────────────────────────────────────────────────
# IDLE STATE — stop here if button not clicked yet
# ─────────────────────────────────────────────────────────────────────────────
if not run_btn:
    if uploaded is None:
        st.markdown(
            '<p style="color:#3a4a60;font-family:\'JetBrains Mono\',monospace;'
            'font-size:0.82rem;margin-top:20px;text-align:center;">'
            '↑ Upload an image to get started</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="color:#3a4a60;font-family:\'JetBrains Mono\',monospace;'
            'font-size:0.82rem;margin-top:12px;text-align:center;">'
            'Image ready — click Run Denoising</p>',
            unsafe_allow_html=True,
        )
        col, _ = st.columns([1, 1])
        with col:
            st.image(Image.open(uploaded).convert("L"),
                     caption="Preview", use_container_width=True)
    st.stop()  # <-- nothing below this runs until button is clicked

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE + METRICS — everything in one try block so all vars are always set
# ─────────────────────────────────────────────────────────────────────────────
try:
    with st.spinner("Denoising…"):
        clean            = read_image(uploaded)
        noisy, denoised  = run_pipeline(G, clean, noise_std)
        pn, pd, sn, sd   = compute_metrics(clean, noisy, denoised)
        delta_psnr       = pd - pn
        delta_ssim       = sd - sn
        fig_bytes        = make_figure(clean, noisy, denoised, pn, pd)
        denoised_bytes   = arr_to_png(denoised)
except Exception as e:
    st.error(f"Inference failed: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS — METRICS
# ─────────────────────────────────────────────────────────────────────────────
psnr_cls = "up"   if delta_psnr >= 0 else "down"
ssim_cls = "up"   if delta_ssim >= 0 else "down"

st.markdown('<div class="sec">Results</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="chips">
  <div class="chip {psnr_cls}">
    <div class="clabel">PSNR (denoised)</div>
    <div class="cval">{pd:.1f}</div>
    <div class="cunit">dB</div>
  </div>
  <div class="chip {ssim_cls}">
    <div class="clabel">SSIM (denoised)</div>
    <div class="cval">{sd:.4f}</div>
    <div class="cunit">&nbsp;</div>
  </div>
  <div class="chip muted">
    <div class="clabel">ΔPSNR</div>
    <div class="cval">{"+" if delta_psnr >= 0 else ""}{delta_psnr:.2f}</div>
    <div class="cunit">dB improvement</div>
  </div>
  <div class="chip muted">
    <div class="clabel">ΔSSIM</div>
    <div class="cval">{"+" if delta_ssim >= 0 else ""}{delta_ssim:.4f}</div>
    <div class="cunit">&nbsp;</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS — IMAGES
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.image(noisy,    clamp=True, use_container_width=True)
    st.markdown(f'<div class="img-label">Noisy · {pn:.1f} dB</div>',
                unsafe_allow_html=True)
with c3:
    st.image(clean, clamp=True, use_container_width=True)
    st.markdown(f'<div class="img-label">Denoised · {pd:.1f} dB</div>',
                unsafe_allow_html=True)
with c2:
    st.image(denoised,    clamp=True, use_container_width=True)
    st.markdown('<div class="img-label">Clean</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOADS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Download</div>', unsafe_allow_html=True)

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "⬇  Denoised image (.png)",
        data=denoised_bytes,
        file_name="denoised.png",
        mime="image/png",
    )
with dl2:
    st.download_button(
        "⬇  Comparison figure (.png)",
        data=fig_bytes,
        file_name="result.png",
        mime="image/png",
    )