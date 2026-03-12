"""
MAE Reconstruction — Streamlit Demo
Masked Autoencoder trained on Tiny ImageNet (ViT-Base backbone)
"""

import io
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms

from model import MaskedAutoEncoder, MAEConfig

# ─── Constants ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224
PATCH_SIZE    = 16
DEVICE        = torch.device("cpu")

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MAE · Image Reconstruction",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

  /* ── Root tokens ── */
  :root {
    --bg:          #080b12;
    --surface:     #0f1219;
    --surface2:    #161b26;
    --surface3:    #1d2233;
    --glass:       rgba(255,255,255,0.035);
    --glass-border:rgba(255,255,255,0.08);
    --accent:      #7c6fff;
    --accent-glow: rgba(124,111,255,0.35);
    --cyan:        #22d3ee;
    --cyan-glow:   rgba(34,211,238,0.25);
    --pink:        #f472b6;
    --amber:       #fbbf24;
    --text:        #e2e8f0;
    --muted:       #64748b;
    --muted2:      #475569;
    --radius:      18px;
    --radius-sm:   10px;
  }

  /* ── App chrome ── */
  .stApp { background: var(--bg) !important; }
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--glass-border) !important;
  }
  section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem !important;
  }

  /* ── Sidebar heading ── */
  .sb-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0 0.25rem 1rem;
    border-bottom: 1px solid var(--glass-border);
    margin-bottom: 1.4rem;
  }
  .sb-header-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #7c6fff, #22d3ee);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
  }
  .sb-header-text { font-size: 0.9rem; font-weight: 600; color: var(--text); }
  .sb-header-sub  { font-size: 0.7rem; color: var(--muted); margin-top: 1px; }

  /* ── Section labels ── */
  .sb-section {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted2);
    margin: 1.4rem 0 0.6rem;
  }

  /* ── Ratio display (above slider) ── */
  .ratio-display {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 0.55rem;
  }
  .ratio-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.02em;
  }
  .ratio-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1;
    background: linear-gradient(125deg, #a89cff 30%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* ── Slider polish ── */
  div[data-testid="stSlider"] > div { padding: 0 !important; }
  div[data-testid="stSlider"] [data-baseweb="slider"] { margin-top: 0.1rem !important; }
  div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { display: none !important; }

  /* Track background */
  div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    height: 5px !important;
    border-radius: 999px !important;
    background: rgba(255,255,255,0.08) !important;
  }
  /* Filled track */
  div[data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"],
  div[data-testid="stSlider"] [data-baseweb="slider"] div[data-baseweb="slider-inner-thumb"],
  div[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] > div:nth-child(2) {
    background: linear-gradient(90deg, #7c6fff, #22d3ee) !important;
    height: 5px !important;
    border-radius: 999px !important;
  }
  /* Thumb */
  div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    width: 18px !important;
    height: 18px !important;
    background: #1d2233 !important;
    border: 2.5px solid #7c6fff !important;
    box-shadow: 0 0 0 3px rgba(124,111,255,0.2), 0 2px 8px rgba(0,0,0,0.5) !important;
    border-radius: 50% !important;
    transition: box-shadow 0.18s, border-color 0.18s !important;
  }
  div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:hover {
    border-color: #22d3ee !important;
    box-shadow: 0 0 0 6px rgba(124,111,255,0.22), 0 2px 12px rgba(0,0,0,0.5) !important;
  }
  /* Hide tick labels */
  div[data-testid="stSlider"] [data-testid="stTickBarMin"],
  div[data-testid="stSlider"] [data-testid="stTickBarMax"] { display: none !important; }

  /* ── Patch grid ── */
  .patch-grid {
    display: grid;
    grid-template-columns: repeat(14, 1fr);
    gap: 2px;
    margin: 0.75rem 0 0.4rem;
  }
  .patch-cell {
    aspect-ratio: 1;
    border-radius: 2px;
    transition: background 0.3s;
  }
  .patch-cell.visible  { background: rgba(124,111,255,0.55); }
  .patch-cell.masked   { background: rgba(255,255,255,0.07); }
  .patch-info {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    margin-top: 0.3rem;
  }
  .patch-info .vis  { color: #a89cff; }
  .patch-info .hid  { color: var(--muted2); }

  /* ── Upload zone ── */
  .upload-zone {
    background: var(--surface2);
    border: 1.5px dashed rgba(124,111,255,0.28);
    border-radius: var(--radius-sm);
    padding: 0.5rem;
    transition: border-color 0.2s;
  }
  .upload-zone:hover { border-color: rgba(124,111,255,0.55); }
  [data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem !important;
  }
  [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
  }

  /* ── Reconstruct button ── */
  div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #7c6fff 0%, #22d3ee 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.72rem 1.2rem !important;
    letter-spacing: 0.01em !important;
    position: relative !important;
    overflow: hidden !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px rgba(124,111,255,0.35) !important;
  }
  div[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,111,255,0.5) !important;
  }
  div[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

  /* ── Hero ── */
  .hero {
    text-align: center;
    padding: 4rem 2rem 3rem;
    position: relative;
    overflow: hidden;
    background: radial-gradient(ellipse 80% 60% at 50% -10%,
      rgba(124,111,255,0.14) 0%, transparent 65%);
    border-bottom: 1px solid var(--glass-border);
    margin-bottom: 2rem;
  }
  .hero::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,111,255,0.4), rgba(34,211,238,0.3), transparent);
  }
  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(124,111,255,0.1);
    border: 1px solid rgba(124,111,255,0.3);
    color: #a89cff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.28rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1.4rem;
  }
  .hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #7c6fff;
    box-shadow: 0 0 8px rgba(124,111,255,0.8);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
  }
  .hero h1 {
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.08;
    background: linear-gradient(135deg, #fff 20%, #c4b5fd 55%, #22d3ee 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 1rem;
  }
  .hero p {
    color: var(--muted);
    font-size: 1rem;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.7;
  }

  /* ── Stat pills row ── */
  .stat-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 2rem;
  }
  .stat-chip {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--surface2);
    border: 1px solid var(--glass-border);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.78rem;
  }
  .stat-chip .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
  }
  .stat-chip .dot.cyan  { background: var(--cyan); box-shadow: 0 0 6px var(--cyan-glow); }
  .stat-chip .dot.violет { background: var(--accent); box-shadow: 0 0 6px var(--accent-glow); }
  .stat-chip .dot.amber { background: var(--amber); }

  /* ── Metric cards ── */
  .metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }
  .metric-card {
    background: var(--surface2);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-sm);
    padding: 1rem 1.1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
  }
  .metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    border-radius: 2px 2px 0 0;
  }
  .metric-card.cyan::before  { background: linear-gradient(90deg, var(--cyan), transparent); }
  .metric-card.violet::before{ background: linear-gradient(90deg, var(--accent), transparent); }
  .metric-card.pink::before  { background: linear-gradient(90deg, var(--pink), transparent); }
  .metric-card.amber::before { background: linear-gradient(90deg, var(--amber), transparent); }
  .metric-card:hover { border-color: rgba(124,111,255,0.3); transform: translateY(-2px); }

  .metric-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
  }
  .metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.02em;
  }
  .metric-val.cyan   { color: var(--cyan); }
  .metric-val.violet { color: #a89cff; }
  .metric-val.pink   { color: var(--pink); }
  .metric-val.amber  { color: var(--amber); }
  .metric-sub {
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 0.25rem;
  }

  /* ── Image panel cards ── */
  .img-card {
    background: var(--surface);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 1.25rem;
    transition: border-color 0.25s, box-shadow 0.25s;
    height: 100%;
  }
  .img-card:hover {
    border-color: rgba(124,111,255,0.35);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }
  .img-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.9rem;
  }
  .img-card-dot {
    width: 8px; height: 8px; border-radius: 50%;
    flex-shrink: 0;
  }
  .img-card-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .img-caption {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted2);
    text-align: center;
    margin-top: 0.6rem;
    letter-spacing: 0.04em;
  }

  /* ── Empty state ── */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
    text-align: center;
  }
  .empty-icon {
    width: 80px; height: 80px;
    background: linear-gradient(135deg, rgba(124,111,255,0.15), rgba(34,211,238,0.08));
    border: 1px solid rgba(124,111,255,0.2);
    border-radius: 24px;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem;
    margin-bottom: 1.4rem;
  }
  .empty-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.5rem;
  }
  .empty-sub {
    font-size: 0.9rem;
    color: var(--muted);
    max-width: 340px;
    line-height: 1.65;
  }
  .empty-sub strong { color: #a89cff; }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: var(--surface2) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
  }
  .streamlit-expanderContent {
    background: var(--surface2) !important;
    border: 1px solid var(--glass-border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
  }

  /* ── Table in expanders ── */
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th {
    text-align: left; padding: 0.4rem 0.6rem;
    font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted2); border-bottom: 1px solid var(--glass-border);
  }
  td { padding: 0.45rem 0.6rem; color: var(--text); border-bottom: 1px solid rgba(255,255,255,0.04); }
  td:last-child { color: #a89cff; font-family: 'JetBrains Mono', monospace; }

  /* ── Download button ── */
  [data-testid="stDownloadButton"] button {
    background: var(--surface2) !important;
    border: 1px solid rgba(124,111,255,0.3) !important;
    color: #a89cff !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: background 0.2s, border-color 0.2s !important;
  }
  [data-testid="stDownloadButton"] button:hover {
    background: rgba(124,111,255,0.12) !important;
    border-color: rgba(124,111,255,0.55) !important;
  }

  /* ── Divider ── */
  hr { border-color: var(--glass-border) !important; margin: 1rem 0 !important; }

  /* ── Hide branding ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Transforms ───────────────────────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

denormalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std =[1 / s  for s  in IMAGENET_STD],
)

# ─── Model loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    cfg   = MAEConfig()
    model = MaskedAutoEncoder(cfg).to(DEVICE)
    state = torch.load("model/mae_best.pth", map_location=DEVICE)
    # Handle DataParallel checkpoint keys
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg

# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, cfg, img_tensor: torch.Tensor, mask_ratio: float):
    """
    Returns:
        original    (H,W,3) float32 [0,1]
        masked      (H,W,3) float32 [0,1]
        reconstructed (H,W,3) float32 [0,1]
    """
    imgs = img_tensor.unsqueeze(0).to(DEVICE)
    _, pred, mask = model(imgs, mask_ratio=mask_ratio)

    # ── Denormalize targets & rebuild reconstruction ──
    target_p = MaskedAutoEncoder.patchify(imgs, cfg.patch_size)
    mean_p   = target_p.mean(dim=-1, keepdim=True)
    std_p    = (target_p.var(dim=-1, keepdim=True) + 1e-6).sqrt()
    pred_px  = pred * std_p + mean_p

    # Blend: keep visible patches, fill masked with prediction
    mask_3d  = mask.unsqueeze(-1).expand_as(pred_px)
    recon_p  = target_p * (1 - mask_3d) + pred_px * mask_3d
    recon_t  = MaskedAutoEncoder.unpatchify(recon_p, cfg.patch_size, cfg.image_size)

    # ── Build masked-view (zero out masked patches) ──
    masked_p = target_p * (1 - mask_3d)
    masked_t = MaskedAutoEncoder.unpatchify(masked_p, cfg.patch_size, cfg.image_size)

    to_np = lambda t: denormalize(t[0].cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
    return to_np(imgs), to_np(masked_t), to_np(recon_t)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((arr * 255).astype(np.uint8))


def compute_psnr(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return 20 * np.log10(1.0 / (mse ** 0.5 + 1e-8))


def compute_ssim_simple(a, b, win=7):
    """Lightweight SSIM approximation (avoids scikit-image dependency)."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sig_a  = np.var(a)
    sig_b  = np.var(b)
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a + sig_b + C2)
    return float(num / (den + 1e-8))


# ─── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-badge">ViT-Base &nbsp;·&nbsp; Tiny ImageNet &nbsp;·&nbsp; MAE</div>
  <h1>Masked Autoencoder<br>Reconstruction</h1>
  <p>Upload any image and watch the model reconstruct it from only 25% of visible
     patches — trained with 75% masking on Tiny ImageNet.</p>
  <div class="stat-row">
    <div class="stat-chip"><div class="dot cyan"></div>PSNR 24.60 dB</div>
    <div class="stat-chip"><div class="dot violет"></div>SSIM 0.7338</div>
    <div class="stat-chip"><div class="dot amber"></div>196 patches · 16 × 16</div>
    <div class="stat-chip"><div class="dot cyan"></div>ViT-Base encoder</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div class="sb-header">
  <div class="sb-header-icon">⚙️</div>
  <div>
    <div class="sb-header-text">Controls</div>
    <div class="sb-header-sub">Adjust inference settings</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Mask Ratio ──────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Masking</div>', unsafe_allow_html=True)

    mask_ratio = st.slider(
        "Mask Ratio",
        min_value=0.1, max_value=0.95, value=0.75, step=0.05,
        help="Fraction of patches hidden from the encoder.",
        label_visibility="collapsed",
    )

    n_masked  = int(mask_ratio * 196)
    n_visible = 196 - n_masked

    # Clean ratio display
    st.markdown(f"""
<div class="ratio-display">
  <span class="ratio-label">Mask Ratio</span>
  <span class="ratio-val">{mask_ratio:.0%}</span>
</div>
""", unsafe_allow_html=True)

    # Patch grid — pseudo-random scatter so visible patches look realistic
    import random as _rng
    _r = _rng.Random(int(mask_ratio * 10000))
    _indices = list(range(196))
    _r.shuffle(_indices)
    _masked_set = set(_indices[:n_masked])
    cells = "".join(
        f'<div class="patch-cell {"masked" if i in _masked_set else "visible"}"></div>'
        for i in range(196)
    )

    st.markdown(f"""
<div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
            border-radius:10px; padding:0.75rem 0.75rem 0.5rem; margin-top:0.2rem;">
  <div style="font-size:0.6rem; font-weight:600; letter-spacing:0.12em;
              text-transform:uppercase; color:#334155; margin-bottom:0.55rem;">
    Patch Map &nbsp;·&nbsp; 14 × 14
  </div>
  <div class="patch-grid">{cells}</div>
  <div class="patch-info" style="margin-top:0.5rem;">
    <span class="vis">▪ {n_visible} visible</span>
    <span class="hid">{n_masked} masked ▪</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Image Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        help="JPG, JPEG, PNG or WebP · up to 200 MB",
    )

    st.markdown("")
    run_btn = st.button("✦  Reconstruct", use_container_width=True)

    # ── Architecture info ────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Model Info</div>', unsafe_allow_html=True)
    with st.expander("ℹ️  Architecture"):
        st.markdown("""
| Component | Config |
|-----------|--------|
| Backbone  | ViT-Base |
| Image size | 224 × 224 |
| Patch size | 16 × 16 |
| Enc depth | 12 blocks |
| Dec depth | 12 blocks |
| Enc dim | 768 |
| Dec dim | 384 |
| Dataset | Tiny ImageNet |
| Epochs | 20 |
""")

    with st.expander("📊  Training Metrics"):
        st.markdown("""
| Metric | Score |
|--------|-------|
| PSNR   | 24.60 dB |
| SSIM   | 0.7338 |
| Val images | 7,680 |
""")


# ─── Main content ─────────────────────────────────────────────────────────────

# Load model once
with st.spinner("Loading model weights…"):
    try:
        model, cfg = load_model()
        model_ok   = True
    except Exception as e:
        model_ok  = False
        model_err = str(e)

if not model_ok:
    st.error(f"**Could not load model:** {model_err}")
    st.info("Make sure `model/mae_best.pth` is placed inside the `model/` folder.")
    st.stop()

# ── State ─────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Run inference ─────────────────────────────────────────────────────────────
if run_btn:
    if uploaded is None:
        st.warning("Please upload an image first.")
    else:
        pil_img    = Image.open(uploaded).convert("RGB")
        img_tensor = preprocess(pil_img)

        with st.spinner("Reconstructing…"):
            orig, masked, recon = run_inference(model, cfg, img_tensor, mask_ratio)

        psnr_val = compute_psnr(orig, recon)
        ssim_val = compute_ssim_simple(
            (orig * 255).astype(np.uint8),
            (recon * 255).astype(np.uint8),
        )
        n_masked = int(mask_ratio * 196)

        st.session_state.results = {
            "orig":   orig,
            "masked": masked,
            "recon":  recon,
            "psnr":   psnr_val,
            "ssim":   ssim_val,
            "n_masked": n_masked,
            "mask_pct": mask_ratio * 100,
        }

# ── Display results ───────────────────────────────────────────────────────────
res = st.session_state.results

if res is None:
    # Empty state
    st.markdown("""
<div class="empty-state">
  <div class="empty-icon">🎭</div>
  <div class="empty-title">Ready to reconstruct</div>
  <div class="empty-sub">
    Upload an image in the sidebar, adjust the mask ratio,
    then hit <strong>Reconstruct</strong>.
  </div>
</div>
""", unsafe_allow_html=True)

else:
    # ── Metrics row ──────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="metric-row">
  <div class="metric-card cyan">
    <div class="metric-label">PSNR</div>
    <div class="metric-val cyan">{res['psnr']:.1f}</div>
    <div class="metric-sub">dB</div>
  </div>
  <div class="metric-card violet">
    <div class="metric-label">SSIM</div>
    <div class="metric-val violet">{res['ssim']:.4f}</div>
    <div class="metric-sub">similarity index</div>
  </div>
  <div class="metric-card pink">
    <div class="metric-label">Mask Ratio</div>
    <div class="metric-val pink">{res['mask_pct']:.0f}<span style="font-size:1rem">%</span></div>
    <div class="metric-sub">of patches hidden</div>
  </div>
  <div class="metric-card amber">
    <div class="metric-label">Hidden</div>
    <div class="metric-val amber">{res['n_masked']}</div>
    <div class="metric-sub">of 196 patches</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Three-column image display ────────────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("""
<div class="img-card">
  <div class="img-card-header">
    <div class="img-card-dot" style="background:#22d3ee; box-shadow:0 0 6px rgba(34,211,238,0.5);"></div>
    <div class="img-card-title">Original</div>
  </div>
""", unsafe_allow_html=True)
        st.image(np_to_pil(res["orig"]), use_column_width=True)
        st.markdown('<div class="img-caption">Input Image</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div class="img-card">
  <div class="img-card-header">
    <div class="img-card-dot" style="background:#f472b6; box-shadow:0 0 6px rgba(244,114,182,0.5);"></div>
    <div class="img-card-title">Masked View</div>
  </div>
""", unsafe_allow_html=True)
        st.image(np_to_pil(res["masked"]), use_column_width=True)
        st.markdown(
            f'<div class="img-caption">{res["n_masked"]} patches hidden &nbsp;·&nbsp; {res["mask_pct"]:.0f}% masked</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class="img-card">
  <div class="img-card-header">
    <div class="img-card-dot" style="background:#a89cff; box-shadow:0 0 6px rgba(168,156,255,0.5);"></div>
    <div class="img-card-title">Reconstructed</div>
  </div>
""", unsafe_allow_html=True)
        st.image(np_to_pil(res["recon"]), use_column_width=True)
        st.markdown(
            f'<div class="img-caption">PSNR {res["psnr"]:.1f} dB &nbsp;·&nbsp; SSIM {res["ssim"]:.4f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Download reconstructed ────────────────────────────────────────────────
    buf = io.BytesIO()
    np_to_pil(res["recon"]).save(buf, format="PNG")
    st.download_button(
        "⬇️  Download Reconstruction",
        data=buf.getvalue(),
        file_name="mae_reconstruction.png",
        mime="image/png",
    )

    # ── Difference map ───────────────────────────────────────────────────────
    with st.expander("🔍 Pixel Difference Map"):
        diff = np.abs(res["orig"].astype(np.float32) - res["recon"].astype(np.float32))
        diff_norm = (diff / diff.max() * 255).astype(np.uint8)
        # Colorize: apply green-to-red heat
        heatmap = np.zeros((*diff_norm.shape[:2], 3), dtype=np.uint8)
        gray = diff_norm.mean(axis=-1)
        heatmap[:, :, 0] = gray          # R channel → errors
        heatmap[:, :, 1] = 255 - gray    # G channel → accurate regions
        c1, c2 = st.columns(2)
        c1.image(Image.fromarray(diff_norm), caption="Absolute Difference (per channel)", use_column_width=True)
        c2.image(Image.fromarray(heatmap), caption="Error Heatmap  (green=good, red=error)", use_column_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.75rem; padding:0.5rem 0 1.2rem;
            font-family:'JetBrains Mono',monospace; letter-spacing:0.05em;">
  MAE &nbsp;·&nbsp; ViT-Base &nbsp;·&nbsp; Tiny ImageNet &nbsp;&nbsp;|&nbsp;&nbsp;
  PSNR 24.60 dB &nbsp;·&nbsp; SSIM 0.7338
</div>
""", unsafe_allow_html=True)
