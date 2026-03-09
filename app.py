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
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  /* ── Root variables ── */
  :root {
    --bg:         #0d0f14;
    --surface:    #13161e;
    --surface2:   #1a1e2a;
    --border:     rgba(255,255,255,0.07);
    --accent:     #6c63ff;
    --accent2:    #00e5ff;
    --accent3:    #ff6584;
    --text:       #e8eaf0;
    --muted:      #6b7280;
    --radius:     16px;
  }

  /* ── App chrome ── */
  .stApp { background: var(--bg) !important; }
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color: var(--text); }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }

  /* ── Hero ── */
  .hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    background: linear-gradient(135deg, #0d0f14 0%, #13161e 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
      radial-gradient(ellipse 60% 40% at 20% 50%, rgba(108,99,255,0.12) 0%, transparent 70%),
      radial-gradient(ellipse 40% 50% at 80% 30%, rgba(0,229,255,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.35);
    color: #a89cff;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
  }
  .hero h1 {
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.1;
    background: linear-gradient(135deg, #fff 30%, #a89cff 70%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.8rem;
  }
  .hero p {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.6;
  }

  /* ── Cards ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
  }
  .card:hover { border-color: rgba(108,99,255,0.3); }
  .card-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* ── Metric pill ── */
  .metric-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
  }
  .metric-pill {
    flex: 1;
    min-width: 120px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
  }
  .metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }
  .metric-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent2);
    letter-spacing: -0.02em;
  }
  .metric-val.purple { color: #a89cff; }
  .metric-val.pink   { color: var(--accent3); }

  /* ── Image panels ── */
  .img-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    color: var(--muted);
    text-align: center;
    margin-top: 0.5rem;
    text-transform: uppercase;
  }

  /* ── Slider track ── */
  .stSlider [data-baseweb="slider"] { margin-top: 0.3rem; }

  /* ── Buttons ── */
  .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6c63ff, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.5rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s, transform 0.1s !important;
    cursor: pointer !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px); }
  .stButton > button:active { transform: translateY(0); }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: var(--surface2) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
  }

  /* ── Upload box ── */
  [data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1.5px dashed rgba(108,99,255,0.35) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
  }

  /* ── Hide Streamlit branding ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Divider ── */
  hr { border-color: var(--border) !important; }
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
  <div class="hero-badge">ViT-Base · Tiny ImageNet · MAE</div>
  <h1>Masked Autoencoder<br>Reconstruction</h1>
  <p>Upload any image and watch the model reconstruct it from only 25% of visible patches —
     trained with 75% masking on Tiny ImageNet.</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    mask_ratio = st.slider(
        "Mask Ratio",
        min_value=0.1, max_value=0.95, value=0.75, step=0.05,
        help="Fraction of patches masked before reconstruction.",
    )
    st.caption(f"**{int(mask_ratio * 196)} / 196** patches hidden from encoder")

    st.markdown("---")
    st.markdown("#### 📤 Upload Image")
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    run_btn = st.button("🎭  Reconstruct")

    st.markdown("---")
    with st.expander("ℹ️ Architecture"):
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
| Training | Tiny ImageNet |
| Epochs | 20 |
""")

    with st.expander("📊 Training Metrics"):
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
<div class="card" style="text-align:center; padding: 4rem 2rem;">
  <div style="font-size:3.5rem; margin-bottom:1rem;">🎭</div>
  <div style="font-size:1.2rem; font-weight:600; margin-bottom:0.5rem; color:#e8eaf0;">
    Ready to reconstruct
  </div>
  <div style="color:#6b7280; font-size:0.95rem; max-width:380px; margin:0 auto;">
    Upload an image in the sidebar, adjust the mask ratio,
    then hit <strong style="color:#a89cff">Reconstruct</strong>.
  </div>
</div>
""", unsafe_allow_html=True)

else:
    # ── Metrics row ──────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="metric-row">
  <div class="metric-pill">
    <div class="metric-label">PSNR</div>
    <div class="metric-val">{res['psnr']:.1f} <span style="font-size:0.8rem; color:#6b7280;">dB</span></div>
  </div>
  <div class="metric-pill">
    <div class="metric-label">SSIM</div>
    <div class="metric-val purple">{res['ssim']:.4f}</div>
  </div>
  <div class="metric-pill">
    <div class="metric-label">Mask</div>
    <div class="metric-val pink">{res['mask_pct']:.0f}<span style="font-size:0.8rem; color:#6b7280;">%</span></div>
  </div>
  <div class="metric-pill">
    <div class="metric-label">Hidden</div>
    <div class="metric-val" style="color:#fbbf24;">{res['n_masked']} <span style="font-size:0.8rem; color:#6b7280;">patches</span></div>
  </div>
</div>
<br/>
""", unsafe_allow_html=True)

    # ── Three-column image display ────────────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📷 &nbsp;Original</div>', unsafe_allow_html=True)
        st.image(np_to_pil(res["orig"]), use_column_width=True)
        st.markdown('<div class="img-label">Input Image</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎭 &nbsp;Masked View</div>', unsafe_allow_html=True)
        st.image(np_to_pil(res["masked"]), use_column_width=True)
        st.markdown(
            f'<div class="img-label">{res["n_masked"]} patches hidden ({res["mask_pct"]:.0f}%)</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✨ &nbsp;Reconstructed</div>', unsafe_allow_html=True)
        st.image(np_to_pil(res["recon"]), use_column_width=True)
        st.markdown(
            f'<div class="img-label">PSNR {res["psnr"]:.1f} dB · SSIM {res["ssim"]:.4f}</div>',
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
<div style="text-align:center; color:#4b5563; font-size:0.8rem; padding: 0.5rem 0 1rem;">
  MAE · ViT-Base · Tiny ImageNet &nbsp;|&nbsp;
  <span style="font-family: 'Space Mono', monospace;">PSNR 24.60 dB · SSIM 0.7338</span>
</div>
""", unsafe_allow_html=True)
