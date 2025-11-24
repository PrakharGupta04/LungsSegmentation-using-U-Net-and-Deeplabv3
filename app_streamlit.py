# app_streamlit.py
"""
Compact Streamlit UI for Lung Segmentation (final)

- Left: fixed sidebar (title, uploader, optional path input, two sliders, Generate button).
- Right: scrollable results area showing UNet & DeepLab prob/mask/overlay, compact plots, and run summary.

Place this file next to infer_compare.py and run:
    streamlit run app_streamlit.py
"""

import streamlit as st
from pathlib import Path
import subprocess, os, sys, json
from tempfile import NamedTemporaryFile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lungs Segmentation", layout="wide")

# ----------------- Config (compact) -----------------
DEFAULT_SAMPLE = ""   # optional default local image path (leave empty if not needed)
OUT_ROOT = "inference_results"
IMG_SIZE = 128        # compact display size (you chose Option 3)
DISPLAY_WIDTH = 120   # thumbnail display width (px)
DEFAULT_UNET_THRESH = 0.5
DEFAULT_DEEPLAB_THRESH = 0.3

# Try to import run_inference from infer_compare (preferred)
try:
    from infer_compare import run_inference
    DIRECT_CALL = True
except Exception:
    DIRECT_CALL = False

# ----------------- Minimal CSS to keep sidebar compact -----------------
st.markdown(
    """
    <style>
      /* compact sidebar paddings */
      .css-1lcbkhc.e1fqkh3o2 { padding-top: 10px; }
      .css-18e3th9 { padding-top: 6px; padding-bottom: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Sidebar (fixed controls) -----------------
st.sidebar.title("Lungs Segmentation")
st.sidebar.write("Upload an X-ray or enter a local path, set thresholds, then click Generate.")

uploaded = st.sidebar.file_uploader("Upload X-ray (png/jpg/bmp)", type=["png","jpg","jpeg","bmp"])
image_path_input = st.sidebar.text_input("Image path (optional)", value=DEFAULT_SAMPLE)

st.sidebar.markdown("**Thresholds (binary masks)**")
th_unet = st.sidebar.slider("UNet threshold", 0.0, 1.0, DEFAULT_UNET_THRESH, 0.01)
th_deeplab = st.sidebar.slider("DeepLab threshold", 0.0, 1.0, DEFAULT_DEEPLAB_THRESH, 0.01)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Generate masks")
st.sidebar.markdown("---")
st.sidebar.caption("Results saved under: inference_results/run_<timestamp>")

# ----------------- Helpers -----------------
def read_prob(path):
    im = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    return np.array(im).astype(np.float32) / 255.0

def make_overlay(orig_pil, mask_np, color=(180,0,0), alpha=0.45):
    orig = orig_pil.convert("RGBA").resize((IMG_SIZE, IMG_SIZE))
    mask_img = Image.fromarray((mask_np*255).astype("uint8")).convert("L")
    colored = Image.new("RGBA", orig.size, color + (0,))
    colored.putalpha(mask_img)
    blended = Image.blend(orig, colored, alpha=alpha)
    return blended.convert("RGB")

def run_subprocess_infer(img_path, out_root="inference_results"):
    cmd = [sys.executable, "infer_compare.py", "--image", str(img_path), "--out", out_root]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy(), text=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr

def parse_run_dir_from_output(text):
    for line in text.splitlines()[::-1]:
        if "Inference finished. Results in:" in line:
            return line.split("Inference finished. Results in:")[-1].strip()
        if "Done. Check folder:" in line:
            return line.split("Done. Check folder:")[-1].strip()
    cand = sorted(Path("inference_results").glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cand[0]) if cand else None

# ----------------- Main UI (results) -----------------
status = st.empty()
results = st.container()

if run_button:
    # Prepare image path (uploaded or provided)
    if uploaded is not None:
        tmp = NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[-1])
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        image_path = tmp.name
    else:
        image_path = image_path_input.strip()
        if not image_path:
            status.error("No image provided. Upload or enter a path in the left sidebar.")
            st.stop()

    status.info("Running inference — please wait...")

    run_dir = None
    if DIRECT_CALL:
        try:
            rd = run_inference(image_path, out_root=OUT_ROOT, show=False)
            run_dir = str(rd)
            status.success(f"Inference finished. Results in: {run_dir}")
        except Exception as e:
            status.warning(f"Direct call failed ({e}). Falling back to subprocess.")
            DIRECT_CALL = False

    if not DIRECT_CALL:
        rc, out = run_subprocess_infer(image_path, out_root=OUT_ROOT)
        if rc != 0:
            status.error("Inference failed. See logs below.")
            st.code(out)
            st.stop()
        run_dir = parse_run_dir_from_output(out)
        status.success(f"Inference finished. Results in: {run_dir}")

    if run_dir is None:
        status.error("No output run folder found.")
        st.stop()

    run_dirp = Path(run_dir)
    unet_prob_path = run_dirp / "unet_prob.png"
    deeplab_prob_path = run_dirp / "deeplab_prob.png"
    summary_path = run_dirp / "summary.json"

    if not unet_prob_path.exists() or not deeplab_prob_path.exists():
        status.error("Expected probability images not found in run directory.")
        st.stop()

    # Load original for overlay (fallback to first prob if not available)
    try:
        orig = Image.open(image_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    except Exception:
        orig = Image.fromarray((read_prob(unet_prob_path)*255).astype("uint8")).convert("L")

    # Read probs and compute masks based on sliders
    prob_u = read_prob(unet_prob_path)
    prob_d = read_prob(deeplab_prob_path)
    mask_u = (prob_u > th_unet).astype(np.uint8)
    mask_d = (prob_d > th_deeplab).astype(np.uint8)
    overlay_u = make_overlay(orig, mask_u, color=(180,0,0))
    overlay_d = make_overlay(orig, mask_d, color=(0,120,0))

    with results:
        st.header("Results")

        st.subheader("UNet outputs")
        c1, c2, c3 = st.columns([1,1,1])
        c1.image((prob_u*255).astype("uint8"), caption="UNet prob", width=DISPLAY_WIDTH, clamp=True, channels="L")
        c2.image((mask_u*255).astype("uint8"), caption=f"UNet mask (thr={th_unet:.2f})", width=DISPLAY_WIDTH, clamp=True, channels="L")
        c3.image(overlay_u, caption="UNet overlay", width=DISPLAY_WIDTH)

        st.subheader("DeepLab outputs")
        d1, d2, d3 = st.columns([1,1,1])
        d1.image((prob_d*255).astype("uint8"), caption="DeepLab prob", width=DISPLAY_WIDTH, clamp=True, channels="L")
        d2.image((mask_d*255).astype("uint8"), caption=f"DeepLab mask (thr={th_deeplab:.2f})", width=DISPLAY_WIDTH, clamp=True, channels="L")
        d3.image(overlay_d, caption="DeepLab overlay", width=DISPLAY_WIDTH)

        # Plots: density (smoothed histogram), violin, mask counts & % area
        st.subheader("Plots & comparison")
        fig, axes = plt.subplots(2,2, figsize=(8.5,5.5))
        for ax in axes.flatten():
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Density
        bins = 200
        hu, edges = np.histogram(prob_u.ravel(), bins=bins, range=(0,1), density=True)
        hd, edges = np.histogram(prob_d.ravel(), bins=bins, range=(0,1), density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        try:
            from scipy.ndimage import gaussian_filter1d
            hu_s = gaussian_filter1d(hu, sigma=2)
            hd_s = gaussian_filter1d(hd, sigma=2)
        except Exception:
            hu_s = hu; hd_s = hd
        axes[0,0].plot(centers, hu_s, label="UNet", color="C0")
        axes[0,0].plot(centers, hd_s, label="DeepLab", color="C1")
        axes[0,0].axvline(th_unet, color="C0", linestyle="--")
        axes[0,0].axvline(th_deeplab, color="C1", linestyle="--")
        axes[0,0].set_title("Probability density")
        axes[0,0].legend(fontsize="small")

        # Violin
        axes[0,1].violinplot([prob_u.ravel(), prob_d.ravel()], showmeans=True)
        axes[0,1].set_xticks([1,2]); axes[0,1].set_xticklabels(["UNet","DeepLab"])
        axes[0,1].set_title("Violin (prob distribution)")

        # Mask counts
        count_u = int(mask_u.sum()); count_d = int(mask_d.sum()); total = IMG_SIZE*IMG_SIZE
        axes[1,0].bar(["UNet","DeepLab"], [count_u, count_d], color=["C0","C1"])
        axes[1,0].set_title("Mask pixel counts")
        axes[1,0].set_ylabel("Pixels (128x128)")

        # Percent area
        pct_u = 100.0 * count_u / total; pct_d = 100.0 * count_d / total
        axes[1,1].barh(["UNet","DeepLab"], [pct_u, pct_d], color=["C0","C1"])
        axes[1,1].set_xlim(0,100)
        axes[1,1].set_title("Mask area (%)")

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Run summary")
        if summary_path.exists():
            try:
                s = json.loads(summary_path.read_text())
                st.json(s)
            except Exception:
                st.write("Could not read summary.json")
        else:
            quick = {
                "unet": {"threshold": th_unet, "mean_prob": float(prob_u.mean()), "max_prob": float(prob_u.max()), "mask_sum": count_u},
                "deeplab": {"threshold": th_deeplab, "mean_prob": float(prob_d.mean()), "max_prob": float(prob_d.max()), "mask_sum": count_d},
                "output_folder": str(run_dirp)
            }
            st.json(quick)

    status.success("Done — left sidebar remains fixed while results appear on the right.")

else:
    with results:
        st.info("Use the left sidebar to upload an X-ray (or enter path), set thresholds, and press Generate. Results appear on the right.")
