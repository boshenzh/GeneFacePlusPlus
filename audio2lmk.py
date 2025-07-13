#!/usr/bin/env python3
"""
GeneFace++ ‑ Audio ➜ MediaPipe‑style Landmark Pipeline
=====================================================
This standalone script takes a **speech audio file** and a **GeneFace++ audio2secc
checkpoint directory** (must contain *config.yaml* and
*model_ckpt_steps_XXXXXX.ckpt*), and produces:

* `<prefix>.npy`  –  NumPy array of shape **(T, 478, 2)** containing 2‑D
  MediaPipe landmarks for every frame.
* `<prefix>.mp4`  –  A white‑background video visualising the landmark motion
  (colour‑coded by facial region).

GPU is optional – add `--device cpu` if CUDA/cuDNN isn’t available.

Usage (GPU):
-------------
```bash
python geneface_lm_infer.py \
  --audio data/raw/val_wavs/coffee_fixed1.wav \
  --ckpt_dir checkpoints/audio2motion_vae \
  --out demo
```

Usage (CPU‑only):
-----------------
```bash
python geneface_lm_infer.py --device cpu ...
```
"""

import os, sys, time, logging, argparse
from pathlib import Path
from typing import List
from functools import lru_cache
from transformers import (
    AutoConfig, AutoModel, AutoProcessor,
    Wav2Vec2FeatureExtractor,
)

# -----------------------------------------------------------------------------
#  If the user forces CPU, hide all CUDA devices **before** torch import.
# -----------------------------------------------------------------------------
if "--device" in sys.argv and "cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import torch
import torchaudio

# GeneFace++ imports (repo must be on PYTHONPATH)
from utils.commons.hparams import set_hparams, hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.pitch_utils import f0_to_coarse
from data_gen.utils.process_audio.extract_hubert import get_hubert_from_16k_wav
from data_gen.utils.process_audio.extract_mel_f0 import (
    extract_mel_from_fname,
    extract_f0_from_wav_and_mel,
)
from data_util.face3d_helper import Face3DHelper
from modules.audio2motion.vae import VAEModel, PitchContourVAEModel


# -----------------------------------------------------------------------------
#  MediaPipe index groups (for colour‑coding in the demo video)
# -----------------------------------------------------------------------------
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
        317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386,
             385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
             160, 161, 246]
LEFT_BROW  = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
IRIS_R = list(range(468, 473))
IRIS_L = list(range(473, 478))

COL = {
    "oval":  (0, 0, 0),        # black
    "lips":  (0, 0, 255),      # red
    "eyes":  (0, 255, 0),      # green
    "brow":  (0, 165, 255),    # orange
    "iris":  (255, 0, 0),      # blue
    "other": (120, 120, 120),
}

# -----------------------------------------------------------------------------
#  Model + feature extraction helpers
# -----------------------------------------------------------------------------

def load_audio2secc(ckpt_dir: Path, device: str = "cuda"):
    """Instantiate GeneFace++ Audio2SECC model from a checkpoint directory."""
    cfg_path = ckpt_dir / "config.yaml"
    ckpt_path = next(ckpt_dir.glob("model_ckpt_steps_*.ckpt"), None)
    if not cfg_path.exists() or ckpt_path is None:
        raise FileNotFoundError("Checkpoint dir must contain config.yaml and model_ckpt_steps_*.ckpt")

    set_hparams(str(cfg_path))
    logging.info("Loaded hparams; motion_type=%s use_pitch=%s", hparams["motion_type"], hparams.get("use_pitch", False))

    in_out_dim = 80 + 64 if hparams["motion_type"] == "id_exp" else 64
    audio_in_dim = 1024  # HuBERT feature dim

    if hparams.get("use_pitch", False):
        model = PitchContourVAEModel(hparams, in_out_dim=in_out_dim, audio_in_dim=audio_in_dim)
    else:
        model = VAEModel(in_out_dim=in_out_dim, audio_in_dim=audio_in_dim)

    load_ckpt(model, str(ckpt_path), model_name="model", strict=True)
    model.to(device).eval()
    return model

@lru_cache()
def _get_speech_encoder(name: str, device: str):
    """
    name ∈ {"hubert", "distilhubert"}
    returns (model, processor)
    """
    if name == "hubert":
        repo = "facebook/hubert-base-ls960"
    elif name == "distilhubert":
        repo = "ntu-spml/distilhubert"
    else:
        raise ValueError(name)
    # processor = AutoProcessor.from_pretrained(mname,use_auth_token=True)
    # model     = AutoModel.from_pretrained(mname).to(device).eval()
    cfg = AutoConfig.from_pretrained(repo)

    # ❶ Try the easy path
    try:
        proc = AutoProcessor.from_pretrained(repo)
    except Exception:                      # tokenizer missing ➜ fallback
        proc = Wav2Vec2FeatureExtractor.from_pretrained(repo)
    mdl  = AutoModel.from_pretrained(repo, config=cfg).to(device).eval()

    return mdl, proc


def run_model(model, hubert: np.ndarray, f0: np.ndarray, device: str):
    """Forward pass through audio2secc and return 3‑D MediaPipe landmarks (T,478,3).

    The audio2secc checkpoint you are using was trained with **motion_type = "exp"**
    so it predicts only 64 expression coefficients.  The 3D‑MM layer still
    expects an 80‑D identity vector; we simply feed a zero vector so the shapes
    match (this produces a neutral identity).
    """
    hubert_t = torch.from_numpy(hubert).float().unsqueeze(0).to(device)
    f0_t     = torch.from_numpy(f0).float().unsqueeze(0).to(device)
    batch = {
        "hubert": hubert_t,
        "f0":     f0_t,
        "x_mask": torch.ones(1, hubert.shape[0],      device=device),
        "y_mask": torch.ones(1, hubert.shape[0] // 2, device=device),
        "audio":  hubert_t,
        "blink":  torch.zeros(1, hubert.shape[0], 1, device=device).long(),
        "eye_amp":   torch.ones(1, 1, device=device),
        "mouth_amp": torch.ones(1, 1, device=device),
    }

    helper = Face3DHelper(keypoint_mode="mediapipe", use_gpu=device.startswith("cuda"))

    with torch.no_grad():
        ret = {}
        _ = model.forward(batch, ret=ret, train=False)
        exp_coef = ret["pred"][0]              # (T,64)
        T = exp_coef.shape[0]
        id_coef = torch.zeros((T, 80), device=exp_coef.device, dtype=exp_coef.dtype)

    lm3d = helper.reconstruct_idexp_lm3d(id_coef, exp_coef)  # (T,478,3)
    return lm3d.cpu().numpy()


# -----------------------------------------------------------------------------
#  Projection + drawing helpers
# -----------------------------------------------------------------------------

def ortho_project(lm3d: np.ndarray, size: int = 720) -> np.ndarray:
    """Orthographically project 3‑D landmarks to 2‑D image coords."""
    xy = lm3d[..., :2].copy()
    # xy[..., 1] *= -1  # flip Y so +Y is downwards in image
    # min_xy = xy.min((0, 1)); max_xy = xy.max((0, 1))
    # scale = 0.9 * size / max(max_xy - min_xy)
    # xy = (xy - min_xy) * scale + 0.05 * size  # margin 5%
    return xy


def draw_points(img: np.ndarray, pts: np.ndarray):
    for idx, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        if idx in FACE_OVAL:        col = COL["oval"]
        elif idx in LIPS:           col = COL["lips"]
        elif idx in LEFT_EYE or idx in RIGHT_EYE: col = COL["eyes"]
        elif idx in LEFT_BROW or idx in RIGHT_BROW: col = COL["brow"]
        elif idx in IRIS_L or idx in IRIS_R: col = COL["iris"]
        else:                       col = COL["other"]
        cv2.circle(img, (xi, yi), 2, col, -1)


def save_video(lm2d: np.ndarray, out_mp4: str, fps: int = 25, size: int = 720):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (size, size))
    for t in range(lm2d.shape[0]):
        frame = np.full((size, size, 3), 255, np.uint8)
        draw_points(frame, lm2d[t])
        vw.write(frame)
    vw.release()
def trim_to_multiple(arr: np.ndarray, mult: int = 8) -> np.ndarray:
    # Remove leftover frames so len is multiple of `mult`
    T = arr.shape[0]
    trim_count = T % mult
    if trim_count != 0:
        arr = arr[:T - trim_count]
    return arr

def extract_features(wav16k: str,device: str = "cuda"):
    hubert = get_hubert_from_16k_wav(wav16k,device=device).detach().cpu().numpy()  # shape=(time,1024)
    wav, mel = extract_mel_from_fname(wav16k)
    f0, _    = extract_f0_from_wav_and_mel(wav, mel)
    # Force f0 to match hubert length
    if f0.shape[0] < hubert.shape[0]:
        f0 = np.pad(f0, (0, hubert.shape[0] - f0.shape[0]))
    else:
        f0 = f0[:hubert.shape[0]]

    # Now trim both to a multiple of 8
    hubert = trim_to_multiple(hubert, 8)
    f0     = trim_to_multiple(f0, 8)
    
    return hubert, f0
# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("GeneFace++ audio→landmark demo")
    parser.add_argument("--audio", required=True, help="Input WAV (any sr)")
    parser.add_argument("--ckpt_dir", required=True, help="audio2motion_vae checkpoint dir")
    parser.add_argument("--out", default="demo", help="Output prefix (.npy/.mp4)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Force device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Ensure 16 kHz mono WAV for feature extractors
    wav16k = args.audio
    if not wav16k.endswith("_16k.wav"):
        wav16k = args.audio[:-4] + "_16k.wav"
        os.system(f"ffmpeg -loglevel quiet -y -i {args.audio} -ar 16000 -ac 1 {wav16k}")

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.info("Running **CPU‑only** (CUDA disabled)")
    t0 = time.time()

    hubert, f0 = extract_features(wav16k,device=device)

    logging.info("Feature extraction time ➜ %.2fs", time.time() - t0)
    t1 = time.time()

    model = load_audio2secc(Path(args.ckpt_dir), device)
    logging.info("audio2secc loading time  ➜ %.2fs", time.time() - t1)
    t2 = time.time()
    lm3d = run_model(model, hubert, f0, device)
    logging.info("Model inference time ➜ %.2fs", time.time() - t2)
    


    
    t3 = time.time()
    # MediaPipe already 478 points; orthographic project to 2‑D
    lm2d = ortho_project(lm3d)
    logging.info("Projection time     ➜ %.2fs", time.time() - t3)
    npy_path = f"{args.out}.npy"; mp4_path = f"{args.out}.mp4"
    np.save(npy_path, lm2d.astype(np.float32))
    logging.info("Saved trajectory ➜ %s", npy_path)
    t4 = time.time()
    save_video(lm2d, mp4_path, fps=args.fps)
    logging.info("Video generation time ➜ %.2fs", time.time() - t4)
    logging.info("Saved video      ➜ %s", mp4_path)
    logging.info("Total runtime    ➜ %.2fs", time.time() - t0)


if __name__ == "__main__":
    main()
