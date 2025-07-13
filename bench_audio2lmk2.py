#!/usr/bin/env python3
"""
GeneFace++ – real‑time latency benchmark
---------------------------------------

Start the script *once*; it initialises

  • HuBERT (facebook/hubert‑large‑ls960‑ft)  
  • Wav2Vec2Processor  
  • GeneFace++ audio2secc checkpoint

Then it **watches an input directory**.  Whenever a new 16 kHz mono WAV
appears, the pipeline runs and reports:

    feat  : feature‑extraction time
    model : audio2secc forward pass
    total : feat + model   (does not include video rendering)

Example
~~~~~~~
    python bench_realtime.py \
        --ckpt_dir checkpoints/audio2motion_vae \
        --in_dir   data/benchmark_wavs \
        --device   cuda
"""

import os, time, argparse, logging, glob
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch, torchaudio
from transformers import HubertModel, Wav2Vec2Processor

# ─────────────────────────────────────────────────────────────────────────────
#  GeneFace++ imports
# ─────────────────────────────────────────────────────────────────────────────
from utils.commons.hparams import set_hparams, hparams
from utils.commons.ckpt_utils import load_ckpt
from modules.audio2motion.vae import VAEModel, PitchContourVAEModel
from data_gen.utils.process_audio.extract_mel_f0 import (
    extract_mel_from_fname, extract_f0_from_wav_and_mel
)
from data_util.face3d_helper import Face3DHelper

# ─────────────────────────────────────────────────────────────────────────────
#  HuBERT loader (cached)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache()
def get_hubert(device: str):
    repo = "facebook/hubert-large-ls960-ft"
    model = HubertModel.from_pretrained(repo).to(device).eval()
    proc  = Wav2Vec2Processor.from_pretrained(repo)
    return model, proc

# ─────────────────────────────────────────────────────────────────────────────
#  GeneFace++ model loader (cached)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache()
def get_audio2secc(ckpt_dir: str, device: str):
    cfg = Path(ckpt_dir) / "config.yaml"
    ckpt = next(Path(ckpt_dir).glob("model_ckpt_steps_*.ckpt"))
    set_hparams(str(cfg))

    in_out_dim   = 80 + 64 if hparams["motion_type"] == "id_exp" else 64
    audio_in_dim = 1024
    if hparams.get("use_pitch", False):
        model = PitchContourVAEModel(hparams, in_out_dim, audio_in_dim)
    else:
        model = VAEModel(in_out_dim, audio_in_dim)

    load_ckpt(model, str(ckpt), model_name="model", strict=True)
    return model.to(device).eval()

# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction (HuBERT + F0)  — identical to your earlier helper
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(wav16k: str, hubert_m, hubert_p, device: str):
    wav, sr = torchaudio.load(wav16k)
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.squeeze(0)

    with torch.inference_mode():
        inp   = hubert_p(wav, sampling_rate=16_000,
                         return_tensors="pt").input_values.to(device)
        hub   = hubert_m(inp).last_hidden_state[0].cpu().numpy()  # (T,1024)
    hub = trim_to_multiple(hub, 24)            # 24 is safe for both VAE & GT

    # F0
    wav_np, mel = wav.numpy(), extract_mel_from_fname(wav16k)[1]
    f0, _ = extract_f0_from_wav_and_mel(wav_np, mel)
    if f0.shape[0] < hub.shape[0]:
        f0 = np.pad(f0, (0, hub.shape[0]-f0.shape[0]))
    else:
        f0 = f0[:hub.shape[0]]

    return hub, f0
def trim_to_multiple(arr: np.ndarray, mult: int) -> np.ndarray:
    """Trim first axis so len % mult == 0 (no padding, keeps RAM small)."""
    return arr[: len(arr) - (len(arr) % mult)]  if len(arr) % mult else arr
# ------------------------------------------------------------------


# ─────────────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--in_dir",   required=True,
                    help="Folder to watch for 16 kHz mono WAV files")
    ap.add_argument("--device",   default="cuda", choices=["cuda","cpu"])
    args = ap.parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("bench")

    # warm‑up load
    hubert_m, hubert_p = get_hubert(device)
    audio2secc = get_audio2secc(args.ckpt_dir, device)
    helper = Face3DHelper(keypoint_mode="mediapipe", use_gpu=False)

    seen = set()
    times = []

    log.info("Initialisation complete.  Watching %s …", args.in_dir)
    while True:
        for wav in sorted(glob.glob(f"{args.in_dir}/*.wav")):
            if wav in seen:             # already processed
                continue
            seen.add(wav)

            t0 = time.time()
            hub, f0 = extract_features(wav, hubert_m, hubert_p, device)
            t_feat = time.time() - t0

            # forward pass
            with torch.no_grad():
                hub_t = torch.from_numpy(hub).unsqueeze(0).to(device).float()
                f0_t  = torch.from_numpy(f0).unsqueeze(0).to(device).float()

                batch = {
                    "hubert": hub_t,
                    "f0":     f0_t,
                    "audio":  hub_t,                 # <- **added**
                    "x_mask": torch.ones_like(hub_t[..., 0]),      # (1, F)
                    "y_mask": torch.ones(1, hub.shape[0] // 2, device=device),
                }

                ret = {}
                audio2secc(batch, ret=ret, train=False)
            t_model = time.time() - t0 - t_feat
            t_total = time.time() - t0
            times.append(t_total)

            log.info("%s | feat %.3f s | model %.3f s | total %.3f s | avg %.3f s",
                     Path(wav).name, t_feat, t_model, t_total,
                     sum(times)/len(times))
        time.sleep(0.2)          # poll interval


if __name__ == "__main__":
    main()
