
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
import tqdm

#extract_hubert
from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import numpy as np
import torch
import os
from utils.commons.hparams import set_hparams, hparams

# GeneFace++ imports (repo must be on PYTHONPATH)
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.pitch_utils import f0_to_coarse
from data_gen.utils.process_audio.extract_mel_f0 import (
    extract_mel_from_fname,
    extract_f0_from_wav_and_mel,
)
from data_util.face3d_helper import Face3DHelper
from modules.audio2motion.vae import VAEModel, PitchContourVAEModel
from deep_3drecon.secc_renderer import SECC_Renderer
from tasks.radnerfs.dataset_utils import RADNeRFDataset, get_boundary_mask, dilate_boundary_mask, get_lf_boundary_mask
from data_gen.utils.mp_feature_extractors.face_landmarker import index_lm68_from_lm478, index_lm131_from_lm478
from modules.postnet.lle import compute_LLE_projection, find_k_nearest_neighbors
from modules.radnerfs.utils import get_audio_features, get_rays, get_bg_coords, convert_poses, nerf_matrix_to_ngp
from inference.infer_utils import smooth_camera_sequence, smooth_features_xd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # registers 3D projection
# # -----------------------------------------------------------------------------
# #  MediaPipe index groups (for colour‑coding in the demo video)
# # -----------------------------------------------------------------------------
# FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
#              379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
#              234, 127, 162, 21, 54, 103, 67, 109]
# LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
#         317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
#         415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386,
#              385, 384, 398]
# RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
#              160, 161, 246]
# LEFT_BROW  = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
# RIGHT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
# IRIS_R = list(range(468, 473))
# IRIS_L = list(range(473, 478))

COL = {
    "oval":  (0, 0, 0),        # black
    "lips":  (0, 0, 255),      # red
    "eyes":  (0, 255, 0),      # green
    "brow":  (0, 165, 255),    # orange
    "iris":  (255, 0, 0),      # blue
    "other": (120, 120, 120),
}

COL = {
    "oval":  (0, 0, 0),        # black
    "lips":  (0, 0, 255),      # red
    "eyes":  (0, 255, 0),      # green
    "brow":  (0, 165, 255),    # orange
    "iris":  (255, 0, 0),      # blue
    "other": (120, 120, 120),
}
LIPS = list(range(48, 68))
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
LEFT_BROW = list(range(17, 22))
RIGHT_BROW = list(range(22, 27))
FACE_OVAL = list(range(0, 17))

debug = True
wav2vec2_processor = None
hubert_model = None #avoid loading model multiple times
secc_renderer = SECC_Renderer(512) 
#parameters
keypoint_mode = 'lm68'
LLE_percent=0.2
blink_mode = "none"
drv_pose='static'
mouth_amp = 0.4
from utils.commons.hparams import set_hparams, hparams
config_path = os.path.join("checkpoints", "motion2video_nerf", "may_head", "config.yaml")
hparams['cond_type'] = 'idexp_lm3d_normalized'
set_hparams(config_path)

dataset_cls = RADNeRFDataset
dataset = dataset_cls('trainval', data_dir='data/binary/videos/May', training=False)


# hparams['binary_data_dir'] = "data/binary/videos"
# hparams['video_id'] = "May"
print("Loading hparams from %s", hparams)
eye_area_percents = torch.tensor(dataset.eye_area_percents)
closed_eye_area_percent = torch.quantile(eye_area_percents, q=0.03).item()
opened_eye_area_percent = torch.quantile(eye_area_percents, q=0.97).item()


def get_hubert_from_16k_wav(wav_16k_name,device="cuda:0"):
    speech_16k, _ = sf.read(wav_16k_name)
    hubert = get_hubert_from_16k_speech(speech_16k,device=device)
    return hubert

@torch.no_grad()
def get_hubert_from_16k_speech(speech, device="cuda:0"):
    global hubert_model, wav2vec2_processor 
    local_path = '/home/boshenzh/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/snapshots/ece5fabbf034c1073acae96d5401b25be96709d8'
    if hubert_model is None:
        if os.path.exists(local_path):
            print("Loading HuBERT from local path...")
            hubert_model = HubertModel.from_pretrained(local_path)
        else:
            print("Loading HuBERT from Hugging Face...")
            hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = hubert_model.to(device)
    if wav2vec2_processor is None:
        if os.path.exists(local_path):
            print("Loading Wav2Vec2 Processor from local path...")
            wav2vec2_processor = Wav2Vec2Processor.from_pretrained(local_path)
        else:
            print("Loading Wav2Vec2 Processor from Hugging Face...")
            wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

    if speech.ndim ==2:
        speech = speech[:, 0] # [T, 2] ==> [T,]
    
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all

    if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it            
        hidden_states = hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]

    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T: # if skipping the last short 
        ret = torch.cat([ret, ret[:, -1:, :].repeat([1,expected_T-ret.shape[0],1])], dim=1)
    else:
        ret = ret[:expected_T]

    return ret


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


#TODO: render the secc just like genefaceapp_infer.py
@torch.no_grad()
def run_model(model, hubert: np.ndarray, f0: np.ndarray, device: str):
    """Forward pass through audio2secc and return 3‑D MediaPipe landmarks (T,478,3).

    The audio2secc checkpoint you are using was trained with **motion_type = "exp"**
    so it predicts only 64 expression coefficients.  The 3D‑MM layer still
    expects an 80‑D identity vector; we simply feed a zero vector so the shapes
    match (this produces a neutral identity).
    """
    helper = Face3DHelper(keypoint_mode="mediapipe", use_gpu=device.startswith("cuda"))
    t_x = hubert.shape[0]
    x_mask = torch.ones([1, t_x]).float() # mask for audio frames
    y_mask = torch.ones([1, t_x//2]).float() # mask for motion/image frames

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


    batch.update({
        'hubert': torch.from_numpy(hubert).float().unsqueeze(0).cuda(),
        'f0': torch.from_numpy(f0).float().reshape([1,-1]).cuda(),
        'x_mask': x_mask.cuda(),
        'y_mask': y_mask.cuda(),
        })
    batch['audio'] = batch['hubert']
    batch['blink'] = torch.zeros([1, t_x, 1]).long().cuda()

    batch['eye_amp'] = torch.ones([1, 1]).cuda() * 1.0
    batch['mouth_amp'] = torch.ones([1, 1]).cuda() * float(mouth_amp)
    # sample['id'] = torch.tensor(self.dataset.ds_dict['id'][0:t_x//2]).cuda()
    batch['id'] = torch.tensor( dataset.ds_dict['id'][0:1]).cuda().repeat([t_x//2, 1])
    pose_lst = []
    euler_lst = []
    trans_lst = []
    rays_o_lst = []
    rays_d_lst = []

    if '-' in drv_pose:
        start_idx, end_idx = inp['drv_pose'].split("-")
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        ds_ngp_pose_lst = [self.dataset.poses[i].unsqueeze(0) for i in range(start_idx, end_idx)]
        ds_euler_lst = [torch.tensor(self.dataset.ds_dict['euler'][i]) for i in range(start_idx, end_idx)]
        ds_trans_lst = [torch.tensor(self.dataset.ds_dict['trans'][i]) for i in range(start_idx, end_idx)]

    for i in range(t_x//2):
        if drv_pose == 'static':
            ngp_pose =  dataset.poses[0].unsqueeze(0)
            euler = torch.tensor( dataset.ds_dict['euler'][0])
            trans = torch.tensor( dataset.ds_dict['trans'][0])
        elif '-' in drv_pose:
            mirror_idx = mirror_index(i, len(ds_ngp_pose_lst))
            ngp_pose = ds_ngp_pose_lst[mirror_idx]
            euler = ds_euler_lst[mirror_idx]
            trans = ds_trans_lst[mirror_idx]
        else:
            ngp_pose =  dataset.poses[i].unsqueeze(0)
            euler = torch.tensor( dataset.ds_dict['euler'][i])
            trans = torch.tensor(dataset.ds_dict['trans'][i])
        rays = get_rays(ngp_pose.cuda(),  dataset.intrinsics,  dataset.H,  dataset.W, N=-1)
        rays_o_lst.append(rays['rays_o'].cuda())
        rays_d_lst.append(rays['rays_d'].cuda())
        pose = convert_poses(ngp_pose).cuda()
        pose_lst.append(pose)
        euler_lst.append(euler)
        trans_lst.append(trans)

    batch['rays_o'] = rays_o_lst
    batch['rays_d'] = rays_d_lst
    batch['poses'] = pose_lst
    batch['euler'] = torch.stack(euler_lst).cuda()
    batch['trans'] = torch.stack(trans_lst).cuda()
    batch['bg_img'] =  dataset.bg_img.reshape([1,-1,3]).cuda()
    batch['bg_coords'] =  dataset.bg_coords.cuda()

    ret = {}
    _ = model.forward(batch, ret=ret, train=False, temperature=0.2) #TODO: set temp from args 
    
    # exp_coef = ret["pred"][0]              # (T,64)
    # T = exp_coef.shape[0]
    # id_coef = torch.zeros((T, 80), device=exp_coef.device, dtype=exp_coef.dtype)
    pred = ret["pred"][0]
    if pred.shape[-1] == 144: #NOTE: the ckpt from github motion_type is exp, so id is not used, set as neutral face
        id_coef = pred[:, :80]
        exp_coef = pred[:, 80:]
    else:
        exp_coef = pred
        T = exp_coef.shape[0]
        id_coef = torch.zeros((T, 80), device=exp_coef.device, dtype=exp_coef.dtype)
    # OPtional: add a SECC render. 
    print("Hubert shape:", hubert.shape)     # Expect approximately (150, 1024)
    print("Expression coefficients shape:", exp_coef.shape)

    # get idexp_lm3d
    id_ds = torch.from_numpy( dataset.ds_dict['id']).float().cuda()
    exp_ds = torch.from_numpy( dataset.ds_dict['exp']).float().cuda()
    idexp_lm3d_ds =  helper.reconstruct_idexp_lm3d(id_ds, exp_ds)
    idexp_lm3d_mean = idexp_lm3d_ds.mean(dim=0, keepdim=True)
    idexp_lm3d_std = idexp_lm3d_ds.std(dim=0, keepdim=True)
    # if hparams.get("normalize_cond", True):
    # 
    idexp_lm3d_ds_normalized = (idexp_lm3d_ds - idexp_lm3d_mean) / idexp_lm3d_std # normalize_cond
    # else:
    #     idexp_lm3d_ds_normalized = idexp_lm3d_ds
    lower = torch.quantile(idexp_lm3d_ds_normalized, q=0.03, dim=0)
    upper = torch.quantile(idexp_lm3d_ds_normalized, q=0.97, dim=0)
    idexp_lm3d = helper.reconstruct_idexp_lm3d(id_coef, exp_coef)  # (T,478,3)
    if keypoint_mode == 'lm68':
        idexp_lm3d = idexp_lm3d[:, index_lm68_from_lm478]
        idexp_lm3d_mean = idexp_lm3d_mean[:, index_lm68_from_lm478]
        idexp_lm3d_std = idexp_lm3d_std[:, index_lm68_from_lm478]
        lower = lower[index_lm68_from_lm478]
        upper = upper[index_lm68_from_lm478]
    elif keypoint_mode == 'lm131':
        idexp_lm3d = idexp_lm3d[:, index_lm131_from_lm478]
        idexp_lm3d_mean = idexp_lm3d_mean[:, index_lm131_from_lm478]
        idexp_lm3d_std = idexp_lm3d_std[:, index_lm131_from_lm478]
        lower = lower[index_lm131_from_lm478]
        upper = upper[index_lm131_from_lm478]
    elif keypoint_mode == 'lm468':
        idexp_lm3d = idexp_lm3d
    else:
        raise NotImplementedError()    
    idexp_lm3d = idexp_lm3d.reshape([-1, 68*3])
    idexp_lm3d_ds_lle = idexp_lm3d_ds[:, index_lm68_from_lm478].reshape([-1, 68*3])
    feat_fuse, _, _ = compute_LLE_projection(feats=idexp_lm3d[:, :68*3], feat_database=idexp_lm3d_ds_lle[:, :68*3], K=10)
     
    idexp_lm3d[:, :68*3] = LLE_percent * feat_fuse + (1-LLE_percent) * idexp_lm3d[:,:68*3]
    idexp_lm3d = idexp_lm3d.reshape([-1, 68, 3])
    idexp_lm3d_normalized = (idexp_lm3d - idexp_lm3d_mean) / idexp_lm3d_std
 
    cano_lm3d = (idexp_lm3d_mean + idexp_lm3d_std * idexp_lm3d_normalized) / 10 +  helper.key_mean_shape[index_lm68_from_lm478].unsqueeze(0)
    eye_area_percent = opened_eye_area_percent * torch.ones([len(cano_lm3d), 1], dtype=cano_lm3d.dtype, device=cano_lm3d.device)
    if blink_mode == 'period':
        cano_lm3d, eye_area_percent = inject_blink_to_lm68(cano_lm3d, self.opened_eye_area_percent, self.closed_eye_area_percent)
        print("Injected blink to idexp_lm3d by directly editting.")
    batch['eye_area_percent'] = eye_area_percent
    idexp_lm3d_normalized = ((cano_lm3d - helper.key_mean_shape[index_lm68_from_lm478].unsqueeze(0)) * 10 - idexp_lm3d_mean) / idexp_lm3d_std
    idexp_lm3d_normalized = torch.clamp(idexp_lm3d_normalized, min=lower, max=upper)
    batch['cano_lm3d'] = cano_lm3d
    idexp_lm3d_normalized_ = idexp_lm3d_normalized    
    # idexp_lm3d_normalized_ = idexp_lm3d_normalized[0:1, :].repeat([len(exp_coef),1,1]).clone()
    idexp_lm3d_normalized_[:, 17:27] = idexp_lm3d_normalized[:, 17:27] # brow
    idexp_lm3d_normalized_[:, 36:48] = idexp_lm3d_normalized[:, 36:48] # eye
    idexp_lm3d_normalized_[:, 27:36] = idexp_lm3d_normalized[:, 27:36] # nose
    idexp_lm3d_normalized_[:, 48:68] = idexp_lm3d_normalized[:, 48:68] # mouth
    idexp_lm3d_normalized_[:, 0:17] = idexp_lm3d_normalized[:, :17] # yaw

    idexp_lm3d_normalized = idexp_lm3d_normalized_
    
    cond_win = idexp_lm3d_normalized.reshape([len(exp_coef), 1, -1])
    cond_wins = [get_audio_features(cond_win, att_mode=2, index=idx) for idx in range(len(cond_win))]
    batch['cond_wins'] = cond_wins

    # face boundark mask, for cond mask
    smo_euler = smooth_features_xd(batch['euler'])
    smo_trans = smooth_features_xd(batch['trans'])
    lm2d = helper.reconstruct_lm2d_nerf(id_coef, exp_coef, smo_euler, smo_trans)
    lm68 = lm2d[:, index_lm68_from_lm478, :]
    batch['lm68'] = lm68.reshape([lm68.shape[0], 68*2])

    return batch

    # return idexp_lm3d.cpu().numpy()


def draw_mp_landmark(frame, pts, canvas_height=720):
    for idx, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(canvas_height - y))
        if idx in FACE_OVAL:       col = COL["oval"]
        elif idx in LIPS:          col = COL["lips"]
        elif idx in LEFT_EYE or idx in RIGHT_EYE: col = COL["eyes"]
        elif idx in LEFT_BROW or idx in RIGHT_BROW: col = COL["brow"]
        else:                      col = COL["other"]
        cv2.circle(frame, (xi, yi), 2, col, -1)
def save_landmark_video(lm68, out_path="mp_landmarks.mp4", size=720, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (size, size))

    for frame_lm in lm68:
        canvas = np.full((size, size, 3), 255, dtype=np.uint8)
        draw_mp_landmark(canvas, frame_lm,size)
        vw.write(canvas)

    vw.release()
    print(f"✅ Saved landmark video to {out_path}")

def ortho_project(lm, size=720):
    # lm is assumed to be a NumPy array with shape (T, 68, 2)
    projected = lm.copy()
    # Flip Y: since OpenCV's coordinate system has y downwards
    projected[..., 1] = size - projected[..., 1]
    
    # Optionally, apply scaling and translation to fit within the image.
    min_xy = projected.min(axis=(0, 1))
    max_xy = projected.max(axis=(0, 1))
    scale = 0.9 * size / (max_xy - min_xy).max()
    projected = (projected - min_xy) * scale + 0.05 * size
    return projected

# user enter file name in data/benchmark_wavs, then model generate landmark.npy and landmark video with audio in data/benchmark__out
def vis_3d(lmk3d):
        # T = number of frames
    T = lmk3d.shape[0]
    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter([], [], [], s=10, c='blue')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("3D MediaPipe Landmarks")

    def init():
        sc._offsets3d = ([], [], [])
        return sc,

    def animate(i):
        x, y, z = lmk3d[i,:,0], lmk3d[i,:,1], lmk3d[i,:,2]
        sc._offsets3d = (x, y, z)
        return sc,

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=T,
        init_func=init,
        interval=100,   # ms between frames → ~10 fps
        blit=False
    )
    ani.save('landmarks.gif', writer='pillow', fps=10)
    print("Save 3D ldandmarks GIF ➜ landmarks.gif")
    plt.show()

def main():
    parser = argparse.ArgumentParser("GeneFace++ audio→landmark demo")
    # parser.add_argument("--audio", required=True, help="Input WAV (any sr)")
    parser.add_argument("--ckpt_dir", required=True, help="audio2motion_vae checkpoint dir")
    # parser.add_argument("--out", default="demo", help="Output prefix (.npy/.mp4)")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Force device")
    parser.add_argument("--temperature", default=0.2) # nearest | random
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    # Ensure 16 kHz mono WAV for feature extractors
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.info("Running **CPU‑only** (CUDA disabled)")
    while True:
        wav_file = input("\n▶ Enter wav file path in data/benchmark_wavs (or 'q' to quit): ").strip()
        wav_file_name = os.path.basename(wav_file).replace(".wav", "_16k.wav")
        if wav_file == "":
            wav_file = "data/benchmark_wavs/coffee_fixed1_16k.wav"
            wav_file_name = "coffee_fixed1_16k.wav"
        elif wav_file.lower() == 'q':
                break
                
        if not os.path.isfile(wav_file):
            print(f"❌ File not found: {wav_file}")
            continue
            
        # try:
        logging.info("Input audio: %s", wav_file)

        # temp_file = wav_file.replace(".wav", "_16k.wav")
        # os.system(f"ffmpeg -loglevel quiet -y -i {wav_file} -ar 16000 -ac 1 {temp_file}")
        t0 = time.time()

        hubert, f0 = extract_features(wav_file,device=device)
        t_feature = time.time()
        model = load_audio2secc(Path(args.ckpt_dir), device) #load audio2secc model
        t_audio2secc = time.time()
        
        batch = run_model(model, hubert, f0, device)
        # 3d visualization for landmarks             
        # vis_3d(batch["cano_lm3d"].cpu().numpy()  )#visualize the 3D landmarks
        
        npy_path = os.path.join("data","out",f"{wav_file_name}_out.npy")
        lmk_mp4_path =os.path.join("data","out",f"{wav_file_name}_out.mp4") 
        projected_landmarks = ortho_project(batch["lm68"].cpu().numpy().reshape(-1, 68, 2), size=720)
        save_landmark_video(projected_landmarks, lmk_mp4_path, size=720, fps=25)
        logging.info("Saved landmark video ➜ %s", lmk_mp4_path)

        # save 3D landmarks video
        # save_landmark_video(batch["lm68"].cpu().numpy().reshape(-1, 68, 2), mp4_path, size=720, fps=25)
        # save 3d landmarks npy 
        # np.save(npy_path, lm2d.astype(np.float32))
        
        
        # logging.info("Saved trajectory ➜ %s", npy_path)
        t_inf = time.time()
        logging.info("Extracted features in %.2f seconds", t_feature - t0)
        logging.info("audio2secc loading time  ➜ %.2fs", t_audio2secc - t_feature)
        logging.info("Model inference time ➜ %.2fs", t_inf - t_audio2secc)

        # except Exception as e:
        #     print(f"❌ Processing failed: {str(e)}")
        #     # Cleanup any partial files
        #     break
        

    



if __name__ == "__main__":
    main()
