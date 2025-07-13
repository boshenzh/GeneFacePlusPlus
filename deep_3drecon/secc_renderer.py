import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from deep_3drecon.util.mesh_renderer import MeshRenderer
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel


class SECC_Renderer(nn.Module):
    def __init__(self, rasterize_size=None, device="cuda"):
        super().__init__()
        self.face_model = ParametricFaceModel('deep_3drecon/BFM')
        self.fov = 2 * np.arctan(self.face_model.center / self.face_model.focal) * 180 / np.pi
        self.znear = 5.
        self.zfar = 15.
        if rasterize_size is None:
            rasterize_size = 2*self.face_model.center 
        self.face_renderer = MeshRenderer(rasterize_fov=self.fov, znear=self.znear, zfar=self.zfar, rasterize_size=rasterize_size, use_opengl=False).cuda()
        face_feat = np.load("deep_3drecon/ncc_code.npy", allow_pickle=True)
        self.face_feat = torch.tensor(face_feat.T).unsqueeze(0).to(device=device)

        del_index_re = np.load('deep_3drecon/bfm_right_eye_faces.npy')
        del_index_re = del_index_re - 1
        del_index_le = np.load('deep_3drecon/bfm_left_eye_faces.npy')
        del_index_le = del_index_le - 1
        face_buf_list = []
        for i in range(self.face_model.face_buf.shape[0]):
            if i not in del_index_re and i not in del_index_le:
                face_buf_list.append(self.face_model.face_buf[i])
        face_buf_arr = np.array(face_buf_list)
        self.face_buf = torch.tensor(face_buf_arr).to(device=device)
    
    def forward(self, id, exp, euler, trans):
        """
        id, exp, euler, euler: [B, C] or [B, T, C]
        return:
            MASK: [B, 1, 512, 512], value[0. or 1.0], 1.0 denotes is face
            SECC MAP: [B, 3, 512, 512], value[0~1]
            if input is BTC format, return [B, C, T, H, W]
        """
        bs = id.shape[0]
        is_btc_flag = id.ndim == 3
        if is_btc_flag:
            t = id.shape[1]
            bs = bs * t
            id, exp, euler, trans = id.reshape([bs,-1]), exp.reshape([bs,-1]), euler.reshape([bs,-1]), trans.reshape([bs,-1])

        face_vertex = self.face_model.compute_face_vertex(id, exp, euler, trans)
        face_mask, _, secc_face = self.face_renderer(
                face_vertex, self.face_buf.unsqueeze(0).repeat([bs, 1, 1]), feat=self.face_feat.repeat([bs,1,1]))
        secc_face = (secc_face - 0.5) / 0.5 # scale to -1~1 

        if is_btc_flag:
            bs = bs // t
            face_mask = rearrange(face_mask, "(n t) c h w -> n c t h w", n=bs, t=t)
            secc_face = rearrange(secc_face, "(n t) c h w -> n c t h w", n=bs, t=t)
        return face_mask, secc_face


# if __name__ == '__main__':
#     import imageio

#     renderer = SECC_Renderer(rasterize_size=512)
#     ret = np.load("data/processed/videos/withheadmotion_clipped/coeff_fit_mp.npy", allow_pickle=True).tolist()
    
#     idx = len(ret['id']) - 1
#     assert idx < len(ret['id']), f"idx={idx} is out of range for loaded id array of shape {ret['id'].shape}"
#     id = torch.tensor(ret['id']).cuda()[idx:idx+1]
#     exp = torch.tensor(ret['exp']).cuda()[idx:idx+1]
#     angle = torch.tensor(ret['euler']).cuda()[idx:idx+1]
#     trans = torch.tensor(ret['trans']).cuda()[idx:idx+1]
#     mask, secc = renderer(id, exp, angle*0, trans*0) # [1, 1, 512, 512], [1, 3, 512, 512]

#     # out_mask = mask[0].permute(1,2,0)
#     # out_mask = (out_mask * 127.5 + 127.5).int().cpu().numpy()
#     # out_mask_rgb = np.repeat(out_mask, 3, axis=2).astype(np.uint8)  # [512, 512, 3]
#     # imageio.imwrite("out_mask.png", out_mask_rgb)
#     # out_img = secc[0].permute(1,2,0)
#     # out_img = (out_img * 127.5 + 127.5).int().cpu().numpy()
#     # imageio.imwrite("out_secc.png", out_img)
    
#     out_mask = mask[0].permute(1,2,0)  # [512, 512, 1]
#     out_mask = (out_mask * 127.5 + 127.5).clamp(0, 255).int().cpu().numpy()
#     out_mask_rgb = np.repeat(out_mask, 3, axis=2).astype(np.uint8)  # [512, 512, 3]
#     imageio.imwrite("out_mask.png", out_mask_rgb)

#     out_img = secc[0].permute(1,2,0)
#     out_img = (out_img * 127.5 + 127.5).clamp(0, 255).int().cpu().numpy().astype(np.uint8)
#     imageio.imwrite("out_secc.png", out_img)


if __name__ == '__main__':
    import imageio
    import os

    renderer = SECC_Renderer(rasterize_size=512)
    ret = np.load("data/processed/videos/withheadmotion_clipped/coeff_fit_mp.npy", allow_pickle=True).tolist()

    # === 设置输出格式 ===
    render_all = True  # 若只想测试一帧，设为 False
    save_gif = True    # True: 保存 gif，False: 保存 mp4
    save_path = "output"  # 输出文件夹
    os.makedirs(save_path, exist_ok=True)

    frames = []
    N = len(ret['id'])

    frame_range = range(N) if render_all else [0]
    for idx in frame_range:
        print(f"Rendering frame {idx}/{N}")

        id = torch.tensor(ret['id']).cuda()[idx:idx+1]
        exp = torch.tensor(ret['exp']).cuda()[idx:idx+1]
        angle = torch.tensor(ret['euler']).cuda()[idx:idx+1]
        trans = torch.tensor(ret['trans']).cuda()[idx:idx+1]

        mask, secc = renderer(id, exp, angle*0, trans*0)  # 去除头动，渲染静态姿态

        out_img = secc[0].permute(1,2,0)  # [512, 512, 3]
        out_img = (out_img * 127.5 + 127.5).clamp(0, 255).int().cpu().numpy().astype(np.uint8)

        frames.append(out_img)

        if not render_all:
            imageio.imwrite(os.path.join(save_path, "secc_single.png"), out_img)

    # === 导出为 gif 或 mp4 ===
    if render_all:
        if save_gif:
            gif_path = os.path.join(save_path, "secc_render.gif")
            imageio.mimsave(gif_path, frames, fps=10)
            print(f"Saved gif to: {gif_path}")
        else:
            mp4_path = os.path.join(save_path, "secc_render.mp4")
            writer = imageio.get_writer(mp4_path, fps=25, codec='libx264')
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"Saved mp4 to: {mp4_path}")
