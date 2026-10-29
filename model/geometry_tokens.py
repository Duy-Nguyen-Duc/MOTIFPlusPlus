import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.embodiment_tags import EMBODIMENT_TAG_MAPPING

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def quat_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1).view(*q.shape[:-1], 3, 3)


class GeometryTokens(nn.Module):
    def __init__(self, mode: str = "plane", calib: str = "assets/wrist_camera/wrist_cameras.json",
                 robots: Optional[Dict[str, str]] = None, hidden_size: int = 512, num_freq: int = 8,
                 coord_range: float = 0.5, patch: int = 14, image_size: int = 224, source_size: int = 256,
                 crop_scale: float = 0.95, wrist_view: int = 1, num_views: int = 2, plane_z: float = 0.0,
                 max_range: float = 1.0, depth_model: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                 depth_input: int = 224):
        super().__init__()
        if mode not in ("plane", "depth"):
            raise ValueError(f"geometry mode {mode}")
        self.mode, self.wrist_view, self.num_views = mode, wrist_view, num_views
        self.coord_range, self.plane_z, self.max_range = coord_range, plane_z, max_range
        self.depth_model, self.depth_input = depth_model, depth_input
        robots = dict(robots or {"panda": "panda_rel", "xarm6": "xarm6_rel", "widowxai": "widowxai_rel"})
        calib = json.load(open(Path(calib)))
        names = sorted(robots)
        ids = [EMBODIMENT_TAG_MAPPING[robots[n]] for n in names]
        emb2robot = torch.full((max(EMBODIMENT_TAG_MAPPING.values()) + 1,), -1, dtype=torch.long)
        for i, e in enumerate(ids):
            emb2robot[e] = i
        self.robot_names = names
        # Effective intrinsics after the data pipeline's crop 
        # (train: random crop of crop_scale, eval: centre crop) and resize to image_size. 
        crop = int(source_size * crop_scale)
        s = image_size / crop
        off = (source_size - crop) / 2.0
        K, T = [], []
        for n in names:
            c = calib[n]
            k = torch.tensor(c["intrinsic_cv"], dtype=torch.float32)
            k = torch.tensor([[k[0, 0] * s, 0.0, (k[0, 2] - off) * s],
                              [0.0, k[1, 1] * s, (k[1, 2] - off) * s],
                              [0.0, 0.0, 1.0]])
            K.append(k)
            T.append(torch.tensor(c["T_tcp_cam_cv"], dtype=torch.float32))
        n_side = image_size // patch
        self.n_side, self.num_patches = n_side, n_side * n_side
        centres = torch.arange(n_side, dtype=torch.float32) * patch + patch / 2.0
        v, u = torch.meshgrid(centres, centres, indexing="ij")                    # row-major like the ViT
        pix = torch.stack([u.flatten(), v.flatten(), torch.ones(n_side * n_side)], dim=-1)   # [N, 3]
        rays = torch.stack([pix @ torch.linalg.inv(k).T for k in K])              # [R, N, 3], z = 1
        self.register_buffer("emb2robot", emb2robot)
        self.register_buffer("T_tcp_cam", torch.stack(T))                         # [R, 4, 4]
        self.register_buffer("rays_cam", rays)
        freqs = (2.0 ** torch.arange(num_freq, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs)
        feat_dim = 3 + 3 * 2 * num_freq + 1                                       # xyz, fourier(xyz), range flag
        self.proj = nn.Sequential(nn.Linear(feat_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        object.__setattr__(self, "_depth_net", None)
        self.last_points = None  

    # -- geometry ---------------------------------------------------------------------------
    def _depth(self, wrist: torch.Tensor) -> torch.Tensor:
        """Frozen monocular depth of the wrist image [B, 3, H, W] in [0, 1] -> per-patch z-depth [B, N]
        (relative scale; the caller anchors it)."""
        if self._depth_net is None:
            from transformers import AutoModelForDepthEstimation
            net = AutoModelForDepthEstimation.from_pretrained(self.depth_model).eval()
            net.requires_grad_(False)
            object.__setattr__(self, "_depth_net", net)      # plain attribute: never a submodule / in the checkpoint
        if next(self._depth_net.parameters()).device != wrist.device:
            self._depth_net.to(wrist.device)
        mean = wrist.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = wrist.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        x = F.interpolate(wrist.float(), size=(self.depth_input, self.depth_input), mode="bilinear", align_corners=False)
        with torch.no_grad(), torch.autocast(device_type=wrist.device.type, enabled=False):
            d = self._depth_net(pixel_values=(x - mean) / std).predicted_depth          # [B, h, w]
        d = F.adaptive_avg_pool2d(d[:, None].float(), self.n_side)[:, 0]                 # [B, n, n]
        return d.flatten(1).clamp_min(1e-4)

    def points(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """tcp-relative world-axis position of every wrist patch [B, N, 3] and a validity flag [B, N]."""
        tcp = batch["tcp_pose"]
        if tcp.dim() == 3:
            tcp = tcp[:, 0]
        tcp = tcp.float()
        ridx = self.emb2robot[batch["embodiment_id"].long().view(-1)]
        if (ridx < 0).any():
            raise ValueError(f"geometry tokens: unknown embodiment id {batch['embodiment_id'].tolist()}")
        R_wt = quat_wxyz_to_matrix(tcp[:, 3:7])                                   # [B, 3, 3]
        T_tc = self.T_tcp_cam[ridx]                                               # [B, 4, 4]
        R_wc = R_wt @ T_tc[:, :3, :3]
        origin = tcp[:, :3] + (R_wt @ T_tc[:, :3, 3:4])[..., 0]                   # camera centre, world
        rays_w = torch.einsum("bij,bnj->bni", R_wc, self.rays_cam[ridx])          # [B, N, 3], cam z = 1 scale
        # plane intersection: origin + lam * ray hits z = plane_z
        rz = rays_w[..., 2]
        lam = (self.plane_z - origin[:, 2:3]) / torch.where(rz.abs() < 1e-4, torch.full_like(rz, -1e-4), rz)
        valid = (rz < -1e-4) & (lam > 0) & (lam < self.max_range)
        lam = torch.where(valid, lam, torch.full_like(lam, self.max_range))
        if self.mode == "depth":
            wrist = batch["images"][:, self.wrist_view, 0]                        # [B, 3, H, W] in [0, 1]
            d = self._depth(wrist)                                                # [B, N] relative z-depth
            anchor = lam.median(dim=1, keepdim=True).values / d.median(dim=1, keepdim=True).values
            z = (d * anchor).clamp(1e-3, self.max_range)
            p_w = origin[:, None] + z[..., None] * rays_w
        else:
            p_w = origin[:, None] + lam[..., None] * rays_w
        rel = (p_w - tcp[:, None, :3]).clamp(-self.coord_range, self.coord_range)
        return rel, valid

    def encode(self, rel: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = rel / self.coord_range                                                # [-1, 1]
        ang = x[..., None] * self.freqs                                           # [B, N, 3, F]
        feats = torch.cat([x, torch.sin(ang).flatten(2), torch.cos(ang).flatten(2), valid[..., None].float()], dim=-1)
        return self.proj(feats)

    def forward(self, img_tokens: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Add the geometry encoding to the wrist view's tokens of img_tokens [B, V*N, D] (t = 1)."""
        N = self.num_patches
        if img_tokens.shape[1] != self.num_views * N:
            raise ValueError(f"geometry tokens expect {self.num_views}x{N} image tokens, got {img_tokens.shape[1]}")
        rel, valid = self.points(batch)
        self.last_points = rel.detach()
        enc = self.encode(rel, valid).to(img_tokens.dtype)
        lo, hi = self.wrist_view * N, (self.wrist_view + 1) * N
        return torch.cat([img_tokens[:, :lo], img_tokens[:, lo:hi] + enc, img_tokens[:, hi:]], dim=1)
