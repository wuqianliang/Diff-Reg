import time as time_lib

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.perspective_n_points import efficient_pnp
import numpy as np
from vision3d.ops import apply_transform, pairwise_distance
from vision3d.utils.opencv import registration_with_pnp_ransac
from vision3d.models.geotransformer import SuperPointMatchingMutualTopk, SuperPointProposalGenerator, SuperPointMatchingDenoising
from vision3d.ops import (
    back_project,
    batch_mutual_topk_select,
    create_meshgrid,
    index_select,
    pairwise_cosine_similarity,
    point_to_node_partition,
    render,
)

# isort: split
from fusion_module import CrossModalFusionModule
from image_backbone import FeaturePyramid, ImageBackbone
from point_backbone import PointBackbone
from utils import get_2d3d_node_correspondences, patchify


from encoders import *

from matching import Matching, log_optimal_transport
from procrustes import SoftProcrustesLayer

from ops.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
from vision3d.ops import mutual_topk_select

import cv2
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import tempfile

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from utils import  multual_nn_correspondence, to_o3d_pcd, to_tsfm, get_correspondences
from vision3d.ops.metrics import compute_isotropic_transform_error
from vision3d.array_ops import (
    evaluate_correspondences,
    evaluate_sparse_correspondences,
    isotropic_registration_error,
    registration_rmse,
)
import open3d as o3d
from sklearn.neighbors import KDTree

import os
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def to_homogeneous(coordinates):
    """
    Convert regular coordinate to homogeneous

    :param coordinates: Regular coordinates [..., 2]
    :return: Homogeneous coordinates [..., 3]
    """
    ones_shape = list(coordinates.shape)
    ones_shape[-1] = 1
    ones = torch.ones(ones_shape).type(coordinates.type())

    return torch.cat([coordinates, ones], dim=-1)

def farthest_point_sampling(points, num_samples):
    farthest_pts = np.zeros((num_samples, points.shape[1]))
    farthest_pts[0] = points[np.random.choice(len(points))]
    tree = KDTree(points)

    for i in range(1, num_samples):
        distances, _ = tree.query(farthest_pts[:i], points.shape[0])
        farthest_pts[i] = points[np.argmax(np.min(distances, axis=0))]

    return farthest_pts



@torch.no_grad()
def predict_depth(model, image):
    return model(image)


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

import math
@torch.no_grad()
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

@torch.no_grad()
# forward diffusion
def q_sample(x_start, t, noise=None, timesteps=1000):
    if noise is None:
        noise = torch.randn_like(x_start)

    betas = cosine_beta_schedule(timesteps).to(x_start.device)
    alphas = 1. - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod =  torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    # print(t.item(), sqrt_alphas_cumprod_t.item(), sqrt_one_minus_alphas_cumprod_t.item())
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise   


import  math
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


resolutions = {"low":(448, 448), "medium":(14*8*5, 14*8*5), "high":(14*8*6, 14*8*6)}

class MATR2D3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.matching_radius_2d = cfg.model.ground_truth_matching_radius_2d
        self.matching_radius_3d = cfg.model.ground_truth_matching_radius_3d
        self.pcd_num_points_in_patch = cfg.model.pcd_num_points_in_patch

        # fixed for now
        # self.img_h_c = 24
        # self.img_w_c = 32

        self.img_h_c = 34
        self.img_w_c = 45
        print('h,w ', self.img_h_c, self.img_w_c)

        self.transform = Compose([
                Resize(
                    width=630,#self.img_w_c,
                    height=476,#self.img_h_c,
                    # width=self.img_w_c,
                    # height=self.img_h_c,                    
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
        ])

        self.img_num_levels_c = 1 
        self.overlap_threshold = 0.3
        self.pcd_min_node_size = 5

        self.img_backbone = ImageBackbone(
            cfg.model.image_backbone.input_dim,
            cfg.model.image_backbone.output_dim,
            cfg.model.image_backbone.init_dim,
            dilation=cfg.model.image_backbone.dilation,
        )

        self.pcd_backbone = PointBackbone(
            cfg.model.point_backbone.input_dim,
            cfg.model.point_backbone.output_dim,
            cfg.model.point_backbone.init_dim,
            cfg.model.point_backbone.kernel_size,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_sigma,
        )

        self.transformer = CrossModalFusionModule(
            cfg.model.transformer.img_input_dim,
            cfg.model.transformer.pcd_input_dim,
            cfg.model.transformer.output_dim,
            cfg.model.transformer.hidden_dim,
            cfg.model.transformer.num_heads,
            cfg.model.transformer.blocks,
            use_embedding=cfg.model.transformer.use_embedding,
        )

        self.denoising_transformer = CrossModalFusionModule(
            cfg.model.transformer.img_input_dim,
            cfg.model.transformer.pcd_input_dim,
            cfg.model.transformer.output_dim,
            cfg.model.transformer.hidden_dim,
            cfg.model.transformer.num_heads,
            cfg.model.transformer.blocks,
            use_embedding=cfg.model.transformer.use_embedding,
        )        


        self.coarse_target = SuperPointProposalGenerator(
            cfg.model.coarse_matching.num_targets,
            cfg.model.coarse_matching.overlap_threshold,
        )


        h,w = resolutions['medium']
        pretrained_backbone = False
        self.encoder = CNNandDinov2(
            cnn_kwargs = dict(
                pretrained=pretrained_backbone,
                amp = True),
            amp = True,
            use_vgg = True,
        )

        self.dino_2_u = nn.Linear(1024, 512)

        ######################################################################################
        # Diffusion
        timesteps = 1000
        self.num_timesteps = int(timesteps)
        sampling_timesteps = cfg.model.coarse_matching.SAMPLE_STEP
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        # self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1. 
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))    

        self.denoising_soft_procrustes = SoftProcrustesLayer(cfg.procrustes)
        self.denoising_coarse_matching = Matching(cfg.model.coarse_matching)
        self.coarse_matching = Matching(cfg.model.coarse_matching)


        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.depth_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14', cache_dir='depth_anything_vitl14').to(self.DEVICE).eval()
        self.depth_coffa = torch.tensor([1.0], requires_grad=True).to(self.DEVICE)
        self.depth_coffb = torch.tensor([0.0], requires_grad=True).to(self.DEVICE)



        ######################################################################################



    def forward(self, data_dict):
        assert data_dict["batch_size"] == 1, "Only batch size of 1 is supported."

        torch.cuda.synchronize()
        start_time = time_lib.time()

        output_dict = {}

        # 1. Unpack data

        # 1.1 Unpack 2D data
        image = data_dict["image"].unsqueeze(1).detach()  # (B, 1, H, W), gray scaling_factor
        image_gray = data_dict["image_gray"].unsqueeze(1).detach()  # (B, 1, H, W), gray scaling_factor
        ori_image = data_dict["ori_image"].unsqueeze(1).detach()
        depth = data_dict["depth"].detach()  # (B, H, W)
        intrinsics = data_dict["intrinsics"].detach()  # (B, 3, 3)
        transform = data_dict["transform"].detach()

        img_h = image.shape[2]
        img_w = image.shape[3]
        img_h_f = img_h
        img_w_f = img_w

        img_points, img_masks = back_project(depth, intrinsics, depth_limit=6.0, transposed=True, return_mask=True)

        img_points = img_points.squeeze(0)  # (B, H, W, 3) -> (H, W, 3)
        img_masks = img_masks.squeeze(0)  # (B, H, W) -> (H, W)
        img_pixels = create_meshgrid(img_h, img_w).float()  # (H, W, 2)

        img_points_f = img_points  # (H, H, 3)
        img_masks_f = img_masks  # (H, H)
        img_pixels_f = img_pixels  # (H, W, 2)

        img_points = img_points.view(-1, 3)  # (H, W, 3) -> (HxW, 3)
        img_pixels = img_pixels.view(-1, 2)  # (H, W, 2) -> (HxW, 2)
        img_masks = img_masks.view(-1)  # (H, W) -> (HxW)
        img_points_f = img_points_f.view(-1, 3)  # (H, W, 3) -> (HxW, 3)
        img_pixels_f = img_pixels_f.view(-1, 2)  # (H/2xW/2, 2)
        img_masks_f = img_masks_f.view(-1)  # (H, W) -> (HxW)

        output_dict["img_points"] = img_points
        output_dict["img_pixels"] = img_pixels
        output_dict["img_masks"] = img_masks
        output_dict["img_points_f"] = img_points_f
        output_dict["img_pixels_f"] = img_pixels_f
        output_dict["img_masks_f"] = img_masks_f

        # 1.2 Unpack 3D data
        pcd_feats = data_dict["feats"].detach()
        pcd_points = data_dict["points"][0].detach()
        pcd_points_f = data_dict["points"][0].detach()
        pcd_points_c = data_dict["points"][-2].detach()
        pcd_pixels_f = render(pcd_points_f, intrinsics, extrinsics=transform, rounding=False)

        output_dict["pcd_points"] = pcd_points
        output_dict["pcd_points_c"] = pcd_points_c
        output_dict["pcd_points_f"] = pcd_points_f
        output_dict["pcd_pixels_f"] = pcd_pixels_f

        # depth anything
        image_for_depth = image.squeeze(0).squeeze(0).cpu().numpy()#cv2.cvtColor(image.squeeze(0).squeeze(0).cpu().numpy(), cv2.COLOR_BGR2RGB) / 255.0
        image_for_depth = self.transform({'image': image_for_depth})['image']    
        image_for_depth = torch.from_numpy(image_for_depth).unsqueeze(0).cuda()
        image_depth_any = predict_depth(self.depth_model, image_for_depth)


        img_points_da, img_masks_da = self.back_project_depth(image_depth_any/100, intrinsics, depth_limit=6.0, scaling_factor_a=self.depth_coffa, scaling_factor_b=self.depth_coffb, transposed=True, return_mask=True)
        img_points_f_da = img_points_da.squeeze(0).view(-1, 3)  # (H, W, 3) -> (HxW, 3)
        img_masks_f_da = img_masks_da.squeeze(0).view(-1)
        # 2. Backbone

        # 2.1.1 DINO v2
        image_ext = image.squeeze(0).permute(0,3,1,2).contiguous()
        img_feats_dino_list =self.encoder(image_ext)
        img_feats_x_dino = img_feats_dino_list[16].float()  # (B, C8, H/14, W/14), aka, (1, 1024, 32, 44)
        img_feats_x_dino_ds = self.dino_2_u(img_feats_x_dino.permute(0,2,3,1).contiguous())

        # 2.1 Image backbone
        img_feats_list = self.img_backbone(image_gray, img_feats_x_dino_ds)
        img_feats_x = img_feats_list[-1]  # (B, C8, H/8, W/8), aka, (1, 512, 60, 80)
        img_feats_f = img_feats_list[0]  # (B, C2, H, W), aka, (1, 128, 480, 640)

        # 2.2 Point backbone
        pcd_feats_list = self.pcd_backbone(pcd_feats, data_dict)
        pcd_feats_c = pcd_feats_list[-1]  # (Nc, 512)
        pcd_feats_f = pcd_feats_list[0]  # (Nf, 128)

        # 3. Transformer

        # 3.1 Prepare image features
        img_shape_c = (self.img_h_c, self.img_w_c)
        img_feats_c = F.interpolate(img_feats_x, size=img_shape_c, mode="bilinear", align_corners=True)  # to (24, 32)
        img_feats_c = img_feats_c.squeeze(0).view(-1, self.img_h_c * self.img_w_c).transpose(0, 1)  # (768, 512)
        img_pixels_c = create_meshgrid(self.img_h_c, self.img_w_c, normalized=True, flatten=True)  # (768, 2)
        # use normalized pixel coordinates for transformer

        img_pixels_c_epnp = create_meshgrid(self.img_h_c, self.img_w_c, flatten=True).float()

        img_feats_x_dino = img_feats_x_dino.squeeze(0).view(-1, self.img_h_c * self.img_w_c).transpose(0, 1)  # (768, 512)


        img_feats_c_backbone, pcd_feats_c_backbone = img_feats_c.clone(), pcd_feats_c.clone()
        # 3.2 Cross-modal fusion transformer
        img_feats_c, pcd_feats_c = self.transformer(
            img_feats_c.unsqueeze(0),
            img_feats_x_dino.unsqueeze(0),
            img_pixels_c.unsqueeze(0),
            pcd_feats_c.unsqueeze(0),
            pcd_points_c.unsqueeze(0),
        )

        img_feats_c = img_feats_c.squeeze(0).view(img_feats_c.shape[1], -1).transpose(0, 1).contiguous()
        img_feats_c = torch.cat([img_feats_c], dim=0).permute(1,0).contiguous()

        # 3.4 Post-processing for point features
        pcd_feats_c = pcd_feats_c.squeeze(0)

        # 4. Coarse-level matching

        # 4.1 Generate 3d patches
        _, pcd_node_sizes, pcd_node_masks, pcd_node_knn_indices, pcd_node_knn_masks = point_to_node_partition(
            pcd_points_f,
            pcd_points_c,
            self.pcd_num_points_in_patch,
            gather_points=True,
            return_count=True,
        )


        pcd_node_masks = torch.logical_and(pcd_node_masks, torch.gt(pcd_node_sizes, self.pcd_min_node_size))
        pcd_padded_points_f = torch.cat([pcd_points_f, torch.ones_like(pcd_points_f[:1]) * 1e10], dim=0)
        pcd_node_knn_points = index_select(pcd_padded_points_f, pcd_node_knn_indices, dim=0)
        pcd_padded_pixels_f = torch.cat([pcd_pixels_f, torch.ones_like(pcd_pixels_f[:1]) * 1e10], dim=0)
        pcd_node_knn_pixels = index_select(pcd_padded_pixels_f, pcd_node_knn_indices, dim=0)

        # 4.2 Generate 2d patches
        all_img_node_knn_points = []
        all_img_node_knn_pixels = []
        all_img_node_knn_indices = []
        all_img_node_knn_masks = []
        all_img_node_masks = []
        all_img_node_levels = []
        all_img_num_nodes = []
        all_img_total_nodes = []
        total_img_num_nodes = 0

        all_gt_img_node_corr_levels = []
        all_gt_img_node_corr_indices = []
        all_gt_pcd_node_corr_indices = []
        all_gt_img_node_corr_overlaps = []
        all_gt_pcd_node_corr_overlaps = []

        img_h_c = self.img_h_c
        img_w_c = self.img_w_c
        for i in range(self.img_num_levels_c):
            (
                img_node_knn_points,  # (N, Ki, 3)
                img_node_knn_points_da,
                img_node_knn_pixels,  # (N, Ki, 2)
                img_node_knn_indices,  # (N, Ki)
                img_node_knn_masks,  # (N, Ki)
                img_node_knn_masks_da,  # (N, Ki)
                img_node_masks,  # (N)
                img_node_masks_da,  # (N)
            ) = patchify(
                img_points_f,
                img_points_f_da,                
                img_pixels_f,
                img_masks_f,
                img_masks_f_da,                
                img_h_f,
                img_w_f,
                img_h_c,
                img_w_c,
                stride=2,
            )

            img_num_nodes = img_h_c * img_w_c
            img_node_levels = torch.full(size=(img_num_nodes,), fill_value=i, dtype=torch.long).cuda()

            all_img_node_knn_points.append(img_node_knn_points)
            all_img_node_knn_pixels.append(img_node_knn_pixels)
            all_img_node_knn_indices.append(img_node_knn_indices)
            all_img_node_knn_masks.append(img_node_knn_masks)
            all_img_node_masks.append(img_node_masks)
            all_img_node_levels.append(img_node_levels)
            all_img_num_nodes.append(img_num_nodes)
            all_img_total_nodes.append(total_img_num_nodes)

            # 4.3 Generate coarse-level ground truth

            (
                gt_img_node_corr_indices,
                gt_pcd_node_corr_indices,
                gt_img_node_corr_overlaps,
                gt_pcd_node_corr_overlaps,
                pcd_centers_c, img_pcd_centers_c, img_pcd_centers_c_da, coarse_match_gt # new added
            ) = get_2d3d_node_correspondences(
                img_node_masks,
                img_node_masks_da,
                img_node_knn_points,
                img_node_knn_points_da,
                img_node_knn_pixels,
                img_node_knn_masks,
                img_node_knn_masks_da,
                pcd_node_masks,
                pcd_node_knn_points,
                pcd_node_knn_pixels,
                pcd_node_knn_masks,
                transform,
                self.matching_radius_2d,
                self.matching_radius_3d,
            )

            gt_img_node_corr_indices += total_img_num_nodes
            gt_img_node_corr_levels = torch.full_like(gt_img_node_corr_indices, fill_value=i)
            all_gt_img_node_corr_levels.append(gt_img_node_corr_levels)
            all_gt_img_node_corr_indices.append(gt_img_node_corr_indices)
            all_gt_pcd_node_corr_indices.append(gt_pcd_node_corr_indices)
            all_gt_img_node_corr_overlaps.append(gt_img_node_corr_overlaps)
            all_gt_pcd_node_corr_overlaps.append(gt_pcd_node_corr_overlaps)

            img_h_c //= 2
            img_w_c //= 2
            total_img_num_nodes += img_num_nodes

        img_node_masks = torch.cat(all_img_node_masks, dim=0)
        img_node_levels = torch.cat(all_img_node_levels, dim=0)

        output_dict["img_num_nodes"] = total_img_num_nodes
        output_dict["pcd_num_nodes"] = pcd_points_c.shape[0]

        gt_img_node_corr_levels = torch.cat(all_gt_img_node_corr_levels, dim=0)
        gt_img_node_corr_indices = torch.cat(all_gt_img_node_corr_indices, dim=0)
        gt_pcd_node_corr_indices = torch.cat(all_gt_pcd_node_corr_indices, dim=0)
        gt_img_node_corr_overlaps = torch.cat(all_gt_img_node_corr_overlaps, dim=0)
        gt_pcd_node_corr_overlaps = torch.cat(all_gt_pcd_node_corr_overlaps, dim=0)

        gt_node_corr_mean_overlaps = (gt_img_node_corr_overlaps+ gt_pcd_node_corr_overlaps)*0.5
        gt_node_corr_min_overlaps = torch.minimum(gt_img_node_corr_overlaps, gt_pcd_node_corr_overlaps)
        gt_node_corr_max_overlaps = torch.maximum(gt_img_node_corr_overlaps, gt_pcd_node_corr_overlaps)

        output_dict["gt_img_node_corr_indices"] = gt_img_node_corr_indices
        output_dict["gt_pcd_node_corr_indices"] = gt_pcd_node_corr_indices
        output_dict["gt_img_node_corr_overlaps"] = gt_img_node_corr_overlaps
        output_dict["gt_pcd_node_corr_overlaps"] = gt_pcd_node_corr_overlaps
        output_dict["gt_img_node_corr_levels"] = gt_img_node_corr_levels
        output_dict["gt_node_corr_min_overlaps"] = gt_node_corr_min_overlaps
        output_dict["gt_node_corr_max_overlaps"] = gt_node_corr_max_overlaps
        output_dict["gt_node_corr_mean_overlaps"] = gt_node_corr_mean_overlaps

        # 5. Fine-leval matching
        img_channels_f = img_feats_f.shape[1]
        img_feats_f = img_feats_f.squeeze(0).view(img_channels_f, -1).transpose(0, 1).contiguous()

        img_feats_f = F.normalize(img_feats_f, p=2, dim=1)
        pcd_feats_f = F.normalize(pcd_feats_f, p=2, dim=1)

        output_dict["img_feats_f"] = img_feats_f
        output_dict["pcd_feats_f"] = pcd_feats_f

        src_mask = pcd_node_masks.unsqueeze(0)
        tgt_mask = img_node_masks.unsqueeze(0)
        tgt_mask_da = img_node_masks_da.unsqueeze(0)

        conf_matrix_pred ,_,_,_ = self.coarse_matching(pcd_feats_c.unsqueeze(0), img_feats_c.unsqueeze(0), src_mask, tgt_mask, True)
        output_dict["conf_matrix_pred"] = conf_matrix_pred


        img_feats_c = F.normalize(img_feats_c, p=2, dim=1)
        pcd_feats_c = F.normalize(pcd_feats_c, p=2, dim=1)

        output_dict["img_feats_c"] = img_feats_c
        output_dict["pcd_feats_c"] = pcd_feats_c


        s_pcd_c = pcd_points_c 
        t_pcd_c = img_pcd_centers_c
        t_pcd_c_da = img_pcd_centers_c_da
        t_pcd_c_2d = img_pixels_c_epnp

        if self.training:

            #############################################################################################
            # generate ground truth correpondences  for coarse level points
            for gt_thr in [0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                coarse_match_gt2 = get_correspondences(to_o3d_pcd(s_pcd_c), to_o3d_pcd(t_pcd_c), transform.cpu().numpy(), gt_thr)
        
                if coarse_match_gt2.shape[0] > 5:
                    
                    coarse_match_gt2 = coarse_match_gt2.permute(1,0)
                    matrix_gt = torch.zeros([s_pcd_c.size(0), t_pcd_c.size(0)]).cuda()
                    matrix_gt[ coarse_match_gt2[0], coarse_match_gt2[1] ] = 1
                    matrix_gt = matrix_gt.unsqueeze(0)
                    R, t, R_forwd, t_forwd, condition, solution_mask = self.denoising_soft_procrustes(matrix_gt, s_pcd_c.unsqueeze(0), t_pcd_c.unsqueeze(0), src_mask, tgt_mask)

                    est_transform = torch.zeros_like(transform)

                    est_transform[:3,:3] = R_forwd.squeeze(0)
                    est_transform[:3,3] = t_forwd.squeeze(0).squeeze(-1)

                    rre, rte = compute_isotropic_transform_error(transform, est_transform) 
                    
                    if rre < 5 and rte < 1:
                        
                        break
                    else:
                        if gt_thr==0.9:
                            print("Very bad gt!! ", gt_thr )
                            output_dict['not_val'] = 1.0
                        else:
                            print("Bad gt!! ", gt_thr )
                
                else:
                    print("Bad gt!! ", gt_thr )
            ############################################################################################# 
            # diffusion    
                    
            pred_matrix_shape = [s_pcd_c.size(0),t_pcd_c.size(0)]
            matrix_gt = torch.zeros(pred_matrix_shape).to(s_pcd_c.device)
            matrix_gt[ coarse_match_gt2[0], coarse_match_gt2[1] ] = 1

            matrix_gt = matrix_gt.unsqueeze(0)
            noise = torch.randn(matrix_gt.shape, device=pcd_feats.device)
            ts = torch.randint(0, self.num_timesteps, (1,), device=pcd_feats.device).long()  
            matrix_gt_disturbed = q_sample(x_start=matrix_gt, t=ts, noise=noise, timesteps = self.num_timesteps)

            pcd_points_c_wrapped, tgt_pcd_wrapped, R_forwd, t_forwd = \
                self.get_warped_from_noising_matching3D3D(s_pcd_c.unsqueeze(0), t_pcd_c_da.unsqueeze(0), src_mask, tgt_mask_da, matrix_gt_disturbed)
        

            # Denoising Transformer
            img_feats_c_denoising, pcd_feats_c_denoising = self.denoising_transformer(
                img_feats_c_backbone.unsqueeze(0),
                img_feats_x_dino.unsqueeze(0),
                img_pixels_c.unsqueeze(0),
                pcd_feats_c_backbone.unsqueeze(0),
                pcd_points_c_wrapped,
            )

            conf_matrix_gt_hat ,_,_,_ = self.denoising_coarse_matching(pcd_feats_c_denoising, img_feats_c_denoising, src_mask, tgt_mask, True)
            output_dict['conf_matrix_gt_hat'] = conf_matrix_gt_hat
            output_dict['src_mask'] = src_mask
            output_dict['tgt_mask'] = tgt_mask
            output_dict['matrix_gt'] = matrix_gt
        
            img_feats_c_denoising = F.normalize(img_feats_c_denoising.squeeze(0), p=2, dim=1)
            pcd_feats_c_denoising = F.normalize(pcd_feats_c_denoising.squeeze(0), p=2, dim=1)

            output_dict["img_feats_c_denoising"] = img_feats_c_denoising
            output_dict["pcd_feats_c_denoising"] = pcd_feats_c_denoising        
            # ################################################################################################
            
        #  Reverse sampling
        if not self.training:

            conf_matrix_pred_shape = torch.zeros(1, s_pcd_c.size(0), t_pcd_c_2d.size(0)).shape
            x = torch.randn(conf_matrix_pred_shape, device=s_pcd_c.device)
      

            total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

            # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1)
            # times = torch.linspace(0, total_timesteps - 1, steps=total_timesteps + 1)[:sampling_timesteps]
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

            for time, time_next in time_pairs:

                time_cond = torch.full((1,), time, device=s_pcd_c.device, dtype=torch.long)
        
                pcd_points_c_wrapped, tgt_pcd_wrapped, R_forwd, t_forwd = \
                    self.get_warped_from_noising_matching3D3D(s_pcd_c.unsqueeze(0), t_pcd_c_da.unsqueeze(0), src_mask, tgt_mask_da, x)

                img_feats_c_denoising, pcd_feats_c_denoising = self.denoising_transformer(
                        img_feats_c_backbone.unsqueeze(0),
                        img_feats_x_dino.unsqueeze(0),
                        img_pixels_c.unsqueeze(0),
                        pcd_feats_c_backbone.unsqueeze(0),
                        pcd_points_c_wrapped,
                    )
                x_start, _, _,  _  = \
                    self.denoising_coarse_matching(pcd_feats_c_denoising, img_feats_c_denoising,  src_mask, tgt_mask, True)
            
                pred_noise = self.predict_noise_from_start(x, time_cond, x_start)

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                
                # noise = torch.randn_like(x)

                x = x_start * alpha_next.sqrt() +  c * pred_noise #+  sigma * noise


            sim_matrix =  x# - x.min()

            if src_mask is not None:
                sim_matrix.masked_fill_(
                    ~(src_mask[..., None] * tgt_mask[:, None]).bool(), float('-inf'))

            log_assign_matrix = log_optimal_transport( sim_matrix, self.denoising_coarse_matching.bin_score, self.denoising_coarse_matching.skh_iters, src_mask, tgt_mask)

            assign_matrix = log_assign_matrix.exp()
            conf_matrix_pred = assign_matrix[:, :-1, :-1].contiguous()

            pcd_node_corr_indices, img_node_corr_indices, node_corr_scores = mutual_topk_select(
                conf_matrix_pred.squeeze(0), 1, largest=True, threshold=None, mutual=False
            )        
        

            # single-pass
            # conf_matrix_pred,  pcd_node_corr_indices, img_node_corr_indices,  node_corr_scores = self.coarse_matching(pcd_feats_c.unsqueeze(0), img_feats_c.unsqueeze(0), src_mask, tgt_mask)            


            img_node_corr_levels = img_node_levels[img_node_corr_indices]

            output_dict["img_node_corr_indices"] = img_node_corr_indices
            output_dict["pcd_node_corr_indices"] = pcd_node_corr_indices
            output_dict["img_node_corr_levels"] = img_node_corr_levels

            pcd_padded_feats_f = torch.cat([pcd_feats_f, torch.zeros_like(pcd_feats_f[:1])], dim=0)

            # 7. Extract patch correspondences
            all_img_corr_indices = []
            all_pcd_corr_indices = []

            for i in range(self.img_num_levels_c):
                node_corr_masks = torch.eq(img_node_corr_levels, i)

                if node_corr_masks.sum().item() == 0:
                    continue

                cur_img_node_corr_indices = img_node_corr_indices[node_corr_masks] - all_img_total_nodes[i]
                cur_pcd_node_corr_indices = pcd_node_corr_indices[node_corr_masks]

                img_node_knn_points = all_img_node_knn_points[i]
                img_node_knn_pixels = all_img_node_knn_pixels[i]
                img_node_knn_indices = all_img_node_knn_indices[i]

                img_node_corr_knn_indices = index_select(img_node_knn_indices, cur_img_node_corr_indices, dim=0)
                img_node_corr_knn_masks = torch.ones_like(img_node_corr_knn_indices, dtype=torch.bool)
                img_node_corr_knn_feats = index_select(img_feats_f, img_node_corr_knn_indices, dim=0)

                pcd_node_corr_knn_indices = pcd_node_knn_indices[cur_pcd_node_corr_indices]  # (P, Kc)
                pcd_node_corr_knn_masks = pcd_node_knn_masks[cur_pcd_node_corr_indices]  # (P, Kc)
                pcd_node_corr_knn_feats = index_select(pcd_padded_feats_f, pcd_node_corr_knn_indices, dim=0)

                similarity_mat = pairwise_cosine_similarity(
                    img_node_corr_knn_feats, pcd_node_corr_knn_feats, normalized=True
                )

                batch_indices, row_indices, col_indices, _ = batch_mutual_topk_select(
                    similarity_mat,
                    k=2,
                    row_masks=img_node_corr_knn_masks,
                    col_masks=pcd_node_corr_knn_masks,
                    threshold=0.75,
                    largest=True,
                    mutual=True,
                )

                img_corr_indices = img_node_corr_knn_indices[batch_indices, row_indices]
                pcd_corr_indices = pcd_node_corr_knn_indices[batch_indices, col_indices]

                all_img_corr_indices.append(img_corr_indices)
                all_pcd_corr_indices.append(pcd_corr_indices)

            img_corr_indices = torch.cat(all_img_corr_indices, dim=0)
            pcd_corr_indices = torch.cat(all_pcd_corr_indices, dim=0)

            # duplicate removal
            num_points_f = pcd_points_f.shape[0]
            corr_indices = img_corr_indices * num_points_f + pcd_corr_indices
            unique_corr_indices = torch.unique(corr_indices)
            img_corr_indices = torch.div(unique_corr_indices, num_points_f, rounding_mode="floor")
            pcd_corr_indices = unique_corr_indices % num_points_f

            img_points_f = img_points_f.view(-1, 3)
            img_pixels_f = img_pixels_f.view(-1, 2)
            img_corr_points = img_points_f[img_corr_indices]
            img_corr_pixels = img_pixels_f[img_corr_indices]
            pcd_corr_points = pcd_points_f[pcd_corr_indices]
            pcd_corr_pixels = pcd_pixels_f[pcd_corr_indices]
            img_corr_feats = img_feats_f[img_corr_indices]
            pcd_corr_feats = pcd_feats_f[pcd_corr_indices]
            corr_scores = (img_corr_feats * pcd_corr_feats).sum(1)

            output_dict["img_corr_points"] = img_corr_points
            output_dict["img_corr_pixels"] = img_corr_pixels
            output_dict["img_corr_indices"] = img_corr_indices
            output_dict["pcd_corr_points"] = pcd_corr_points
            output_dict["pcd_corr_pixels"] = pcd_corr_pixels
            output_dict["pcd_corr_indices"] = pcd_corr_indices
            output_dict["corr_scores"] = corr_scores

        torch.cuda.synchronize()
        duration = time_lib.time() - start_time
        output_dict["duration"] = duration



        # ######################
        # # 创建图像

        # # output_dict["pcd_points_f"] = pcd_points_f
        # # output_dict["pcd_pixels_f"] = pcd_pixels_f  


        # min_depth = (pcd_points_f[:,2].cpu().numpy()).min()
        # max_depth = (pcd_points_f[:,2].cpu().numpy()).max()
        # normalized_depth_values = (pcd_points_f[:,2].cpu().numpy() - min_depth) / (max_depth - min_depth)  # 归一化深度值
        # colors = [(int(value * 255), int(value * 255), int(value * 255)) for value in normalized_depth_values]       

        # image_size = (img_h,img_w)
        # # 创建带深度颜色的图像
        # image = Image.new('RGB', image_size, color='white')
        # draw = ImageDraw.Draw(image)

        # # 绘制投影结果
        # point_size = 1.5
        # for i, point in enumerate(pcd_pixels_f.cpu().numpy()):
        #     x, y = point
        #     draw.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill=colors[i])
        #     # draw.point(point, fill=colors[i])


        # # 保存图片
        # transposed_image = image.transpose(Image.TRANSPOSE)
        # transposed_image.save(save_dir+"/pcd_proj_"+data_dict['cloud_file'].split("/")[-1]+'.png')     

        # np.savez(save_dir+"/da_match_"+data_dict['image_id']+'_'+data_dict['cloud_id'],
        #         img_corr_pixels=output_dict["img_corr_pixels"].cpu().numpy(),
        #         pcd_corr_pixels=output_dict["pcd_corr_pixels"].cpu().numpy(),
        #         corr_scores = output_dict["corr_scores"].cpu().numpy())
        # ########################
        return output_dict
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def get_warped_from_noising_matching3D3D(self, s_pcd, t_pcd, src_mask, tgt_mask, matrix_gt_disturbed):

        if src_mask is not None:
            matrix_gt_disturbed.masked_fill_(
                    ~(src_mask[..., None] * tgt_mask[:, None]).bool(), float('-inf'))

            log_assign_matrix_disturbed = log_optimal_transport( matrix_gt_disturbed, self.denoising_coarse_matching.bin_score, self.denoising_coarse_matching.skh_iters, src_mask, tgt_mask)

            assign_matrix_disturbed = log_assign_matrix_disturbed.exp()
            conf_matrix_disturbed = assign_matrix_disturbed[:, :-1, :-1].contiguous().type(torch.float32)
        
        R, t, R_forwd, t_forwd, condition, solution_mask = self.denoising_soft_procrustes(conf_matrix_disturbed, s_pcd, t_pcd, src_mask, tgt_mask)
        # print(R_forwd, "\n" ,t_forwd)
        src_pcd_wrapped = (torch.matmul(R_forwd.type(torch.float32), s_pcd.transpose(1, 2)) + t_forwd.type(torch.float32)).transpose(1, 2)
        tgt_pcd_wrapped = t_pcd.type(torch.float32)        

        return src_pcd_wrapped, tgt_pcd_wrapped  , R_forwd, t_forwd   
     
    from typing import Optional, Tuple, Union

    import torch
    from torch import Tensor             
    def back_project_depth(self,
        depth_mat: Tensor,
        intrinsics: Tensor,
        scaling_factor_a: float = 1000.0,
        scaling_factor_b: float = 1000.0,
        depth_limit: Optional[float] = None,
        transposed: bool = False,
        return_mask: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Back project depth image to point cloud.

        Args:
            depth_mat (Tensor): the depth image in the shape of (B, H, W).
            intrinsics (Tensor): the intrinsic matrix in the shape of (B, 3, 3).
            scaling_factor (float): the depth scaling factor. Default: 1000.
            depth_limit (float, optional): ignore the pixels further than this value.
            transposed (bool): if True, the resulting point matrix is in the shape of (B, H, W, 3).
            return_mask (bool): if True, return a mask matrix where 0-depth points are False. Default: False.

        Returns:
            A Tensor of the point image in the shape of (B, 3, H, W).
            A Tensor of the mask image in the shape of (B, H, W).
        """
        focal_x = intrinsics[..., 0:1, 0:1]
        focal_y = intrinsics[..., 1:2, 1:2]
        center_x = intrinsics[..., 0:1, 2:3]
        center_y = intrinsics[..., 1:2, 2:3]

        batch_size, height, width = depth_mat.shape
        coords = torch.arange(height * width).view(height, width).to(depth_mat.device).unsqueeze(0).expand_as(depth_mat)
        u = coords % width  # (B, H, W)
        v = torch.div(coords, width, rounding_mode="floor")  # (B, H, W)

        z = depth_mat * scaling_factor_a + scaling_factor_b  # (B, H, W)
        if depth_limit is not None:
            z.masked_fill_(torch.gt(z, depth_limit), 0.0)
        x = (u - center_x) * z / focal_x  # (B, H, W)
        y = (v - center_y) * z / focal_y  # (B, H, W)

        if transposed:
            points = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)
        else:
            points = torch.stack([x, y, z], dim=1)  # (B, 3, H, W)

        if not return_mask:
            return points

        masks = torch.gt(z, 0.0)

        return points, masks

    
    import math
    @torch.no_grad()
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    @torch.no_grad()
    # forward diffusion
    def q_sample(x_start, t, noise=None, timesteps=1000):
        if noise is None:
            noise = torch.randn_like(x_start)

        betas = cosine_beta_schedule(timesteps).to(x_start.device)
        alphas = 1. - betas

        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sqrt_alphas_cumprod =  torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise    

    def match_2_conf_matrix(self, matches_gt, matrix_pred):
        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate (matches_gt) :
            matrix_gt [ b][ match[0],  match[1] ] = 1
        return matrix_gt


def create_model(cfg):
    model = MATR2D3D(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == "__main__":
    main()
