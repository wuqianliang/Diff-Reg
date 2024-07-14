from typing import Tuple

import torch
from torch import Tensor

from vision3d.ops import apply_transform, index_select, knn, masked_mean, pairwise_distance
import numpy as np

def batchify(knn_inputs, block_h, block_w, stride):
    squeeze_last = False
    if knn_inputs.dim() == 2:
        knn_inputs = knn_inputs.unsqueeze(-1)
        squeeze_last = True

    num_inputs, num_neighbors, num_channels = knn_inputs.shape
    assert num_neighbors == block_h * block_w
    knn_inputs = knn_inputs.view(num_inputs, block_h // stride, stride, block_w // stride, stride, num_channels)
    knn_inputs = knn_inputs.permute(0, 2, 4, 1, 3, 5).contiguous()
    knn_inputs = knn_inputs.view(num_inputs * stride * stride, (block_h // stride) * (block_w // stride), num_channels)

    if squeeze_last:
        knn_inputs = knn_inputs.squeeze(-1)

    return knn_inputs


@torch.no_grad()
def patchify(
    img_points: Tensor,
    img_points_da: Tensor,
    img_pixels: Tensor,
    img_masks: Tensor,
    img_masks_da: Tensor,
    img_h_f: int,
    img_w_f: int,
    img_h_c: int,
    img_w_c: int,
    stride: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert img_h_f % img_h_c == 0, f"Image height must be divisible by patch height ({img_h_f} vs {img_h_c})."
    assert img_w_f % img_w_c == 0, f"Image width must be divisible by patch width ({img_w_f} vs {img_w_c})."
    indices = torch.arange(img_h_f * img_w_f).cuda().view(img_h_f, img_w_f)
    knn_indices = indices.view(img_h_c, img_h_f // img_h_c, img_w_c, img_w_f // img_w_c)  # (H', H/H', W', W/W')
    knn_indices = knn_indices.permute(0, 2, 1, 3).contiguous()  # (H', W', H/H', W/W')
    if stride > 1:
        knn_indices = knn_indices[..., ::stride, ::stride].contiguous()  # (H', W', H/H'/S, W/W'/S)
    knn_indices = knn_indices.view(img_h_c * img_w_c, -1)  # (H'xW', BhxBw)
    knn_points = index_select(img_points, knn_indices, dim=0)
    knn_points_da = index_select(img_points_da, knn_indices, dim=0)
    knn_pixels = index_select(img_pixels, knn_indices, dim=0)
    knn_masks = index_select(img_masks, knn_indices, dim=0)
    knn_masks_da = index_select(img_masks_da, knn_indices, dim=0)
    masks = torch.any(knn_masks, dim=1)
    masks_da = torch.any(knn_masks_da, dim=1)
    return knn_points, knn_points_da, knn_pixels, knn_indices, knn_masks, knn_masks_da, masks, masks_da


@torch.no_grad()
def get_2d3d_node_correspondences(
    img_masks: Tensor,
    img_masks_da: Tensor,
    img_knn_points: Tensor,
    img_knn_points_da: Tensor,
    img_knn_pixels: Tensor,
    img_knn_masks: Tensor,
    img_knn_masks_da: Tensor,
    pcd_masks: Tensor,
    pcd_knn_points: Tensor,
    pcd_knn_pixels: Tensor,
    pcd_knn_masks: Tensor,
    transform: Tensor,
    pos_radius_2d: float,
    pos_radius_3d: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate 2D-3D ground-truth superpoint/patch correspondences.

    Each patch is composed of at most k-nearest points of the corresponding superpoint.
    A pair of points match if their 3D distance is below `pos_radius_3d` AND their 2D distance is below `pos_radius_2d`.

    Args:
        img_masks (tensor[bool]): (M,)
        img_knn_points (tensor): (M, Ki, 3)
        img_knn_pixels (tensor): (M, Ki, 2)
        img_knn_masks (tensor[bool]): (M, Ki)
        pcd_masks (tensor[bool]): (N,)
        pcd_knn_points (tensor): (N, Kc, 3)
        pcd_knn_pixels (tensor): (N, Kc, 3)
        pcd_knn_masks (tensor[bool]): (N, Kc)
        transform (tensor): (4, 4)
        pos_radius_2d (float)
        pos_radius_3d (float)

    Returns:
        src_corr_indices (LongTensor): (C,)
        tgt_corr_indices (LongTensor): (C,)
        corr_overlaps (Tensor): (C,)
    """
    pcd_knn_points = apply_transform(pcd_knn_points, transform)  # (N, Kc, 3)
    # 在相机坐标空间的3D点, patch level
    img_centers = masked_mean(img_knn_points, img_knn_masks)  # (M, 3)
    img_centers_da = masked_mean(img_knn_points_da, img_knn_masks_da)  # (M, 3)
    pcd_centers = masked_mean(pcd_knn_points, pcd_knn_masks)  # (N, 3)

    coarse_match_gt = torch.from_numpy( multual_nn_correspondence(pcd_centers.cpu().numpy(), img_centers.cpu().numpy(), search_radius=0.06))
    # print(coarse_match_gt.shape)
    # filter out non-overlapping patches using enclosing sphere
    img_knn_dists = torch.linalg.norm(img_knn_points - img_centers.unsqueeze(1), dim=-1)  # (M, K)
    img_knn_dists[~img_knn_masks] = 0.0
    img_max_dists = img_knn_dists.max(1)[0]  # (M,)
    pcd_knn_dists = torch.linalg.norm(pcd_knn_points - pcd_centers.unsqueeze(1), dim=-1)  # (N, K)
    pcd_knn_dists[~pcd_knn_masks] = 0.0
    pcd_max_dists = pcd_knn_dists.max(1)[0]  # (N,)
    dist_mat = torch.sqrt(pairwise_distance(img_centers, pcd_centers))  # (M, N)
    intersect_mat = torch.gt(img_max_dists.unsqueeze(1) + pcd_max_dists.unsqueeze(0) + pos_radius_3d - dist_mat, 0.0)
    intersect_mat = torch.logical_and(intersect_mat, img_masks.unsqueeze(1))
    intersect_mat = torch.logical_and(intersect_mat, pcd_masks.unsqueeze(0))
    candidate_img_indices, candidate_pcd_indices = torch.nonzero(intersect_mat, as_tuple=True)

    num_candidates = candidate_img_indices.shape[0]

    # select potential patch pairs, compute correspondence matrix
    img_knn_points = img_knn_points[candidate_img_indices]  # (B, Ki, 3)
    img_knn_pixels = img_knn_pixels[candidate_img_indices]  # (B, Ki, 2)
    img_knn_masks = img_knn_masks[candidate_img_indices]  # (B, Ki)
    pcd_knn_points = pcd_knn_points[candidate_pcd_indices]  # (B, Kc, 3)
    pcd_knn_pixels = pcd_knn_pixels[candidate_pcd_indices]  # (B, Kc, 2)
    pcd_knn_masks = pcd_knn_masks[candidate_pcd_indices]  # (B, Ki)

    # compute 2d overlap masks
    img_knn_min_distances_3d, img_knn_min_indices_3d = knn(img_knn_points, pcd_knn_points, k=1, return_distance=True)
    img_knn_min_distances_3d = img_knn_min_distances_3d.squeeze(-1)
    img_knn_min_indices_3d = img_knn_min_indices_3d.squeeze(-1)
    img_knn_batch_indices_3d = torch.arange(num_candidates).cuda().unsqueeze(1).expand_as(img_knn_min_indices_3d)
    img_knn_min_pcd_pixels = pcd_knn_pixels[img_knn_batch_indices_3d, img_knn_min_indices_3d]
    img_knn_min_distances_2d = torch.linalg.norm(img_knn_pixels - img_knn_min_pcd_pixels, dim=-1)
    img_knn_min_pcd_masks = pcd_knn_masks[img_knn_batch_indices_3d, img_knn_min_indices_3d]
    img_knn_overlap_masks_3d = torch.lt(img_knn_min_distances_3d, pos_radius_3d)
    img_knn_overlap_masks_2d = torch.lt(img_knn_min_distances_2d, pos_radius_2d)
    img_knn_overlap_masks = torch.logical_and(img_knn_overlap_masks_2d, img_knn_overlap_masks_3d)
    img_knn_overlap_masks = torch.logical_and(img_knn_overlap_masks, img_knn_min_pcd_masks)
    img_knn_overlap_masks = torch.logical_and(img_knn_overlap_masks, img_knn_masks)

    # compute 3d overlap masks
    pcd_knn_min_distances_3d, pcd_knn_min_indices_3d = knn(pcd_knn_points, img_knn_points, k=1, return_distance=True)
    pcd_knn_min_distances_3d = pcd_knn_min_distances_3d.squeeze(-1)
    pcd_knn_min_indices_3d = pcd_knn_min_indices_3d.squeeze(-1)
    pcd_knn_batch_indices_3d = torch.arange(num_candidates).cuda().unsqueeze(1).expand_as(pcd_knn_min_indices_3d)
    pcd_knn_min_img_pixels = img_knn_pixels[pcd_knn_batch_indices_3d, pcd_knn_min_indices_3d]
    pcd_knn_min_distances_2d = torch.linalg.norm(pcd_knn_pixels - pcd_knn_min_img_pixels, dim=-1)
    pcd_knn_min_img_masks = img_knn_masks[pcd_knn_batch_indices_3d, pcd_knn_min_indices_3d]
    pcd_knn_overlap_masks_3d = torch.lt(pcd_knn_min_distances_3d, pos_radius_3d)
    pcd_knn_overlap_masks_2d = torch.lt(pcd_knn_min_distances_2d, pos_radius_2d)
    pcd_knn_overlap_masks = torch.logical_and(pcd_knn_overlap_masks_2d, pcd_knn_overlap_masks_3d)
    pcd_knn_overlap_masks = torch.logical_and(pcd_knn_overlap_masks, pcd_knn_min_img_masks)
    pcd_knn_overlap_masks = torch.logical_and(pcd_knn_overlap_masks, pcd_knn_masks)

    # compute overlaps
    img_overlap_counts = img_knn_overlap_masks.sum(1)  # (B,)
    pcd_overlap_counts = pcd_knn_overlap_masks.sum(1)  # (B,)
    img_total_counts = img_knn_masks.sum(-1)  # (B,)
    pcd_total_counts = pcd_knn_masks.sum(-1)  # (B,)
    img_overlap_ratios = img_overlap_counts.float() / img_total_counts.float()  # (B,)
    pcd_overlap_ratios = pcd_overlap_counts.float() / pcd_total_counts.float()  # (B,)

    img_overlap_masks = torch.gt(img_overlap_ratios, 0.0)
    pcd_overlap_masks = torch.gt(pcd_overlap_ratios, 0.0)
    overlap_masks = torch.logical_and(img_overlap_masks, pcd_overlap_masks)
    img_corr_indices = candidate_img_indices[overlap_masks]
    pcd_corr_indices = candidate_pcd_indices[overlap_masks]
    img_corr_overlaps = img_overlap_ratios[overlap_masks]
    pcd_corr_overlaps = pcd_overlap_ratios[overlap_masks]

    return img_corr_indices, pcd_corr_indices, img_corr_overlaps, pcd_corr_overlaps, pcd_centers, img_centers, img_centers_da, coarse_match_gt




def partition_arg_topK(matrix, K, axis=0):
    """ find index of K smallest entries along a axis
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

def knn_point_np(k, reference_pts, query_pts):
    '''
    :param k: number of k in k-nn search
    :param reference_pts: (N, 3) float32 array, input points
    :param query_pts: (M, 3) float32 array, query points
    :return:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''

    N, _ = reference_pts.shape
    M, _ = query_pts.shape
    reference_pts = reference_pts.reshape(1, N, -1).repeat(M, axis=0)
    query_pts = query_pts.reshape(M, 1, -1).repeat(N, axis=1)
    dist = np.sum((reference_pts - query_pts) ** 2, -1)
    idx = partition_arg_topK(dist, K=k, axis=1)
    val = np.take_along_axis ( dist , idx, axis=1)
    return np.sqrt(val), idx


def blend_scene_flow (query_loc, reference_loc, reference_flow , knn=3) :
    '''approximate flow on query points
    this function assume query points are sub-/un-sampled from reference locations
    @param query_loc:[m,3]
    @param reference_loc:[n,3]
    @param reference_flow:[n,3]
    @param knn:
    @return:
        blended_flow:[m,3]
    '''
    dists, idx = knn_point_np (knn, reference_loc, query_loc)
    dists[dists < 1e-10] = 1e-10
    weight = 1.0 / dists
    weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
    blended_flow = np.sum (reference_flow [idx] * weight.reshape ([-1, knn, 1]), axis=1, keepdims=False)

    return blended_flow


def multual_nn_correspondence(src_pcd_deformed, tgt_pcd, search_radius=0.3, knn=1):

    src_idx = np.arange(src_pcd_deformed.shape[0])

    s2t_dists, ref_tgt_idx = knn_point_np (knn, tgt_pcd, src_pcd_deformed)
    s2t_dists, ref_tgt_idx = s2t_dists[:,0], ref_tgt_idx [:, 0]
    valid_distance = s2t_dists < search_radius

    _, ref_src_idx = knn_point_np (knn, src_pcd_deformed, tgt_pcd)
    _, ref_src_idx = _, ref_src_idx [:, 0]

    cycle_src_idx = ref_src_idx [ ref_tgt_idx ]

    is_mutual_nn = cycle_src_idx == src_idx

    mutual_nn = np.logical_and( is_mutual_nn, valid_distance)
    correspondences = np.stack([src_idx [ mutual_nn ], ref_tgt_idx[mutual_nn] ] , axis=0)

    return correspondences


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):

    src_pcd.transform(trans)
    correspondences =  KDTree_corr ( src_pcd, tgt_pcd, search_voxel_size, K=None)
    correspondences = torch.from_numpy(correspondences)
    return correspondences


import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d

_EPS = 1e-7  # To prevent division by zero


# def viz_coarse_nn_correspondence_mayavi(s_pc, t_pc, good_c, bad_c, f_src_pcd=None, f_tgt_pcd=None, scale_factor=0.02):
#     '''
#     @param s_pc:  [S,3]
#     @param t_pc:  [T,3]
#     @param correspondence: [2,K]
#     @param f_src_pcd: [S1,3]
#     @param f_tgt_pcd: [T1,3]
#     @param scale_factor:
#     @return:
#     '''

#     import mayavi.mlab as mlab
#     c_red = (224. / 255., 0 / 255., 0 / 255.)
#     c_pink = (224. / 255., 75. / 255., 232. / 255.)
#     c_blue = (0. / 255., 0. / 255., 255. / 255.)
#     c_green = (0. / 255., 255. / 255., 0. / 255.)
#     c_gray1 = (255 / 255., 255 / 255., 125 / 255.)
#     c_gray2 = (125. / 255., 125. / 255., 255. / 255.)

#     if f_src_pcd is not None:
#         mlab.points3d(f_src_pcd[:, 0], f_src_pcd[:, 1], f_src_pcd[:, 2], scale_factor=scale_factor * 0.25,
#                       color=c_gray1)
#     else:
#         mlab.points3d(s_pc[:, 0], s_pc[:, 1], s_pc[:, 2], scale_factor=scale_factor * 0.75, color=c_gray1)

#     if f_tgt_pcd is not None:
#         mlab.points3d(f_tgt_pcd[:, 0], f_tgt_pcd[:, 1], f_tgt_pcd[:, 2], scale_factor=scale_factor * 0.25,
#                       color=c_gray2)
#     else:
#         mlab.points3d(t_pc[:, 0], t_pc[:, 1], t_pc[:, 2], scale_factor=scale_factor * 0.75, color=c_gray2)

#     s_cpts_god = s_pc[good_c[0]]
#     t_cpts_god = t_pc[good_c[1]]
#     flow_good = t_cpts_god - s_cpts_god

#     s_cpts_bd = s_pc[bad_c[0]]
#     t_cpts_bd = t_pc[bad_c[1]]
#     flow_bad = t_cpts_bd - s_cpts_bd

#     def match_draw(s_cpts, t_cpts, flow, color):

#         mlab.points3d(s_cpts[:, 0], s_cpts[:, 1], s_cpts[:, 2], scale_factor=scale_factor * 0.35, color=c_blue)
#         mlab.points3d(t_cpts[:, 0], t_cpts[:, 1], t_cpts[:, 2], scale_factor=scale_factor * 0.35, color=c_pink)
#         mlab.quiver3d(s_cpts[:, 0], s_cpts[:, 1], s_cpts[:, 2], flow[:, 0], flow[:, 1], flow[:, 2],
#                       scale_factor=1, mode='2ddash', line_width=1., color=color)

#     match_draw(s_cpts_god, t_cpts_god, flow_good, c_green)
#     match_draw(s_cpts_bd, t_cpts_bd, flow_bad, c_red)

#     mlab.show()


# def correspondence_viz(src_raw, tgt_raw, src_pcd, tgt_pcd, corrs, inlier_mask, max=200):
#     perm = np.random.permutation(corrs.shape[1])
#     ind = perm[:max]

#     corrs = corrs[:, ind]
#     inlier_mask = inlier_mask[ind]

#     good_c = corrs[:, inlier_mask]
#     bad_c = corrs[:, ~inlier_mask]

#     offset = np.array([[1.45, 0, 0]])
#     # src_pcd = src_pcd + offset
#     # src_raw = src_raw + offset
#     tgt_pcd = tgt_pcd + offset
#     tgt_raw = tgt_raw + offset

#     viz_coarse_nn_correspondence_mayavi(src_pcd, tgt_pcd, good_c, bad_c, src_raw, tgt_raw, scale_factor=0.07)


def fmr_wrt_distance(data,split,inlier_ratio_threshold=0.05):
    """
    calculate feature match recall wrt distance threshold
    """
    fmr_wrt_distance =[]
    for distance_threshold in range(1,21):
        inlier_ratios =[]
        distance_threshold /=100.0
        for idx in range(data.shape[0]):
            inlier_ratio = (data[idx] < distance_threshold).mean()
            inlier_ratios.append(inlier_ratio)
        fmr = 0
        for ele in split:
            fmr += (np.array(inlier_ratios[ele[0]:ele[1]]) > inlier_ratio_threshold).mean()
        fmr /= 8
        fmr_wrt_distance.append(fmr*100)
    return fmr_wrt_distance

def fmr_wrt_inlier_ratio(data, split, distance_threshold=0.1):
    """
    calculate feature match recall wrt inlier ratio threshold
    """
    fmr_wrt_inlier =[]
    for inlier_ratio_threshold in range(1,21):
        inlier_ratios =[]
        inlier_ratio_threshold /=100.0
        for idx in range(data.shape[0]):
            inlier_ratio = (data[idx] < distance_threshold).mean()
            inlier_ratios.append(inlier_ratio)
        
        fmr = 0
        for ele in split:
            fmr += (np.array(inlier_ratios[ele[0]:ele[1]]) > inlier_ratio_threshold).mean()
        fmr /= 8
        fmr_wrt_inlier.append(fmr*100)

    return fmr_wrt_inlier



def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_tsfm(rot,trans):
    tsfm = np.eye(4)
    tsfm[:3,:3]=rot
    tsfm[:3,3]=trans.flatten()
    return tsfm
    
def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats

def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):

    src_pcd.transform(trans)
    correspondences =  KDTree_corr ( src_pcd, tgt_pcd, search_voxel_size, K=None)
    correspondences = torch.from_numpy(correspondences)
    return correspondences


def KDTree_corr ( src_pcd_transformed, tgt_pcd, search_voxel_size, K=None):

    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    correspondences = []
    for i, point in enumerate(src_pcd_transformed.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])
    correspondences = np.array(correspondences)
    return correspondences



def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]

def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]

def random_sample(pcd, feats, N):
    """
    Do random sampling to get exact N points and associated features
    pcd:    [N,3]
    feats:  [N,C]
    """
    if(isinstance(pcd,torch.Tensor)):
        n1 = pcd.size(0)
    elif(isinstance(pcd, np.ndarray)):
        n1 = pcd.shape[0]

    if n1 == N:
        return pcd, feats

    if n1 > N:
        choice = np.random.permutation(n1)[:N]
    else:
        choice = np.random.choice(n1, N)

    return pcd[choice], feats[choice]
    
def get_angle_deviation(R_pred,R_gt):
    """
    Calculate the angle deviation between two rotaion matrice
    The rotation error is between [0,180]
    Input:
        R_pred: [B,3,3]
        R_gt  : [B,3,3]
    Return: 
        degs:   [B]
    """
    R=np.matmul(R_pred,R_gt.transpose(0,2,1))
    tr=np.trace(R,0,1,2) 
    rads=np.arccos(np.clip((tr-1)/2,-1,1))  # clip to valid range
    degs=rads/np.pi*180

    return degs

def ransac_pose_estimation(src_pcd, tgt_pcd, src_feat, tgt_feat, mutual = False, distance_threshold = 0.05, ransac_n = 3):
    """
    RANSAC pose estimation with two checkers
    We follow D3Feat to set ransac_n = 3 for 3DMatch and ransac_n = 4 for KITTI. 
    For 3DMatch dataset, we observe significant improvement after changing ransac_n from 4 to 3.
    """
    if(mutual):
        if(torch.cuda.device_count()>=1):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        src_feat, tgt_feat = to_tensor(src_feat), to_tensor(tgt_feat)
        scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0,1).to(device)).cpu()
        selection = mutual_selection(scores[None,:,:])[0]
        row_sel, col_sel = np.where(selection)
        corrs = o3d.utility.Vector2iVector(np.array([row_sel,col_sel]).T)
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
            source=src_pcd, target=tgt_pcd,corres=corrs, 
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            criteria=o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    else:
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        src_feats = to_o3d_feats(src_feat)
        tgt_feats = to_o3d_feats(tgt_feat)

        result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
            src_pcd, tgt_pcd, src_feats, tgt_feats,distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), ransac_n,
            [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.registration.RANSACConvergenceCriteria(50000, 1000))
            
    return result_ransac.transformation

def get_inlier_ratio(src_pcd, tgt_pcd, src_feat, tgt_feat, rot, trans, inlier_distance_threshold = 0.1):
    """
    Compute inlier ratios with and without mutual check, return both
    """
    src_pcd = to_tensor(src_pcd)
    tgt_pcd = to_tensor(tgt_pcd)
    src_feat = to_tensor(src_feat)
    tgt_feat = to_tensor(tgt_feat)
    rot, trans = to_tensor(rot), to_tensor(trans)

    results =dict()
    results['w']=dict()
    results['wo']=dict()

    if(torch.cuda.device_count()>=1):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    src_pcd = (torch.matmul(rot, src_pcd.transpose(0,1)) + trans).transpose(0,1)
    scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0,1).to(device)).cpu()

    ########################################
    # 1. calculate inlier ratios wo mutual check
    _, idx = scores.max(-1)
    dist = torch.norm(src_pcd- tgt_pcd[idx],dim=1)
    results['wo']['distance'] = dist.numpy()

    c_inlier_ratio = (dist < inlier_distance_threshold).float().mean()
    results['wo']['inlier_ratio'] = c_inlier_ratio

    ########################################
    # 2. calculate inlier ratios w mutual check
    selection = mutual_selection(scores[None,:,:])[0]
    row_sel, col_sel = np.where(selection)
    dist = torch.norm(src_pcd[row_sel]- tgt_pcd[col_sel],dim=1)
    results['w']['distance'] = dist.numpy()

    c_inlier_ratio = (dist < inlier_distance_threshold).float().mean()
    results['w']['inlier_ratio'] = c_inlier_ratio

    return results


def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column
    
    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N] 
    """
    score_mat=to_array(score_mat)
    if(score_mat.ndim==2):
        score_mat=score_mat[None,:,:]
    
    mutuals=np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]): # loop through the batch
        c_mat=score_mat[i]
        flag_row=np.zeros_like(c_mat)
        flag_column=np.zeros_like(c_mat)

        max_along_row=np.argmax(c_mat,1)[:,None]
        max_along_column=np.argmax(c_mat,0)[None,:]
        np.put_along_axis(flag_row,max_along_row,1,1)
        np.put_along_axis(flag_column,max_along_column,1,0)
        mutuals[i]=(flag_row.astype(np.bool)) & (flag_column.astype(np.bool))
    return mutuals.astype(np.bool)  

