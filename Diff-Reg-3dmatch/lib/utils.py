import os,re,sys,json,yaml,random, argparse, torch, pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
_EPS = 1e-7  # To prevent division by zero


class Logger:
    def __init__(self, path):
        self.path = path
        log_path = self.path + '/log'
        if os.path.exists(log_path):
            os.remove(log_path)
        self.fw = open(log_path,'a')

    def write(self, text):
        self.fw.write(text)
        self.fw.flush()

    def close(self):
        self.fw.close()

def save_obj(obj, path ):
    """
    save a dictionary to a pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(path):
    """
    read a dictionary from a pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_config(path):
    """
    Loads config file:

    Args:
        path (str): path to the config file

    Returns: 
        config (dict): dictionary of the configuration parameters, merge sub_dicts

    """
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)
    
    config = dict()
    for key, value in cfg.items():
        for k,v in value.items():
            config[k] = v

    return config


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def square_distance(src, dst, normalised = False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if(normalised):
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist
    

def validate_gradient(model):
    """
    Confirm all the gradients are non-nan and non-inf
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                return False
            if torch.any(torch.isinf(param.grad)):
                return False
    return True


def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]



def to_o3d_pcd(pcd):
    '''
    Transfer a point cloud of numpy.ndarray to open3d point cloud
    :param pcd: point cloud of numpy.ndarray in shape[N, 3]
    :return: open3d.geometry.PointCloud()
    '''
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    return pcd_


def calc_ppf_gpu(points, point_normals, patches, patch_normals):
    '''
    Calculate ppf gpu
    points: [n, 3]
    point_normals: [n, 3]
    patches: [n, nsamples, 3]
    patch_normals: [n, nsamples, 3]
    '''
    points = torch.unsqueeze(points, dim=1).expand(-1, patches.shape[1], -1)
    point_normals = torch.unsqueeze(point_normals, dim=1).expand(-1, patches.shape[1], -1)
    vec_d = patches - points #[n, n_samples, 3]
    d = torch.sqrt(torch.sum(vec_d ** 2, dim=-1, keepdim=True)) #[n, n_samples, 1]
    # angle(n1, vec_d)
    y = torch.sum(point_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(point_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle1 = torch.atan2(x, y) / np.pi

    # angle(n2, vec_d)
    y = torch.sum(patch_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(patch_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle2 = torch.atan2(x, y) / np.pi

    # angle(n1, n2)
    y = torch.sum(point_normals * patch_normals, dim=-1, keepdim=True)
    x = torch.cross(point_normals, patch_normals, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle3 = torch.atan2(x, y) / np.pi

    ppf = torch.cat([d, angle1, angle2, angle3], dim=-1) #[n, samples, 4]
    return ppf


def group_all(feats):
    '''
    all-to-all grouping
    feats: [n, c]
    out: grouped feat: [n, n, c]
    '''
    grouped_feat = torch.unsqueeze(feats, dim=0)
    grouped_feat = grouped_feat.expand(feats.shape[0], -1, -1) #[n, n, c]
    return grouped_feat


def point_to_node_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.
    Fixed knn bug.
    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`
    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """
    sq_dist_mat = square_distance(nodes[None, ::], points[None, ::])[0]  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(points.shape[0]).cuda()  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


# def get_node_occlusion_score(
#         ref_knn_point_ids: torch.Tensor,
#         src_knn_point_ids: torch.Tensor,
#         ref_points: torch.Tensor,
#         src_points: torch.Tensor,
#         rot: torch.Tensor,
#         trans: torch.Tensor,
#         ref_masks: Optional[torch.Tensor] = None,
#         src_masks: Optional[torch.Tensor] = None,
#         ref_knn_masks: Optional[torch.Tensor] = None,
#         src_knn_masks: Optional[torch.Tensor] = None,
#         overlap_thres: Optional[float] = 0.0375,
# ):
#     r"""
#     Compute the occlusion scores for each node. Scores are in range of [0, 1], 0 for completely occluded,
#     while 1 for completely visible, depending on vicinity points.
#     Args:
#         ref_knn_point_ids: torch.Tensor (M, K)
#         src_knn_point_ids: torch.Tensor (N, K)
#         ref_points: torch.Tensor (N1, 3)
#         src_points: torch.Tensor (N2, 3)
#         rot: torch.Tensor (3, 3)
#         trans: torch.Tensor (3, 1)
#         ref_masks (optional): torch.BoolTensor (M,) (default: None)
#         src_masks (optional): torch.BoolTensor (N,) (default: None)
#         ref_knn_masks (optional): torch.BoolTensor (M, K) (default: None)
#         src_knn_masks (optional): torch.BoolTensor (N, K) (default: None)
#     Returns:
#         ref_overlap_score: torch.Tensor (M,)
#         src_overlap_score: torch.Tensor (N,)
#     """
#     src_points = torch.matmul(src_points, rot.T) + trans.T
#     ref_o, src_o = torch.from_numpy(np.array([ref_points.shape[0]])).to(ref_points).int(), torch.from_numpy(np.array([src_points.shape[0]])).to(src_points).int()

#     _, ref_dist = knnquery(1, src_points, ref_points, src_o, ref_o)
#     _, src_dist = knnquery(1, ref_points, src_points, ref_o, src_o)

#     ref_overlap = torch.lt(ref_dist, overlap_thres).float().squeeze(1) #(M, )
#     src_overlap = torch.lt(src_dist, overlap_thres).float().squeeze(1) #(N, )

#     M, K = ref_knn_point_ids.shape
#     N, _ = src_knn_point_ids.shape
#     ref_knn_point_ids = ref_knn_point_ids.view(-1).contiguous()
#     src_knn_point_ids = src_knn_point_ids.view(-1).contiguous()

#     ref_knn_overlap = ref_overlap[ref_knn_point_ids].reshape((M, K))
#     src_knn_overlap = src_overlap[src_knn_point_ids].reshape((N, K))

#     ref_overlap_score = torch.sum(ref_knn_overlap * ref_knn_masks, dim=1) / (torch.sum(ref_knn_masks, dim=1) + 1e-10)
#     src_overlap_score = torch.sum(src_knn_overlap * src_knn_masks, dim=1) / (torch.sum(src_knn_masks, dim=1) + 1e-10)

#     ref_overlap_score = ref_overlap_score * ref_masks
#     src_overlap_score = src_overlap_score * src_masks
#     return ref_overlap_score, src_overlap_score


# def get_node_correspondences(
#     ref_nodes: torch.Tensor,
#     src_nodes: torch.Tensor,
#     ref_knn_points: torch.Tensor,
#     src_knn_points: torch.Tensor,
#     rot: torch.Tensor,
#     trans: torch.Tensor,
#     pos_radius: float,
#     ref_masks: Optional[torch.Tensor] = None,
#     src_masks: Optional[torch.Tensor] = None,
#     ref_knn_masks: Optional[torch.Tensor] = None,
#     src_knn_masks: Optional[torch.Tensor] = None,
# ):
#     r"""Generate ground-truth superpoint/patch correspondences.
#     Each patch is composed of at most k nearest points of the corresponding superpoint.
#     A pair of points match if the distance between them is smaller than `self.pos_radius`.
#     Args:
#         ref_nodes: torch.Tensor (M, 3)
#         src_nodes: torch.Tensor (N, 3)
#         ref_knn_points: torch.Tensor (M, K, 3)
#         src_knn_points: torch.Tensor (N, K, 3)
#         rot: torch.Tensor (3, 3)
#         trans: torch.Tensor (3, 1)
#         pos_radius: float
#         ref_masks (optional): torch.BoolTensor (M,) (default: None)
#         src_masks (optional): torch.BoolTensor (N,) (default: None)
#         ref_knn_masks (optional): torch.BoolTensor (M, K) (default: None)
#         src_knn_masks (optional): torch.BoolTensor (N, K) (default: None)
#     Returns:
#         corr_indices: torch.LongTensor (C, 2)
#         corr_overlaps: torch.Tensor (C,)
#     """
#     src_nodes = torch.matmul(src_nodes, rot.T) + trans.T
#     src_knn_points = torch.matmul(src_knn_points, rot.T) + (trans.T)[None, ::]

#     # generate masks
#     if ref_masks is None:
#         ref_masks = torch.ones(size=(ref_nodes.shape[0],), dtype=torch.bool).cuda()
#     if src_masks is None:
#         src_masks = torch.ones(size=(src_nodes.shape[0],), dtype=torch.bool).cuda()
#     if ref_knn_masks is None:
#         ref_knn_masks = torch.ones(size=(ref_knn_points.shape[0], ref_knn_points.shape[1]), dtype=torch.bool).cuda()
#     if src_knn_masks is None:
#         src_knn_masks = torch.ones(size=(src_knn_points.shape[0], src_knn_points.shape[1]), dtype=torch.bool).cuda()

#     node_mask_mat = torch.logical_and(ref_masks.unsqueeze(1), src_masks.unsqueeze(0))  # (M, N)

#     # filter out non-overlapping patches using enclosing sphere
#     ref_knn_dists = torch.linalg.norm(ref_knn_points - ref_nodes.unsqueeze(1), dim=-1)  # (M, K)
#     ref_knn_dists.masked_fill_(~ref_knn_masks, 0.0)
#     ref_max_dists = ref_knn_dists.max(1)[0]  # (M,)
#     src_knn_dists = torch.linalg.norm(src_knn_points - src_nodes.unsqueeze(1), dim=-1)  # (N, K)
#     src_knn_dists.masked_fill_(~src_knn_masks, 0.0)
#     src_max_dists = src_knn_dists.max(1)[0]  # (N,)
#     dist_mat = torch.sqrt(square_distance(ref_nodes[None, ::], src_nodes[None, ::])[0])  # (M, N)
#     intersect_mat = torch.gt(ref_max_dists.unsqueeze(1) + src_max_dists.unsqueeze(0) + pos_radius - dist_mat, 0)
#     intersect_mat = torch.logical_and(intersect_mat, node_mask_mat)
#     sel_ref_indices, sel_src_indices = torch.nonzero(intersect_mat, as_tuple=True)

#     # select potential patch pairs
#     ref_knn_masks = ref_knn_masks[sel_ref_indices]  # (B, K)
#     src_knn_masks = src_knn_masks[sel_src_indices]  # (B, K)
#     ref_knn_points = ref_knn_points[sel_ref_indices]  # (B, K, 3)
#     src_knn_points = src_knn_points[sel_src_indices]  # (B, K, 3)

#     point_mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))  # (B, K, K)

#     # compute overlaps
#     dist_mat = square_distance(ref_knn_points, src_knn_points) # (B, K, K)
#     dist_mat.masked_fill_(~point_mask_mat, 1e12)
#     point_overlap_mat = torch.lt(dist_mat, pos_radius ** 2)  # (B, K, K)
#     ref_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-1), dim=-1).float()  # (B,)
#     src_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-2), dim=-1).float()  # (B,)
#     ref_overlaps = ref_overlap_counts / ref_knn_masks.sum(-1).float()  # (B,)
#     src_overlaps = src_overlap_counts / src_knn_masks.sum(-1).float()  # (B,)
#     overlaps = (ref_overlaps + src_overlaps) / 2  # (B,)

#     overlap_masks = torch.gt(overlaps, 0)
#     ref_corr_indices = sel_ref_indices[overlap_masks]
#     src_corr_indices = sel_src_indices[overlap_masks]
#     corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
#     #corr_indices = torch.stack([src_corr_indices, ref_corr_indices], dim=1)
#     corr_overlaps = overlaps[overlap_masks]

#     return corr_indices, corr_overlaps