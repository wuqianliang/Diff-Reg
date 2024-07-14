from models.blocks import *
from models.backbone import KPFCN
from models.transformero import RepositioningTransformer
from models.matching import Matching, log_optimal_transport
from models.procrustes import SoftProcrustesLayer

from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences
from datasets.utils import blend_scene_flow, multual_nn_correspondence
import time as time_lib


def mutual_topk_select(
    score_mat,
    k,
    largest,
    threshold,
    mutual,
    reduce_result =True,
):
    """Mutual Top-k Selection.

    Args:
        score_mat (Tensor): score matrix. (N, M)
        k (int): the top-k entries from both sides are selected.
        largest (bool=True): use largest top-k.
        threshold (float=0.0): only scores >(<) threshold are selected if (not) largest.
        mutual (bool=True): If True, only entries that are within the top-k of both sides are selected.
        reduce_result (bool=True): If True, return correspondences indices and scores. If False, return corr_mat.

    Returns:
        row_corr_indices (LongTensor): row indices of the correspondences.
        col_corr_indices (LongTensor): col indices of the correspondences.
        corr_scores (Tensor): scores of the correspondences.
        corr_mat (BoolTensor): correspondences matrix.  (N, M)
    """
    num_rows, num_cols = score_mat.shape

    row_topk_indices = score_mat.topk(k=k, largest=largest, dim=1)[1]  # (N, K)
    row_indices = torch.arange(num_rows).cuda().view(num_rows, 1).expand(-1, k)  # (N, K)
    row_corr_mat = torch.zeros_like(score_mat, dtype=torch.bool)  # (N, M)
    row_corr_mat[row_indices, row_topk_indices] = True

    col_topk_indices = score_mat.topk(k=k, largest=largest, dim=0)[1]  # (K, M)
    col_indices = torch.arange(num_cols).cuda().view(1, num_cols).expand(k, -1)  # (K, M)
    col_corr_mat = torch.zeros_like(score_mat, dtype=torch.bool)  # (N, M)
    col_corr_mat[col_topk_indices, col_indices] = True

    if mutual:
        corr_mat = torch.logical_and(row_corr_mat, col_corr_mat)
    else:
        corr_mat = torch.logical_or(row_corr_mat, col_corr_mat)

    if threshold is not None:
        if largest:
            masks = torch.gt(score_mat, threshold)
        else:
            masks = torch.lt(score_mat, threshold)
        corr_mat = torch.logical_and(corr_mat, masks)

    if reduce_result:
        row_corr_indices, col_corr_indices = torch.nonzero(corr_mat, as_tuple=True)
        corr_scores = score_mat[row_corr_indices, col_corr_indices]
        return row_corr_indices, col_corr_indices, corr_scores

    return corr_mat

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

class Pipeline(nn.Module):

    def __init__(self, config):
        super(Pipeline, self).__init__()
        self.config = config
        self.backbone = KPFCN(config['kpfcn_config'])
        self.pe_type = config['coarse_transformer']['pe_type']
        self.coarse_transformer = RepositioningTransformer(config['coarse_transformer'])        
        self.coarse_matching = Matching(config['coarse_matching'])
        self.soft_procrustes = SoftProcrustesLayer(config['coarse_transformer']['procrustes'])


        config['coarse_transformer']['layer_types'] = ['self', 'cross', 'self', 'cross','self', 'cross']
        self.denoising_transformer = RepositioningTransformer(config['coarse_transformer'])
        self.denoising_coarse_matching = Matching(config['coarse_matching'])
        self.denoising_soft_procrustes = SoftProcrustesLayer(config['coarse_transformer']['procrustes'])
        timesteps = 1000
        self.num_timesteps = int(timesteps)

        sampling_timesteps = config['SAMPLE_STEP']
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        
        self.ddim_sampling_eta = 1.  

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('alphas_cumprod', alphas_cumprod)

        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))              

    def forward(self, data,  timers=None, eval_flag=False):

        self.timers = timers
        # start_time = time_lib.time()
        if self.timers: self.timers.tic('kpfcn backbone encode')
        coarse_feats = self.backbone(data, phase="coarse")
        if self.timers: self.timers.toc('kpfcn backbone encode')

        if self.timers: self.timers.tic('coarse_preprocess')
        src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask = self.split_feats (coarse_feats, data)
        data.update({ 's_pcd': s_pcd, 't_pcd': t_pcd })
        if self.timers: self.timers.toc('coarse_preprocess')

        src_feats_backbone, tgt_feats_backbone = src_feats, tgt_feats

        # end_time = time_lib.time() 
        # execution_time = end_time - start_time  # 计算执行时间
        # print(f"Backbone Execution time: {execution_time} seconds")
        if self.training:

            if self.timers: self.timers.tic('coarse feature transformer')
            src_feats, tgt_feats, src_pe, tgt_pe = self.coarse_transformer(src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, data, timers=timers)
            if self.timers: self.timers.toc('coarse feature transformer')

            if self.timers: self.timers.tic('match feature coarse')
            conf_matrix_pred, coarse_match_pred = self.coarse_matching(src_feats, tgt_feats, src_pe, tgt_pe, src_mask, tgt_mask, data, pe_type = self.pe_type)
            data.update({'conf_matrix_pred': conf_matrix_pred, 'coarse_match_pred': coarse_match_pred })
            if self.timers: self.timers.toc('match feature coarse')

            if self.timers: self.timers.tic('procrustes_layer')
            R, t, _, _, _, _ = self.soft_procrustes(conf_matrix_pred, s_pcd, t_pcd, src_mask, tgt_mask)
            data.update({"R_s2t_pred": R, "t_s2t_pred": t})
            if self.timers: self.timers.toc('procrustes_layer')

            ts = torch.randint(0, self.num_timesteps, (1,), device=src_feats.device).long()
            matrix_gt = torch.zeros(src_feats.size(0), src_feats.size(1), tgt_feats.size(1)).to(src_feats.device)
        
            for b, match in enumerate (data['coarse_matches']):
                matrix_gt[b] [ match[0],  match[1] ] = 1

            # 生成服从标准正态分布的随机数
            random_number = torch.randn(matrix_gt.shape, device=src_feats.device)  # 生成一个标准正态分布的随机数
            noise = (random_number.abs() % 1) * (random_number.abs()/random_number)*1.5  # 将绝对值取模映射到0到1之间

            matrix_gt_disturbed = q_sample(x_start=matrix_gt, t=ts, noise=noise, timesteps = self.num_timesteps)
            matrix_gt_disturbed = matrix_gt_disturbed - matrix_gt_disturbed.min()
            matrix_gt_disturbed = torch.nan_to_num(matrix_gt_disturbed,nan=0)

            src_pcd_wrapped, tgt_pcd_wrapped = self.get_warped_from_noising_matching(s_pcd, t_pcd, src_mask, tgt_mask, matrix_gt_disturbed)
            src_feats_noising, tgt_feats_noising, src_pe, tgt_pe = self.denoising_transformer(src_feats_backbone, tgt_feats_backbone, src_pcd_wrapped, tgt_pcd_wrapped, src_mask, tgt_mask, data, timers=timers)
            conf_matrix_gt_hat, coarse_match_gt_hat = self.denoising_coarse_matching(src_feats_noising, tgt_feats_noising, src_pe, tgt_pe, src_mask, tgt_mask, data, pe_type = self.pe_type)
            
            data.update({'conf_matrix_gt_hat': conf_matrix_gt_hat, 'coarse_match_gt_hat': coarse_match_gt_hat })

            ################################################################################################


        if not self.training and not eval_flag:
                
            conf_matrix_pred_shape = torch.zeros(1, src_feats_backbone.size(1), tgt_feats_backbone.size(1)).shape
            x = torch.randn(conf_matrix_pred_shape, device=src_feats_backbone.device)

            total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

            # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1)
            # times = torch.linspace(0, total_timesteps - 1, steps=total_timesteps + 1)[:sampling_timesteps]
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]


            for time, time_next in time_pairs:
                
                time_cond = torch.full((1,), time, device=src_feats_backbone.device, dtype=torch.long)

                x = x - x.min()

                src_pcd_wrapped, tgt_pcd_wrapped = self.get_warped_from_noising_matching(s_pcd, t_pcd, src_mask, tgt_mask, x)

                src_feats_noising, tgt_feats_noising, src_pe, tgt_pe = self.denoising_transformer(src_feats_backbone, tgt_feats_backbone, src_pcd_wrapped, tgt_pcd_wrapped, src_mask, tgt_mask, data, timers=timers)
                x_start, _ = self.denoising_coarse_matching(src_feats_noising, tgt_feats_noising, src_pe, tgt_pe, src_mask, tgt_mask, data, pe_type = self.pe_type)

                pred_noise = self.predict_noise_from_start(x, time_cond, x_start)

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                
                noise = torch.randn_like(x)

                x = x_start * alpha_next.sqrt() +  c * pred_noise #+  sigma * noise
           
            # end_time = time_lib.time() 
            # execution_time = end_time - start_time  # 计算执行时间
            # print(f"Denoising Execution time: {execution_time} seconds")

            # conf_matrix_pred =  torch.sigmoid(x)

            sim_matrix = x - x.min()

            if src_mask is not None:
                sim_matrix.masked_fill_(
                    ~(src_mask[..., None] * tgt_mask[:, None]).bool(), float('-inf'))

            log_assign_matrix = log_optimal_transport( sim_matrix, self.denoising_coarse_matching.bin_score, self.denoising_coarse_matching.skh_iters, src_mask, tgt_mask)

            assign_matrix = log_assign_matrix.exp()
            conf_matrix_pred = assign_matrix[:, :-1, :-1].contiguous()

            src_indices, tgt_indices, weights = mutual_topk_select(
                conf_matrix_pred.squeeze(0), 1, largest=True, threshold=None, mutual=False
            )
            match_pred = torch.cat([torch.zeros_like(src_indices).unsqueeze(-1),src_indices.unsqueeze(-1),tgt_indices.unsqueeze(-1)],dim=-1)

            data.update({'conf_matrix_pred': conf_matrix_pred, 'match_pred':match_pred})

            R, t, _, _, _, _ = self.soft_procrustes(conf_matrix_pred, s_pcd, t_pcd, src_mask, tgt_mask)
            data.update({"R_s2t_pred": R, "t_s2t_pred": t})
            
        return data

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def get_warped_from_noising_matching(self, s_pcd, t_pcd, src_mask, tgt_mask, matrix_gt_disturbed):

        if src_mask is not None:
            matrix_gt_disturbed.masked_fill_(
                    ~(src_mask[..., None] * tgt_mask[:, None]).bool(), float('-inf'))

            log_assign_matrix_disturbed = log_optimal_transport( matrix_gt_disturbed, self.denoising_coarse_matching.bin_score, self.denoising_coarse_matching.skh_iters, src_mask, tgt_mask)

            assign_matrix_disturbed = log_assign_matrix_disturbed.exp()
            conf_matrix_disturbed = assign_matrix_disturbed[:, :-1, :-1].contiguous().type(torch.float32)
        
        R, t, R_forwd, t_forwd, condition, solution_mask = self.denoising_soft_procrustes(conf_matrix_disturbed, s_pcd, t_pcd, src_mask, tgt_mask)
   
        src_pcd_wrapped = (torch.matmul(R_forwd.type(torch.float32), s_pcd.transpose(1, 2)) + t_forwd.type(torch.float32)).transpose(1, 2)
        tgt_pcd_wrapped = t_pcd.type(torch.float32)        

        return src_pcd_wrapped, tgt_pcd_wrapped

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

    def split_feats(self, geo_feats, data):

        pcd = data['points'][self.config['kpfcn_config']['coarse_level']]

        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']
        src_ind_coarse_split = data[ 'src_ind_coarse_split']
        tgt_ind_coarse_split = data['tgt_ind_coarse_split']
        src_ind_coarse = data['src_ind_coarse']
        tgt_ind_coarse = data['tgt_ind_coarse']

        b_size, src_pts_max = src_mask.shape
        tgt_pts_max = tgt_mask.shape[1]

        src_feats = torch.zeros([b_size * src_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        tgt_feats = torch.zeros([b_size * tgt_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        src_pcd = torch.zeros([b_size * src_pts_max, 3]).type_as(pcd)
        tgt_pcd = torch.zeros([b_size * tgt_pts_max, 3]).type_as(pcd)

        src_feats[src_ind_coarse_split] = geo_feats[src_ind_coarse]
        tgt_feats[tgt_ind_coarse_split] = geo_feats[tgt_ind_coarse]
        src_pcd[src_ind_coarse_split] = pcd[src_ind_coarse]
        tgt_pcd[tgt_ind_coarse_split] = pcd[tgt_ind_coarse]

        return src_feats.view( b_size , src_pts_max , -1), \
               tgt_feats.view( b_size , tgt_pts_max , -1), \
               src_pcd.view( b_size , src_pts_max , -1), \
               tgt_pcd.view( b_size , tgt_pts_max , -1), \
               src_mask, \
               tgt_mask