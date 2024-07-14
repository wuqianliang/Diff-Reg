import torch
import torch.nn as nn

from vision3d.loss import CircleLoss
from vision3d.ops import apply_transform, pairwise_distance, random_choice
from vision3d.ops.metrics import compute_isotropic_transform_error


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = CircleLoss(
            cfg.loss.coarse_loss.positive_margin,
            cfg.loss.coarse_loss.negative_margin,
            cfg.loss.coarse_loss.positive_optimal,
            cfg.loss.coarse_loss.negative_optimal,
            cfg.loss.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.loss.coarse_loss.positive_overlap
        self.negative_overlap = cfg.loss.coarse_loss.negative_overlap
        self.eps = 1e-8

        self.pos_w = 1.0
        self.neg_w = 1.0

        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.match_type = 'sinkhorn'
        
    def forward(self, output_dict):
        img_feats = output_dict["img_feats_c"]
        pcd_feats = output_dict["pcd_feats_c"]
        gt_img_node_corr_indices = output_dict["gt_img_node_corr_indices"]
        gt_pcd_node_corr_indices = output_dict["gt_pcd_node_corr_indices"]
        gt_node_corr_min_overlaps = output_dict["gt_node_corr_min_overlaps"]
        gt_node_corr_max_overlaps = output_dict["gt_node_corr_min_overlaps"]

        feat_dists = torch.sqrt(pairwise_distance(img_feats, pcd_feats, normalized=True) + self.eps)

        min_overlaps = torch.zeros_like(feat_dists)
        min_overlaps[gt_img_node_corr_indices, gt_pcd_node_corr_indices] = gt_node_corr_min_overlaps
        pos_masks = torch.gt(min_overlaps, self.positive_overlap)
        pos_scales = torch.sqrt(min_overlaps * pos_masks.float())

        max_overlaps = torch.zeros_like(feat_dists)
        max_overlaps[gt_img_node_corr_indices, gt_pcd_node_corr_indices] = gt_node_corr_max_overlaps
        neg_masks = torch.lt(max_overlaps, self.negative_overlap)

        # 1.
        loss_circle = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        src_mask = output_dict['src_mask']
        tgt_mask = output_dict['tgt_mask']  
        conf_matrix_gt = output_dict['matrix_gt'] 
        c_weight = (src_mask[:, :, None] * tgt_mask[:, None, :]).float()     
      
        # 2. matching loss
        loss_focal = self.compute_correspondence_loss(output_dict['conf_matrix_pred'], conf_matrix_gt, weight=c_weight)

        # ##########################################################################
        # 3. 
        img_feats_denoising = output_dict["img_feats_c_denoising"]
        pcd_feats_denoising = output_dict["pcd_feats_c_denoising"]

        feat_dists_denoising = torch.sqrt(pairwise_distance(img_feats_denoising, pcd_feats_denoising, normalized=True) + self.eps)
        loss_circle_denoising = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists_denoising, pos_scales)

        # 4. matching loss for denoising ####
        conf_matrix_gt_hat = output_dict['conf_matrix_gt_hat']          
        loss_matrix_gt_hat = self.compute_correspondence_loss(conf_matrix_gt_hat, conf_matrix_gt, weight=c_weight)           
        # ##########################################################################
        


        return loss_circle, loss_circle_denoising, loss_focal, loss_matrix_gt_hat

    def match_2_conf_matrix(self, matches_gt, matrix_pred):
        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate (matches_gt.long()) :
            matrix_gt [ b][ match[0],  match[1] ] = 1
        return matrix_gt
    
    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        #corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma

        if self.match_type == "dual_softmax":
            pos_conf = conf[pos_mask]
            loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
            loss =  pos_w * loss_pos.mean()
            return loss

        elif self.match_type == "sinkhorn":
            # no supervision on dustbin row & column.
            loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
            loss = pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
            return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()

        self.max_correspondences = cfg.loss.fine_loss.max_correspondences
        self.pos_radius_3d = cfg.loss.fine_loss.positive_radius_3d
        self.neg_radius_3d = cfg.loss.fine_loss.negative_radius_3d
        self.pos_radius_2d = cfg.loss.fine_loss.positive_radius_2d
        self.neg_radius_2d = cfg.loss.fine_loss.negative_radius_2d

        self.circle_loss = CircleLoss(
            cfg.loss.fine_loss.positive_margin,
            cfg.loss.fine_loss.negative_margin,
            cfg.loss.fine_loss.positive_optimal,
            cfg.loss.fine_loss.negative_optimal,
            cfg.loss.fine_loss.log_scale,
        )

    @torch.no_grad()
    def get_recall(self, gt_corr_mat, fdist_mat):
        # Get feature match recall, divided by number of points which has inlier matches
        num_gt_corr = torch.gt(gt_corr_mat.sum(-1), 0).float().sum() + 1e-12
        src_indices = torch.arange(fdist_mat.shape[0]).cuda()
        src_nn_indices = fdist_mat.min(-1)[1]
        pred_corr_mat = torch.zeros_like(fdist_mat)
        pred_corr_mat[src_indices, src_nn_indices] = 1.0
        recall = (pred_corr_mat * gt_corr_mat).sum() / num_gt_corr
        return recall

    def forward(self, data_dict, output_dict):
        assert data_dict["batch_size"] == 1, "Only support the batch_size of 1."

        # 1. unpack data
        img_points = output_dict["img_points_f"]  # (HxW, 3)
        img_feats = output_dict["img_feats_f"]  # (HxW, C)

        pcd_points = output_dict["pcd_points_f"]  # (N, 3)
        pcd_pixels = output_dict["pcd_pixels_f"]  # (N, 3)
        pcd_feats = output_dict["pcd_feats_f"]  # (N, C)

        transform = data_dict["transform"]  # (4, 4)
        img_corr_pixels = data_dict["img_corr_pixels"]
        pcd_corr_indices = data_dict["pcd_corr_indices"]

        image_w = data_dict["image_w"]

        pcd_points = apply_transform(pcd_points, transform)  # (N, 3)

        # 2. sample correspondences
        if pcd_corr_indices.shape[0] > self.max_correspondences:
            sel_indices = random_choice(pcd_corr_indices.shape[0], size=self.max_correspondences, replace=False)
            img_sel_pixels = img_corr_pixels[sel_indices]
            pcd_sel_indices = pcd_corr_indices[sel_indices]
        else:
            img_sel_pixels = img_corr_pixels
            pcd_sel_indices = pcd_corr_indices

        img_sel_v_coords = img_sel_pixels[:, 0]
        img_sel_u_coords = img_sel_pixels[:, 1]
        img_sel_indices = img_sel_v_coords * image_w + img_sel_u_coords
        img_sel_points = img_points[img_sel_indices]  # (M, 3)
        img_sel_pixels = img_sel_pixels.float()
        img_sel_feats = img_feats[img_sel_indices]  # (M, 3)

        pcd_sel_points = pcd_points[pcd_sel_indices]  # (M, C)
        pcd_sel_pixels = pcd_pixels[pcd_sel_indices]  # (M, C)
        pcd_sel_feats = pcd_feats[pcd_sel_indices]  # (M, C)

        dist3d_mat = pairwise_distance(img_sel_points, pcd_sel_points, squared=False, strict=True)
        dist2d_mat = pairwise_distance(img_sel_pixels, pcd_sel_pixels, squared=False, strict=True)
        pos_masks = torch.logical_and(
            torch.lt(dist3d_mat, self.pos_radius_3d),
            torch.lt(dist2d_mat, self.pos_radius_2d),
        )
        neg_masks = torch.logical_or(
            torch.gt(dist3d_mat, self.neg_radius_3d),
            torch.gt(dist2d_mat, self.neg_radius_2d),
        )
        fdist_mat = pairwise_distance(img_sel_feats, pcd_sel_feats, normalized=False)

        # 3. circle loss
        loss = self.circle_loss(pos_masks, neg_masks, fdist_mat)

        # 5. matching recall
        recall = self.get_recall(pos_masks.float(), fdist_mat)


        return loss, recall


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.c_loss = CoarseMatchingLoss(cfg)
        self.f_loss = FineMatchingLoss(cfg)
        self.weight_c_loss = cfg.loss.coarse_loss.weight
        self.weight_f_loss = cfg.loss.fine_loss.weight

    def forward(self, data_dict, output_dict):
        c_loss_circle, _, _, c_loss_matrix_gt_hat = self.c_loss(output_dict)
        f_loss, f_recall = self.f_loss(data_dict, output_dict)

    
        c_loss = (c_loss_circle + c_loss_matrix_gt_hat) * self.weight_c_loss
        f_loss = f_loss * self.weight_f_loss

        loss = c_loss + f_loss 


        return {"loss": loss, "c_loss_circle": c_loss_circle, "f_loss": f_loss, 'c_loss_matrix_gt_hat': c_loss_matrix_gt_hat, \
                'f_recall': f_recall}


class EvalFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse_matching(self, output_dict):
        img_length_c = output_dict["img_num_nodes"]
        pcd_length_c = output_dict["pcd_num_nodes"]
        gt_node_corr_min_overlaps = output_dict["gt_node_corr_min_overlaps"]
        gt_img_node_corr_indices = output_dict["gt_img_node_corr_indices"]
        gt_pcd_node_corr_indices = output_dict["gt_pcd_node_corr_indices"]
        img_node_corr_indices = output_dict["img_node_corr_indices"]
        pcd_node_corr_indices = output_dict["pcd_node_corr_indices"]

        masks = torch.gt(gt_node_corr_min_overlaps, self.acceptance_overlap)
        gt_img_node_corr_indices = gt_img_node_corr_indices[masks]
        gt_pcd_node_corr_indices = gt_pcd_node_corr_indices[masks]
        gt_node_corr_mat = torch.zeros(img_length_c, pcd_length_c).cuda()
        gt_node_corr_mat[gt_img_node_corr_indices, gt_pcd_node_corr_indices] = 1.0

        precision = gt_node_corr_mat[img_node_corr_indices, pcd_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine_matching(self, data_dict, output_dict):
        transform = data_dict["transform"]
        img_corr_points = output_dict["img_corr_points"]
        pcd_corr_points = output_dict["pcd_corr_points"]
        # only evaluate the correspondences with depth
        corr_masks = torch.gt(img_corr_points[..., -1], 0.0)
        img_corr_points = img_corr_points[corr_masks]
        pcd_corr_points = pcd_corr_points[corr_masks]
        pcd_corr_points = apply_transform(pcd_corr_points, transform)
        corr_distances = torch.linalg.norm(pcd_corr_points - img_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean().nan_to_num_()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, data_dict, output_dict):
        transform = data_dict["transform"]
        est_transform = output_dict["estimated_transform"]
        pcd_points = output_dict["pcd_points"]

        rre, rte = compute_isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.linalg.inv(transform), est_transform)
        realigned_pcd_points_f = apply_transform(pcd_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_pcd_points_f - pcd_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def forward(self, data_dict, output_dict):
        c_precision = self.evaluate_coarse_matching(output_dict)
        f_precision = self.evaluate_fine_matching(data_dict, output_dict)

        return {"PIR": c_precision, "IR": f_precision}
