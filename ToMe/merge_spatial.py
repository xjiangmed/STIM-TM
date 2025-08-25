'''
Adapted from https://github.com/facebookresearch/ToMe
'''

import math
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


def do_nothing(x, mode=None):
    return x

#######################################################################################################
def cal_depth_score(sim_scores):
    left_peak = torch.cummax(sim_scores, dim=1).values
    right_peak = torch.flip(
        torch.cummax(torch.flip(sim_scores, dims=[1]), dim=1).values,
        dims=[1]
    )
    depth_scores = left_peak + right_peak - 2 * sim_scores
    return depth_scores
 
def temporal_segmentation(features, alpha=0.5, k=None):
    b, t, n, d = features.shape
    
    
    sim_scores = F.cosine_similarity(features[:, :-1, :, :], features[:, 1:, :, :], dim=-1)
    sim_scores = sim_scores.mean(dim=2) #[b, t-1]
    depth_scores = cal_depth_score(sim_scores)
    
    last_pos = t - 1 
    
    if k==0:
        boundaries = [ 
            [last_pos] for bidx in range(b)
        ]
    
        return boundaries 
        
    
    if k is not None:
        _, topk_indices = torch.topk(depth_scores, k, dim=1)
        sorted_indices = topk_indices.sort(dim=1).values
        boundaries = sorted_indices.tolist()
    else:
        mean = depth_scores.mean(dim=1, keepdim=True)  # [b, 1]
        std = depth_scores.std(dim=1, keepdim=True)    # [b, 1]
        thresholds = mean + alpha * std                # [b, 1]
        mask = depth_scores > thresholds
        
        candidates = [row.nonzero().squeeze(-1).tolist() for row in mask]
        
        boundaries = []
        for i, cand in enumerate(candidates):
            if len(cand) > 15:
                _, indices = torch.topk(depth_scores[i], 15)
                boundaries.append(indices.sort().values.tolist())
            else:
                boundaries.append(cand)

    boundaries = [ 
        bnd + [last_pos] if not bnd or bnd[-1] != last_pos else bnd
        for bnd in boundaries
    ]
    
    return boundaries 

def bipartite_soft_matching_SIM_TM(
    x: torch.Tensor,
    size: torch.Tensor,
    source: torch.Tensor,
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    merging_type: str = 'patch',
    saliency_aware: bool = False, 
    segment_num: int = 1,
) -> Tuple[Callable, Callable]:
    
    assert merging_type == 'patch'
    
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    
    t = metric.shape[-2]  
    r = min(r, (t - protected) // 2)
    
    if r <= 0:
        return do_nothing, do_nothing
    
    def merging(data, unm_idx, src_idx, dst_idx, src_so, mask, mode="mean"):
        dominant_tokens = data.masked_select(~mask.expand(-1,-1,-1,data.shape[3])).view(
            data.shape[0], data.shape[1], -1, data.shape[3])
        data_filtered = data.masked_select(mask.expand(-1,-1,-1,data.shape[3])).view(
                data.shape[0], data.shape[1], -1, data.shape[3]) 
        
        src, dst = data_filtered[..., ::2, :], data_filtered[..., 1::2, :]
        B, n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(B, n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(B, n, r, c))
        
        if src_so is not None:
            src = (src * src_so).to(dst.dtype) 
            
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, n, r, c), src, reduce=mode)
        if distill_token:
            return torch.cat([dominant_tokens[:,:1], unm[:, :1], dst[:, :1], dominant_tokens[:,1:], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([dominant_tokens, unm, dst], dim=-2)
    
    def tome_merging(m_, x_, size_, source_, score_obj_=None, static_score=None, dominant_ratio= 0.65):
        assert class_token == True
        
        candidate_num = 2*r # inference-time
        # candidate_num = 4*r # training w/ token merging
        
        if class_token:
            dominant_num = int((m_.shape[-2]-1)-candidate_num)
        else:
            dominant_num = int(m_.shape[-2]-candidate_num)
        
        static_score_wocls = static_score[:,:,1:,:]
        topk_indices = static_score_wocls.topk(dominant_num, dim=2, largest=False).indices+1 #smallest: low similarity-dynamic tokens
        
        
        all_indices = torch.cat([
            torch.zeros((topk_indices.shape[0], topk_indices.shape[1], 1, topk_indices.shape[3]), dtype=topk_indices.dtype, device=topk_indices.device), 
            topk_indices
        ], dim=2).squeeze(-1)
        
        # 创建掩码：标记需要保留的令牌位置为False
        mask = torch.ones_like(m_[:, :, :, 0], dtype=torch.bool, device=m_.device).scatter_(2, all_indices, False).unsqueeze(-1)
        
        ### Filter: 过滤已选令牌，处理剩余令牌
        m_filtered = m_[mask.expand(-1,-1,-1,m_.shape[3])].view(
            m_.shape[0], m_.shape[1], -1, m_.shape[3])
        
        # 归一化剩余令牌的metric（用于相似度计算）
        m_filtered = m_filtered / m_filtered.norm(dim=-1, keepdim=True) 
        a, b = m_filtered[..., ::2, :], m_filtered[..., 1::2, :] 
        scores = a @ b.transpose(-1, -2)  # |Set A| * |Set B| edges [B, T, N/2, N/2]
       
        if merging_type == 'patch' and class_token:
            scores[..., 0, :] = -math.inf
        if merging_type == 'patch' and distill_token:
            scores[..., :, 0] = -math.inf
            
        node_max, node_idx = scores.max(dim=-1)  # keep edge with the highest sim for every node in Set A
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # sort |Set A| edges based on sim

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # node_idx: idx for Set B
        
        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=-2)[0]
        
        src_so = None
        if score_obj_ is not None:
            src_so, dst_so = score_obj_[..., ::2, :], score_obj_[..., 1::2, :] 
            src_so = src_so.gather(dim=-2, index=src_idx) 
            
        merged_x_ = merging(x_, unm_idx, src_idx, dst_idx, src_so, mask, mode="sum")
        merged_size_ = merging(size_, unm_idx, src_idx, dst_idx, src_so, mask, mode="sum")
        if source_ is not None:
            merged_source_ = merging(source_, unm_idx, src_idx, dst_idx, src_so, mask, mode="amax")
        else:
            merged_source_ = None
        return merged_x_, merged_size_, merged_source_
    
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        B, T, N, C = metric.shape
        # 划分segments
        # boundaries = temporal_segmentation(metric) #threshold-based
        boundaries = temporal_segmentation(metric, k=segment_num-1)
        
        merged_xs, merged_sizes, merged_sources = [], [], []
        for b in range(B): 
            start_idx = 0
            # 在每个segment内，优先合并静态的区域
            segment_merged_xs, segment_merged_sizes, segment_merged_sources = [], [], []
            for bidx in boundaries[b]: 
                
                segment_score_obj = None
                
                segment_metric = metric[b:b+1, start_idx:bidx+1, :, :] #[B, W, N, C]
                segment_x = x[b:b+1, start_idx:bidx+1, :, :]
                segment_size = size[b:b+1, start_idx:bidx+1, :, :]
                if source is not None:
                    segment_source = source[b:b+1, start_idx:bidx+1, :, :]
                else:
                    segment_source = None
                
                start_idx = bidx+1
                
                # 时间冗余（全局一致性）：计算每个token的帧间相似度 
                window_size = segment_metric.shape[1]
                frames_normed = F.normalize(segment_metric, p=2, dim=-1)
                if window_size==1:
                    #局部帧间差异
                    valid_pairs = []
                    if start_idx-1>=0:
                        left_neighbor = F.normalize(metric[b:b+1, start_idx-1:start_idx, :, :], p=2, dim=-1)
                        left_local_sim = einsum('b w n c, b w n c -> b w n', left_neighbor, frames_normed)
                        valid_pairs.append(left_local_sim)
                    if bidx+1<metric.shape[1]:
                        right_neighbor = F.normalize(metric[b:b+1, bidx+1:bidx+2, :, :], p=2, dim=-1)
                        right_local_sim = einsum('b w n c, b w n c -> b w n', frames_normed, right_neighbor)
                        valid_pairs.append(right_local_sim)
                    frames_sim = sum(valid_pairs) / len(valid_pairs) # B 1 N
                    frames_sim = frames_sim.unsqueeze(-1) #[B, 1, N, 1] 
                    frames_sim = frames_sim.repeat(1, window_size, 1, 1) #[B, W, N, 1]
                else:
                    #段内全局冗余
                    frames_sim = einsum('b w n c, b t n c -> b w t n', frames_normed, frames_normed)
                    frames_sim = (frames_sim.sum(dim=-2) - 1).sum(dim=-2) / (window_size*(window_size-1)) # B N
                    frames_sim = frames_sim.unsqueeze(1).unsqueeze(-1) #[B, 1, N, 1]
                    frames_sim = frames_sim.repeat(1, window_size, 1, 1) #[B, W, N, 1]
                    
                ###################################
                # # !!!!! task importance
                # caam = segment_x.abs().sum(-1) # (B, W, N)
                # cam_min = caam.min(dim=-1, keepdim=True)[0]
                # cam_max = caam.max(dim=-1, keepdim=True)[0]
                # caam = (caam - cam_min)/(cam_max - cam_min) # 归一化到[0,1]
                # caam = caam.unsqueeze(-1)
                # frames_sim = frames_sim*(1-caam)
                ###################################
                
                curr_merged_x, curr_merged_size, curr_merged_source = \
                    tome_merging(segment_metric, segment_x, segment_size, segment_source, segment_score_obj, static_score=frames_sim)
                
                segment_merged_xs.append(curr_merged_x)
                segment_merged_sizes.append(curr_merged_size)
                if source is not None:
                    segment_merged_sources.append(curr_merged_source)
            
            segment_merged_xs = torch.cat(segment_merged_xs, dim=1)
            segment_merged_sizes = torch.cat(segment_merged_sizes, dim=1)
            if source is not None:
                segment_merged_sources = torch.cat(segment_merged_sources, dim=1)
            
            merged_xs.append(segment_merged_xs)
            merged_sizes.append(segment_merged_sizes)
            merged_sources.append(segment_merged_sources)
            
        
        merged_xs = torch.cat(merged_xs, dim=0)
        merged_sizes = torch.cat(merged_sizes, dim=0)
        if source is not None:
            merged_sources = torch.cat(merged_sources, dim=0)        
        else:
            merged_sources = None
              
    return merged_xs, merged_sizes, merged_sources
    
    
    
    
    