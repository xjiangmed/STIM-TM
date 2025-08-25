'''
Adapted from https://github.com/facebookresearch/ToMe
'''

import math
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching_TIM_TM(
    metric: torch.Tensor,
    score_obj: torch.Tensor = None,
    r: int = 1,
    class_token: bool = False,
    distill_token: bool = False,
    merging_type: str = 'patch',
    frame_average: bool = False,
) -> Tuple[Callable, Callable]:
    """
    [B, N, T, C] -> [B, N, T-r, C]
    """
    assert merging_type == 'frame'
    
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    
    t = metric.shape[-2]  # dimension for reduction
    r = min(r, (t - protected) // 1)
    
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        # metric: [B, N, T, C]
        B, N, T, _ = metric.shape
        current_metric = metric.clone()
        if score_obj is None:
            current_score_obj = None
        else:
            current_score_obj = score_obj.clone()
        all_src_indices = []
        all_dst_indices = []
        all_max_sim_indices = []
        all_src_sos = []
        
        for i in range(r):
            # print('*****Iteration--', i)
            # Calculate the cosine similarity between adjacent frames
            _, _, T1, _ = current_metric.shape
            similarity_matrix = F.cosine_similarity(current_metric[:, :, :-1, :], current_metric[:, :, 1:, :], dim=-1) #[B, N, T-1]
            if frame_average:
                similarity_matrix = similarity_matrix.mean(1, keepdim=True).expand(-1, N, -1)
                
            # Select the frame indices with the top-1 similarity 
            _, max_similarity_indices = torch.max(similarity_matrix, dim=2, keepdim=True)
            
            # Calculate source and dst indices for compression
            src_indices = max_similarity_indices + 1
            dst_indices = torch.arange(T1 - 1).to(metric.device)[None, None, :].repeat(B, N, 1) #[B, N, T-1]
            dst_indices[dst_indices > max_similarity_indices] += 1
            
            src_so = None
            if current_score_obj is not None:
                C_s = current_score_obj.shape[-1]
                src_so = current_score_obj.gather(dim=2, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C_s))
                dst_so = current_score_obj.gather(dim=2, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C_s))
                # print('src_so.shape:', src_so.shape)
                
                #合并当前score_obj，准备下一次迭代
                dst_so.scatter_add_(dim=2, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C_s), src=src_so)
                current_score_obj = dst_so
            
            # 保存当前的合并索引
            all_src_indices.append(src_indices)
            all_dst_indices.append(dst_indices)
            all_max_sim_indices.append(max_similarity_indices)
            all_src_sos.append(src_so)
            
            # 合并当前metric，准备下一次迭代
            C_m = current_metric.shape[-1]
            src_metric = current_metric.gather(dim=2, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C_m))
            dst_metric = current_metric.gather(dim=2, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C_m))
            dst_metric.scatter_add_(dim=2, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C_m), src=src_metric)
            current_metric = dst_metric
           
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # x: [B, N, T, C]
        current_x = x.clone()
        C = x.shape[-1]
        for i in range(r):
            src_indices = all_src_indices[i]
            dst_indices = all_dst_indices[i]
            max_similarity_indices = all_max_sim_indices[i]
            src_so = all_src_sos[i]
            
            src = current_x.gather(dim=2, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C))
            dst = current_x.gather(dim=2, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C))
            if score_obj is not None:
                src = (src * src_so).to(dst.dtype) #对源令牌（source tokens）进行重要性加权
            
            # dst.scatter_add_(dim=2, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C), src=src)
            dst = dst.scatter_reduce(dim=2, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C), src=src, reduce=mode)
            
            current_x = dst
        
        return dst
    return merge, None 

