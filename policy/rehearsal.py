"""Independent OPT rehearsal operations; Torch is the only dependency.
Mathematical provenance: local InfiniGen performance reference and HillServe paper.
The external reference is used only by the optional validation path.
"""
import torch
import torch.nn.functional as F

def reform_hidden_states(x):
    return torch.cat([x,torch.ones((*x.shape[:-1],1),device=x.device,dtype=x.dtype)],dim=-1)

def weight_bias_concat(w,b,scaling=False,head_dim=1):
    joined=torch.cat([w,b.to(w.device).reshape(-1,1)],dim=1)
    return joined*(head_dim**-.5) if scaling else joined

def skew(query,key,wq,wk,n_head,head_dim):
    result_q=wq.clone();result_k=wk.clone()
    for head in range(n_head):
        qsvd=torch.svd(query[0,:,head].float());ksvd=torch.svd(key[0,:,head].float())
        importance=qsvd.S.half()*ksvd.S.half()
        # Reference scatters basis columns into sorted-index positions.
        # Inverse permutation expresses the same transform without scatter.
        inverse=torch.argsort(torch.argsort(importance))
        basis=qsvd.V.half().index_select(1,inverse)
        section=slice(head*head_dim,(head+1)*head_dim)
        result_q[section]=basis.T@wq[section]
        result_k[section]=basis.T@wk[section]
    return result_q,result_k

def partial_weight_index_generation(query,n_head,head_dim,partial_weight_ratio):
    shaped=query[0].reshape(-1,n_head,head_dim)
    # A separate per-head reduction preserves the reference's F16 summation order.
    indices=[torch.topk(shaped[:,head,:].abs().sum(0),int(head_dim*partial_weight_ratio)).indices for head in range(n_head)]
    return torch.stack(indices).unsqueeze(0).expand(query.shape[0],-1,-1).contiguous()

def set_partial_weight(w,index,n_head,head_dim):
    rows=index[0]+torch.arange(n_head,device=w.device).reshape(-1,1)*head_dim
    return w.index_select(0,rows.flatten())

def set_partial_cache(keys,index,n_head,head_dim):
    n,bh,_=keys.shape
    shaped=keys.reshape(n,bh//n_head,n_head,head_dim)
    gather_index=index.unsqueeze(0).expand(n,-1,-1,-1)
    return shaped.gather(-1,gather_index).reshape(n,bh,-1)

def speculate_attention(hidden,p_w_q,p_k_c,n_head,alpha,max_num_kv):
    q=F.linear(hidden,p_w_q).reshape(hidden.shape[0],1,n_head,-1).permute(0,2,1,3).flatten(0,1)
    scores=torch.bmm(q,p_k_c.permute(1,2,0))
    threshold=scores.amax(-1,keepdim=True)-alpha
    counts=(scores>threshold).to(scores.dtype).sum(-1)
    keep=min(int(counts.mean().item()),max_num_kv)
    return torch.topk(scores.permute(2,1,0),keep,dim=0).indices

def select_kv(ids,k,v):
    per_head=ids[:,0,:]
    flat=per_head*k.shape[1]+torch.arange(k.shape[1],device=ids.device)
    shape=(*per_head.shape,k.shape[-1])
    return k.flatten(0,1).index_select(0,flat.flatten()).reshape(shape),v.flatten(0,1).index_select(0,flat.flatten()).reshape(shape)
