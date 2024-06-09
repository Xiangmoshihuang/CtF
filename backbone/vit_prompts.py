import torch
import torch.nn as nn
import copy

class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_pool, prompt_length, ortho_weight,
                 prompt_mode='prompt-tune', key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.prompt_mode = prompt_mode
        self._init_smart(emb_d, prompt_pool, prompt_length, ortho_weight)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d, prompt_pool, prompt_length, ortho_weight):
        assert self.prompt_mode == 'prompt-tune', "prompt_mode should be 'prompt-tune' in CODA-Prompt!"

        # prompt basic param
        self.e_pool_size = int(prompt_pool)
        self.e_p_length = int(prompt_length)
        self.e_layers = [0,1,2,3,4]

        # strenth of ortho penalty
        self.ortho_mu = ortho_weight
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks)) # how many prompt component that a task equipt
            s = int(self.task_count * pt)   # prompt component for 0 - T-1
            f = int((self.task_count + 1) * pt) # prompt component for 0 - T
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, e_prompt_pool, e_prompt_length, g_prompt_length, 
                 prompt_mode='prompt-tune', key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.prompt_mode = prompt_mode
        
        assert self.prompt_mode == 'prompt-tune', "prompt_mode should be 'prompt-tune' in Dual-Prompt!"
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [2,3,4]

        # prompt pool size
        self.g_p_length = g_prompt_length
        self.e_p_length = e_prompt_length
        self.e_pool_size = e_prompt_pool

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)


    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:,task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:,k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]
                
            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length/2)
                Ek = P_[:, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, i:, :].reshape((B, -1, self.emb_d))
            else:
                i = int(self.e_p_length/2)
                Ek = P_[:, :, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, :, i:, :].reshape((B, -1, self.emb_d))
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_pool, prompt_length, prompt_mode,
                 top_k=None, layer_ids:list=None, key_dim=768):
        super().__init__()
        assert prompt_mode in ['prompt_tune', 'prefix_tune'], "prompt_mode should be 'prompt_tune' or 'prefix_tune' in L2P !"
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.prompt_mode = prompt_mode
        self.top_k = 5 if top_k is None else  top_k
        
        # prompt locations
        if layer_ids is None:
            self.layer_ids = [0]
        else:
            self.layer_ids = layer_ids

        # prompt pool size
        self.p_length = int(prompt_length)
        self.pool_size = int(prompt_pool)

        # e prompt init
        for e in self.layer_ids:
            p = tensor_prompt(self.pool_size, self.p_length, emb_d)
            k = tensor_prompt(self.pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
    
    def process_task_count(self):
        self.task_count += 1
    
    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        p_return = None
        loss = 0
        # e prompts
        if l in self.layer_ids:
            B, C = x_querry.shape
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                loss = (1.0 - cos_sim[:,k_idx]).sum()
                P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]
                
            # select prompts
            if self.prompt_mode == 'prompt_tune':
                i = int(self.p_length/2)
                Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))
                # combine prompts for prompt tuning
                p_return = [Ek, Ev]
            elif self.prompt_mode == 'prefix_tune':
                p_return = P_.reshape(B, -1, self.emb_d)

        return p_return, loss, x_block

# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p

class MoEPrompt(nn.Module):
    def __init__(self, mode, input_dim, prompt_dim, prompt_length, prompt_pool_size, topk=None, expert_mid=None):
        super().__init__()
        self.mode = mode
        self.topk = topk

        self.gate_net = GateNet_normal(mode, input_dim, prompt_length, self.topk)
        self.expert_nets = ExpertNet(mode, input_dim, prompt_dim, prompt_length, prompt_pool_size, expert_mid)
    
    def process_task_count(self):
        self.expert_nets.add_task_count()
        self.gate_net.add_task_count()

    def forward(self, img_feature, train=False, task_id=None):
        gate_out = self.gate_net(img_feature.detach())
        gate_topk = torch.topk(gate_out, self.topk)
        expert_out = self.expert_nets(img_feature.detach(), gate_topk)
        return expert_out, 0, img_feature

class ExpertNet(nn.Module):
    def __init__(self, mode, input_dim, output_dim, prompt_length, mid_dim) -> None:
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prompt_length = prompt_length
        self.mid_dim = mid_dim

        self.task_count = 0

        self.w_a_list = nn.ParameterList()
        self.w_b_list = nn.ParameterList()

        self.w_a_list.append(nn.Parameter(torch.FloatTensor(self.input_dim, self.prompt_length, self.mid_dim), requires_grad=True))
        self.w_b_list.append(nn.Parameter(torch.FloatTensor(self.prompt_length, self.mid_dim, self.output_dim), requires_grad=True))
    
    def add_task_count(self):
        if 'freeze_old_experts' in self.mode:
            self.w_a_list[-1].requires_grad = False
            self.w_b_list[-1].requires_grad = False
        self.w_a_list.append(nn.Parameter(torch.FloatTensor(self.input_dim, self.prompt_length, self.mid_dim), requires_grad=True))
        self.w_b_list.append(nn.Parameter(torch.FloatTensor(self.prompt_length, self.mid_dim, self.output_dim), requires_grad=True))

    def forward(self, x, idxs=None):
        # x: [batch_size, input_dim]
        # w_a: [input_dim, prompt_pool, mid_dim]
        # w_b: [prompt_pool, mid_dim, output_dim]
        w_a = torch.cat(self.w_a_list, dim=1)
        w_b = torch.cat(self.w_b_list, dim=0)
        # x_mid: [batch_size, prompt_pool*mid_dim]
        x_mid = x @ w_a.reshape(w_a.shape[0], -1)
        if 'MLP_expert' in self.mode:
            x_mid = torch.relu(x_mid)
        # x_mid: [batch_size, prompt_pool, mid_dim]
        x_mid = x_mid.reshape(x_mid.shape[0], w_b.shape[0], -1)
        # out: [prompt_pool, batch_size, output_dim]
        out = x_mid.permute(1, 0, 2) @ w_b
        return out.permute(1, 0, 2) # [batch_size, prompt_pool, output_dim]

class GateNet_normal(nn.Module):
    def __init__(self, mode, input_dim, prompt_length, topk) -> None:
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.prompt_length = prompt_length
        self.topk = topk
        self.prototype_list = nn.ParameterList()

        self.prototype_list.append(nn.Parameter(torch.FloatTensor(self.prompt_length), requires_grad=True))
    
    def add_task_count(self):
        if 'freeze_old_prototype' in self.mode:
            self.prototype_list[-1].requires_grad = False
        self.prototype_list.append(nn.Parameter(torch.FloatTensor(self.prompt_length), requires_grad=True))
    
    def forward(self, x):
        # x: [batch_size, input_dim]
        x = nn.functional.normalize(x, dim=-1)
        
        # prototype: [prompt_pool, input_dim]
        prototype = nn.functional.normalize(torch.stack(self.prototype_list, dim=0), dim=-1)

        # out: [batch_size, prompt_pool]
        out = x @ prototype.T
        return out

class GateNet_transform(nn.Module):
    def __init__(self, mode, input_dim, prompt_length, mid_dim) -> None:
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.prompt_length = prompt_length
        self.mid_dim = mid_dim
        self.transform_weight_list = nn.ParameterList()
        self.transform_bias_list = nn.ParameterList()
        self.prototype_list = nn.ParameterList()

        self.transform_weight_list.append(nn.Parameter(torch.FloatTensor(self.input_dim, self.mid_dim), requires_grad=True))
        self.transform_bias_list.append(nn.Parameter(torch.FloatTensor(self.mid_dim), requires_grad=True))
        self.prototype_list.append(nn.Parameter(torch.FloatTensor(self.prompt_length, self.mid_dim), requires_grad=True))
    
    def add_task_count(self):
        if 'freeze_old_prototype' in self.mode:
            self.prototype_list[-1].requires_grad = False
        if 'freeze_old_transform' in self.mode:
            self.transform_weight_list[-1].requires_grad = False
            self.transform_bias_list[-1].requires_grad = False
        self.transform_weight_list.append(nn.Parameter(torch.FloatTensor(self.input_dim, self.mid_dim), requires_grad=True))
        self.transform_bias_list.append(nn.Parameter(torch.FloatTensor(self.mid_dim), requires_grad=True))
        self.prototype_list.append(nn.Parameter(torch.FloatTensor(self.prompt_length, self.mid_dim), requires_grad=True))
    
    def forward(self, x):
        # transform_weight: [input_dim, task_num * mid_dim]
        # transform_bias: [task_num * mid_dim]
        transform_weight = torch.cat(self.transform_weight_list, dim=1)
        transform_bias = torch.cat(self.transform_bias_list, dim=0)
        
        # x: [batch_size, input_dim]
        # transformed_x: [batch_size, task_num * mid_dim]
        transformed_x = x @ transform_weight + transform_bias
        # transformed_x: [batch_size, task_num, mid_dim]
        transformed_x = transformed_x.reshape(transformed_x.shape[0], -1, self.mid_dim)
        transformed_x = nn.functional.normalize(transformed_x, dim=-1)
        
        # prototype: [prompt_pool, mid_dim]
        prototype = nn.functional.normalize(torch.cat(self.prototype_list, dim=0), dim=-1)

        # out: [batch_size, task_num, prompt_pool]
        out = transformed_x @ prototype.T
        return out