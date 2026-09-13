import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)
    
class Project(nn.Module):
    def __init__(self, n_inputs, n_outputs, mlp_width) -> None:
        super().__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.output = nn.Linear(mlp_width, n_outputs)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.output(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = None
    def forward(self, x, use_attn=True):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        self.attn = attn
        attn = self.attn_drop(attn)
        x = ( attn @ v ) if use_attn else v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_dim=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        xa_attn = self.attn(self.norm1(x))
        xa = x + self.drop_path(xa_attn)
        xa = xa + self.drop_path(self.mlp(self.norm2(xa)))

        return xa

class Featurizer(nn.Module):
    def __init__(self, n_bands=30, in_channels=1, num_classes=16, num_tokens=4, dim=64, depth=1, heads=8,  
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.):
        super(Featurizer, self).__init__()
        self.L = num_tokens
        self.fea_dim = dim
        self.token_len = num_tokens
        self.conv_a = nn.Conv2d(dim, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*n_bands, out_channels=dim,
                      kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(depth)])
        
        self.norm = nn.LayerNorm(dim)
        
        self.to_cls_token = nn.Identity()

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens
    
    def _tokenize(self, x):
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        T = self._forward_semantic_tokens(x)
        return T
    
    def forward(self, x):
        T = self._tokenize(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)
        return self.to_cls_token(x[:, 0])

class PrependPrompt():
    def __init__(self, network, domain_token, em_token):
        self.featurizer = network
        self.domain_tokens = domain_token
        self.em_tokens = em_token
        self.pos_embeddings = self.featurizer.pos_embedding
        
    def add_domain_prompt(self):
        domain_tokens = self.domain_tokens
        em_tokens = self.em_tokens
        def _add_domain_prompt(model, x):
            act = x[0]
            x_new = torch.cat([act, domain_tokens, em_tokens], dim=1)
            return (x_new, )
        return _add_domain_prompt

    def __enter__(self):
        self.hook = self.featurizer.dropout.register_forward_pre_hook(self.add_domain_prompt())
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

class SUPnet(nn.Module):
    def __init__(self, num_domains, opts):
        super().__init__()
        self.featurizer = Featurizer(n_bands=opts.num_bands, num_classes=opts.num_classes, 
                                    num_tokens=opts.num_tokens, dim=opts.dim, depth=opts.depth)
        self.classifier = Classifier(self.featurizer.fea_dim, opts.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.opts = opts
        self.num_domains = num_domains + 1
        self.hidden_dim = self.featurizer.fea_dim
        self.cls_prompt_dim = opts.num_cls_prompt
        self.em_prompt_dim = opts.num_em_prompt
        self.mlp_dim = 1024
        self.dev = int(opts.device)

        # prompt tokens
        self.prompt_tokens = nn.Parameter(
            torch.empty(self.num_domains, self.cls_prompt_dim, self.featurizer.fea_dim).normal_(std=0.02)
        )
        self.EM_prompt_tokens = nn.Parameter(
            torch.empty(self.num_domains, self.em_prompt_dim, self.featurizer.fea_dim).normal_(std=0.02)
        )
        num_em_dim = self.featurizer.fea_dim*self.em_prompt_dim*opts.num_em
        self.EM_prompt_pro = Project(num_em_dim, self.featurizer.fea_dim*self.em_prompt_dim, num_em_dim//2) 

        # prompt adapter
        self.project = Project(self.hidden_dim, self.num_domains * self.cls_prompt_dim, self.mlp_dim) 
        self.EM_project = Project(self.hidden_dim, self.num_domains * self.em_prompt_dim, self.mlp_dim) 
        
        # optimizer
        if opts.optim == 'Adam':
            self.prompt_opt = torch.optim.Adam(
                [self.prompt_tokens, self.EM_prompt_tokens],
                lr=opts.lr_prompt,
                weight_decay=opts.l2_decay
            )
            self.project_opt = torch.optim.Adam(
                self.project.parameters(),
                lr=opts.lr_project
            )
            self.EM_project_opt = torch.optim.Adam(
                self.EM_project.parameters(),
                lr=opts.lr_project
            )
            self.optimizer = torch.optim.Adam(
                [
                    {'params': self.featurizer.parameters(), 'lr': opts.lr},
                    {'params': self.classifier.parameters(), 'lr': opts.lr_cls},
                    {'params': self.EM_prompt_pro.parameters(), 'lr': opts.lr_prompt},
                    {'params': self.prompt_tokens, 'lr': opts.lr_prompt},
                    {'params': self.EM_prompt_tokens, 'lr': opts.lr_prompt}
                ]
            )
        elif opts.optim == 'AdamW':
            self.project_opt = torch.optim.AdamW(
                self.project.parameters(),
                lr=opts.lr_project
            )
            self.EM_project_opt = torch.optim.AdamW(
                self.EM_project.parameters(),
                lr=opts.lr_project
            )
            self.optimizer = torch.optim.AdamW(
                [
                    {'params': self.featurizer.parameters(), 'lr': opts.lr},
                    {'params': self.classifier.parameters(), 'lr': opts.lr_cls},
                    {'params': self.EM_prompt_pro.parameters(), 'lr': opts.lr_prompt},
                    {'params': self.prompt_tokens, 'lr': opts.lr_prompt},
                    {'params': self.EM_prompt_tokens, 'lr': opts.lr_prompt}
                ]
            )
        elif opts.optim == 'SGD':
            self.project_opt = torch.optim.SGD(
                self.project.parameters(),
                lr=opts.lr_project,
                weight_decay=opts.l2_decay,
            )
            self.EM_project_opt = torch.optim.SGD(
                self.EM_project.parameters(),
                lr=opts.lr_project,
                weight_decay=opts.l2_decay,
            )
            self.optimizer = torch.optim.SGD(
                [
                    {'params': self.featurizer.parameters(), 'lr': opts.lr},
                    {'params': self.classifier.parameters(), 'lr': opts.lr_cls},
                    {'params': self.EM_prompt_pro.parameters(), 'lr': opts.lr_prompt},
                    {'params': self.prompt_tokens, 'lr': opts.lr_prompt},
                    {'params': self.EM_prompt_tokens, 'lr': opts.lr_prompt}
                ]
            )
        
    def x_domain_prompt_comb(self, all_bias, em=False):
        if em:
            prompt_tokens = self.EM_prompt_tokens
        else:
            prompt_tokens = self.prompt_tokens
        tokens = prompt_tokens[None, ...]
        bias = all_bias[..., None]
        comb_prompt = tokens * bias
        comb_prompt = comb_prompt.sum(dim=1)
        return comb_prompt

    def all_bias_gt(self, disc_labels, all_y, all_bias, 
                    in_domain_weight=0.8, out_domain_base=0.5, distribution_factor=0.3):
        batch_size = all_bias.shape[0]
        num_domains = all_bias.shape[1]
        
        gt = torch.zeros(all_bias.shape, dtype=torch.float32, device=self.dev)
        
        class_domain_count = torch.zeros((all_y.max()+1, num_domains), device=self.dev)
        for i in range(batch_size):
            class_domain_count[all_y[i], disc_labels[i]] += 1
        
        for i in range(batch_size):
            current_class = all_y[i]
            current_domain = disc_labels[i]
            gt[i, current_domain, :] = in_domain_weight
            other_domains = [d for d in range(num_domains) if d != current_domain]
            other_domain_weights = class_domain_count[current_class, other_domains]
            if other_domain_weights.sum() > 0:
                other_domain_weights = other_domain_weights / other_domain_weights.sum()
                for j, domain in enumerate(other_domains):
                    gt[i, domain, :] = out_domain_base + distribution_factor * other_domain_weights[j]
        return gt

    def forward_prompt(self, all_x, domain_labels, em_labels):
        domain_prompts = self.prompt_tokens[domain_labels]
        tmp_bs = domain_prompts.shape[0]
        em_prompts = self.EM_prompt_tokens[em_labels]
        em_prompts = self.EM_prompt_pro(em_prompts.view(tmp_bs,-1)).view(tmp_bs,self.em_prompt_dim,-1)
        with PrependPrompt(self.featurizer, domain_prompts, em_prompts):
            all_logit = self.network(all_x)
        return all_logit
    
    def forward_second(self, all_x, all_z, all_z_em):
        hint = all_z.detach()
        all_bias = self.project(hint).reshape(-1, self.num_domains, self.cls_prompt_dim)
        all_bias = F.softmax(all_bias, dim=1)
        domain_prompts = self.x_domain_prompt_comb(all_bias)
        tmp_bs = domain_prompts.shape[0]
        em_hint = all_z_em.detach()
        em_all_bias = self.EM_project(em_hint).reshape(-1, self.num_domains, self.em_prompt_dim)
        em_all_bias = F.softmax(em_all_bias, dim=1)
        em_prompts = self.x_domain_prompt_comb(em_all_bias, em=True)
        em_prompts = self.EM_prompt_pro(em_prompts.view(tmp_bs,-1)).view(tmp_bs,self.em_prompt_dim,-1)
        with PrependPrompt(self.featurizer, domain_prompts, em_prompts):
            all_logit = self.network(all_x)
            return all_logit, all_bias, em_all_bias
    
    @torch.no_grad()
    def forward_raw(self, all_x):
        all_z = self.featurizer(all_x)
        return all_z
        
    def update(self, all_x, all_y, all_em, domain_labels, em_labels, opts, all_em_y):
        self.optimizer.zero_grad()
        self.project_opt.zero_grad()
        self.EM_project_opt.zero_grad()
        
        # domain prompt learning
        all_logit = self.forward_prompt(all_x, domain_labels, em_labels)
        loss_cls = F.cross_entropy(all_logit, all_y)
        loss_cls.backward(retain_graph=True)
        
        # prompt adapter learning
        self.network.eval()
        all_z = self.forward_raw(all_x)
        all_z_em = self.forward_raw(all_em.unsqueeze(1))
        self.network.train()
        
        all_logit, all_bias, em_all_bias = self.forward_second(all_x, all_z, all_z_em)
        gt_bias = self.all_bias_gt(domain_labels, all_y, all_bias)
        em_gt_bias = self.all_bias_gt(em_labels, all_em_y, em_all_bias)
        loss_prompt = F.cross_entropy(all_logit, all_y)
        loss_w = F.binary_cross_entropy(all_bias, gt_bias)
        loss_w_em = F.binary_cross_entropy(em_all_bias, em_gt_bias)
        loss_MPL = opts.lambda_2*(loss_prompt + loss_w + loss_w_em)
        loss_MPL.backward(retain_graph=True)

        self.optimizer.step()
        self.project_opt.step()
        self.EM_project_opt.step()

        return loss_cls.item(), opts.lambda_2*loss_prompt.item(), opts.lambda_2*(loss_w+loss_w_em).item()
    
    def predict(self, x, em):
        all_z = self.forward_raw(x)
        all_z_em = self.forward_raw(em.unsqueeze(1)) 
        all_logit, _, _ = self.forward_second(x, all_z, all_z_em)
        return all_logit


