import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s): 
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class Randomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x, idx_swap):
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True)
        
        x = (x - mean) / (var + self.eps).sqrt()
        x = x[idx_swap].detach()
        x = x * (var + self.eps).sqrt() + mean
        return x

class SUGenerator(nn.Module):
    def __init__(self, P, Channel, z_dim, num_domains):
        super(SUGenerator, self).__init__()
        self.P = P
        self.Channel = Channel
        self.num_domains = num_domains
        # encoder
        self.fc1 = nn.Linear(Channel, 32 * P)
        self.bn1 = nn.BatchNorm1d(32 * P)

        self.fc2 = nn.Linear(32 * P, 16 * P)
        self.bn2 = nn.BatchNorm1d(16 * P)

        self.fc3 = nn.Linear(16 * P, 4 * P)
        self.bn3 = nn.BatchNorm1d(4 * P)

        self.fc4 = nn.Linear(4 * P, z_dim)
        self.fc5 = nn.Linear(4 * P, z_dim)

        self.fc9 = nn.Linear(Channel, 32 * P)
        self.bn9 = nn.BatchNorm1d(32 * P)

        self.fc10 = nn.Linear(32 * P, 16 * P)
        self.bn10 = nn.BatchNorm1d(16 * P)

        self.fc11 = nn.Linear(16 * P, 4 * P)
        self.bn11 = nn.BatchNorm1d(4 * P)

        self.fc12 = nn.Linear(4 * P, 4 * P)
        self.bn12 = nn.BatchNorm1d(4 * P)

        self.fc13 = nn.Linear(4 * P, 1 * P)

        ### decoder
        self.fc6 = nn.Linear(z_dim, P * 4)
        self.bn6 = nn.BatchNorm1d(P * 4)

        self.fc7 = nn.Linear(P * 4, P * 64)
        self.bn7 = nn.BatchNorm1d(P * 64)

        self.fc8 = nn.Linear(P * 64, Channel * P)

        self.adain2 = AdaIN2d(z_dim, z_dim)
        self.rand_sug = Randomization()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    def encoder_z(self, x):
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc2(h1)
        h1 = self.bn2(h1)
        h11 = F.leaky_relu(h1, 0.00)

        h1 = self.fc3(h11)
        h1 = self.bn3(h1)
        h1 = F.leaky_relu(h1, 0.00)

        mu = self.fc4(h1)
        log_var = self.fc5(h1)
        return mu, log_var

    def encoder_abu(self, x):
        h1 = self.fc9(x)
        h1 = self.bn9(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc10(h1)
        h1 = self.bn10(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc11(h1)
        h1 = self.bn11(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc12(h1)
        h1 = self.bn12(h1)

        h1 = F.leaky_relu(h1, 0.00)
        h1 = self.fc13(h1)

        a = F.softmax(h1, dim=1)
        return a

    def reparameterize(self, mu, log_var, device):
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=device)
        return mu + eps * std

    def decoder(self, z):
        h1 = self.fc6(z)
        h1 = self.bn6(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc7(h1)
        h1 = self.bn7(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc8(h1)
        em = torch.sigmoid(h1)
        return em

    def forward(self, inputs, device, gen=False):
        mu, log_var = self.encoder_z(inputs)
        abu = self.encoder_abu(inputs)
        z = self.reparameterize(mu, log_var, device)
        em = self.decoder(z)
        em_tensor = em.view([-1, self.P, self.Channel])
        abu_tensor = abu.view([-1, 1, self.P])
        img_hat = abu_tensor @ em_tensor
        img_hat = torch.squeeze(img_hat, dim=1)
        if gen:
            img_gen_all, EM_interp_all = [], []
            z_all, z_rand_all = [], []
            for i in range(self.num_domains):
                idx_swap = torch.randperm(mu.shape[0])
                z_tmp_ = self.rand_sug(z, idx_swap).reshape(-1, mu.shape[1])
                EM_interp = self.decoder(z_tmp_)
                EM_interp = torch.reshape(EM_interp, [-1, self.P, self.Channel])
                img_gen_all.append((abu_tensor @ EM_interp).squeeze(1))
                EM_interp_all.append((EM_interp).squeeze(1))
                z_rand_all.append(z_tmp_)
                z_all.append(z)
            return em_tensor, img_gen_all, EM_interp_all
        else:
            return img_hat, mu, log_var, abu, em_tensor