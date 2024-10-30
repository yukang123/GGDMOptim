import math
import torch
from torch import nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from unet_1d import Unet1D

# gaussian diffusion trainer class
def extract(a, t, x_shape):
    return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start, beta_end = scale * 0.0001, scale * 0.02

    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


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


class GaussianDiffusion(nn.Module):
    '''
    Adapted from https://github.com/openai/improved-diffusion/blob/1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5/improved_diffusion/gaussian_diffusion.py#L101
    '''
    def __init__(
            self, model, image_size, timesteps=1000, beta_schedule='cosine', 
            ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.num_timesteps = timesteps
        self.denoise_timesteps = range(self.num_timesteps - 1, -1, -1)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))


    def p_mean_variance(self, x, t, classes=None):
        pred_noise = self.model(x, t, classes)
        x_start = extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - extract(self.sqrt_recipm1_alphas_cumprod,
                                                                                    t, x.shape) * pred_noise
        posterior_mean = extract(self.posterior_mean_coef1, t, x.shape) * x_start + extract(self.posterior_mean_coef2,
                                                                                            t, x.shape) * x
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x.shape)

        return posterior_mean, posterior_log_variance_clipped

    # @torch.no_grad()
    def p_sample(self, x, t, classes=None):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, model_log_variance = self.p_mean_variance(x=x, t=batched_times, classes=classes)
        noise = torch.randn_like(x) if t > 0 else 0.

        return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, classes=None, num_samples=None):
        # DDPM Sampler
        assert (classes is not None or num_samples is not None)
        N = classes.shape[0] if classes is not None else num_samples
        img = torch.randn((N, self.image_size), device=self.betas.device)
        for t in tqdm(range(self.num_timesteps - 1, -1, -1), desc='sampling'):
            img = self.p_sample(img, t, classes)
        return img

    def p_losses(self, x_start, t, classes=None):
        noise = torch.randn_like(x_start)
        x = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

        return F.mse_loss(self.model(x, t, classes), noise)

    def forward(self, img, classes=None):
        t = torch.randint(0, self.num_timesteps, (img.shape[0],), device=img.device).long()

        return self.p_losses(img, t, classes)
    

class GuidedGaussianDiffusion(GaussianDiffusion):
    def __init__(
            self, 
            sigma = 0.1, 
            beta = None, 
            beta_coef = 1, 
            cond_score_version = "v1", # v1: naive gradients G, v2: gradient guidance G_loss
            **kwargs
            ):
        '''
        Params:
            sigma: noise level in Assumption 2 Y = f(x) + epsilon, epsilon ~ N(0, sigma^2)
            beta: constant value for beta_t at all timesteps (default is None, optional), could be useful for tuning beta_t
            beta_coef: coefficient for beta_t (default is 1, optional), could be useful for tuning beta_t
            cond_score_version: version of guidance
                v1: naive gradients G
                v2: gradient guidance G_loss
        
        '''
        super().__init__(**kwargs)
        self.g = None
        self.sigma = sigma 
        self.beta = beta
        self.beta_coef = beta_coef
        self.cond_score_version = cond_score_version

    def update_grad(self, g, batch_size=32):
        if len(g.shape) == 2: # Each sample has its own gradient for parallel optimization 
            self.g = g # N * dim
        else: # [Deprecated] The gradient should be taken at the mean of all samples in a batch (may need changes in main.py)
            g = g.view(1, -1)
            self.g = g.repeat(batch_size, 1) # sample a batch with the conditional score based on the same gradient g

    def cond_score_v2(self, x, t, classes):
        r"""
        Gradient Guidance G_loss
        Following Definiton 1/ Lemma 1 in the original paper

        G_loss = - beta_t * \nabla_{x_t} (y - g^T * E[x_0 | x_t])^2
        """

        alpha_t = extract(self.sqrt_alphas_cumprod, t, (x.shape[0], 1))
        h_t = extract(self.sqrt_one_minus_alphas_cumprod, t, (x.shape[0], 1)) ** 2

        #### compute the gradient of g^T * E[x_0 | x_t] w.r.t. x_t
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True) # x_t, batched samples
            pred_noise = self.model(x_in, t, classes=None) # predict noise
            noise_svd = extract(
                self.sqrt_one_minus_alphas_cumprod, t, pred_noise.shape)
            uncond_score = -1 * pred_noise * 1 / noise_svd
            x_0_hat = (x_in + h_t * uncond_score) / alpha_t # get E[x_0 | x_t]
            value = torch.sum(x_0_hat * self.g, dim=1, keepdim=True)
            gradient = torch.autograd.grad(value.sum(), x_in)[0] # Get graidents for each sample in a batch parallelly

        ## 
        # beta_t 
        ## theoretical value: a variant of the beta_t shown in Lemma 1
        ## beta_t := alpha_t^2/2 / (alpha_t^2 * \sigma^2 + h(t) ||\nabla_{x_t} (g^T * E[x_0 | x_t])||^2)
        beta_t = (alpha_t ** 2 / 2) / (
            alpha_t ** 2 * self.sigma ** 2 + h_t * torch.norm(gradient, dim=1, keepdim=True) ** 2
            ) if self.beta is None else torch.full((x.shape[0], 1), self.beta, device=x.device)
        beta_t = beta_t * self.beta_coef 

        ## gradient of the loss (y - g^T * E[x_0 | x_t]) w.r.t. x_t
        gradient_2 = 2 * (value - classes.view(-1, 1)) * gradient 
        
        G = uncond_score - beta_t * gradient_2 # get the conditional score
        pred_noise = -1 * G * noise_svd # transform the conditional score to noise
        return pred_noise

    def cond_score_v1(self, x, classes, uncond_score, t):
        r'''
        Naive Gradients G: following (20) in Proposition 3 of Appendix in the original paper with covaraince Σ = I

        G: = beta_t * (y - g^T * E[x_0 | x_t]) * g
        '''

        alpha_t = extract(self.sqrt_alphas_cumprod, t, (x.shape[0], 1))
        h_t = extract(self.sqrt_one_minus_alphas_cumprod, t, (x.shape[0], 1)) ** 2

        if not self.beta:
            ### theoretical value defined in Proposition 3 of Appendix in the original paper, covaraince Σ = I is omitted
            ### beta_t = alpha_t / (sigma^2 + h_t * ||g||^2)
            beta_t = alpha_t / (self.sigma ** 2 + h_t * torch.norm(self.g, dim=1, keepdim=True) ** 2)  
        else:
            ## beta is constant for all timesteps
            beta_t = torch.full((x.shape[0], 1), self.beta, device=x.device)
        
        x_0_hat = (x + h_t * uncond_score) / alpha_t # get E[x_0 | x_t]

        ### get the condtional score
        G = uncond_score + beta_t * (classes - torch.sum(x_0_hat * self.g, dim=1)).view(-1, 1) * self.g * self.beta_coef # (adding beta_coef for tuning flexibility, default is 1)
        return G

    def p_mean_variance(self, x, t, classes=None):
        if classes is None:
            pred_noise = self.model(x, t, classes=None)
        else:
            if self.cond_score_version == "v1":
                # Naive Gradients: G
                pred_noise = self.model(x, t, classes=None)
                noise_svd = extract(
                    self.sqrt_one_minus_alphas_cumprod, t, pred_noise.shape)
                uncond_score = -1 * pred_noise * 1 / noise_svd
                assert self.g is not None
                cond_score = self.cond_score_v1(x, classes, uncond_score, t)
                pred_noise = -1 * cond_score * noise_svd
            else:
                # Gradient Guidance of Look-Ahead Loss: G_loss
                pred_noise = self.cond_score_v2(x, t, classes)

        ## DDPM Sampler
        x_start = extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - extract(self.sqrt_recipm1_alphas_cumprod,
                                                                                    t, x.shape) * pred_noise
        posterior_mean = extract(self.posterior_mean_coef1, t, x.shape) * x_start + extract(self.posterior_mean_coef2,
                                                                                            t, x.shape) * x
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x.shape)

        return posterior_mean, posterior_log_variance_clipped

if __name__ == "__main__":
    dim_x = 64
    model = Unet1D(dim=64, conditional=False)
    diffusion = GuidedGaussianDiffusion(
        model=model, image_size=dim_x, timesteps=200,
        )
    samples = diffusion.sample(num_samples=32)
    

