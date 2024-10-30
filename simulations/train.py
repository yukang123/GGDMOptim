import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from scipy.stats import ortho_group
import numpy as np
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

####### Hyperparameter Loader for data distribution / reward function #######

class BasicData(object):
    def __init__(self, hyper_path="data/hyperparameters.npz", **kwargs):
        '''
        1. Generating/Loading the hyperparameters for data distribution and reward functions,
        2. Sampling from the data distribution to build a dataset for diffusion model pretraining
        Params:
            hyper_path: str, the path to save/load the hyperparameters
            kwargs: dict, the kwargs for generating hyperparameters (if hyper_path does not exist)
        '''
        if os.path.exists(hyper_path):
            self.load_hyper(hyper_path)
        else:
            self.generate_hyperparameters(**kwargs)
            os.makedirs(os.path.dirname(hyper_path), exist_ok=True)
            self.save_hyper(hyper_path)
        self.hyper_path = hyper_path

    def generate_hyperparameters(self):
        pass

    def load_hyper(self, hyper_path):
        pass

    def save_hyper(self, hyper_path):
        pass

    def generate_x(self, N): ## sampling data from the predefined distribution
        pass

    def off_support_ratio(self, samples):
        pass


class LinearLatentData(BasicData):
    '''
    For the experiments on the linear latent space data structure
    Data Sampler + Hyperparameters for reward functions
    '''
    
    def generate_hyperparameters(self, d_inner=16, d_outer=64, r_off=9):
        '''
        d_outer: data's ambient dimension (default: 64)
        d_inner: linear subspace dimension (default: 16)
        r_off: off/on-support ratio (default: 9)
        '''
        ### generate latent space A
        self.d_inner, self.d_outer = d_inner, d_outer # d_inner < d_outer
        self.A = ortho_group.rvs(dim=d_outer)[:d_inner, :] # (d_inner x d_outer); please notice that self.A is A^T

        #### two different thetas for f1(x) = 10 − (theta^T * x − 3)^2
        #### (1) A * beta_star
        beta = np.random.randn(d_inner, 1)
        beta /= np.linalg.norm(beta) # uniformly sampling from the unit sphere
        self.theta_1 = self.A.T.dot(beta) # (d_outer x 1), theta_1 = A * beta_star

        #### (2) off/on-support ratio r_off of theta is 9. 
        theta = np.random.randn(d_outer, 1)
        parallel, vertical, _ = self.off_support_ratio(theta, return_components=True)
        theta = r_off * vertical / np.linalg.norm(vertical, axis=-1) + 1 * parallel / np.linalg.norm(parallel, axis=-1)
        theta = theta / np.linalg.norm(theta)
        self.theta_2 = theta # (d_outer x 1), theta_2
        ## validate the off/on-support ratio
        r_off_theta = self.off_support_ratio(theta)[0]
        assert np.abs(r_off_theta - r_off) < 1e-6

        ##### b for f2(x) = 5 − 0.5||x − b||
        # two different b's
        # (1) b = 4 * 1_D
        self.b_1 = np.array(4.)
        # (2) b ~ N(4, 9 * I_D)
        self.b_2 = 4. + 3. * np.random.randn(d_outer)

    ### load and save hyperparameters
    def load_hyper(self, hyper_path):
        content = np.load(hyper_path)
        self.d_inner, self.d_outer = content["d_inner"], content["d_outer"] # d_inner < d_outer
        self.A = content["A"]
        self.theta_1 = content["theta_1"]
        self.theta_2 = content["theta_2"]
        self.b_1 = content["b_1"]
        self.b_2 = content["b_2"]

    def save_hyper(self, hyper_path):
        np.savez(
            hyper_path, 
            d_inner = self.d_inner, 
            d_outer = self.d_outer,
            A = self.A, 
            theta_1 = self.theta_1,
            theta_2 = self.theta_2,
            b_1 = self.b_1,
            b_2 = self.b_2,
            )
        
    def off_support_ratio(self, samples, return_components=False):
        '''
        Compute the off/on-support ratio: 
            the ratio between the norms of samples' vertical and parallel components with respect to the latent space A
            r = ||x_v|| / ||x_p||
        Params:
            samples: np.ndarray or torch.Tensor, samples to be computed, shape: (N, d_outer) or (d_outer, )
            return_components: bool, whether to return the components
        Return:
            (optional)
                parallel: np.ndarray, on-support component for samples
                vertical: np.ndarray, off-support component for samples
            ratio: np.ndarray, off/on-support ratio for samples
        '''

        if isinstance(samples, torch.Tensor):
            samples = samples.view(-1, self.d_outer).cpu().numpy()
        else:
            samples = samples.reshape(-1, self.d_outer)
        parallel = samples.dot(self.A.T).dot(self.A) # projection to A
        vertical = samples - parallel
        ratio = np.linalg.norm(vertical, axis=-1) / (np.linalg.norm(parallel, axis=-1) + 1e-8)
        if return_components:
            return parallel, vertical, ratio
        return ratio
    
    def generate_x(self, N):
        '''
        Data Sampler
        Params:
            N: int, number of samples
        '''
        # if self.nonlinear:
        #     x = np.random.multivariate_normal(mean=self.mean, cov=np.identity(self.d_outer), size=(N))
        #     x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        # else:
        x = np.random.randn(N, self.d_inner).dot(self.A) 
        return x
    

class UnitBallData(BasicData):
    '''
    For the experiments on the nonlinear data structure (unit ball)
    Data Sampler + Hyperparameters for reward functions
    '''

    def generate_hyperparameters(self, d_outer=64):
        '''
        d_outer: data's ambient dimension (default: 64)
        '''
        self.d_outer = d_outer
        ### parameters for data sampler
        self.mean = np.zeros(d_outer)

        ### parameters for reward function:  f(x) = theta^T * x
        theta = np.random.randn(d_outer, 1)
        theta = theta / np.linalg.norm(theta)
        self.theta = theta

    ### load and save hyperparameters
    def load_hyper(self, hyper_path):
        content = np.load(hyper_path)
        self.d_outer = content["d_outer"] 
        self.theta = content["theta"]
        self.mean = content["mean"]

    def save_hyper(self, hyper_path):
        np.savez(
            hyper_path, 
            d_outer = self.d_outer,
            theta = self.theta,
            mean = self.mean,   
            )
        
    def generate_x(self, N):
        x = np.random.multivariate_normal(mean=self.mean, cov=np.identity(self.d_outer), size=(N))
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        return x

    def off_support_ratio(self, samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        return np.linalg.norm(samples, axis=-1) - 1
      

###### Trainer ######
class SingleStepIterator(object):
    '''
    Training unconditional diffusion model 
    '''
    def __init__(self, diffusion, optimizer_diff, log_dir=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.diffusion = diffusion.to(self.device)
        self.optimizer_diff = optimizer_diff
        self.logger = SummaryWriter(log_dir=log_dir)
        self.stats = defaultdict(list)

    def log(self, **kwargs):
        for key, val in kwargs.items():
            self.stats[key].append(val)
            self.logger.add_scalar(f'stats/{key}', val, len(self.stats[key]) - 1)

    def train(self, dataset, num_episodes):
        for e in range(num_episodes):
            for sample_batch, in tqdm(dataset, desc=f'diffusion training epoch {e}'):
                sample_batch = sample_batch.to(self.device)
                loss_diffusion = self.diffusion(sample_batch)
                # optimizing
                self.optimizer_diff.zero_grad()
                loss_diffusion.backward()
                self.optimizer_diff.step()
                self.log(loss_diffusion=loss_diffusion.item())

###### Logger ######
class CustomLogger:
    def __init__(self, log_dir=None, comment=None):
        self.logger = SummaryWriter(log_dir=log_dir, comment=comment)
        self.stats = defaultdict(list)

    def log(self, **kwargs):
        for key, val in kwargs.items():
            self.stats[key].append(val)
            self.logger.add_scalar(f'stats/{key}', val, len(self.stats[key]) - 1)
