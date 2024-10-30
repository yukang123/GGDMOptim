<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Gradient Guidance for Diffusion Models: </br> An Optimization Perspective </h1>

<p align='center' style="text-align:center;font-size:1.25em;"> 
    Yingqing Guo*, &nbsp; 
    Hui Yuan*, &nbsp; 
    Yukang Yang, &nbsp; 
    <a href="https://minshuochen.github.io/" target="_blank" style="text-decoration: none;">Minshuo Chen</a>, &nbsp; 
    <a href="https://mwang.princeton.edu/" target="_blank" style="text-decoration: none;">Mengdi Wang</a>
    <br/>  
Princeton University (*Equal Contribution)
</p>

<p align='center';>
<b>
<em>NeurIPS 2024</em> <br> 
</b>
</p>


<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://arxiv.org/abs/2404.14743" target="_blank" style="text-decoration: none;">arXiv</a>&nbsp;
</b>
</p>


## :wrench: Installation
Clone this repo: 
```
git clone https://github.com/yukang123/GGDMOptim.git
cd GGDMOptim
```

Install required Python packages

```
(Optional) conda create -n guided_diffusion python=3.10
conda activate guided_diffusion

pip install -r requirements.txt
```

Computational Resource Requirements: 

one NVIDIA 80G A100 GPU is recommended. But for the numerical simulations, the maximum GPU usage is less than 3GB; while for the image generation, the experiments could be accomodated within a smaller GPU memory (e.g., 24GB) by decreasing the batch size or image size.

## :mag_right: Numerical Simulations
Experiments on two different data distributions.

**Prerequisites**: Please download ```data``` (hyperparameters and dataset for pretraining) and ```checkpoints``` (pretrained checkpoints) from this [huggingface](https://huggingface.co/PDMR/GGDMOptim/tree/main) repo and put them into the ```simulations``` folder (replace the current placeholder). Then
``` cd simulations ```.

Please check the ```main.py``` file to understand the meanings of different parameters. Please use **tensorboard** to visualize the results based on log files saved in ```logs``` folder.

**A. Linear Latent Space:** 
> (Optional) Pretraining Unconditional Diffusion Model: 

```
python main.py --x_type linear_latent --pretrain --pre_lr 1e-4 --pre_num_episodes 20 --pretrain_data data/linear_latent/pretrain_65536.npy --pretrain_ckpt_folder checkpoints/pretrained_n
```
You can also generate new training data by setting ```--pretrain_data``` as None and trying different random seeds.

> Comparisons between two different guidances $G$ and $G_{loss}$ (Figure 4 of the paper)

Reward Function: $$ f(x) = 10 - (\theta^\top x - 3)^2, \frac{\Vert{\theta_\bot}\Vert}{\Vert{\theta_\Vert}\Vert}=9.$$ You may add ```--seed xxxx``` in the below commands (try seeds in $[1234, 2345,3456,4567,5678]$)

**Algorithm 1**: Gradient-Guided Diffusion for Generative Optimization


Please add ```--optimize``` to enable generative optimization. ```--interval``` means the expected increment of reward values per iteration, i.e., $\delta$ in Appendix E.1 of the paper.

$G$: naive gradient (```--cond_score_version v1```)
```
python main.py --optimize --x_type linear_latent --cond_score_version v1 --func_type quadratic --pretrain_ckpt checkpoints/pretrained/linear_latent_pretrain_epoch_20.pth --generate_bs 32 --opt_rounds 1000 --interval 0.9 
```
And you may save the generated samples by adding ```--save_samples```. 

$G_{loss}$: Gradient Guidance of Look-Ahead Loss (```--cond_score_version v2```)
```
python main.py --optimize --x_type linear_latent --cond_score_version v2 --func_type quadratic --pretrain_ckpt checkpoints/pretrained/linear_latent_pretrain_epoch_20.pth --generate_bs 32 --opt_rounds 1000 --interval 0.2 --save_samples (optional)
```

**Algorithm 2**: Gradient-Guided Diffusion with Adaptive Fine-tuning

Please add ```--score_matching_finetune``` to finetune the score model with the score matching loss.

$G$:
```
python main.py --optimize --x_type linear_latent --cond_score_version v1 --func_type quadratic --pretrain_ckpt checkpoints/pretrained/linear_latent_pretrain_epoch_20.pth --generate_bs 32 --opt_rounds 1200 --interval 0.9 --score_matching_finetune --sm_ft_lr 1e-6 --sm_ft_start_round 40 --save_samples (optional)
```

$G_{loss}$:
```
python main.py --optimize --x_type linear_latent --cond_score_version v2 --func_type quadratic --pretrain_ckpt checkpoints/pretrained/linear_latent_pretrain_epoch_20.pth --generate_bs 32 --opt_rounds 1200 --interval 0.2 --score_matching_finetune --sm_ft_lr 1e-6 --sm_ft_start_round 60 --save_samples (optional)
```

> Algorithm 1 with $G_{loss}$ for optimizing different functions (Figure 9 of the paper)

Reward Functions:

(a) $f_1(x) = 10 - (\theta^\top x - 3)^2$.  (```--opt_rounds 1000```)

Please set ```--func_type quadratic```. When $\frac{\Vert{\theta_\bot}\Vert}{\Vert{\theta_\Vert}\Vert}=9$, set ```--interval 0.2``` . When $\theta=A\beta^*$, add ```--use_theta_1``` and set ```--interval 0.05```. 

(b) $f_2(x)= 5 - 0.5 \lVert x - b \rVert.$  (```--opt_rounds 50```)

Please set ```--func_type negnormplus```. 
When $b \sim \mathcal{N}( 4\cdot\mathbf{1}, 9 \cdot I_{D})$, specify ```--interval 1```. 
When $b =4 \cdot \mathbf{1}_{D}$, add ```--use_b_1``` and set ```--interval 1```. 


B. **Nonlinear**: Unit Ball

Please set ```--x_type unit_ball```.

> (Optional) Pretraining Unconditional Diffusion Model: 

```
python main.py --x_type unit_ball --pretrain --pre_lr 1e-4 --pre_num_episodes 20 --pretrain_data data/unit_ball/pretrain_65536.npy --pretrain_ckpt_folder checkpoints/pretrained_n --seed 1
```

> $G$ vs $G_{loss}$

The reward function is $\theta^\top x$. 

$G$:
```
python main.py --optimize --x_type unit_ball --cond_score_version v1 --pretrain_ckpt checkpoints/pretrained/unit_ball_pretrain_epoch_20.pth --generate_bs 256 --opt_rounds 25 --interval 0.2 --seed 1
```

$G_{loss}$:
```
python main.py --optimize --x_type unit_ball --cond_score_version v2 --pretrain_ckpt checkpoints/pretrained/unit_ball_pretrain_epoch_20.pth --generate_bs 256 --opt_rounds 25 --interval 0.2 --seed 1
```

Please try different ```--interval``` values in a large range (maybe 0.02-0.52) to control the guidance strength and get the $(reward, ratio)$ after the optimization at each interval. You may use these values to plot the parato front curves like the right figure in Figure 11 for the comparison between $G$ and $G_{loss}$. 

<!-- $$
beta in G_loss
\beta_{\gamma}(t,g) := \frac{\alpha^2(t)/2}{\alpha^2(t) \sigma^2 + h(t)v_{\gamma}(t)\left| \nabla_{x_t}\left(g^\top \hat{x}_0\right)\right|^2}
$$

$$
\hat{X}_0 = \frac{1}{\alpha(t)}\left(x_t + h(t){s}_{\theta}(x_t, t) \right)
$$ -->


## :mag_right: Image Generation
Codes are adapted from [RCGDM](https://github.com/Kaffaljidhmah2/RCGDM.git). 

```
- cd image_generation
- python main.py --target 10 --guidance 100 --opt_steps 8 --repeat_epoch 4 --seed 5 --bs 5 --prompt 'fox' 
```

You could tune the ```--target``` (eg., 2, 4) and ```--guidance``` to control the guidance strength, and also try out different ```--opt_steps``` to see different optimization processes. You may also specify different text prompts with detailed descriptions.

**Additional Comments**: The runtime of the above experiments may vary with different GPU conditions and batch sizes. But the relative increase in time cost between gradient-guided generation and unguided generation remains as shown in Table 1 of Appendix F.3 of the paper.


## :e-mail: Contact

For help or issues about the github codes, please email Yukang Yang (yy1325@princeton.edu) and Yingqing Guo (yg6736@princeton.edu) or submit a GitHub issue.


## :mailbox_with_mail: Citation
If you find this code useful in your research, please consider citing:
```
@article{guo2024gradient,
  title={Gradient Guidance for Diffusion Models: An Optimization Perspective},
  author={Guo, Yingqing and Yuan, Hui and Yang, Yukang and Chen, Minshuo and Wang, Mengdi},
  journal={arXiv preprint arXiv:2404.14743},
  year={2024}
}

```
