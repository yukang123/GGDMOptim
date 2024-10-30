import os
if os.getcwd() != os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

import torch
import numpy as np
from torch.utils.data import DataLoader

from diffusion import GuidedGaussianDiffusion
from unet_1d import Unet1D
from train import LinearLatentData, UnitBallData, SingleStepIterator, CustomLogger, set_seed
from torch.utils.data import TensorDataset
from argparse import ArgumentParser
from test_func import BatchGrad, Linear, NegNormPlus, QuadraticOnLinear, BatchGradV2    
import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type = int,
        default = 2345,
        help = "random seed"
    )

    ## data distribution
    parser.add_argument(
        "--x_type",
        type = str,
        default = "linear_latent", # "linear_latent" (linear data structure) or "unit_ball" (nonlinear data structure)
        help = "the type of distribution of X"
    )
    parser.add_argument(
        "--d_outer",
        type = int,
        default = 64,
        help = "the dimension of the ambient space"
    )
    parser.add_argument(
        "--d_inner",
        type = int,
        default = 16,
        help = "the dimension of the linear subspace (d_inner < d_outer, for linear_latent)"
    )

    # A. Pretrain
    parser.add_argument(
        "--pretrain",
        action = "store_true",
        help = "whether to pretrain the score model"
    )
    parser.add_argument(
        "--pretrain_ckpt_folder",  
        type=str,   
        default="pretrained",   
        help="the folder to save the pretrained checkpoint"
    )
    parser.add_argument(
        "--pretrain_data",
        type = str,
        default = None,
        help = "the path of pretrained data"
    )
    parser.add_argument(
        "--N_0",
        type = int,
        default = 65536,
        help = "sample number of the initial dataset D_0"
    )
    parser.add_argument(
        "--T",
        type = int,
        default = 200,
        help = "the total timestep of forward diffusion process"
    )
    parser.add_argument(
        "--pre_lr",
        type = float,
        default = 1e-3,
        help = "the learning rate to train the diffusion model on D_0"
    )
    parser.add_argument(
        "--pre_bs",
        type = int,
        default = 32,
        help = "the batch size for pretraining the diffusion model"
    )
    parser.add_argument(
        "--pre_num_episodes",
        type = int,
        default = 50,
        help = "the total num of training epochs for diffusion model pretraining"
    )

    # B. Reward Function
    parser.add_argument(
        "--func_type",
        type = str,
        default = "quadratic",
        help = "the type of function"
    ) 
    ## Hyperparameters for the experiments on the linear latent space data structure
    # 1. Quadratic Functions 10 - (theta^T * x - 3)^2
    parser.add_argument(
        "--r_off",
        type = int,
        default = 9,
        help = "off/on-support ratio for theta_2 in LinearLatentData"
    )
    parser.add_argument(
        "--use_theta_1",
        action = "store_true",
        help = "whether to use theta_1"
    )
    # 2. NegNormPlus Functions 5 - 0.5 * ||x - b||
    parser.add_argument(
        "--use_b_1",
        action = "store_true",
        help = "whether to use b_1"
    )


    ## C. Reward-guided Optimization
    parser.add_argument(
        "--optimize",
        action = "store_true",
        help = "whether to optimize the reward function based on guided diffusion model"
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type = str,
        default = None,
        help = "the path of pretrained checkpoint"
    )
    ### guidance
    parser.add_argument(
        "--cond_score_version",
        type = str,
        default = "v1",
        help = "the version of score guidance"
    )       
    parser.add_argument(
        "--beta_coef",
        type = float,
        default = 1.0,
        help = "the coefficient before the guidance scale beta_t defined in the paper"
    )
    parser.add_argument(
        "--beta",
        type = float,
        default = None,
        help = "the constant beta to replace the beta_t defined in the paper"
    )
    parser.add_argument(
        "--generate_bs",
        type = int,
        default = 32,
        help = "the number of samples which are generated from the diffusion model (independently) per round"
    )
    parser.add_argument(
        "--opt_rounds",
        type = int,
        default = 20,
        help = "the total iterations/rounds of optimization"
    )
    parser.add_argument(
        "--interval",
        type = float,
        default = 0.05,
        help = "the expected increment of y per iteration"
    )

    # Score Matching Fine-tuning (Algorithm 2)
    parser.add_argument(
        "--score_matching_finetune",
        action = "store_true",
        help = "whether to fine-tune the score model using score matching loss"
    )
    parser.add_argument(
        "--sm_ft_lr",
        type = float,
        default = 1e-5,
        help = "the learning rate to fine-tune the score model"
    )
    parser.add_argument(
        "--sm_ft_round",
        type = int,
        default = None,
        help = "the number of rounds to fine-tune the score model"
    )
    parser.add_argument(
        "--sm_ft_start_round",
        type = int,
        default = 0,
        help = "the round when to start fine-tuning the score model with score-matching loss"
    )     
    parser.add_argument(
        "--sm_ft_step_per_round",
        type = int,
        default = 1,
        help = "finetune steps per round"
    )

    # Logging
    parser.add_argument(
        "--log_dir",
        type = str,
        default = "logs",
        help = "the folder to store the tensorboard logs"
    )    
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="the directory to save the results"
    )
    parser.add_argument(
        "--save_samples",
        action = "store_true",
        help = "whether to save samples"
    )
    parser.add_argument(
        "--reward_K",
        type = int,
        default = 4,
        help = "select top K reward to visualize"
    )
    parser.add_argument(
        "--other_remark",
        type = str,
        default = "",
        help = "other remarks for the experiment to be put into the folder name"
    )
    parser.add_argument(
        "--disable_guidance",
        action = "store_true",
        help = "disable the guidance (only for getting the sample time for multiple rounds of unguided sampling)"
    )
    args = parser.parse_args()
    return args

def save_checkpoint(model, step, x_type, save_folder="checkpoints/pretrained", phase="pretrain"):
    ckpt = {
        "x_type": x_type,
        "model_state_dict": model.state_dict()
    }
    if phase == "pretrain":
        ckpt["epoch"] = step
        step_info = f"epoch_{step}"
    elif phase == "finetune":
        ckpt["iteration"] = step
        step_info = f"iteration_{step}"

    os.makedirs(save_folder, exist_ok=True)
    torch.save(ckpt, os.path.join(save_folder, f"{x_type}_{phase}_{step_info}.pth"))

def pretrain(args, generator, model, diffusion, log_dir=None):
    '''
    Unconditional diffusion model pretraining
    '''
    print("====================Pretraining Starts====================")
    # 1. Generate training data D_0
    if args.pretrain_data is None or not os.path.exists(args.pretrain_data):
        data_diff = generator.generate_x(args.N_0)
        os.makedirs(f"data/{args.x_type}", exist_ok=True)
        np.save(f"data/{args.x_type}/pretrain_{args.N_0}.npy", data_diff)
    else:
        data_diff = np.load(args.pretrain_data)
    data_diff = torch.from_numpy(data_diff).float()

    # 2. Train diffusion model on D_0
    model.train()

    ## optimizer and dataloader
    optimizer_diff = torch.optim.Adam(diffusion.model.parameters(), lr=args.pre_lr, betas=(0.9, 0.99))
    dataset_diff = TensorDataset(data_diff)
    dataloader_diff = DataLoader(dataset_diff, batch_size=args.pre_bs)

    ## train
    trainer = SingleStepIterator(diffusion, optimizer_diff, log_dir=log_dir)

    start_time = time.time()
    trainer.train(dataloader_diff, num_episodes=args.pre_num_episodes) # 10
    train_time = time.time() - start_time
    save_checkpoint(diffusion, step=args.pre_num_episodes, save_folder=args.pretrain_ckpt_folder, x_type=args.x_type)
    return model, diffusion, train_time

def optimize(args, model, diffusion, generator, func, func_type, query_samples, interval, opt_logger, reward_K):
    '''
    Reward optimization with guided diffusion model
    '''
    print("====================Optimization Starts====================")
    if args.score_matching_finetune:
        optimizer_ft_sm = torch.optim.Adam(diffusion.model.parameters(), lr=args.sm_ft_lr, betas=(0.9, 0.99))
        sm_ft_round = args.sm_ft_round if args.sm_ft_round is not None and args.sm_ft_round > 0 else args.opt_rounds - args.sm_ft_start_round
        finetune_rounds = 0
        
    samples_list = []
    reward_list = []
    top_k_samples_list = []

    sample_time = 0
    opt_start_time = time.time()
    for i in range(args.opt_rounds):
        ## 1. get y_k (B_k, d_outer)
        # Query gradient g_k (get gradient for each sample z_(k,i))
        bs_gradient, bs_value = BatchGrad(query_samples, func)         
        diffusion.update_grad(bs_gradient) #, batch_size=bs_gradient.shape[0])
        # y_k = delta + g_k^T * z_k
        guidance = interval + torch.sum(bs_gradient * query_samples, dim=1)
        opt_logger.log(guidance=guidance[0].item(), mean_guidance=torch.mean(guidance).item())

        if args.score_matching_finetune:
            ## Alg.2: Fine-tuning Diffusion Model with Score Matching Loss
            if i >= args.sm_ft_start_round and finetune_rounds < sm_ft_round:
                model.train()
                steps = args.sm_ft_step_per_round # [Deprecated] default: 1
                batch_size = len(query_samples) // steps
                for j in range(steps):
                    loss_diffusion = diffusion(query_samples[j*batch_size: (j+1)*batch_size, :])
                    optimizer_ft_sm.zero_grad()
                    loss_diffusion.backward()
                    optimizer_ft_sm.step()
                    opt_logger.log(loss_diffusion=loss_diffusion.item())
                finetune_rounds += 1
                print("finetune the pretrained score model | round: ", finetune_rounds)
                model.eval()

                if finetune_rounds == sm_ft_round:
                    save_checkpoint(
                        diffusion, step=sm_ft_round, x_type=args.x_type, 
                        save_folder=f"checkpoints/score_finetuned/{func_type}_{args.cond_score_version}", phase="finetune"
                        )
        
        start_time = time.time()
        query_samples = diffusion.sample(
            num_samples=args.generate_bs,
            classes= guidance if not args.disable_guidance else None,
            )
        sample_time += time.time() - start_time

        reward = func(query_samples)
        mean_reward = torch.mean(reward)
        ratio = generator.off_support_ratio(query_samples)

        opt_logger.log(reward=mean_reward.item())
        opt_logger.log(ratio=np.mean(ratio))

        topk_value, topk_pos = torch.topk(reward.view(-1), reward_K)
        opt_logger.log(top_k_reward=torch.mean(topk_value).item())
        topk_ratio = generator.off_support_ratio(query_samples[topk_pos, :])
        opt_logger.log(top_k_ratio=np.mean(topk_ratio))

        if args.save_samples:
            samples_list.append(query_samples.cpu().numpy())
            top_k_samples_list.append(query_samples[topk_pos, :].cpu().numpy())
        reward_list.append(reward.cpu().numpy())

    end_time = time.time()
    optimize_time = end_time - opt_start_time
    return samples_list, reward_list, top_k_samples_list, sample_time, optimize_time

def main(args, seed=2345):

    all_start_time = time.time()

    set_seed(seed)
    d_outer = args.d_outer

    ### build score model and diffusion
    model = Unet1D(dim=d_outer, conditional=False)
    print(f"guidance version: {args.cond_score_version}")
    diffusion = GuidedGaussianDiffusion(
        model=model, 
        image_size=d_outer, 
        timesteps=args.T, 
        cond_score_version=args.cond_score_version,
        beta_coef=args.beta_coef, beta=args.beta,
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    diffusion.to(device = device)

    ### load hyperparameters for data distribution and reward functions
    hyper_path = f"data/{args.x_type}/hyperparameters.npz"
    if args.x_type == "linear_latent": ## linear data structure (default)
        generator = LinearLatentData(hyper_path=hyper_path, d_inner=args.d_inner, d_outer=d_outer, r_off=args.r_off)
    elif args.x_type == "unit_ball": ## nonlinear data structure
        generator = UnitBallData(hyper_path=hyper_path, d_outer=d_outer)
    else:
        raise ValueError("Invalid x_type")
    
    if args.pretrain:
        total_epoch = args.pre_num_episodes

        exp_info = f"{args.x_type}/pretrain_{total_epoch}"
        log_dir = f"{args.log_dir}/{exp_info}{args.other_remark}"

        ### pretrain diffusion model
        model, diffusion, train_time = pretrain(args, generator, model, diffusion, log_dir=log_dir)
        print(f"------------Pretraining Time Summary--------------")
        print(f"{os.path.basename(__file__)} | {args.seed}")
        print(f"pretrain diffusion model (epoch={total_epoch})")
        print(f"training time: {train_time}")
        print("-------------------------------------------------")

    else:
        ### load pretrained checkpoint
        assert args.pretrain_ckpt is not None
        pretrain_info = torch.load(args.pretrain_ckpt, map_location=device)
        total_epoch = pretrain_info['epoch']
        assert pretrain_info["x_type"] == args.x_type, f"pretrain x_type: {pretrain_info['x_type']} | current x_type: {args.x_type}"
        
        print(f"load pretrained diffusion model (epoch={total_epoch})")
        diffusion.load_state_dict(pretrain_info["model_state_dict"])
        
        exp_info = f"{args.x_type}/pretrain_{total_epoch}"
        
    model.eval()
    if args.x_type == "linear_latent": 
        func_type = args.func_type
        if args.func_type == "quadratic":

            theta_ = generator.theta_1 if args.use_theta_1 else generator.theta_2 # [default] theta_2
            print(f"theta norm: {np.linalg.norm(theta_):.2f}, off/on-support ratio with respect to A: {generator.off_support_ratio(theta_)[0]:.2f}")
            func = QuadraticOnLinear(theta=theta_, negative=False)
            func_remark = f"{func_type}_theta_1" if args.use_theta_1 else f"{func_type}_theta_2"

        elif args.func_type == "negnormplus":
            b_ = generator.b_1 if args.use_b_1 else generator.b_2 # [default] b_2
            func = NegNormPlus(b=b_, negative=False)
            func_remark = f"{func_type}_b_1" if args.use_b_1 else f"{func_type}_b_2"

        else:
            raise ValueError(f"{args.func_type} is not supported")

    elif args.x_type == "unit_ball":
        print(f"theta norm: {np.linalg.norm(generator.theta):.2f}")
        func = Linear(generator.theta, negative=False)
        func_type = "linear"  
        func_remark = "linear" 

    start_time = time.time()
    query_samples = diffusion.sample(num_samples=args.generate_bs) # args.generate_bs: B_k
    uncond_sample_time = time.time() - start_time

    reward = func(query_samples)
    mean_reward = torch.mean(reward)
    ratio = generator.off_support_ratio(query_samples)

    topk_value, topk_pos =  torch.topk(reward.view(-1), args.reward_K)
    topk_ratio = generator.off_support_ratio(query_samples[topk_pos, :])

    if args.save_samples:
        samples_list = []
        samples_list.append(query_samples.cpu().numpy())
        top_k_samples_list = []
        top_k_samples_list.append(query_samples[topk_pos, :].cpu().numpy())
    reward_list = []
    reward_list.append(reward.cpu().numpy())


    print(f"Unconditional Generation: sample_num: {len(query_samples)} | mean_ratio: {np.mean(ratio)} | sample_time: {uncond_sample_time}")
    print(f"Reward type: {func_remark} | mean_reward: {mean_reward}")
    remark_ = func_remark + "_"

    if args.optimize:
        ### Guided Optimization
        interval = args.interval
        alg_type = "alg1" if not args.score_matching_finetune else "alg2"
        guidance_version = "G" if args.cond_score_version == "v1" else "G_loss"
        exp_info = f"{args.x_type}/optimize_{alg_type}_guidance_{guidance_version}/func_{func_remark}_interval_{interval}"
        log_dir = f"{args.log_dir}/{exp_info}_seed_{args.seed}{args.other_remark}"

        opt_logger = CustomLogger(log_dir=log_dir)
        ## record the initial rewards of the samples generated by the diffusion model
        opt_logger.log(reward=mean_reward.item())
        opt_logger.log(ratio=np.mean(ratio))
        opt_logger.log(top_k_reward=torch.mean(topk_value).item())
        opt_logger.log(top_k_ratio=np.mean(topk_ratio))

        samples_list_op, reward_list_op, top_k_samples_list_op, sample_time, optimize_time = optimize(args, model, diffusion, generator, func, func_type, query_samples, interval, opt_logger, args.reward_K)
        
        reward_list = reward_list + reward_list_op
        if args.save_samples:
            samples_list = samples_list + samples_list_op
            top_k_samples_list = top_k_samples_list + top_k_samples_list_op

        print(f"------------Optimization Time Summary-----------------")
        print(f"{os.path.basename(__file__)} | data type: {args.x_type} | function type: {func_remark} | guidance type: {args.cond_score_version} | score matching finetune: {args.score_matching_finetune} | {args.seed}")
        print(f"The total optimizing time is: {optimize_time} ")
        print(f"Sample time: {sample_time} (total rounds: {args.opt_rounds})")
        print(f"Each round:\n optimizing time: {optimize_time/args.opt_rounds} | sample time: {sample_time/args.opt_rounds}")
        print("-------------------------------------------------")
        remark_ = ""

    save_dir = f"{args.save_dir}/{exp_info}{args.other_remark}"    
    os.makedirs(save_dir, exist_ok=True)

    reward_array = np.concatenate(reward_list, axis=1)
    np.save(f"{save_dir}/{remark_}reward_seed_{args.seed}.npy", reward_array)
    if args.save_samples:
        saved_samples = np.stack(samples_list, axis=0)
        np.save(f"{save_dir}/samples_seed_{args.seed}.npy", saved_samples)
        ## To be removed
        top_k_saved_samples = np.stack(top_k_samples_list, axis=0)
        np.save(f"{save_dir}/top_{args.reward_K}_samples_seed_{args.seed}.npy", top_k_saved_samples)

    print(f"The total running time is: {time.time()-all_start_time} ")


if __name__ == "__main__":
    args = parse_args()
    main(args, seed=args.seed)

    