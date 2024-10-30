'''
The codes in this folder are adapted from https://github.com/Kaffaljidhmah2/RCGDM.git
'''
import os
if os.getcwd() != os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

from gradguided_sdpipeline import GradGuidedSDPipeline
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import wandb
import argparse
from scorer import RCGDMScorer
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="device")

    parser.add_argument("--target", type=float, default=0., help="target reward value y")
    parser.add_argument("--guidance", type=float, default=1., help="guidance strength beta(t)") 
    parser.add_argument("--opt_steps", type=int, default=100, help="number of optimization steps")

    parser.add_argument("--prompt", type=str, default=None, help="text prompt for image generation")
    parser.add_argument("--out_dir", type=str, default= "results", help="output directory")
    parser.add_argument("--bs", type=int, default=5, help="batch size for optimization")
    parser.add_argument("--seed", type=int, default=-1, help="random seed")
    parser.add_argument("--repeat_epoch", type=int, default=1, help="repeat the optimization for multiple times")
    parser.add_argument("--image_size", type=int, default=512, help="image size for optimization")

    parser.add_argument("--wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--save_samples", action="store_true", help="save the generated images")
    args = parser.parse_args()
    return args

def get_grad_eval(ims, reward_model, device='cuda'):    
    ims = ims.to(device)
    ims.requires_grad = True

    rewards = reward_model(ims)
    rewards_sum = rewards.sum()
    rewards_sum.backward() # get the gradients for each sample in the batch

    grads = ims.grad
    biases = - torch.einsum('bijk,bijk->b', grads, ims) + rewards # r(x) - <grad, x>
    return grads, biases, rewards

def concat_from_2d_list(lst, size=128):
    ## concatenate images in both row and column directions
    img = Image.new('RGB', (size*len(lst[0]), size*len(lst)))
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            img.paste(lst[i][j], (size*j, size*i))
    return img

def main():
    ######### preparation ##########
    args = parse_args()
    ### Set up device, out_dir, and wandb
    device= args.device if torch.cuda.is_available() else "cpu"
    total_bs = args.bs * args.repeat_epoch
    remark = f'target{args.target}guidance{args.guidance}seed{args.seed}totalbs{total_bs}_{args.prompt}'
    out_dir = args.out_dir + '/' + remark
    os.makedirs(out_dir, exist_ok=True)

    print("-"*50)
    print("target: ", args.target)
    print("guidance: ", args.guidance)
    print("opt_steps: ", args.opt_steps)
    print("prompt: ", args.prompt)
    
    print("bs: ", args.bs)
    print("repeat_epoch: ", args.repeat_epoch)
    print("seed: ", args.seed)
    print("device: ", device)
    print("out_dir: ", out_dir)
    print("-"*50)

    if args.wandb:
        wandb.init(project = "gradient_guided_sd", 
            name = remark,
            config = {
            'target': args.target,
            'guidance': args.guidance, 
            'seed': args.seed
            }
        )
    
    image_size_kwargs = {
        "height": args.image_size,
        "width": args.image_size
    }

    ### Set up prompts
    prompts = None
    if args.prompt != None:
        ## repeat the same prompt for multiple times
        prompts = [args.prompt] * args.repeat_epoch
    else:
        ## if prompt is not specified, randomly sample (args.repeat_epoch,) prompts from imagenet classes
        with open("imagenet_classes.txt", "r") as file:
            imagenet_classes = file.readlines() ### 1000 classes in total
        imagenet_classes = [class_name.strip() for class_name in imagenet_classes]
        random.seed(args.seed)
        prompts = random.sample(imagenet_classes, args.repeat_epoch)

    ### Load Reward Model borrowed from RCGDM 
    reward_model = RCGDMScorer().to(device)

    ### 0. Image Generation with Pretrained Stable Diffusion Model
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to(device)
    pretrain_gen_start = time.time()
    pretrain_images = pipe([prompts[0]] * args.bs, **image_size_kwargs).images
    pretrain_gen_end = time.time()
    ## inference time with the pretrained diffusion model
    pretrain_gen_time = pretrain_gen_end - pretrain_gen_start


    ### 1. Gradient Guided Optimization ###
    targets =  [args.target] * args.opt_steps
    guidances = [args.guidance] * args.opt_steps
    sd_model = GradGuidedSDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
    sd_model.to(device)
    image_rewards = np.zeros([args.opt_steps, args.repeat_epoch, args.bs])
    ## (opt_steps, repeat_epoch, bs)

    if args.seed > 0:
        torch.manual_seed(args.seed)
        latent_size = int(args.image_size) // sd_model.vae_scale_factor
        print(latent_size)
        shape = (args.repeat_epoch, args.bs , 4, latent_size, latent_size)
        ### sample the latent variable at t=T from N(0, I) as the start of reverse diffusion model
        init_latents = torch.randn(shape, device=device)
    else:
        init_latents = None

    opt_start_time = time.time()
    opt_images_step = defaultdict(list)
    for n in range(args.repeat_epoch):
    # conduct optimization for each prompt in prompts (args.repeat_epoch, )
        sd_model.set_linear_reward_model(is_init = True, batch_size = args.bs, **image_size_kwargs)
        prompt = prompts[n]
        for k in range(args.opt_steps):
            sd_model.set_target(targets[k]) # set target reward value y
            sd_model.set_guidance(guidances[k]) # set guidance strength beta(t)
            if init_latents is None:
                init_i = None
            else:
                init_i = init_latents[n]
            image_, image_eval_ = sd_model(prompt, num_images_per_prompt=args.bs, latents=init_i, **image_size_kwargs)
            image_ = image_.images
            ### query reward gradient with regard to the generated images
            grads, biases, rewards = get_grad_eval(image_eval_, reward_model)
            grads = grads.clone().detach()
            biases = biases.clone().detach()
            sd_model.set_linear_reward_model(gradients = grads, biases = biases, **image_size_kwargs)
            rewards = rewards.detach().cpu().numpy()
            if len(rewards.shape) > 1:
                rewards = rewards.squeeze()
            image_rewards[k, n] = rewards
            if args.wandb:
                wandb.log({'step': k, 'reward_mean': rewards.mean(), 'reward_std': rewards.std()})
            
            if args.save_samples:
                opt_images_step[k].append([img.resize([args.image_size//4, args.image_size//4]) for img in image_]) # (128, 128) images

    if args.save_samples:
        os.makedirs(out_dir + "/generated_images", exist_ok=True)
        imgs = concat_from_2d_list([[img.resize([args.image_size//4, args.image_size//4]) for img in pretrain_images]], size=args.image_size//4)
        imgs.save(out_dir + "/generated_images/pretrain.png")
        for k in range(args.opt_steps):
            img = concat_from_2d_list(opt_images_step[k], size=args.image_size//4)
            img.save(out_dir + f"/generated_images/step_{k}.png")

    end_time = time.time()
    total_optim_time = end_time - opt_start_time
    total_optim_round = args.repeat_epoch * args.opt_steps

    print(f"------------Time Summary-----------------")
    print(f"Total rounds: {total_optim_round}")
    print(f"The total optimizing time is: {total_optim_time} ")
    print(f"Each round:\n optimizing time: {total_optim_time/total_optim_round} ")
    print(f"Time for unguided generation with pretrained model (Single Round): {pretrain_gen_time}")


    # calculate mean along batch dimension
    image_rewards_mean = np.mean(image_rewards, axis=2)
    # save mean to csv opt_steps x repeat_epoch 
    np.savetxt(out_dir + '/rewards_mean_repeat.csv', image_rewards_mean, delimiter=',')
    
    # calculate std for image_rewards_mean along repeat_epoch dimension
    image_rewards_std = np.std(image_rewards_mean, axis=1)
    # save std to csv
    np.savetxt(out_dir + '/rewards_std_in_repeats.csv', image_rewards_std, delimiter=',')

    # reshape image_rewards to [opt_steps, repeat_epoch * bs]
    image_rewards = image_rewards.reshape(args.opt_steps, -1)
    # save image_rewards to csv opt_steps x (repeat_epoch * bs)
    np.savetxt(out_dir + '/rewards.csv', image_rewards, delimiter=',')

    # calculate mean rewards and std
    mean_rewards = np.mean(image_rewards, axis=1)
    std_rewards = np.std(image_rewards, axis=1)
    # save mean_rewards and std_rewards to csv
    np.savetxt(out_dir + '/mean_rewards.csv', mean_rewards, delimiter=',')
    np.savetxt(out_dir + '/std_rewards.csv', std_rewards, delimiter=',')

    ## plot mean rewards and std error bar
    x = np.arange(args.opt_steps)
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_rewards, color='dodgerblue')
    plt.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, color='dodgerblue', alpha=0.2)
    plt.xlabel('Optimization Steps')
    plt.ylabel('Reward')
    # save
    plt.savefig(out_dir + '/reward_plot.png')
    plt.close()

    # plot mean_rewards and use image_rewards_std as error bar
    plt.figure(figsize=(10, 6))
    # plot mean_rewards curve
    plt.plot(x, mean_rewards, color='dodgerblue')
    plt.errorbar(x, mean_rewards, yerr=image_rewards_std, fmt='o', color='dodgerblue')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Reward')
    # save
    plt.savefig(out_dir + '/reward_plot_errorbar.png')
    plt.close()


if __name__ == "__main__":
    main()