import torch
import torchvision

class RCGDMScorer(torch.nn.Module):
    ### Reward model borrowed from RCGDM
    def __init__(self):
        super().__init__()
        ## load the synthetic reward model which is based on a ResNet-18 model
        self.reward_model = torch.load('reward_model.pth') 
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()

    def __call__(self, images):
        ## input: generated images
        target_size = 224
        normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                           std=[0.26862954, 0.26130258, 0.27577711])
        
        images = ((images + 1.0) / 2.0).clamp(0, 1) # convert from [-1, 1] to [0, 1]
        ## image transform
        im_pix = torchvision.transforms.Resize(target_size)(images)
        im_pix = normalize(im_pix).to(images.dtype)
           
        return self.reward_model(im_pix)



    