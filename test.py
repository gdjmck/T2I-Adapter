from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from omegaconf import OmegaConf
from configs.utils import instantiate_from_config
import torch

if __name__ == '__main__':
    # configs
    config = OmegaConf.load('configs/train/Adapter-XL-layout.yaml')
    # load adapter
    adapter_config = config.model.params.adapter_config
    adapter = instantiate_from_config(adapter_config).cuda()
    adapter.load_state_dict(torch.load('experiments/adapter_sketch_xl/checkpoint-9000/model-00.pth'))
    
    # load euler_a scheduler
    model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder=None,
    )
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    control_image = load_image("./conditioning_image_1.png")
    prompt = "A layout plan of residential community, with a total of 3 buildings, a floor area ratio of 2.50, a building density of 30.8%, and an average number of floors of 6.7"

    negative_prompt = "colorful"
    gen_images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=30,
        adapter_conditioning_scale=0.8,
        guidance_scale=7.5, 
    ).images[0]
    gen_images.save('out_lin.png')
