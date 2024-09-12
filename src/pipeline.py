import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

# 输出图片的大小为512*512
WIDTH = 512
HEIGHT = 512

# 特征空间的大小根据Encoder的输出，是512/8
LATENTS_WIDTH = 512 // 8
LATENTS_HEIGHT = 512 // 8

def generate(prompt: str, uncond_prompt: str, input_image=None, strength=0.8, do_cfg = True, cfg_scale=7.5,
             sampler_name="ddpm", n_inference_steps=50,models={}, seed=None, device=None,
             idle_device=None, tokenizer=None):
    with torch.no_grad():
        # strength是在输入图片时，我们想要关注原图片的程度。strength越大，往输入图片添加的噪声就越多
        # 生成的图片就和输入图片的差距越大
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed() # 生成一个随机种子
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg: # 使用Classifier-free Guidance
            # convert the prompt into tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (Batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_size, seq_len) -> (Batch_size, seq_len, dim)
            # dim = 768
            cond_context = clip(cond_tokens)

            # cond_tokens是我们的prompt，而uncond_tokens一般是空字符串，或者不想要的东西的prompt
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            # (2, seq_len, dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77, 768)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((HEIGHT, WIDTH))
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, channels)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, channels) -> (Batch_size, Height, Width, channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_size, Height, Width, channels) -> (Batch_size, channels, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            # 生成一个tensor，维度是latents_shape, 每个值从标准正态分布采样
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            latents = encoder(input_image_tensor, encoder_noise)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # If we are doing text-to-image, we start from a random noise N(0, I)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timesteps).to(device)

            # (Batch_size, 4, height, width)
            # model_input这个就是encoder的输出
            model_input = latents

            if do_cfg:
                # (Batch_size, 4, height, width) -> (2*Batch_size, 4, height, width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise of the UNET
            model_output = diffusion(model_input, context, timestep)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                # 根据cfg的公式
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # remove noise predicted by UNET
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_size, channels, Height, Width) -> (Batch_size, Height, Width, channels)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
      x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)