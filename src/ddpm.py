import torch
import numpy as np

class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float=0.00085, beta_end: float=0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        # 999，998，997, ... 0 = 1000 steps
        # 999, 999-20, 999-40 ... 0 = 50 steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep:int) -> int:
        prev_t = timestep - self.num_training_steps // self.num_inference_steps
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.startstep = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_outputs: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # x0
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_outputs) / alpha_prod_t ** 0.5

        # coefficients for x0 and xt
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t)  / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # predicted previous sample mean
        pred_prev_sample = pred_original_sample * pred_original_sample_coeff + latents * current_sample_coeff

        variance = 0
        if t > 0:
            device = model_outputs.device
            noise = torch.randn(model_outputs.shape, generator=self.generator, device=device, dtype=model_outputs.dtype)
            variance = self._get_variance(t)
            variance = variance ** 0.5 * noise

        # 为了从一个正态分布X~N(mu，sigma^2）中sample，我们先从Z～N(0，1）sample，再变换成N（mu，sigma）
        # 利用公式 X = mu + sigma * Z
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        """
        Build the mean and the variance, and sample from it.
        """
        # alpha_cumprod = \bar{\alpha_t}
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # mean
        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 噪声和图片的维度需要一致
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod) ** 0.5 # 标准差
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        # z= N(0,1) -> x= N(mean, variance)
        # x = mean + stddev * z
        noisy_samples = (original_samples + sqrt_alpha_prod) + sqrt_one_minus_alpha_prod * noise
        return noisy_samples