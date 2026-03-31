import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, LCMScheduler


class DiffusionScheduler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.num_timesteps = config.num_timesteps
        self.scheduler_type = config.scheduler_type
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end

        if self.scheduler_type == "ddpm":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=self.num_timesteps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                prediction_type="epsilon"
            )
        elif self.scheduler_type == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=self.num_timesteps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                prediction_type="epsilon"
            )
        elif self.scheduler_type == "dpm":
            self.scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.num_timesteps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                prediction_type="epsilon"
            )
        elif self.scheduler_type == "pndm":
            self.scheduler = PNDMScheduler(
                num_train_timesteps=self.num_timesteps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                prediction_type="epsilon"
            )
        elif self.scheduler_type == "lcm":
            self.scheduler = LCMScheduler(
                num_train_timesteps=self.num_timesteps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                prediction_type="epsilon"
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def q_sample(self, x_start, t, noise):
        return self.scheduler.add_noise(x_start, noise, t)

    def step(self, model_output, t, sample):
        return self.scheduler.step(model_output, t, sample)
