"""Conditional DDPM in the frozen encoder's latent coordinates, Eqs. (10)-(11)."""
import math
import torch
from torch import nn
from .networks import mlp


class DisplacementDiffusion(nn.Module):
    def __init__(self, latent, context, hidden, steps):
        super().__init__()
        self.steps = steps
        self.latent = latent
        # Cosine schedule is an explicit implementation choice; PDF gives no schedule.
        t = torch.linspace(0, steps, steps+1, dtype=torch.float64)/steps
        abar = torch.cos((t+.008)/1.008*math.pi/2).square()
        abar = abar/abar[0]
        beta = (1-abar[1:]/abar[:-1]).clamp(.0001, .999).float()
        alpha = 1-beta; cumulative = torch.cumprod(alpha, 0)
        previous = torch.cat([torch.ones(1), cumulative[:-1]])
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("cumulative", cumulative)
        self.register_buffer("posterior_variance", beta*(1-previous)/(1-cumulative))
        self.time = nn.Embedding(steps, 16)
        self.denoiser = mlp(latent+context+16, latent, hidden, depth=3)

    def predict(self, x, context, step):
        return self.denoiser(torch.cat([x, context, self.time(step)], -1))

    def loss(self, displacement, context):
        x = displacement.detach()
        t = torch.randint(self.steps, (len(x),), device=x.device)
        noise = torch.randn_like(x); a = self.cumulative[t, None]
        noisy = a.sqrt()*x+(1-a).sqrt()*noise
        # Eq. (11) sums the squared vector error, then averages observations.
        return (self.predict(noisy, context, t)-noise).square().sum(-1).mean()

    @torch.no_grad()
    def sample(self, context):
        x = torch.randn(len(context), self.latent, device=context.device)
        for step in reversed(range(self.steps)):
            t = torch.full((len(x),), step, device=x.device, dtype=torch.long)
            eps = self.predict(x, context, t)
            # Standard DDPM posterior mean, no hidden clean-state clipping.
            x = (x-self.beta[step]/(1-self.cumulative[step]).sqrt()*eps)/self.alpha[step].sqrt()
            if step:
                x = x+self.posterior_variance[step].sqrt()*torch.randn_like(x)
        return x
