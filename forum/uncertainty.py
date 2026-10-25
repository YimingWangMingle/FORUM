"""Within-member Monte Carlo averaging BEFORE across-member disagreement."""
import torch


def disagreement(member_rewards, member_continuations, gamma, lambda_reward, lambda_value):
    # Population std under a uniformly sampled ensemble member, Eq. (14).
    u_reward = member_rewards.std(dim=0, unbiased=False)
    u_value = member_continuations.std(dim=0, unbiased=False)
    return lambda_reward*u_reward+lambda_value*gamma*u_value, u_reward, u_value


@torch.no_grad()
def estimate_uncertainty(models, target_raw_value, z, action, config, continuation_allowed=None):
    samples, batch = config.uncertainty_samples, len(z)
    repeated_z = z.repeat_interleave(samples, dim=0)
    repeated_a = action.repeat_interleave(samples, dim=0)
    rewards, continuations = [], []
    for model in models:
        reward, done, nxt = model.simulate(repeated_z, repeated_a)
        continuation = torch.zeros_like(reward)
        alive = ~done
        if continuation_allowed is not None:
            alive &= continuation_allowed.repeat_interleave(samples)
        if alive.any():
            continuation[alive] = target_raw_value(nxt[alive])
        rewards.append(reward.reshape(batch, samples).mean(-1))
        continuations.append(continuation.reshape(batch, samples).mean(-1))
    return disagreement(torch.stack(rewards), torch.stack(continuations), config.gamma,
                        config.lambda_reward, config.lambda_value)
