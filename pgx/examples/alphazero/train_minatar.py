# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import datetime
import os
import pickle
import time
import glob
from functools import partial
from typing import NamedTuple, Optional, Dict, Any, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
import wandb
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

from network import MiniAtarNet  # unchanged: policy+value network module

devices = jax.local_devices()
num_devices = len(devices)


class Config(BaseModel):
    # Use a single-player MiniAtar PGX env; change default as needed
    env_id: pgx.EnvId = "minatar-breakout" # specified via args
    seed: int = 0 # [123, 124, 125] are used in the submission
    max_num_iters: int = 100
    # network params
    num_channels: int = 32
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 256
    num_simulations: int = 64
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 1024
    learning_rate: float = 0.001
    gamma: float = 0.99  # single-player discount
    # eval params
    eval_interval: int = 5
    # policy
    policy: str = "puct" # Options: puct, p-uct, p-uct-bayes, p-uct-v, p-uct-tuned
    # checkpoint directory (for resuming from preemption)
    checkpoint_dir: str = None

    class Config:
        extra = "forbid"

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=False)
    p.add_argument("--env_id", type=str, required=True)
    p.add_argument("--policy", type=str, required=True,
                   choices=["puct", "p-uct", "p-uct-v", "puct-v"])
    p.add_argument("--checkpoint_dir", type=str, help="Directory for storing/loading checkpoints")
    args = p.parse_args()
    return Config(**vars(args))

config = parse_args()
print(config)

env = pgx.make(config.env_id)

# Find the latest checkpoint in the checkpoint directory
def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        return None

    # Sort by iteration number (assuming format: "iteration.ckpt")
    checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    return checkpoint_files[-1]

# Load checkpoint and return model state and training state
def load_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, Any], jnp.ndarray, Any, Any, int, int, float, Optional[str]]:
    print(f"Loading checkpoint from {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    return (
        checkpoint["model"],
        checkpoint.get("rng_key", jax.random.PRNGKey(0)),
        checkpoint["opt_state"],
        checkpoint["config"],
        checkpoint["iteration"],
        checkpoint["frames"],
        checkpoint["hours"],
        checkpoint.get("wandb_run_id", None)
    )

# Save checkpoint with wandb run ID
def save_checkpoint(checkpoint_dir: str, iteration: int, config: Config, rng_key: jnp.ndarray,
                   model, opt_state, frames: int, hours: float, wandb_run_id: str):
    checkpoint_path = os.path.join(checkpoint_dir, f"{iteration:06d}.ckpt")
    model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))

    with open(checkpoint_path, "wb") as f:
        dic = {
            "config": config,
            "rng_key": rng_key,
            "model": jax.device_get(model_0),
            "opt_state": jax.device_get(opt_state_0),
            "iteration": iteration,
            "frames": frames,
            "hours": hours,
            "pgx.__version__": pgx.__version__,
            "env_id": env.id,
            "env_version": env.version,
            "wandb_run_id": wandb_run_id,
        }
        pickle.dump(dic, f)

    print(f"Saved checkpoint to {checkpoint_path}")

def forward_fn(x, is_eval=False):
    # Keep the original AZNet head/signatures
    net = MiniAtarNet(
        num_actions=env.num_actions,
        # num_channels=config.num_channels,
        # num_blocks=config.num_layers,
        # resnet_v2=config.resnet_v2,
    )
    policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
    return policy_out, value_out

forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
optimizer = optax.adam(learning_rate=config.learning_rate)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state):
    """Called by mctx; receives a single rng_key for the whole batch.
    We split it into per-env keys before calling batched env.step."""
    model_params, model_state = model

    # Split the single PRNGKey into per-batch keys
    batch_size = state.observation.shape[0]
    step_keys = jax.random.split(rng_key, batch_size)

    # Step the environment (MiniAtar needs RNG)
    state = jax.vmap(env.step)(state, action, step_keys)

    # Inference
    (logits, value), _ = forward.apply(
        model_params, model_state, state.observation, is_eval=True
    )

    # Mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    # Single-player scalar reward
    reward = jnp.squeeze(state.rewards)

    # Zero value on terminals; positive discounting
    value = jnp.where(state.terminated, 0.0, value)
    discount = jnp.full_like(value, config.gamma)
    discount = jnp.where(state.terminated, 0.0, discount)

    return mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
        variance=jnp.zeros_like(value),
    ), state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    """Self-play/rollouts using Gumbel MuZero MCTS. Unchanged, except scalar rewards and γ."""
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        (logits, value), _ = forward.apply(model_params, model_state, observation, is_eval=True)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, variance=jnp.zeros_like(value), embedding=state)

        if config.policy == "p-uct":
            policy_output = mctx.muzero_p_uct_policy(
                params=model,
                rng_key=key1,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=~state.legal_action_mask,
                qtransform=mctx.qtransform_by_parent_and_siblings,
            )
        if config.policy == "puct":
            policy_output = mctx.muzero_puct_policy(
                params=model,
                rng_key=key1,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=~state.legal_action_mask,
                qtransform=mctx.qtransform_by_parent_and_siblings,
            )
        elif config.policy == "p-uct-v":
            policy_output = mctx.muzero_p_uct_v_policy(
                params=model,
                rng_key=key1,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=~state.legal_action_mask,
                qtransform=mctx.q_var_transform_by_parent_and_siblings,
            )
        elif config.policy == "puct-v":
            policy_output = mctx.muzero_puct_v(
                params=model,
                rng_key=key1,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=~state.legal_action_mask,
                qtransform=mctx.q_var_transform_by_parent_and_siblings,
            )

                    # step env with chosen actions; use auto_reset + RNG keys for batched stepping
        keys = jax.random.split(key2, batch_size)
        next_state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)

        # scalar reward and γ discount for targets
        reward = jnp.squeeze(next_state.rewards)
        discount = jnp.full_like(value, config.gamma)
        discount = jnp.where(next_state.terminated, 0.0, discount)

        return next_state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=reward,
            terminated=next_state.terminated,
            discount=discount,
        )

    # Run for max_num_steps
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices

    # mask value loss if episode truncated (positions after first terminal)
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # discounted return targets (backward scan): v_t = r_t + γ * v_{t+1}, zeroed after terminal
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(body_fn, jnp.zeros(batch_size), jnp.arange(config.max_num_steps))
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )

def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

    return policy_loss + value_loss, (model_state, policy_loss, value_loss)

@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss

@jax.pmap
def evaluate(rng_key, my_model):
    """Single-player evaluation by sampling the policy network (lightweight).
    For thorough eval, you could mirror selfplay with MCTS; this keeps the original pattern minimal."""
    my_model_params, my_model_state = my_model

    # batch of eval envs = selfplay batch per device
    batch_size = config.selfplay_batch_size // num_devices
    rng_key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    total_R = jnp.zeros(batch_size)

    def cond_fn(tup):
        _key, _state, _R = tup
        return ~_state.terminated.all()

    def body_fn(tup):
        key, state, R = tup
        (logits, _), _ = forward.apply(my_model_params, my_model_state, state.observation, is_eval=True)
        key, _rng = jax.random.split(key)
        action = jax.random.categorical(_rng, logits, axis=-1)

        # MiniAtar step needs RNG; provide per-env keys
        key, _rng2 = jax.random.split(key)
        step_keys = jax.random.split(_rng2, state.observation.shape[0])
        state = jax.vmap(env.step)(state, action, step_keys)

        R = R + jnp.squeeze(state.rewards)
        return (key, state, R)

    _, _, R = jax.lax.while_loop(cond_fn, body_fn, (rng_key, state, total_R))
    return R  # per-env episode return


if __name__ == "__main__":
    # Try to load a checkpoint and resume.
    wandb_run_id = None
    latest_checkpoint = None
    if config.checkpoint_dir:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        latest_checkpoint = find_latest_checkpoint(config.checkpoint_dir)
    if latest_checkpoint:
        # A checkpoint exists, so we're resuming.
        _, _, _, _, _, _, _, wandb_run_id = load_checkpoint(latest_checkpoint)

    wandb.init(
        project="pgx-az-minatar-extra",
        config=config.model_dump(),
        name=f"{config.policy}-{config.env_id}",
        group=config.env_id,
        job_type=config.policy,
        tags=[config.env_id, config.policy, "minatar", "pgx"],
        id=wandb_run_id,
        resume="allow",
    )
    if wandb_run_id is None:
        wandb_run_id = wandb.run.id

    # Initialize model and optimizer state
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = dummy_state.observation
    model = forward.init(jax.random.PRNGKey(0), dummy_input)  # (params, state)
    opt_state = optimizer.init(params=model[0])

    if latest_checkpoint:
        # Resume from checkpoint
        model_dict, rng_key, opt_state, loaded_config, iteration, frames, hours, _ = load_checkpoint(latest_checkpoint)
        print(f"Resuming from iteration {iteration}, frames {frames}")
        iteration += 1  # Start from the next iteration

        # The loaded model is a (params, state) tuple.
        model = model_dict
        # Use loaded config except for checkpoint_dir
        for key, value in loaded_config.model_dump().items():
            if key != "checkpoint_dir":
                setattr(config, key, value)
    else:
        # Logging state
        iteration: int = 0
        hours: float = 0.0
        frames: int = 0
        rng_key = jax.random.PRNGKey(config.seed)

    # replicate to devices
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    while True:
        if iteration % config.eval_interval == 0:
            # Evaluation: report returns (mean/std/min/max)
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R = evaluate(keys, model)
            log.update(
                {
                    "eval/return/mean": R.mean().item(),
                    "eval/return/std": R.std().item(),
                    "eval/return/min": R.min().item(),
                    "eval/return/max": R.max().item(),
                }
            )

            # Store checkpoint
            if config.checkpoint_dir:
                save_checkpoint(
                    config.checkpoint_dir,
                    iteration,
                    config,
                    rng_key,
                    model,
                    opt_state,
                    frames,
                    hours,
                    wandb_run_id
                )

        print(log)
        wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Selfplay with MCTS
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model, keys)
        samples: Sample = compute_loss_input(data)

        # Shuffle samples and make minibatches
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        # Training
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        )
