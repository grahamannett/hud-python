from __future__ import annotations

import os
import time
from pathlib import Path
from contextlib import nullcontext
import pickle
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from hud.rl.config import Config, TrainingConfig
from hud.rl.logger import console
from hud.rl.loss import compute_loss, get_per_token_logps, entropy_from_logits
from hud.rl.metrics import MetricsCollector
from hud.rl.model import build_model
from hud.rl.optimizer import get_optimizer
from hud.rl.parallel_dims import ParallelDims
from hud.rl.utils import get_world_size, setup_distributed, is_main_process, get_rank
from hud.rl.checkpoint import CheckpointManager
from hud.rl.utils import save_step_metrics
from hud.rl.types import TrainingSample


def get_batch(step: int, root: str) -> list[TrainingSample]:
    """Load the batch for the given step from ``<output_dir>/step_{step}/rollouts``."""

    rank = get_rank()
    output_root = Path(root)
    step_rollout_dir = output_root / f"step_{step:05d}" / "rollouts"
    batch_path = step_rollout_dir / f"rank_{rank}.pt"

    console.info(f"Waiting for batch: {batch_path}")
    while not batch_path.exists():
        time.sleep(0.5)

    console.info_log(f"Loading batch from {batch_path}")
    return torch.load(batch_path, weights_only=False)

def train(
    config: TrainingConfig,
    max_steps: int,
) -> None:

    setup_distributed()
    world_size = get_world_size()

    console.section_title("Initializing trainer")

    parallel_dims = ParallelDims(
        dp_replicate=config.dp_replicate,
        dp_shard=config.dp_shard,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        world_size=world_size,
    )

    model = build_model(config, parallel_dims)

    ref_model: torch.nn.Module | None = None
    if config.loss.kl_beta > 0:
        console.info("Initializing reference model for KL regularization")
        ref_model = build_model(config, parallel_dims)
        ref_model.eval()

    optimizer = get_optimizer(config.optimizer, model)

    checkpoint_manager = CheckpointManager(
        output_dir=config.output_dir,
        save_last_n=config.save_last_n,
    )

    collector = MetricsCollector(distributed=(world_size > 1))

    record_function_ctx = nullcontext
    if config.profile:
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True).__enter__()
        torch.cuda.memory._record_memory_history(max_entries=100000)
        record_function_ctx = record_function

    for step in range(max_steps):

        torch.cuda.reset_peak_memory_stats()

        collector.reset()
        # Save checkpoint from previous step (skip first step since no training yet)
        if step > 0:
            console.info(f"Saving checkpoint for step {step - 1}...")
            checkpoint_manager.save(model, step - 1)
        
        batch = get_batch(step, config.output_dir)
                
        if ref_model is not None:
            with console.progress("Computing reference log probabilities...") as progress, torch.no_grad():
                for i, minibatch in enumerate(batch):
                    sample = minibatch.to_device(torch.device("cuda"))
                    logits = ref_model(**sample.inputs).logits
                    logits = torch.cat(
                        [torch.zeros_like(logits[:, :1, :]), logits[:, :-1, :]], dim=1
                    )
                    logits = logits / sample.temperature.view(-1, 1, 1)
                    sample.ref_logprobs = get_per_token_logps(
                        logits,
                        sample.inputs["input_ids"],
                    ).cpu()
                    del logits
                    progress.update(f"Computing reference log probabilities... {i + 1}/{len(batch)}")

        

        with console.progress("Computing old log probabilities...") as progress, torch.no_grad():
            for i, minibatch in enumerate(batch):
                sample = minibatch.to_device(torch.device("cuda"))
                with record_function_ctx("recompute_forward"):
                    logits = model(**sample.inputs).logits
                logits = torch.cat(
                    [torch.zeros_like(logits[:, :1, :]), logits[:, :-1, :]], dim=1
                )
                logits = logits / sample.temperature.view(-1, 1, 1)
                sample.old_logprobs = get_per_token_logps(
                    logits,
                    sample.inputs["input_ids"],
                ).cpu()
                del logits
                progress.update(f"Computing old log probabilities... {i + 1}/{len(batch)}")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        model.train()
        training_start_time = time.time()

        if config.loss.importance_sampling_level == "token":
            loss_norm = int(sum(
                minibatch.inputs["assistant_mask"].sum().item() for minibatch in batch
            ))
        elif config.loss.importance_sampling_level == "sequence":
            loss_norm = len(batch)

        with console.progress("Training...") as progress:
            for idx, minibatch in enumerate(batch):
                model.set_requires_all_reduce(idx == len(batch) - 1)

                sample = minibatch.to_device(torch.device("cuda"))
                with record_function_ctx("recompute_forward"):
                    logits = model(**sample.inputs).logits
                logits = torch.cat(
                    [torch.zeros_like(logits[:, :1, :]), logits[:, :-1, :]], dim=1
                )
                logits = logits / sample.temperature.view(-1, 1, 1)
                logprobs = get_per_token_logps(
                    logits,
                    sample.inputs["input_ids"],
                )

                # Compute entropy for masked tokens in chunks
                with torch.no_grad():
                    mask = sample.inputs["assistant_mask"].bool()
                    if mask.any():
                        ent = entropy_from_logits(logits, mask, chunk_size=128)
                        collector.log(entropy=ent)

                # old_logprobs is set in the previous loop, so it should not be None
                assert sample.old_logprobs is not None, "old_logprobs must be computed before training"

                loss, loss_tensors = compute_loss(
                    logprobs,
                    sample.old_logprobs,
                    sample.ref_logprobs,
                    sample.advantage,
                    sample.inputs["assistant_mask"],
                    config.loss,
                    loss_norm,
                )

                collector.log(loss=loss.detach().cpu())

                mask = sample.inputs["assistant_mask"].bool()
                for name, tensor in loss_tensors.items():
                    if isinstance(tensor, torch.Tensor) and tensor.shape == mask.shape:
                        collector.log(**{name: tensor[mask]})
                
                del logits
                with record_function_ctx("backward"):
                    loss.backward()
                progress.update(f"Trained... {idx + 1}/{len(batch)}")

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm
        )
        collector.log(grad_norm=grad_norm.detach())

        optimizer.step()
        optimizer.zero_grad()
        
        step_duration = time.time() - training_start_time
        console.info(f"Step {step} training took {step_duration:.2f} seconds")

        stats = collector.get_stats()
        if is_main_process():
            save_step_metrics(config.output_dir, step, stats)

        if config.profile:
            memory_path = Path(config.output_dir) / f"step_{step:05d}" / f"memory_{get_rank()}.pickle"
            with open(memory_path, "wb") as f:
                pickle.dump(torch.cuda.memory._snapshot(), f)

            prof.__exit__(None, None, None) 
            prof.export_chrome_trace(Path(config.output_dir) / f"step_{step:05d}" / f"trace_{get_rank()}.json.gz")
        
        
        torch.cuda.empty_cache()
    
    # Save final checkpoint after last training step
    if max_steps > 0:
        console.info(f"Saving final checkpoint for step {max_steps - 1}...")
        checkpoint_manager.save(model, max_steps - 1)


def main() -> None:
    """Main entry point for training script."""
    config, _ = Config.from_argv()
    
    if config.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    train(config.training, config.training_steps)


if __name__ == "__main__":
    main()
