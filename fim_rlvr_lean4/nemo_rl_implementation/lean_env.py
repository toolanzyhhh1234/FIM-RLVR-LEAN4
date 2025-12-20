from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import os

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from fim_rlvr_lean4.curriculum import CurriculumManager
from fim_rlvr_lean4.lean_verifier import LeanVerifier


class LeanEnvConfig(TypedDict):
    num_workers: int


@dataclass
class LeanEnvMetadata:
    theorem_id: str
    fim_prefix: str
    fim_suffix: str
    ground_truth: str


@ray.remote  # pragma: no cover
class LeanVerifyWorker:
    def __init__(self, lean_env_path: str) -> None:
        self.verifier = LeanVerifier(lean_env_path)

    def verify(self, code: str) -> bool:
        ok, _ = self.verifier.verify(code)
        return bool(ok)

    def verify_many(self, codes: list[str]) -> list[bool]:
        return [self.verify(code) for code in codes]


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class LeanEnvironment(EnvironmentInterface[LeanEnvMetadata]):
    def __init__(
        self,
        cfg: LeanEnvConfig,
        lean_env_path: str,
        curriculum_state_path: str,
    ) -> None:
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.curriculum_state_path = curriculum_state_path
        self.curriculum = (
            CurriculumManager.load(curriculum_state_path)
            if curriculum_state_path and curriculum_state_path
            else CurriculumManager()
        )

        self.workers = [
            LeanVerifyWorker.options(  # type: ignore
                runtime_env={
                    "py_executable": get_actor_python_env(
                        "nemo_rl.environments.code_environment.CodeEnvironment"
                    ),
                    "env_vars": dict(os.environ),
                }
            ).remote(lean_env_path)
            for _ in range(self.num_workers)
        ]

    def _verify_batch(self, codes: list[str]) -> list[bool]:
        if not codes:
            return []
        if self.num_workers <= 1:
            return ray.get(self.workers[0].verify_many.remote(codes))
        chunk_size = (len(codes) + self.num_workers - 1) // self.num_workers
        chunks = [
            codes[i : i + chunk_size] for i in range(0, len(codes), chunk_size)
        ]
        futures = []
        for worker, chunk in zip(self.workers, chunks):
            futures.append(worker.verify_many.remote(chunk))
        results: list[bool] = []
        for res in ray.get(futures):
            results.extend(res)
        return results

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[LeanEnvMetadata],
    ) -> EnvironmentReturn[LeanEnvMetadata]:
        completions = [m[-1]["content"] if m else "" for m in message_log_batch]
        codes: list[str] = []
        for completion, meta in zip(completions, metadata):
            full = (meta.get("fim_prefix", "") or "") + (completion or "") + (
                meta.get("fim_suffix", "") or ""
            )
            codes.append(full)

        results = self._verify_batch(codes)
        rewards = torch.tensor([1.0 if ok else 0.0 for ok in results]).cpu()
        done = torch.ones_like(rewards).cpu()

        for ok, meta in zip(results, metadata):
            self.curriculum.update_outcome(meta["theorem_id"], ok)

        observations = [
            {
                "role": "environment",
                "content": "Environment: correct" if ok else "Environment: incorrect",
            }
            for ok in results
        ]

        # save curriculum state for robustness
        if self.curriculum_state_path:
            self.curriculum.save(self.curriculum_state_path)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(metadata),
            rewards=rewards,
            terminateds=done,
            answers=[None] * len(metadata),
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        metrics = {
            "accuracy": batch["rewards"].mean().item(),
            "num_samples_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }
        return batch, metrics
