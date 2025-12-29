# Copyright 2025
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

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import torch

from fim_rlvr_lean4.lean_verifier import LeanVerifier
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

_MISSING_BLOCK = "[MISSING_BLOCK]"


def _extract_user_content(raw_prompt: Any) -> str | None:
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, dict):
        content = raw_prompt.get("content")
        return content if isinstance(content, str) else None
    if isinstance(raw_prompt, Iterable):
        for message in reversed(list(raw_prompt)):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
        # fallback to last message content
        for message in reversed(list(raw_prompt)):
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return None


def _split_fim_prompt(text: str) -> tuple[str, str] | None:
    if _MISSING_BLOCK not in text:
        return None
    prefix, suffix = text.split(_MISSING_BLOCK, 1)
    return prefix, suffix


@register("lean_verifier")
class LeanVerifierRewardManager(AbstractRewardManager):
    """Reward manager that compiles Lean code and rewards successful proofs."""

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any | None = None,
        reward_fn_key: str = "data_source",
        *,
        lean_env_path: str | None = None,
        verification_timeout: float | None = None,
        parallel_workers: int = 1,
        reward_success: float = 1.0,
        reward_failure: float = 0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.verification_timeout = verification_timeout
        self.parallel_workers = max(int(parallel_workers or 1), 1)
        self.reward_success = float(reward_success)
        self.reward_failure = float(reward_failure)

        if lean_env_path is None:
            lean_env_path = "verification_env"
        lean_env_path = os.path.abspath(os.path.expanduser(lean_env_path))
        if not os.path.isdir(lean_env_path):
            raise ValueError(
                f"lean_env_path does not exist or is not a directory: {lean_env_path}. "
                "Point it to the Lean project folder containing lakefile.lean."
            )
        self.verifier = LeanVerifier(lean_env_path)

    def _build_full_code(self, data_item, prompt_str: str, response_str: str) -> str:
        extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
        fim_prefix = data_item.non_tensor_batch.get("fim_prefix") or extra_info.get("fim_prefix")
        fim_suffix = data_item.non_tensor_batch.get("fim_suffix") or extra_info.get("fim_suffix")

        if fim_prefix is not None or fim_suffix is not None:
            return (fim_prefix or "") + (response_str or "") + (fim_suffix or "")

        raw_prompt = data_item.non_tensor_batch.get("raw_prompt")
        user_content = _extract_user_content(raw_prompt)
        if user_content:
            split = _split_fim_prompt(user_content)
            if split is not None:
                pre, suf = split
                return pre + (response_str or "") + suf

        split = _split_fim_prompt(prompt_str)
        if split is not None:
            pre, suf = split
            return pre + (response_str or "") + suf

        if prompt_str:
            return prompt_str + (response_str or "")
        return response_str or ""

    def _verify_code(self, code: str) -> bool:
        if not code.strip():
            return False
        ok, _ = self.verifier.verify(code, timeout=self.verification_timeout)
        return bool(ok)

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        codes: list[str] = []
        valid_response_lengths: list[int] = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            full_code = self._build_full_code(data_item, prompt_str, response_str)
            codes.append(full_code)
            valid_response_lengths.append(int(valid_response_length))

        if self.parallel_workers <= 1 or len(codes) <= 1:
            results = [self._verify_code(code) for code in codes]
        else:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                results = list(executor.map(self._verify_code, codes))

        for i, (ok, valid_response_length) in enumerate(zip(results, valid_response_lengths, strict=True)):
            reward = self.reward_success if ok else self.reward_failure
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {"lean_verifier_success": results},
            }
        else:
            return reward_tensor
