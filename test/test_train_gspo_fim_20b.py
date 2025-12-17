import sys
import types
import unittest

# Stub heavy dependencies if they are missing so we can import the training script.


def _ensure_stubbed_dependencies():
    if "unsloth" not in sys.modules:
        unsloth = types.ModuleType("unsloth")

        class DummyFastLanguageModel:
            pass

        unsloth.FastLanguageModel = DummyFastLanguageModel
        sys.modules["unsloth"] = unsloth

    if "trl" not in sys.modules:
        trl = types.ModuleType("trl")
        trl.GRPOConfig = object
        trl.GRPOTrainer = object
        sys.modules["trl"] = trl

    if "datasets" not in sys.modules:
        datasets = types.ModuleType("datasets")

        def _fake_load_dataset(*args, **kwargs):
            raise RuntimeError("load_dataset should not be invoked in unit tests")

        datasets.load_dataset = _fake_load_dataset
        sys.modules["datasets"] = datasets


_ensure_stubbed_dependencies()

from train_gspo_fim_20b import build_dynamic_transform, lean_validity_reward_factory


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.calls.append(messages)
        return f"TEMPLATE::{messages[-1]['content']}"


class RecordingCurriculum:
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.ratio_requests = []
        self.updates = []

    def get_mask_ratio(self, theorem_id):
        self.ratio_requests.append(theorem_id)
        return self.ratio

    def update_outcome(self, theorem_id, success):
        self.updates.append((theorem_id, success))


class RecordingVerifier:
    def __init__(self, results):
        # results is a list of (success, output) tuples returned in order
        self.results = list(results)
        self.calls = []

    def verify(self, code):
        self.calls.append(code)
        return self.results.pop(0)


class TrainGspoFIM20BTests(unittest.TestCase):
    def test_dynamic_transform_builds_prompts_and_metadata(self):
        tokenizer = FakeTokenizer()
        curriculum = RecordingCurriculum(ratio=0.4)
        captured_full_codes = []

        def fake_reconstruct(raw_prompt):
            return "PRE_", "_SUF"

        def fake_mask(full_code, ratio):
            captured_full_codes.append((full_code, ratio))
            return "mask_pre\n", "mask_suf\n", "mask_mid\n"

        transform = build_dynamic_transform(
            tokenizer, curriculum, reconstruct_fn=fake_reconstruct, mask_fn=fake_mask
        )

        batch = {
            "prompt": ["<PFX>PRE_<SFX>_SUF<MID>"],
            "completion": ["MID"],
            "metadata": [{"theorem_name": "thm_custom"}],
        }

        output = transform(batch)

        self.assertEqual(len(output["prompt"]), 1)
        self.assertEqual(output["fim_prefix"][0], "mask_pre\n")
        self.assertEqual(output["fim_suffix"][0], "mask_suf\n")
        self.assertEqual(output["theorem_id"][0], "thm_custom")
        self.assertEqual(
            output["prompt"][0], "TEMPLATE::mask_pre\n[MISSING_BLOCK]\nmask_suf\n"
        )

        self.assertEqual(
            captured_full_codes,
            [("PRE_" + "MID" + "_SUF", 0.4)],
            "Masker should receive reconstructed full code and ratio",
        )
        self.assertEqual(curriculum.ratio_requests, ["thm_custom"])
        self.assertEqual(len(tokenizer.calls), 1)

    def test_dynamic_transform_defaults_theorem_id_to_index(self):
        tokenizer = FakeTokenizer()
        curriculum = RecordingCurriculum(ratio=0.1)

        transform = build_dynamic_transform(
            tokenizer,
            curriculum,
            reconstruct_fn=lambda raw: ("P_", "_S"),
            mask_fn=lambda full, ratio: ("pre", "suf", "mid"),
        )

        batch = {"prompt": ["<PFX>P_<SFX>_S<MID>"], "completion": ["X"], "metadata": [None]}
        output = transform(batch)

        self.assertEqual(output["theorem_id"][0], "0")
        self.assertEqual(curriculum.ratio_requests, ["0"])

    def test_lean_validity_reward_updates_curriculum_and_scores(self):
        verifier = RecordingVerifier(results=[(True, "ok")])
        curriculum = RecordingCurriculum()
        reward_fn = lean_validity_reward_factory(verifier, curriculum)

        completions = ["COMPLETE", "IGNORED"]
        fim_prefix = ["pre_", ""]
        fim_suffix = ["_suf", ""]
        theorem_ids = ["thm1", "thm2"]

        scores = reward_fn(
            completions, fim_prefix=fim_prefix, fim_suffix=fim_suffix, theorem_id=theorem_ids
        )

        self.assertEqual(scores, [2.0, 0.0])
        self.assertEqual(verifier.calls, ["pre_COMPLETE_suf"])
        self.assertEqual(
            curriculum.updates, [("thm1", True), ("thm2", False)], "Curriculum should be updated per sample"
        )


if __name__ == "__main__":
    unittest.main()
