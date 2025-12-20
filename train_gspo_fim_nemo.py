import argparse
import os
import pprint
import sys
from functools import partial

from omegaconf import OmegaConf


def _add_nemo_rl_to_path() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    nemo_rl_root = os.path.join(repo_root, "RL")
    if nemo_rl_root not in sys.path:
        sys.path.append(nemo_rl_root)


_add_nemo_rl_to_path()

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup  # noqa: E402
from nemo_rl.algorithms.utils import get_tokenizer  # noqa: E402
from nemo_rl.data.datasets import AllTaskProcessedDataset  # noqa: E402
from nemo_rl.data.interfaces import TaskDataSpec  # noqa: E402
from nemo_rl.distributed.virtual_cluster import init_ray  # noqa: E402
from nemo_rl.distributed.ray_actor_environment_registry import (  # noqa: E402
    get_actor_python_env,
)
from nemo_rl.models.generation import configure_generation_config  # noqa: E402
from nemo_rl.utils.config import load_config, parse_hydra_overrides  # noqa: E402
from nemo_rl.utils.logger import get_next_experiment_dir  # noqa: E402

from fim_rlvr_lean4.curriculum import CurriculumManager  # noqa: E402
from fim_rlvr_lean4.nemo_rl_implementation.fim_data import (  # noqa: E402
    CurriculumStateReader,
    filter_valid_rows,
    fim_data_processor,
    load_fim_dataset,
)
from fim_rlvr_lean4.nemo_rl_implementation.lean_env import (  # noqa: E402
    LeanEnvironment,
)

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run GSPO training for FIM Lean4 with NeMo-RL"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "fim_rlvr_lean4",
            "nemo_rl_implementation",
            "configs",
            "gspo_fim_lean4.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO/GSPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    fim_cfg = config["fim"]
    parquet_path = fim_cfg.get(
        "parquet_path",
        os.environ.get(
            "FIM_PARQUET_PATH",
            "hf://datasets/AI-MO/NuminaMath-LEAN/data/train-00000-of-00001.parquet",
        ),
    )
    curriculum_state_path = fim_cfg.get(
        "curriculum_state_path",
        os.path.join(config["checkpointing"]["checkpoint_dir"], "curriculum_state.json"),
    )

    curriculum_dir = os.path.dirname(curriculum_state_path)
    if curriculum_dir:
        os.makedirs(curriculum_dir, exist_ok=True)
    if not os.path.exists(curriculum_state_path):
        CurriculumManager().save(curriculum_state_path)

    dataset = filter_valid_rows(load_fim_dataset(parquet_path))

    curriculum_reader = CurriculumStateReader(curriculum_state_path)
    task_spec = TaskDataSpec(task_name="fim_lean4")
    processor = partial(
        fim_data_processor,
        curriculum_reader=curriculum_reader,
    )

    train_dataset = AllTaskProcessedDataset(
        dataset,
        tokenizer,
        task_spec,
        processor,
        max_seq_length=config["data"]["max_input_seq_length"],
    )

    val_dataset = None

    lean_env = LeanEnvironment.options(  # type: ignore
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.code_environment.CodeEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(
        config["env"]["lean"],
        fim_cfg["lean_env_path"],
        curriculum_state_path,
    )

    (
        policy,
        policy_generation,
        _cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    print("🚀 Running GSPO training (NeMo-RL GRPO with GSPO loss config)")

    task_to_env = {"fim_lean4": lean_env}
    val_task_to_env = {"fim_lean4": lean_env}

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
