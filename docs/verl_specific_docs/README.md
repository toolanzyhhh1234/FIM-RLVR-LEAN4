# verl notes

- 2025-12-24: The GSPO example script `verl/examples/gspo_trainer/run_qwen30b_gspo.sh` sets
  `trainer.n_gpus_per_node=$ARNOLD_WORKER_GPU` and `trainer.nnodes=$ARNOLD_WORKER_NUM`.
  If those env vars are unset, Hydra receives empty strings and `validate_config` raises
  `TypeError: can't multiply sequence by non-int of type 'str'`. Fix by exporting
  `ARNOLD_WORKER_GPU=1` and `ARNOLD_WORKER_NUM=1`, or default them in the script.
