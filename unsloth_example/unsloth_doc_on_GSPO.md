# GSPO Reinforcement Learning

We're introducing GSPO which is a variant of [GRPO](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/..#from-rlhf-ppo-to-grpo-and-rlvr) made by the Qwen team at Alibaba. They noticed the observation that when GRPO takes importance weights for each token, even though inherently advantages do not scale or change with each token. This lead to the creation of GSPO, which now assigns the importance on the sequence likelihood rather than the individual token likelihoods of the tokens.

* Use our free GSPO notebooks for: [**gpt-oss-20b**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-\(20B\)-GRPO.ipynb) and [**Qwen2.5-VL**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_5_7B_VL_GRPO.ipynb)

Enable GSPO in Unsloth by setting `importance_sampling_level = "sequence"` in the GRPO config. The difference between these two algorithms can be seen below, both from the GSPO paper from Qwen and Alibaba:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-45d743dd5dcd590626777ce09cfab61808aa8c24%2Fimage.png?alt=media" alt="" width="563"><figcaption><p>GRPO Algorithm, Source: <a href="https://arxiv.org/abs/2507.18071">Qwen</a></p></figcaption></figure>

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-ee755850cbe17482ce240dde227d55c62e9a3e64%2Fimage.png?alt=media" alt="" width="563"><figcaption><p>GSPO algorithm, Source: <a href="https://arxiv.org/abs/2507.18071">Qwen</a></p></figcaption></figure>

In Equation 1, it can be seen that the advantages scale each of the rows into the token logprobs before that tensor is sumed. Essentially, each token is given the same scaling even though that scaling was given to the entire sequence rather than each individual token. A simple diagram of this can be seen below:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-b3c944808a15dde0a7ff45782f9f074993304bf1%2FCopy%20of%20GSPO%20diagram%20(1).jpg?alt=media" alt="" width="286"><figcaption><p>GRPO Logprob Ratio row wise scaled with advantages</p></figcaption></figure>

Equation 2 shows that the logprob ratios for each sequence is summed and exponentiated after the Logprob ratios are computed, and only the resulting now sequence ratios get row wise multiplied by the advantages.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-62fc5b50921e79cce155d2794201c9b96faf941e%2FGSPO%20diagram%20(1).jpg?alt=media" alt="" width="313"><figcaption><p>GSPO Sequence Ratio row wise scaled with advantages</p></figcaption></figure>

Enabling GSPO is simple, all you need to do is set the `importance_sampling_level = "sequence"` flag in the GRPO config.

```python
training_args = GRPOConfig(
    output_dir = "vlm-grpo-unsloth",
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    # beta = 0.00,
    epsilon = 3e-4,
    epsilon_high = 4e-4,
    num_generations = 8,    
    max_prompt_length = 1024,
    max_completion_length = 1024,
    log_completions = False,
    max_grad_norm = 0.1,
    temperature = 0.9,
    # report_to = "none", # Set to "wandb" if you want to log to Weights & Biases
    num_train_epochs = 2, # For a quick test run, increase for full training
    report_to = "none"
    
    # GSPO is below:
    importance_sampling_level = "sequence",
    
    # Dr GRPO / GAPO etc
    loss_type = "dr_grpo",
)
```
