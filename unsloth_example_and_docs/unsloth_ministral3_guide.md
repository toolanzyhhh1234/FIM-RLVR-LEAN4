# Ministral 3 - How to Run Guide

istral releases Ministral 3, their new multimodal models in Base, Instruct, and Reasoning variants, available in **3B**, **8B**, and **14B** sizes. They offer best-in-class performance for their size, and are fine-tuned for instruction and chat use cases. The multimodal models support **256K context** windows, multiple languages, native function calling, and JSON output.

The full unquantized 14B Ministral-3-Instruct-2512 model fits in **24GB RAM**/VRAM. You can now run, fine-tune and RL on all Ministral 3 models with Unsloth:

<a href="#run-ministral-3-tutorials" class="button primary">Run Ministral 3 Tutorials</a><a href="#fine-tuning" class="button primary">Fine-tuning Ministral 3</a>

We've also uploaded Mistral Large 3 [GGUFs here](https://huggingface.co/unsloth/Mistral-Large-3-675B-Instruct-2512-GGUF). For all Ministral 3 uploads (BnB, FP8), [see here](https://huggingface.co/collections/unsloth/ministral-3).

| Ministral-3-Instruct GGUFs:                                                                                                                                                                                               | Ministral-3-Reasoning GGUFs:                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [3B](https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF) • [8B](https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF) • [14B](https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF) | [3B](https://huggingface.co/unsloth/Ministral-3-3B-Reasoning-2512-GGUF) • [8B](https://huggingface.co/unsloth/Ministral-3-8B-Reasoning-2512-GGUF) • [14B](https://huggingface.co/unsloth/Ministral-3-14B-Reasoning-2512-GGUF) |

### ⚙️ Usage Guide

To achieve optimal performance for **Instruct**, Mistral recommends using lower temperatures such as `temperature = 0.15` or `0.1`<br>

For **Reasoning**, Mistral recommends `temperature = 0.7` and `top_p = 0.95`.

| Instruct:                     | Reasoning:          |
| ----------------------------- | ------------------- |
| `Temperature = 0.15` or `0.1` | `Temperature = 0.7` |
| `Top_P = default`             | `Top_P = 0.95`      |

**Adequate Output Length**: Use an output length of `32,768` tokens for most queries for the reasoning variant, and `16,384` for the instruct variant. You can increase the max output size for the reasoning model if necessary.

The maximum context length Ministral 3 can reach is `262,144`

The chat template format is found when we use the below:

{% code overflow="wrap" %}

```python
tokenizer.apply_chat_template([
    {"role" : "user", "content" : "What is 1+1?"},
    {"role" : "assistant", "content" : "2"},
    {"role" : "user", "content" : "What is 2+2?"}
    ], add_generation_prompt = True
)
```

{% endcode %}

#### Ministral *Reasoning* chat template:

{% code overflow="wrap" lineNumbers="true" %}

```
<s>[SYSTEM_PROMPT]# HOW YOU SHOULD THINK AND ANSWER

First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response.[/SYSTEM_PROMPT][INST]What is 1+1?[/INST]2</s>[INST]What is 2+2?[/INST]
```

{% endcode %}

#### Ministral *Instruct* chat template:

{% code overflow="wrap" lineNumbers="true" expandable="true" %}

```
<s>[SYSTEM_PROMPT]You are Ministral-3-3B-Instruct-2512, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.
The current date is {today}.

When you're not sure about some information or when the user's request requires up-to-date or specific data, you must use the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or complete response. If no relevant tools are available, then clearly state that you don't have the information and avoid making up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have.

# WEB BROWSING INSTRUCTIONS

You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.

# MULTI-MODAL INSTRUCTIONS

You have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.
You cannot read nor transcribe audio files or videos.

# TOOL CALLING INSTRUCTIONS

You may have access to tools that you can use to fetch information or perform actions. You must use these tools in the following situations:

1. When the request requires up-to-date information.
2. When the request requires specific data that you do not have in your knowledge base.
3. When the request involves actions that you cannot perform without tools.

Always prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the user that you cannot perform the requested action at the moment.[/SYSTEM_PROMPT][INST]What is 1+1?[/INST]2</s>[INST]What is 2+2?[/INST]
```

{% endcode %}

## 📖 Run Ministral 3 Tutorials

Below are guides for the [Reasoning](#reasoning-ministral-3-reasoning-2512) and [Instruct](#instruct-ministral-3-instruct-2512) variants of the model.

### Instruct: Ministral-3-Instruct-2512

To achieve optimal performance for **Instruct**, Mistral recommends using lower temperatures such as `temperature = 0.15` or `0.1`

#### :sparkles: Llama.cpp: Run Ministral-3-14B-Instruct Tutorial

{% stepper %}
{% step %}
Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

{% code overflow="wrap" %}

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

{% endcode %}
{% endstep %}

{% step %}
You can directly pull from Hugging Face via:

```bash
./llama.cpp/llama-cli \
    -hf unsloth/Ministral-3-14B-Instruct-2512-GGUF:Q4_K_XL \
    --jinja -ngl 99 --threads -1 --ctx-size 32684 \
    --temp 0.15
```

{% endstep %}

{% step %}
Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose `UD_Q4_K_XL` or other quantized versions.

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Ministral-3-14B-Instruct-2512-GGUF",
    local_dir = "Ministral-3-14B-Instruct-2512-GGUF",
    allow_patterns = ["*UD-Q4_K_XL*"],
)
```

{% endstep %}
{% endstepper %}

### Reasoning: Ministral-3-Reasoning-2512

To achieve optimal performance for **Reasoning**, Mistral recommends using `temperature = 0.7` and `top_p = 0.95`.

#### :sparkles: Llama.cpp: Run Ministral-3-14B-Reasoning Tutorial

{% stepper %}
{% step %}
Obtain the latest `llama.cpp` on [GitHub](https://github.com/ggml-org/llama.cpp). You can also use the build instructions below. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

{% code overflow="wrap" %}

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

{% endcode %}
{% endstep %}

{% step %}
You can directly pull from Hugging Face via:

```bash
./llama.cpp/llama-cli \
    -hf unsloth/Ministral-3-14B-Reasoning-2512-GGUF:Q4_K_XL \
    --jinja -ngl 99 --threads -1 --ctx-size 32684 \
    --temp 0.6 --top-p 0.95
```

{% endstep %}

{% step %}
Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose `UD_Q4_K_XL` or other quantized versions.

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Ministral-3-14B-Reasoning-2512-GGUF",
    local_dir = "Ministral-3-14B-Reasoning-2512-GGUF",
    allow_patterns = ["*UD-Q4_K_XL*"],
)
```

{% endstep %}
{% endstepper %}

## 🛠️ Fine-tuning Ministral 3 <a href="#fine-tuning" id="fine-tuning"></a>

Unsloth now supports fine-tuning of all Ministral 3 models, including vision support. To train, you must use the latest 🤗Hugging Face `transformers` v5 and `unsloth` which includes our our recent [ultra long context](https://docs.unsloth.ai/new/500k-context-length-fine-tuning) support. The large 14B Ministral 3 model should fit on a free Colab GPU.

We made free Unsloth notebooks to fine-tune Ministral 3. Change the name to use the desired model.

* Ministral-3B-Instruct [Vision notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Ministral_3_VL_\(3B\)_Vision.ipynb) (vision)
* Ministral-3B-Instruct [GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Ministral_3_\(3B\)_Reinforcement_Learning_Sudoku_Game.ipynb)

{% columns %}
{% column %}
Ministral Vision finetuning notebook

{% embed url="<https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Ministral_3_VL_(3B)_Vision.ipynb>" %}
{% endcolumn %}

{% column %}
Ministral Sudoku GRPO RL notebook

{% embed url="<https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Ministral_3_(3B)_Reinforcement_Learning_Sudoku_Game.ipynb>" %}
{% endcolumn %}
{% endcolumns %}

### :sparkles:Reinforcement Learning (GRPO)

Unsloth now supports RL and GRPO for the Mistral models as well. As usual, they benefit from all of Unsloth's enhancements and tomorrow, we are going to release a notebook soon specifically for autonomously solving the sudoku puzzle.

* Ministral-3B-Instruct [GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Ministral_3_\(3B\)_Reinforcement_Learning_Sudoku_Game.ipynb)

**To use the latest version of Unsloth and transformers v5, update via:**

{% code overflow="wrap" %}

```
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
```

{% endcode %}

The goal is to auto generate strategies to complete Sudoku!

{% columns %}
{% column %}

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2F2qDbhHfpuhNAHOtIernm%2Fimage.png?alt=media&#x26;token=9a3d4bb2-3994-4ec8-aeb8-16bc2bcb77c4" alt=""><figcaption></figcaption></figure>
{% endcolumn %}

{% column %}

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FLZlHHeAjoVAeO6juQDiC%2Fimage.png?alt=media&#x26;token=45abbb30-b705-4eec-81fc-fb99dd0c2621" alt=""><figcaption></figcaption></figure>
{% endcolumn %}
{% endcolumns %}

For the reward plots for Ministral, we get the below. We see it works well!

{% columns %}
{% column %}
![](https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FqpfPNKkSF2O1T0flshEi%2Funknown.png?alt=media\&token=a2f14139-bcab-40bf-a054-f189de5d23df)

![](https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fe8TBzOVVn5iYhlJ6nh63%2Funknown.png?alt=media\&token=520699f9-ffd0-43a5-a0ef-263fa678b4bd)
{% endcolumn %}

{% column %}
![](https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FudSxKSBuSOIXONrtarmp%2Funknown.png?alt=media\&token=beefcbce-67df-4ce2-92b8-3e0adc240df6)

![](https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FgwwlcVjMt9nqyqVC6xqD%2Funknown.png?alt=media\&token=b5b390b6-c9e6-4926-9a70-d4aa365caa86)
{% endcolumn %}
{% endcolumns %}
