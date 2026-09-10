# SageMaker GRPO Model Trainer

This module provides a framework for training language models using Group Relative Policy Optimization (GRPO) on Amazon SageMaker. It supports distributed training, quantization, and various optimization techniques including LoRA (Low-Rank Adaptation).

## Overview

The model trainer implements reinforcement learning for language models using the following key components:

- GRPO (Group Relative Policy Optimization) for training
- LoRA for efficient fine-tuning
- MLflow for experiment tracking
- Weights & Biases for monitoring
- Pluggable reward functions, including LLM-as-judge (RLAIF)

## Directory Structure

```
fsdp/
├── scripts/
│   ├── reward_function/          # One reward per module, auto-discovered
│   │   ├── __init__.py           #   package docs + load_reward_functions()
│   │   ├── _registry.py          #   @register_reward decorators + discovery
│   │   ├── _common.py            #   completion/tool-call rendering helpers
│   │   ├── _judge.py             #   shared judge plumbing (rubric, retries)
│   │   ├── format_reward.py      #   "format"     <think>/<answer> blocks
│   │   ├── length_reward.py      #   "length"     completion length target
│   │   ├── rouge_reward.py       #   "rouge"      ROUGE-L vs an answer column
│   │   ├── judge_reward.py       #   "judge"      Bedrock Converse (RLAIF)
│   │   └── judge_http_reward.py  #   "judge_http" OpenAI-compatible endpoint
│   ├── requirements.txt          # Python dependencies
│   └── train.py                  # Main training script
├── model-trainer-notebook.ipynb  # Example notebook
└── README.md                     # This file
```

## Prerequisites

Install the required dependencies:

```bash
pip install -r scripts/requirements.txt
```

Key dependencies include:

- transformers==5.13.1
- peft==0.19.1
- accelerate==1.14.0
- trl==1.5.1
- sagemaker==3.16.0
- mlflow
- wandb
- rouge (optional, only for the `rouge` reward)

## Features

### Reward Functions

Rewards are selected with `reward_funcs`, a comma-separated list in `args.yaml`:

```yaml
reward_funcs: "format,rouge"
```

| Name | What it scores | Needs |
| --- | --- | --- |
| `format` | 1.0 per well-formed `<think>` / `<answer>` block, so 0.0–2.0 | — |
| `length` | Completion length as a fraction of `length_reward_target` characters | — |
| `rouge` | ROUGE-L precision of the answer against the reference | an `answer` dataset column, `rouge` installed |
| `judge` | An LLM scores the completion against a rubric, via Amazon Bedrock's Converse API | Bedrock access, `--judge_*` settings |
| `judge_http` | Same, against any OpenAI-compatible endpoint | `httpx`, an endpoint |

Adding a reward means adding a file to `scripts/reward_function/`; the decorator registers it and the package discovers it at import time, so no change to `train.py` is needed:

```python
# scripts/reward_function/my_reward.py
from ._common import extract_completion_text
from ._registry import register_reward

@register_reward("my_reward")
def my_reward_func(completions, **kwargs):
    return [1.0 if "yes" in extract_completion_text(c) else 0.0 for c in completions]
```

It is then selectable as `reward_funcs: "my_reward"`. A reward defined outside the
package can be referenced as `module.path:function_name`.

Two contract details worth knowing:

- A reward may return `None` for a sample to mean "does not apply here"; TRL turns that into `NaN` and drops it from that sample's reward sum instead of scoring it 0.0.
- TRL logs each reward as `rewards/<function __name__>/mean`, so a function's name is part of its observable contract.

### Training Optimizations

- 4-bit quantization using bitsandbytes
- Gradient checkpointing
- Flash Attention 2 support
- Distributed training with FSDP
- LoRA fine-tuning

### Monitoring & Tracking

- MLflow integration for experiment tracking
- Weights & Biases integration for training monitoring
- Custom GPU metrics logging

## Usage

1. Configure your training parameters in `args.yaml`
2. Prepare your dataset in JSON format
3. Run the training script:

```bash
python scripts/train.py --config args.yaml
```

### Key Configuration Parameters

- `model_id`: Hugging Face model identifier
- `reward_funcs`: comma-separated reward names (see the table above)
- `length_reward_target`: character count at which the `length` reward saturates
- `num_generations`: completions sampled per prompt — the group GRPO compares within
- `beta`: KL coefficient toward the reference policy (GRPO's, unrelated to DPO's beta)
- `lora_r`: LoRA rank
- `lora_alpha`: LoRA alpha parameter
- `lora_dropout`: LoRA dropout rate
- `temperature`: Sampling temperature — keep at 1.0; a low value makes every completion in a group agree, and a group with no reward variance produces no gradient
- `top_p`: Top-p sampling parameter
- `mlflow_uri`: MLflow tracking server URI
- `mlflow_experiment_name`: MLflow experiment name

The effective batch size (`num_processes × per_device_train_batch_size × gradient_accumulation_steps`) must be divisible by `num_generations`; TRL raises at startup otherwise. On one `ml.p4d.24xlarge` with the notebook's defaults that is 8 × 2 × 2 = 32 completions per step, i.e. 4 prompts per step at `num_generations: 8`.

## Model Saving

The trainer supports two saving modes:

1. Saving adapter weights separately (default)
2. Merging adapter weights with the base model (when `merge_weights=True`)

## Monitoring

### MLflow Integration

When enabled, the following is tracked:

- Training metrics
- Model parameters
- Dataset versions
- System metrics

### Weights & Biases Integration

When configured, monitors:

- Training progress
- GPU utilization
- Loss metrics
- Per-GPU metrics

## Error Handling

The trainer includes comprehensive error handling and logging:

- Detailed error messages
- Training state recovery
- Checkpoint management

## Contributing

When contributing to this project:

1. Follow the existing code style
2. Add appropriate error handling
3. Update documentation as needed
4. Add tests for new features
