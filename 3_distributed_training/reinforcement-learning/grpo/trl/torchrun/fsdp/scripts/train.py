from accelerate import Accelerator
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import datetime
from functools import lru_cache
from huggingface_hub import snapshot_download
import importlib
import json
import logging
import mlflow
from mlflow.models import infer_signature
import os
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
)
import subprocess
import sys
import textwrap
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Mxfp4Config,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer, TrlParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import WandbCallback
import contextlib
from typing import Any, Callable, Dict, List, Optional, Tuple
import wandb

try:
    from distutils.util import strtobool
except ImportError:  # distutils was removed from the stdlib in Python 3.12

    def strtobool(val):
        """String truthy/falsey -> 1/0 (distutils.util.strtobool replacement)."""
        val = str(val).strip().lower()
        if val in ("y", "yes", "t", "true", "on", "1"):
            return 1
        if val in ("n", "no", "f", "false", "off", "0"):
            return 0
        raise ValueError(f"invalid truth value {val!r}")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def resolve_trust_remote_code(
    model_id: str, token: Optional[str] = None
) -> Optional[bool]:
    """Whether the repo's custom modeling code must be executed for `model_id`.

    Returns `True` when the architecture has no native implementation, and `None`
    when the argument should not be passed at all - either because a native
    implementation exists or because `config.json` could not be read.

    Transformers gives the remote `modeling_*.py` precedence over its own
    implementation whenever `trust_remote_code=True` and the repo ships an
    `auto_map`. That pins the model to the Transformers snapshot it was uploaded
    with, which typically predates the AttentionInterface: no
    `_supports_flash_attn` / `_supports_sdpa` flags (so `attn_implementation` is
    rejected and only `eager` works), and legacy attention masks. So only opt in
    when there is no native implementation for the architecture.
    """
    try:
        from transformers import PreTrainedConfig
    except ImportError:  # transformers < 5.0
        from transformers import PretrainedConfig as PreTrainedConfig
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        )
    except ImportError:
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = {}

    try:
        # Reads config.json only - executes no repository code.
        config_dict, _ = PreTrainedConfig.get_config_dict(model_id, token=token)
    except Exception as e:
        logger.warning(
            f"Could not inspect config.json for {model_id} ({e}); leaving "
            "trust_remote_code unset so Transformers resolves it"
        )
        return None

    model_type = config_dict.get("model_type")
    if (
        model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        or model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    ):
        return None
    return True if config_dict.get("auto_map") else None


def trust_remote_code_kwargs(
    model_id: str,
    token: Optional[str] = None,
    override: Optional[bool] = None,
) -> Dict[str, Any]:
    """`{"trust_remote_code": ...}` only when the argument needs to be passed.

    Omitting it is not the same as passing `False`: unset lets Transformers use
    its own resolution (native class when available, and an actionable
    "Please pass the argument `trust_remote_code=True`" when custom code is
    genuinely required), while an explicit `False` refuses custom code outright
    and reports the vaguer "Unrecognized configuration class". Loaders that do
    not accept the argument also stay unaffected when it is left out.
    """
    value = (
        override if override is not None else resolve_trust_remote_code(model_id, token)
    )
    return {} if value is None else {"trust_remote_code": bool(value)}


@dataclass
class ScriptArguments:
    """Arguments for the script execution."""

    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "Attention implementation"}
    )
    auto_calculate_lengths: bool = field(
        default=False,
        metadata={
            "help": "Auto-calculate max_completion_length from dataset prompt lengths"
        },
    )
    checkpoint_dir: str = field(default=None, metadata={"help": "Checkpoint directory"})
    deserialize_messages: bool = field(
        default=False, metadata={"help": "Deserialize JSON-encoded prompt field"}
    )
    early_stopping: bool = field(
        default=False, metadata={"help": "Whether to use early stopping"}
    )
    use_checkpoints: bool = field(
        default=False, metadata={"help": "Whether to use checkpointing"}
    )
    load_in_4bit: bool = field(
        default=True, metadata={"help": "Load model in 4-bit quantization"}
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "lora_r"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora_alpha"})
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "lora_dropout"}
    )
    merge_weights: Optional[bool] = field(
        default=False, metadata={"help": "Merge adapter with base model"}
    )
    mlflow_uri: Optional[str] = field(
        default=None, metadata={"help": "MLflow tracking ARN"}
    )
    mlflow_experiment_name: Optional[str] = field(
        default=None, metadata={"help": "MLflow experiment name"}
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for GRPO training"}
    )
    prompt_field: str = field(
        default="prompt", metadata={"help": "Field name for prompt in dataset"}
    )
    reward_funcs: Optional[str] = field(
        default="format,length",
        metadata={
            "help": "Comma-separated reward functions: format,length,rouge or module:path.to.func"
        },
    )
    target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Target modules for LoRA"}
    )
    token: str = field(default=None, metadata={"help": "Hugging Face API token"})
    trust_remote_code: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Execute the repository's custom modeling code. Leave unset to "
                "auto-detect: the native Transformers implementation is preferred "
                "whenever one exists, and custom code is only used for "
                "architectures Transformers does not implement. Forcing true on a "
                "natively supported model silently downgrades it to a stale "
                "snapshot (breaks flash_attention_2/3 and sdpa)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Torch dtype (auto, bfloat16, float16, float32)"},
    )
    train_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the training dataset"}
    )
    use_mxfp4: bool = field(
        default=False,
        metadata={"help": "Use MXFP4 quantization instead of BitsAndBytes"},
    )
    use_peft: bool = field(default=True, metadata={"help": "Use PEFT for training"})
    use_snapshot_download: bool = field(
        default=False,
        metadata={"help": "Use snapshot download instead of Hugging Face Hub"},
    )
    val_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the val dataset"}
    )
    wandb_token: str = field(default="", metadata={"help": "Wandb API token"})
    wandb_project: str = field(
        default="project", metadata={"help": "Wandb project name"}
    )
    patch_peft_fsdp_auto_wrap_policy: bool = field(
        default=False,
        metadata={
            "help": (
                "Patch PEFT's FSDP auto-wrap policy for architectures PEFT doesn't "
                "recognize. FSDP + LoRA only."
            )
        },
    )
    cast_parameters_to_uniform_dtype: bool = field(
        default=False,
        metadata={
            "help": (
                "Cast all model parameters to uniform dtype. Required for models "
                "with mixed float32/bfloat16 parameters. "
                "Needed for both FSDP and DeepSpeed."
            )
        },
    )


def trust_remote_code_for(script_args: ScriptArguments) -> Dict[str, Any]:
    """`trust_remote_code` kwargs for the run's model: explicit config value wins."""
    return trust_remote_code_kwargs(
        script_args.model_id,
        script_args.token,
        override=script_args.trust_remote_code,
    )


class ModelConfigBuilder:
    """Centralized model configuration builder to eliminate duplicate logic."""

    def __init__(self, script_args: ScriptArguments, training_args: GRPOConfig):
        self.script_args = script_args
        self.training_args = training_args
        self._torch_dtype = None
        self._quantization_config = None
        self._use_deepspeed = None
        self._use_fsdp = None
        self._trust_remote_code = None

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get torch dtype with single source of truth."""
        if self._torch_dtype is None:
            if self.script_args.torch_dtype in ["auto", None]:
                self._torch_dtype = (
                    torch.bfloat16 if self.training_args.bf16 else torch.float32
                )
            else:
                self._torch_dtype = getattr(torch, self.script_args.torch_dtype)
        return self._torch_dtype

    @property
    def trust_remote_code(self) -> Dict[str, Any]:
        """Resolve the `trust_remote_code` kwargs once per run (may be empty)."""
        if self._trust_remote_code is None:
            self._trust_remote_code = trust_remote_code_for(self.script_args)
            logger.info(
                f"Model loading kwargs {self._trust_remote_code or '{}'} for "
                f"{self.script_args.model_id}"
            )
        return self._trust_remote_code

    @property
    def use_deepspeed(self) -> bool:
        """Check if DeepSpeed is enabled."""
        if self._use_deepspeed is None:
            self._use_deepspeed = strtobool(
                os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")
            )
        return self._use_deepspeed

    @property
    def use_fsdp(self) -> bool:
        """Check if FSDP is enabled."""
        if self._use_fsdp is None:
            self._use_fsdp = strtobool(os.environ.get("ACCELERATE_USE_FSDP", "false"))
        return self._use_fsdp

    @property
    def quantization_config(self) -> Optional[Any]:
        """Get quantization configuration."""
        if self._quantization_config is None and self.script_args.load_in_4bit:
            if self.script_args.use_mxfp4:
                self._quantization_config = Mxfp4Config(dequantize=True)
                logger.info("Using MXFP4 quantization")
            else:
                self._quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_quant_storage=self.torch_dtype,
                )
                logger.info("Using BitsAndBytes quantization")
        return self._quantization_config

    def build_model_kwargs(self) -> Dict[str, Any]:
        """Build complete model loading arguments."""
        if (
            self.script_args.attn_implementation is not None
            and self.script_args.attn_implementation != ""
        ):
            model_kwargs = {
                "attn_implementation": self.script_args.attn_implementation,
                "torch_dtype": self.torch_dtype,
                "use_cache": False,  # GRPO requires use_cache=False
                "cache_dir": "/tmp/.cache",
                **self.trust_remote_code,
            }
        else:
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "use_cache": False,  # GRPO requires use_cache=False
                "cache_dir": "/tmp/.cache",
                **self.trust_remote_code,
            }

        # Set low_cpu_mem_usage based on DeepSpeed usage
        if not self.use_deepspeed:
            model_kwargs["low_cpu_mem_usage"] = True

        # Add quantization config if enabled
        if self.quantization_config is not None:
            model_kwargs["quantization_config"] = self.quantization_config

        return model_kwargs

    def build_trainer_kwargs(self) -> Dict[str, Any]:
        """Build trainer-specific configuration."""
        trainer_kwargs = {}

        if self.use_fsdp or (self.training_args.fsdp and self.training_args.fsdp != ""):
            logger.info("Using FSDP configuration")
            if self.training_args.gradient_checkpointing_kwargs is None:
                trainer_kwargs["gradient_checkpointing_kwargs"] = {
                    "use_reentrant": False
                }
        elif self.use_deepspeed:
            logger.info("Using DeepSpeed configuration")
        else:
            logger.info("Using DDP configuration")
            if self.training_args.gradient_checkpointing_kwargs is None:
                trainer_kwargs["gradient_checkpointing_kwargs"] = {
                    "use_reentrant": False
                }

        return trainer_kwargs


class CustomWandbCallback(WandbCallback):
    """Custom Wandb callback that logs metrics for all GPUs."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            logs = {f"gpu_{i}_{k}": v for i in range(8) for k, v in logs.items()}
            super().on_log(args, state, control, model, logs, **kwargs)


@contextlib.contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory cleanup."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                f"GPU memory freed: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated"
            )


@contextlib.contextmanager
def model_lifecycle(model_name: str):
    """Context manager for model loading/cleanup lifecycle."""
    model = None
    try:
        logger.info(f"Loading model: {model_name}")
        yield model
    except Exception as e:
        logger.error(f"Error in model lifecycle for {model_name}: {e}")
        raise
    finally:
        if model is not None:
            logger.info(f"Cleaning up model: {model_name}")
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def download_model(model_name):
    print("Downloading model ", model_name)
    os.makedirs("/tmp/tmp_folder", exist_ok=True)
    snapshot_download(repo_id=model_name, local_dir="/tmp/tmp_folder")
    print(f"Model {model_name} downloaded under /tmp/tmp_folder")


def set_custom_env(env_vars: Dict[str, str]) -> None:
    """Set custom environment variables."""
    if not isinstance(env_vars, dict):
        raise TypeError("env_vars must be a dictionary")

    for key, value in env_vars.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("All keys and values in env_vars must be strings")

    os.environ.update(env_vars)
    print("Updated environment variables:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")


def is_mlflow_enabled(script_args: ScriptArguments) -> bool:
    """Check if MLflow is enabled based on script arguments."""
    return (
        script_args.mlflow_uri is not None
        and script_args.mlflow_experiment_name is not None
        and script_args.mlflow_uri != ""
        and script_args.mlflow_experiment_name != ""
    )


def setup_mlflow(script_args: ScriptArguments) -> None:
    """Set up MLflow tracking."""
    if not is_mlflow_enabled(script_args):
        return

    logger.info("Initializing MLflow")
    mlflow.enable_system_metrics_logging()
    mlflow.autolog()
    mlflow.set_tracking_uri(script_args.mlflow_uri)
    mlflow.set_experiment(script_args.mlflow_experiment_name)

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    set_custom_env(
        {
            "MLFLOW_RUN_NAME": f"GRPO-{formatted_datetime}",
            "MLFLOW_EXPERIMENT_NAME": script_args.mlflow_experiment_name,
        }
    )


def setup_wandb(script_args: ScriptArguments) -> None:
    """Set up Weights & Biases tracking."""
    if script_args.wandb_token and script_args.wandb_token != "":
        logger.info("Initializing Wandb")
        set_custom_env({"WANDB_API_KEY": script_args.wandb_token})
        wandb.init(project=script_args.wandb_project)
        return [CustomWandbCallback()]
    else:
        set_custom_env({"WANDB_DISABLED": "true"})
        return None


def patch_peft_fsdp_auto_wrap_policy():
    """Patch PEFT's fsdp_auto_wrap_policy for model architectures that PEFT doesn't recognize.

    PEFT's implementation inspects the model to find the transformer layer class but fails
    on newer architectures (e.g. Qwen3.5). This patch catches the exception and auto-detects
    the decoder layer class by scanning for modules with 'DecoderLayer' in their class name.

    This is safe to call unconditionally — if PEFT's original function works, the patch
    is a no-op pass-through.
    """
    import functools
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        _or_policy,
        lambda_auto_wrap_policy,
    )
    import peft.utils.other

    _original_fsdp_auto_wrap_policy = peft.utils.other.fsdp_auto_wrap_policy

    def _patched_fsdp_auto_wrap_policy(model):
        try:
            return _original_fsdp_auto_wrap_policy(model)
        except Exception:
            base = model.base_model.model if hasattr(model, "base_model") else model
            decoder_layer_cls = None
            for _, module in base.named_modules():
                cls_name = type(module).__name__
                if "DecoderLayer" in cls_name:
                    decoder_layer_cls = type(module)
                    break
            if decoder_layer_cls is None:
                raise
            logger.info(
                f"Patched FSDP auto-wrap policy to use {decoder_layer_cls.__name__}"
            )
            from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

            peft_prompt_learning_cls = [PrefixEncoder, PromptEmbedding, PromptEncoder]
            try:
                from peft.tuners import CartridgeEncoder

                peft_prompt_learning_cls.append(CartridgeEncoder)
            except ImportError:
                pass

            def _leaf_with_trainable_weight(module):
                # Matches PEFT's real lambda_policy_fn: wrap any leaf module that
                # owns a trainable weight (e.g. LoRA's lora_A/lora_B) as its own
                # FSDP unit, separate from the frozen decoder layer around it.
                return (
                    len(list(module.named_children())) == 0
                    and getattr(module, "weight", None) is not None
                    and module.weight.requires_grad
                )

            lambda_policy = functools.partial(
                lambda_auto_wrap_policy, lambda_fn=_leaf_with_trainable_weight
            )
            transformer_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={decoder_layer_cls, *peft_prompt_learning_cls},
            )
            return functools.partial(
                _or_policy, policies=[lambda_policy, transformer_policy]
            )

    peft.utils.other.fsdp_auto_wrap_policy = _patched_fsdp_auto_wrap_policy
    logger.info("PEFT FSDP auto-wrap policy patch applied")


def cast_parameters_to_uniform_dtype(
    model,
    target_dtype: torch.dtype,
    cast_buffers: bool = True,
    exclude_buffer_pattern: Optional[str] = None,
) -> int:
    """Cast ordinary floating-point base-model params (and optionally buffers) to a
    uniform dtype so FSDP1 can flatten them into a single FlatParameter.

    FSDP1 needs a uniform dtype only among parameters flattened into the same FSDP
    unit. In HF/Accelerate that unit is a whole layer class, so an fp32 island inside
    an otherwise-bf16 layer -- a module transformers keeps in fp32
    (`_keep_in_fp32_modules`), or an fp32 rotary `inv_freq` buffer / Mamba-MoE router
    param -- triggers `Must flatten tensors with uniform dtype ... float32 and
    bfloat16`. Mixed dtypes also cause gradient-checkpointing recomputation
    mismatches (PyTorch issue #159359), which is why floating-point buffers are cast
    too (they are checked via named_buffers(), e.g. non-persistent rotary inv_freq).

    Call this BEFORE apply_lora_config(): PEFT's get_peft_model() upcasts LoRA adapter
    weights to float32 (autocast_adapter_dtype=True, the default) for training
    stability; casting afterwards would silently undo that and train LoRA in bf16.

    Safety:
      * Only floating-point tensors are cast; quantized/packed params (bitsandbytes
        Params4bit/Int8Params, etc.) and meta/DTensor tensors are skipped so their
        packed storage is never corrupted.
      * When target_dtype != float32, modules transformers deliberately keeps in fp32
        (`_keep_in_fp32_modules`) are downcast and warned about -- fine for a frozen
        LoRA base, riskier for full fine-tuning (prefer FSDP2 / MixedPrecision there).
      * Pass exclude_buffer_pattern=r"inv_freq|rotary" on long-context runs to keep
        rotary frequencies in fp32 (RoPE phase error grows with position).
      * Tied weights are re-tied after casting, since .to() allocates new tensors.

    Returns the number of parameters/buffers that were cast.
    """
    quantized_param_types = {
        "Params4bit",
        "Int8Params",
        "Params8bit",
        "FP8Parameter",
    }
    keep_fp32 = set(getattr(model, "_keep_in_fp32_modules", None) or [])
    buf_exclude = None
    if exclude_buffer_pattern:
        import re

        buf_exclude = re.compile(exclude_buffer_pattern)

    cast_count = 0
    downcast_kept_fp32 = []
    for name, param in model.named_parameters():
        if (
            type(param).__name__ in quantized_param_types
            or not param.is_floating_point()
            or param.is_meta
            or type(param.data).__name__ == "DTensor"
            or param.dtype == target_dtype
        ):
            continue
        if (
            keep_fp32
            and target_dtype != torch.float32
            and any(k in name for k in keep_fp32)
        ):
            downcast_kept_fp32.append(name)
        param.data = param.data.to(target_dtype)
        cast_count += 1

    if cast_buffers:
        for name, buf in model.named_buffers():
            if (
                not buf.is_floating_point()
                or buf.is_meta
                or buf.dtype == target_dtype
                or (buf_exclude is not None and buf_exclude.search(name))
            ):
                continue
            buf.data = buf.data.to(target_dtype)
            cast_count += 1

    if cast_count > 0 and hasattr(model, "tie_weights"):
        model.tie_weights()

    if downcast_kept_fp32:
        logger.warning(
            f"Downcast {len(downcast_kept_fp32)} module(s) transformers keeps in fp32 "
            f"({', '.join(sorted(set(downcast_kept_fp32))[:5])}) to {target_dtype}; "
            "fine for a frozen LoRA base, riskier for full fine-tuning."
        )
    if cast_count > 0:
        logger.info(
            f"Cast {cast_count} parameters/buffers from mixed dtypes to {target_dtype} for FSDP"
        )
    return cast_count


def apply_lora_config(
    model: AutoModelForCausalLM, script_args: ScriptArguments
) -> AutoModelForCausalLM:
    """Apply LoRA configuration to the model."""
    config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        target_modules=(
            "all-linear"
            if script_args.target_modules is None
            else script_args.target_modules
        ),
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_model(
    config_builder: ModelConfigBuilder, script_args: ScriptArguments
) -> AutoModelForCausalLM:
    """Load model using centralized configuration."""
    model_kwargs = config_builder.build_model_kwargs()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id, **model_kwargs
        )

        # Apply gradient checkpointing configuration.
        # User-provided gradient_checkpointing_kwargs in the YAML wins. If the user
        # didn't pin use_reentrant, fall back to the strategy-appropriate default:
        # FSDP/DDP -> non-reentrant, DeepSpeed ZeRO-3 -> reentrant (non-reentrant's
        # saved_tensors_hooks see partitioned weights on backward and raise CheckpointError).
        if config_builder.training_args.gradient_checkpointing:
            gc_kwargs = dict(
                config_builder.training_args.gradient_checkpointing_kwargs or {}
            )
            if "use_reentrant" not in gc_kwargs:
                gc_kwargs["use_reentrant"] = bool(config_builder.use_deepspeed)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

        return model
    except Exception as e:
        logger.error(f"Error loading model {script_args.model_id}: {e}")
        raise


def load_tokenizer(script_args: ScriptArguments) -> AutoTokenizer:
    """Load tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_id,
            **trust_remote_code_for(script_args),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer {script_args.model_id}: {e}")
        raise


# Built-in reward functions
def _extract_completion_text(completion: Any) -> str:
    """Normalize a single GRPO completion to plain text.

    GRPOTrainer passes completions in one of two shapes depending on the prompt
    format:
      - conversational prompt -> completion is a list of message dicts, e.g.
        ``[{"role": "assistant", "content": "..."}]``
      - standard (plain string) prompt -> completion is a plain ``str``

    This helper handles both (plus a bare dict, defensively) so reward functions
    never crash with ``'str' object has no attribute 'get'`` on standard-format
    datasets.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if not completion:
            return ""
        first = completion[0]
        return first.get("content", "") if isinstance(first, dict) else str(first)
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def format_reward_func(completions: List, **kwargs) -> List[float]:
    """Rewards completions with <think>...</think> and <answer>...</answer> tags."""
    import re

    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [_extract_completion_text(c) for c in completions]
    return [1.0 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]


def length_reward_func(
    completions: List, target_length: int = 512, **kwargs
) -> List[float]:
    """Rewards completions based on length relative to target."""
    responses = [_extract_completion_text(c) for c in completions]
    return [min(len(r), target_length) / target_length for r in responses]


def rouge_reward_func(
    completions: List, answer: Optional[List[str]] = None, **kwargs
) -> List[float]:
    """Rewards completions based on ROUGE-L precision against reference answers.

    Requires an ``answer`` column in the dataset. Returns 0.0 for every completion
    when references are missing or the ``rouge`` package is not installed, and
    scores each pair independently so a single bad sample cannot zero out the whole
    batch.
    """
    if answer is None:
        logger.warning(
            "rouge reward selected but no 'answer' column in dataset; returning 0.0"
        )
        return [0.0] * len(completions)

    try:
        from rouge import Rouge
    except ImportError:
        logger.warning("rouge package not installed; rouge reward returns 0.0")
        return [0.0] * len(completions)

    rouge = Rouge()
    responses = [_extract_completion_text(c) for c in completions]
    scores = []
    for response, reference in zip(responses, answer):
        # Rouge raises on empty strings; fall back to a space so the pair scores 0.
        hyp = response if response and response.strip() else " "
        ref = reference if isinstance(reference, str) and reference.strip() else " "
        try:
            scores.append(rouge.get_scores(hyp, ref)[0]["rouge-l"]["p"])
        except Exception:
            scores.append(0.0)
    return scores


BUILTIN_REWARDS = {
    "format": format_reward_func,
    "length": length_reward_func,
    "rouge": rouge_reward_func,
}


def load_reward_functions(reward_funcs_str: str) -> List[Callable]:
    """Load reward functions from string specification."""
    reward_funcs = []
    for spec in reward_funcs_str.split(","):
        spec = spec.strip()
        if spec in BUILTIN_REWARDS:
            reward_funcs.append(BUILTIN_REWARDS[spec])
            logger.info(f"Loaded built-in reward function: {spec}")
        elif ":" in spec:
            module_path, func_name = spec.rsplit(":", 1)
            module = importlib.import_module(module_path)
            reward_funcs.append(getattr(module, func_name))
            logger.info(f"Loaded custom reward function: {spec}")
        else:
            logger.warning(f"Unknown reward function: {spec}")
    return reward_funcs


def _coerce_tied_weights_keys(model):
    """Compatibility shim for save_pretrained across transformers versions.

    transformers >= 5.x expects each module's ``_tied_weights_keys`` to be a dict
    (``_get_tied_weight_keys`` calls ``.keys()`` on it during ``save_pretrained``).
    Some remote-code models (e.g. NVIDIA Nemotron-H, which sets
    ``_tied_weights_keys = ["lm_head.weight"]``) still use the old list convention,
    triggering ``AttributeError: 'list' object has no attribute 'keys'`` at save time.

    Convert any list/tuple/set form to ``{key: key}`` in place. transformers only
    consumes the keys (as regex patterns matched against pointer-shared tensors), so
    the mapping value is irrelevant, and this is a no-op on versions/models that
    already use a dict.
    """
    for module in model.modules():
        tied = getattr(module, "_tied_weights_keys", None)
        if isinstance(tied, (list, tuple, set)):
            module._tied_weights_keys = {k: k for k in tied}
    return model


def _patch_peft_weight_converter_compat() -> None:
    """Let `WeightConverter` tolerate the kwargs peft 0.19.x passes to it.

    `peft.utils.transformers_weight_conversion.build_peft_weight_mapping` rebuilds a
    model's weight converters with
    `orig_conversion.__class__(..., distributed_operation=..., quantization_operation=...)`.
    That matched the old dataclass `WeightConverter`, but transformers rewrote it with an
    explicit `(source_patterns, target_patterns, operations)` signature - both fields are
    now runtime state initialised to `None` by `WeightTransform.__init__`. Loading any
    LoRA adapter for a model whose `model_type` has a registered conversion mapping (MoE
    architectures that merge per-expert checkpoint weights into 3-D tensors, e.g.
    `nemotron_h`) therefore raises `TypeError: WeightConverter.__init__() got an
    unexpected keyword argument 'distributed_operation'`.

    peft `main` fixes this by dropping both kwargs; peft 0.19.1 is the latest release and
    still has the bug. Forwarding them after `__init__` is equivalent, since the values
    are only populated during a tensor-parallel or quantized load. Remove this shim once
    the requirements pin a peft release that contains the upstream fix.
    """
    from transformers.core_model_loading import WeightConverter

    orig_init = WeightConverter.__init__
    if getattr(orig_init, "_peft_compat", False):
        return

    def __init__(
        self, *args, distributed_operation=None, quantization_operation=None, **kwargs
    ):
        orig_init(self, *args, **kwargs)
        self.distributed_operation = distributed_operation
        self.quantization_operation = quantization_operation

    __init__._peft_compat = True
    WeightConverter.__init__ = __init__


def _merge_adapter_in_process(
    temp_dir: str,
    final_output_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    trc_kwargs: Optional[Dict[str, Any]] = None,
) -> AutoModelForCausalLM:
    """Merge LoRA adapter in the current process (for FSDP/DDP)."""
    _patch_peft_weight_converter_compat()
    with gpu_memory_manager():
        model = AutoPeftModelForCausalLM.from_pretrained(
            temp_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **(trc_kwargs or {}),
        )
        model = model.merge_and_unload()
        _coerce_tied_weights_keys(model)
        model.save_pretrained(
            final_output_dir, safe_serialization=True, max_shard_size="2GB"
        )
        return model


def _merge_adapter_via_subprocess(
    temp_dir: str,
    final_output_dir: str,
    torch_dtype_str: str = "bfloat16",
    trc_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Merge LoRA adapter in a clean subprocess to avoid DeepSpeed env conflicts.

    `trc_kwargs` carries `trust_remote_code` only when it has to be passed; an
    empty mapping lets Transformers resolve the implementation itself.
    """
    merge_script = textwrap.dedent(f"""\
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers.core_model_loading import WeightConverter

        # peft 0.19.x passes distributed_operation / quantization_operation into
        # WeightConverter.__init__, which the current signature does not accept
        # (fixed on peft main). Both are runtime fields defaulting to None, so
        # forwarding them post-init matches upstream. Without this, loading a LoRA
        # adapter for a model whose model_type has a registered conversion mapping
        # (MoE 3-D expert weights, e.g. nemotron_h) raises TypeError.
        _orig_wc_init = WeightConverter.__init__
        if not getattr(_orig_wc_init, "_peft_compat", False):
            def _wc_init(self, *a, distributed_operation=None, quantization_operation=None, **kw):
                _orig_wc_init(self, *a, **kw)
                self.distributed_operation = distributed_operation
                self.quantization_operation = quantization_operation
            _wc_init._peft_compat = True
            WeightConverter.__init__ = _wc_init

        trc = {trc_kwargs or {}!r}

        print("Loading adapter for merging...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            "{temp_dir}",
            torch_dtype=getattr(torch, "{torch_dtype_str}"),
            low_cpu_mem_usage=True,
            **trc,
        )

        print("Merging LoRA weights...")
        model = model.merge_and_unload()

        print("Saving merged model...")
        # Compat: transformers >=5.x calls .keys() on each module's _tied_weights_keys;
        # some remote-code models (e.g. Nemotron-H) declare it as a list. Coerce to a dict.
        for _m in model.modules():
            _tied = getattr(_m, "_tied_weights_keys", None)
            if isinstance(_tied, (list, tuple, set)):
                _m._tied_weights_keys = {{k: k for k in _tied}}
        model.save_pretrained(
            "{final_output_dir}",
            safe_serialization=True,
            max_shard_size="2GB",
        )

        print("Merge complete!")
    """)

    clean_env = {
        k: v
        for k, v in os.environ.items()
        if "DEEPSPEED" not in k and "ACCELERATE" not in k
    }

    result = subprocess.run(
        [sys.executable, "-c", merge_script],
        env=clean_env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Merge subprocess failed: {result.stderr}")
        raise RuntimeError(f"Merge failed: {result.stderr}")

    logger.info(f"Merge subprocess output: {result.stdout}")


def _detect_distributed_strategy(trainer: GRPOTrainer) -> Tuple[bool, bool]:
    """Detect whether DeepSpeed or FSDP is active."""
    use_deepspeed = (
        hasattr(trainer.accelerator.state, "deepspeed_plugin")
        and trainer.accelerator.state.deepspeed_plugin is not None
    )
    use_fsdp = trainer.is_fsdp_enabled
    return use_deepspeed, use_fsdp


def save_model(
    trainer: GRPOTrainer,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    script_args: ScriptArguments,
    accelerator: Accelerator,
    mlflow_enabled: bool,
    final_output_dir: str,
) -> None:
    """Save the trained model with proper DeepSpeed ZeRO-3 handling and online merging."""
    logger.info("STARTING MODEL SAVE PROCESS")

    accelerator.wait_for_everyone()

    use_deepspeed, use_fsdp = _detect_distributed_strategy(trainer)
    logger.info(f"Distributed strategy - DeepSpeed: {use_deepspeed}, FSDP: {use_fsdp}")

    if use_fsdp:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if script_args.use_peft and script_args.merge_weights:
        temp_dir = "/tmp/adapter_temp"
        os.makedirs(temp_dir, exist_ok=True)

        if use_deepspeed:
            # Trainer.save_model handles ZeRO-3 state dict gathering
            trainer.save_model(temp_dir)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                torch.cuda.empty_cache()
                dtype_str = (
                    script_args.torch_dtype
                    if script_args.torch_dtype not in ["auto", None]
                    else "bfloat16"
                )
                _merge_adapter_via_subprocess(
                    temp_dir,
                    final_output_dir,
                    torch_dtype_str=dtype_str,
                    trc_kwargs=trust_remote_code_for(script_args),
                )
                tokenizer.save_pretrained(final_output_dir)
                if mlflow_enabled:
                    logger.info(
                        "Skipping MLflow registration (model merged in subprocess)"
                    )
        else:
            # FSDP/DDP: use trainer.save_model so Accelerate honors the
            # FULL_STATE_DICT type set above and gathers a full, unsharded adapter
            # of plain tensors (this call must run on all ranks for the gather's
            # collective). Calling trainer.model.save_pretrained directly bypasses
            # that gather and pickles sharded FSDP DTensors; reloading them makes
            # merge_and_unload()'s weight_B @ weight_A a DTensor reshard (all-to-all)
            # on CPU, which fails with "No backend type associated with device type
            # cpu" because the process group only has the GPU-only NCCL backend.
            trainer.save_model(temp_dir)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                del model, trainer
                save_dtype = (
                    getattr(torch, script_args.torch_dtype)
                    if script_args.torch_dtype not in ["auto", None]
                    else torch.bfloat16
                )
                merged_model = _merge_adapter_in_process(
                    temp_dir,
                    final_output_dir,
                    torch_dtype=save_dtype,
                    trc_kwargs=trust_remote_code_for(script_args),
                )
                tokenizer.save_pretrained(final_output_dir)
                if mlflow_enabled:
                    register_model_in_mlflow(merged_model, tokenizer, script_args)

        accelerator.wait_for_everyone()

    else:
        # Covers both PEFT without merge and non-PEFT models
        trainer.save_model(final_output_dir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            tokenizer.save_pretrained(final_output_dir)
            if mlflow_enabled:
                register_model_in_mlflow(trainer.model, tokenizer, script_args)

        accelerator.wait_for_everyone()

    logger.info("MODEL SAVE PROCESS COMPLETED SUCCESSFULLY")


def register_model_in_mlflow(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, script_args: ScriptArguments
) -> None:
    """Register the model in MLflow."""
    logger.info(f"MLflow model registration under {script_args.mlflow_experiment_name}")

    try:
        params = {"top_p": 0.9, "temperature": 0.2, "max_new_tokens": 1024 * 4}
        signature = infer_signature("inputs", "generated_text", params=params)

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            signature=signature,
            name="model",
            task="text-generation",
            registered_model_name=f"model-{os.environ.get('MLFLOW_RUN_NAME', '').split('GRPO-')[-1]}",
        )
    except Exception as e:
        logger.error(f"Error registering model in MLflow: {e}")
        raise


def _align_to_multiple(value: int, multiple: int = 64) -> int:
    """Round up to the next multiple for hardware efficiency."""
    return ((value + multiple - 1) // multiple) * multiple


def calculate_optimal_grpo_lengths(
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    prompt_field: str = "prompt",
    sample_size: int = 1000,
    percentile: float = 0.95,
) -> int:
    """Calculate optimal max_completion_length for GRPO.

    Tokenizes prompts using apply_chat_template to match GRPOTrainer's internal
    tokenization and logs prompt length statistics. Returns only max_completion_length
    since GRPOConfig does not have a max_prompt_length parameter.

    Note: max_completion_length controls generation length, not dataset length.
    This function provides a data-informed starting point but you may need to
    adjust based on your expected output length.
    """
    sample_indices = torch.randperm(len(dataset))[: min(sample_size, len(dataset))]
    sample_data = dataset.select(sample_indices)

    prompt_lengths = []
    errors = 0

    for i, sample in enumerate(sample_data):
        try:
            prompt = sample.get(prompt_field, "")

            if isinstance(prompt, list):
                prompt_ids = tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=True
                )
            else:
                prompt_ids = tokenizer.encode(str(prompt), add_special_tokens=True)

            prompt_lengths.append(len(prompt_ids))

            if i == 0:
                logger.info(f"Sample 0: prompt={len(prompt_ids)} tokens")

        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning(f"Length calc error on sample {i}: {e}")

    if not prompt_lengths:
        raise ValueError("Could not compute lengths for any samples")

    if errors > 0:
        logger.warning(f"Skipped {errors}/{len(sample_data)} samples due to errors")

    p95_prompt = int(sorted(prompt_lengths)[int(percentile * len(prompt_lengths))])
    # Heuristic: completion length as 2x prompt for reasoning tasks.
    # Adjust based on your expected output length.
    max_completion_length = _align_to_multiple(p95_prompt * 2)

    logger.info(f"Analyzed {len(prompt_lengths)} samples ({errors} errors)")
    logger.info(
        f"Average prompt length: {sum(prompt_lengths) / len(prompt_lengths):.1f}"
    )
    logger.info(f"{percentile*100}th percentile prompt length: {p95_prompt}")
    logger.info(f"Estimated max_completion_length: {max_completion_length}")

    return max_completion_length


def deserialize_prompt(dataset: Dataset, prompt_field: str = "prompt") -> Dataset:
    """Deserialize JSON-encoded prompt field in-place using dataset.map().

    Handles the case where the prompt is stored as a JSON string
    (e.g., '[{"role": "user", "content": "..."}]') and needs to be
    parsed into a list of message dicts for GRPOTrainer's conversational format.
    """

    def process(sample):
        prompt = sample[prompt_field]
        # Only parse strings that look like a serialized structure ('[' or '{'), so
        # standard plain-text prompts (and bare scalars like "4") are left as-is
        # rather than raising or being coerced to non-string types.
        if isinstance(prompt, str) and prompt.lstrip()[:1] in ("[", "{"):
            try:
                prompt = json.loads(prompt)
            except (json.JSONDecodeError, ValueError):
                pass  # Keep as plain string (standard format)
        return {prompt_field: prompt}

    return dataset.map(process)


def _is_hf_dataset_dir(path: str) -> bool:
    """Check if path is a HuggingFace dataset directory (Arrow format)."""
    return os.path.isdir(path) and os.path.exists(
        os.path.join(path, "dataset_info.json")
    )


def _load_dataset_auto(path: str) -> Dataset:
    """Load a dataset from path, automatically detecting format (JSON, JSONL, or Arrow)."""
    if path.endswith(".jsonl") or path.endswith(".json"):
        return load_dataset("json", data_files=path, split="train")
    if path.endswith(".arrow"):
        logger.info(f"Loading Arrow file from {path}")
        return load_dataset("arrow", data_files=path, split="train")
    if _is_hf_dataset_dir(path):
        logger.info(f"Detected HuggingFace Arrow dataset format at {path}")
        ds = load_from_disk(path)
        if isinstance(ds, DatasetDict):
            split = "train" if "train" in ds else list(ds.keys())[0]
            logger.info(f"DatasetDict detected, using split '{split}'")
            ds = ds[split]
        return ds
    # Fallback: look for JSON/JSONL files in directory
    import glob as _glob

    json_files = sorted(
        _glob.glob(os.path.join(path, "*.json"))
        + _glob.glob(os.path.join(path, "*.jsonl"))
    )
    if json_files:
        logger.info(f"Found JSON file(s) in directory: {json_files}")
        return load_dataset("json", data_files=json_files, split="train")
    raise FileNotFoundError(
        f"No supported dataset files found in '{path}'. "
        "Expected .json, .jsonl, .arrow files or a HuggingFace dataset directory."
    )


def load_datasets(script_args: ScriptArguments) -> Tuple[Dataset, Optional[Dataset]]:
    """Load training and test datasets."""
    try:
        logger.info(f"Loading training dataset from {script_args.train_dataset_path}")
        train_ds = _load_dataset_auto(script_args.train_dataset_path)

        if script_args.deserialize_messages:
            logger.info("Deserializing JSON-encoded prompt field")
            train_ds = deserialize_prompt(train_ds, script_args.prompt_field)

        test_ds = None
        if script_args.val_dataset_path:
            logger.info(f"Loading test dataset from {script_args.val_dataset_path}")
            test_ds = _load_dataset_auto(script_args.val_dataset_path)

            if script_args.deserialize_messages:
                logger.info("Deserializing JSON-encoded prompt field for test dataset")
                test_ds = deserialize_prompt(test_ds, script_args.prompt_field)

        return train_ds, test_ds
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise


def train(script_args, training_args, train_ds, test_ds):
    """Train the model using centralized configuration."""
    set_seed(training_args.seed)

    # Create centralized config builder
    config_builder = ModelConfigBuilder(script_args, training_args)
    mlflow_enabled = is_mlflow_enabled(script_args)

    if script_args.token is not None:
        os.environ.update({"HF_TOKEN": script_args.token})
        if dist.is_initialized():
            logger.info("Waiting for all processes after setting HF token")
            dist.barrier()

    if script_args.use_snapshot_download:
        download_model(script_args.model_id)
        if dist.is_initialized():
            logger.info("Waiting for all processes after model download")
            dist.barrier()
        script_args.model_id = "/tmp/tmp_folder"

    # Load model and tokenizer using centralized config
    model = load_model(config_builder, script_args)
    tokenizer = load_tokenizer(script_args)

    # Auto-calculate lengths if enabled
    if script_args.auto_calculate_lengths:
        logger.info("Auto-calculating optimal lengths from dataset...")
        max_completion_length = calculate_optimal_grpo_lengths(
            tokenizer, train_ds, script_args.prompt_field
        )
        training_args.max_completion_length = max_completion_length
        logger.info(f"Set max_completion_length={max_completion_length}")

    # Cast the frozen base model's params/buffers to a uniform dtype BEFORE LoRA is
    # applied, so PEFT's fp32 upcast of the adapter weights (see apply_lora_config /
    # get_peft_model) is preserved instead of being immediately overwritten.
    if script_args.cast_parameters_to_uniform_dtype:
        cast_parameters_to_uniform_dtype(model, config_builder.torch_dtype)

    # Apply PEFT before trainer (same as SFT) for FSDP compatibility
    if script_args.use_peft:
        model = apply_lora_config(model, script_args)

    if (
        script_args.patch_peft_fsdp_auto_wrap_policy
        and script_args.use_peft
        and training_args.fsdp
        and training_args.fsdp != ""
    ):
        patch_peft_fsdp_auto_wrap_policy()

    callbacks = setup_wandb(script_args)
    if script_args.early_stopping:
        if callbacks is None:
            callbacks = []
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.01
            )
        )

        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False

    # Load reward functions
    reward_funcs = load_reward_functions(script_args.reward_funcs)

    # Apply trainer kwargs from centralized config
    trainer_kwargs = config_builder.build_trainer_kwargs()
    for key, value in trainer_kwargs.items():
        setattr(training_args, key, value)

    # Set report_to based on enabled tracking services
    report_to = []
    if os.environ.get("WANDB_DISABLED", "false").lower() != "true":
        report_to.append("wandb")
    if is_mlflow_enabled(script_args):
        report_to.append("mlflow")
    training_args.report_to = report_to

    # Initialize GRPO trainer
    # Note: peft_config is NOT passed here — model is already wrapped with PEFT above.
    # GRPOTrainer auto-detects PeftModel and handles reference model accordingly.
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        reward_funcs=reward_funcs,
        callbacks=callbacks,
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    if script_args.checkpoint_dir is not None:
        os.makedirs(script_args.checkpoint_dir, exist_ok=True)

        original_output_dir = training_args.output_dir
        training_args.output_dir = script_args.checkpoint_dir
    else:
        original_output_dir = training_args.output_dir

    # Start training
    if mlflow_enabled:
        logger.info(f"MLflow tracking under {script_args.mlflow_experiment_name}")
        mlflow.set_system_metrics_node_id(
            f"node_{trainer.accelerator.process_index // torch.cuda.device_count()}"
        )
        if trainer.accelerator.is_main_process:
            mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", None))
            mlflow.log_params(
                {
                    "total_gpus": trainer.accelerator.num_processes,
                    "nodes": trainer.accelerator.num_processes
                    // torch.cuda.device_count(),
                    "gpus_per_node": torch.cuda.device_count(),
                }
            )
            try:
                train_dataset_mlflow = mlflow.data.from_pandas(
                    train_ds.to_pandas(), name="train_dataset"
                )
                mlflow.log_input(train_dataset_mlflow, context="train")
            except Exception as e:
                logger.warning(f"Failed to log dataset to MLflow: {e}")

    if (
        script_args.checkpoint_dir
        and get_last_checkpoint(script_args.checkpoint_dir) is not None
        and script_args.use_checkpoints
    ):
        train_result = trainer.train(resume_from_checkpoint=True)
    else:
        train_result = trainer.train()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    save_model(
        trainer,
        model,
        tokenizer,
        script_args,
        trainer.accelerator,
        mlflow_enabled,
        original_output_dir,
    )
    trainer.accelerator.wait_for_everyone()


def main() -> None:
    """Main function to parse arguments and start training."""
    parser = TrlParser((ScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_and_config()

    set_custom_env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    setup_mlflow(script_args)

    train_ds, test_ds = load_datasets(script_args)
    train(script_args, training_args, train_ds, test_ds)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
