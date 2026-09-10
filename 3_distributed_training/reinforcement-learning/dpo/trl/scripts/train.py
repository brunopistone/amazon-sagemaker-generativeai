from accelerate import Accelerator
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import datetime
from functools import lru_cache
from huggingface_hub import snapshot_download
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
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    GenerationConfig,
    Mxfp4Config,
    set_seed,
)

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
from trl import DPOConfig, DPOTrainer, TrlParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import WandbCallback
import contextlib
from typing import Any, Dict, List, Optional, Tuple
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
            "help": (
                "Auto-calculate DPOConfig.max_length (prompt + completion) from the "
                "95th-percentile token length of the dataset, aligned to a multiple of 64. "
                "Ignored when max_length is set explicitly in the config."
            )
        },
    )
    checkpoint_dir: str = field(default=None, metadata={"help": "Checkpoint directory"})
    deserialize_messages: bool = field(
        default=False,
        metadata={"help": "Deserialize JSON-encoded prompt, chosen, rejected fields"},
    )
    use_checkpoints: bool = field(
        default=False, metadata={"help": "Whether to use checkpointing"}
    )
    early_stopping: bool = field(
        default=False, metadata={"help": "Whether to use early stopping"}
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
    modality_type: str = field(
        default="text",
        metadata={
            "help": (
                "Input modality: 'text' (text-only, default), "
                "'image' (image+text multi-modal). "
                "When set to 'image', loads model with AutoModelForImageTextToText "
                "and passes the processor to DPOTrainer."
            )
        },
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for DPO training"}
    )
    vlm_base_model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Original full VLM used to restore vision weights when model_id or "
                "the adapter base is a text-only intermediate checkpoint."
            )
        },
    )
    token: str = field(default=None, metadata={"help": "Hugging Face API token"})
    # NOTE: `trust_remote_code` deliberately does NOT live here. TRL 1.10.0 added
    # the same field to its configs (DPOConfig/SFTConfig/GRPOConfig), and TrlParser
    # builds one argparse namespace from both dataclasses, so defining it in both
    # aborts at startup with:
    #   argparse.ArgumentError: argument --trust_remote_code/--trust-remote-code:
    #   conflicting option strings
    # The YAML key still works - it binds to the trainer config instead - and
    # `trust_remote_code_for()` reads the value back off `script_args` after
    # `main()` copies it across. Same fix as `max_length` in the SFT script.
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
    target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Target modules for LoRA"}
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Torch dtype (auto, bfloat16, float16, float32)"},
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
                "with mixed float32/bfloat16 parameters."
                "Needed for both FSDP and DeepSpeed."
            )
        },
    )


def trust_remote_code_for(
    script_args: ScriptArguments, training_args: Optional["DPOConfig"] = None
) -> Dict[str, Any]:
    """`trust_remote_code` kwargs for the run's model: explicit config value wins.

    The flag lives on the trainer config rather than ScriptArguments, because TRL
    1.10.0 defines it too and TrlParser rejects the duplicate (see the note in
    ScriptArguments). TRL types it as a plain `bool` defaulting to False, so False
    is indistinguishable from "not set": only an explicit true is honoured as an
    override, and everything else falls through to auto-detection. That is the only
    useful direction anyway - auto-detect already omits the argument when it is not
    needed, which lets Transformers raise its own actionable "Please pass the
    argument `trust_remote_code=True`" instead of failing opaquely.

    `training_args` is optional so this stays callable from a test, or from another
    module importing this script, without constructing a trainer config.
    """
    override = True if getattr(training_args, "trust_remote_code", False) else None
    return trust_remote_code_kwargs(
        script_args.model_id, script_args.token, override=override
    )


class ModelConfigBuilder:
    """Centralized model configuration builder to eliminate duplicate logic."""

    def __init__(self, script_args: ScriptArguments, training_args: DPOConfig):
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
            self._trust_remote_code = trust_remote_code_for(
                self.script_args, self.training_args
            )
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
                "cache_dir": "/tmp/.cache",
                **self.trust_remote_code,
            }
        else:
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
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


def patch_dpo_trainer_dtype(trainer):
    """Patch DPOTrainer to fix input_ids dtype issue with tool calling.

    This is a known bug in DPOTrainer where concatenated_forward incorrectly
    casts input_ids to the model's dtype (bfloat16) instead of keeping them as long.
    See: https://github.com/huggingface/trl/issues/2101

    NOTE: This patch is not needed for trl >= 0.29.0, where the bug was fixed upstream
    and concatenated_forward was removed.
    """
    if not hasattr(trainer, "concatenated_forward"):
        logger.info("Skipping DPOTrainer dtype patch (not needed for this TRL version)")
        return trainer

    original_concatenated_forward = trainer.concatenated_forward

    def safe_concatenated_forward(model, batch, is_ref_model=False):
        # Fix all tensor dtypes in the batch before forward pass
        for key in batch.keys():
            if batch[key] is not None and isinstance(batch[key], torch.Tensor):
                # Input IDs, labels, and attention masks must be long
                if any(x in key for x in ["input_ids", "labels", "attention_mask"]):
                    if batch[key].dtype != torch.long:
                        batch[key] = batch[key].long()

        return original_concatenated_forward(model, batch, is_ref_model)

    trainer.concatenated_forward = safe_concatenated_forward
    return trainer


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
            "MLFLOW_RUN_NAME": f"DPO-{formatted_datetime}",
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
    on newer architectures. This patch catches the exception and auto-detects
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
    model: AutoModelForCausalLM,
    script_args: ScriptArguments,
    is_vlm: bool = False,
) -> AutoModelForCausalLM:
    """Apply LoRA configuration to the model.

    For VLMs with target_modules='all-linear', the vision encoder is excluded
    to avoid gradient checkpointing recomputation mismatches — LoRA on vision
    encoder layers causes shape/dtype conflicts during FSDP recomputation
    (PyTorch issue #159359).
    """
    lora_kwargs = dict(
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

    if is_vlm and script_args.target_modules is None:
        vision_prefixes = [
            "visual",
            "vision_tower",
            "vision_model",
            "img_processor",
            "vpm",
        ]
        lora_kwargs["exclude_modules"] = vision_prefixes
        logger.info(
            f"VLM detected: excluding vision encoder from LoRA targets ({vision_prefixes})"
        )

    config = LoraConfig(**lora_kwargs)
    return get_peft_model(model, config)


def load_model(
    config_builder: ModelConfigBuilder, script_args: ScriptArguments
) -> AutoModelForCausalLM:
    """Load model using centralized configuration.

    When modality_type is 'image', loads with AutoModelForImageTextToText.
    Otherwise loads with AutoModelForCausalLM.
    """
    model_kwargs = config_builder.build_model_kwargs()

    try:
        if (
            script_args.modality_type == "image"
            and AutoModelForImageTextToText is not None
        ):
            model = AutoModelForImageTextToText.from_pretrained(
                script_args.model_id, **model_kwargs
            )
            logger.info(f"Loaded model with {AutoModelForImageTextToText.__name__}")
        else:
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


def load_tokenizer(
    script_args: ScriptArguments, training_args: "DPOConfig"
) -> AutoTokenizer:
    """Load tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_id,
            **trust_remote_code_for(script_args, training_args),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer {script_args.model_id}: {e}")
        raise


def load_processor(script_args: ScriptArguments, training_args: "DPOConfig"):
    """Load processor for multimodal models. Returns None if unavailable."""
    try:
        processor = AutoProcessor.from_pretrained(
            script_args.model_id,
            **trust_remote_code_for(script_args, training_args),
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Loaded processor for {script_args.model_id}")
        return processor
    except Exception as e:
        logger.warning(f"No processor found for {script_args.model_id}: {e}")
        return None


def extract_tools_from_dataset(
    dataset: Dataset, sample_size: int = 512
) -> Optional[List[Dict]]:
    """Tools for the run, taken from the first sample that declares them.

    ``DPOConfig.tools`` is ONE value for the whole run - it reaches the chat template as a
    single ``tools=`` argument - so a dataset whose rows declare different tool sets cannot
    be represented faithfully here. Row 0 still wins, because that is all the config can
    carry, but a mismatch is now reported rather than applied in silence: a row rendered
    with another row's tools teaches the model to call functions that were never offered
    for that request.
    """
    if "tools" not in dataset.column_names:
        return None

    tools = dataset[0]["tools"]
    if not tools:
        return None

    total = len(dataset)
    limit = min(total, sample_size)
    distinct = {
        json.dumps(dataset[i]["tools"], sort_keys=True, default=str)
        for i in range(limit)
    }
    if len(distinct) > 1:
        logger.warning(
            f"The `tools` column holds {len(distinct)} different tool sets in the first "
            f"{limit} of {total} row(s), but DPOConfig.tools is a single run-wide value. "
            "Row 0's tools will be used for EVERY row, so rows declaring a different set "
            "are rendered with tools they never had. Split the dataset per tool set, or "
            "carry the schemas in each row's prompt text instead."
        )
    return tools


def _was_trained_as_vlm(adapter_dir: str) -> bool:
    """Check if the adapter was trained on a VLM by inspecting adapter weight keys.

    When a model is loaded with AutoModelForImageTextToText, the language model
    layers are nested under 'language_model' (e.g. model.language_model.layers.X).
    When loaded with AutoModelForCausalLM, they are directly under 'model'
    (e.g. model.layers.X). We check the saved adapter weights for this prefix.
    """
    if AutoModelForImageTextToText is None:
        return False
    try:
        import glob

        # Check safetensors files first
        safetensor_files = glob.glob(
            os.path.join(adapter_dir, "adapter_model*.safetensors")
        )
        if safetensor_files:
            from safetensors import safe_open

            with safe_open(safetensor_files[0], framework="pt") as sf:
                for key in sf.keys():
                    if "language_model" in key:
                        return True
                return False

        # Fall back to pytorch bin files
        bin_files = glob.glob(os.path.join(adapter_dir, "adapter_model*.bin"))
        if bin_files:
            state_dict = torch.load(bin_files[0], map_location="cpu", weights_only=True)
            for key in state_dict.keys():
                if "language_model" in key:
                    return True
            return False
    except Exception as e:
        logger.warning(f"Could not inspect adapter weights: {e}")
    return False


def _is_vlm_from_config(model_id: Optional[str]) -> bool:
    """Check if a model is a VLM by inspecting its config."""
    if not model_id or AutoModelForImageTextToText is None:
        return False
    try:
        from transformers import AutoConfig
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        )

        config = AutoConfig.from_pretrained(
            model_id, **trust_remote_code_kwargs(model_id)
        )
        return config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    except Exception as e:
        logger.warning(f"Could not inspect VLM config for {model_id}: {e}")
        return False


def _resolve_vlm_base_model_id(
    adapter_base_model_id: Optional[str],
    explicit_vlm_base_model_id: Optional[str],
) -> Optional[str]:
    """Resolve the complete VLM used to preserve vision weights during export."""
    if explicit_vlm_base_model_id:
        if not _is_vlm_from_config(explicit_vlm_base_model_id):
            raise ValueError(
                "vlm_base_model_id must reference a full supported VLM, got: "
                f"{explicit_vlm_base_model_id}"
            )
        return explicit_vlm_base_model_id
    if _is_vlm_from_config(adapter_base_model_id):
        return adapter_base_model_id
    return None


def _transplant_into_vlm(
    merged_causal_state: dict,
    base_model_id: str,
    torch_dtype: torch.dtype,
):
    """Transplant merged CausalLM weights into a full VLM to preserve vision encoder.

    When an adapter is trained with AutoModelForCausalLM on a VLM base, the merged
    language model weights use CausalLM key paths (model.layers.X...). This function
    loads the full VLM and replaces the language model weights with the merged ones,
    keeping the vision encoder and projector intact.
    """
    logger.info(
        "Transplanting merged CausalLM weights into full VLM to preserve vision encoder"
    )
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **trust_remote_code_kwargs(base_model_id),
    )
    vlm_state = vlm_model.state_dict()

    causal_layer_key = next((k for k in merged_causal_state if ".layers.0." in k), None)
    if causal_layer_key is None:
        raise RuntimeError(
            "Could not find layer keys in merged state dict; VLM export aborted"
        )

    causal_prefix = causal_layer_key.split("layers.0.")[0]
    vlm_layer_key = next(
        (k for k in vlm_state if ".layers.0." in k and "language_model" in k), None
    )
    if vlm_layer_key is None:
        raise RuntimeError(
            "Could not find language_model layer keys in VLM; export aborted"
        )

    vlm_prefix = vlm_layer_key.split("layers.0.")[0]
    logger.info(f"Key mapping: CausalLM '{causal_prefix}*' -> VLM '{vlm_prefix}*'")

    updated = 0
    updated_core = 0
    missing_core = []
    for causal_key, value in merged_causal_state.items():
        is_core = causal_key.startswith(causal_prefix)
        if is_core:
            vlm_key = vlm_prefix + causal_key[len(causal_prefix) :]
        elif causal_key in vlm_state:
            vlm_key = causal_key
        else:
            vlm_key = next((vk for vk in vlm_state if vk.endswith(causal_key)), None)
        if vlm_key and vlm_key in vlm_state:
            if tuple(vlm_state[vlm_key].shape) != tuple(value.shape):
                raise RuntimeError(
                    f"Shape mismatch during VLM transplant: {causal_key} -> "
                    f"{vlm_key}: {tuple(value.shape)} != "
                    f"{tuple(vlm_state[vlm_key].shape)}"
                )
            vlm_state[vlm_key] = value
            updated += 1
            updated_core += int(is_core)
        elif is_core:
            missing_core.append(causal_key)

    if updated_core == 0 or missing_core:
        raise RuntimeError(
            "Incomplete CausalLM-to-VLM transplant: "
            f"{updated_core} core tensors updated, {len(missing_core)} missing "
            f"(examples: {missing_core[:3]})"
        )

    logger.info(
        f"Transplanted {updated}/{len(merged_causal_state)} language model weights into VLM"
    )
    vlm_model.load_state_dict(vlm_state)
    return vlm_model


def _patch_peft_weight_converter_compat() -> None:
    """Let `WeightConverter` tolerate the kwargs peft 0.19.x passes to it.

    `peft.utils.transformers_weight_conversion.build_peft_weight_mapping` rebuilds a
    model's weight converters with
    `orig_conversion.__class__(..., distributed_operation=..., quantization_operation=...)`,
    but `WeightConverter.__init__` only ever accepted
    `(source_patterns, target_patterns, operations)` - both fields are runtime state
    initialised to `None` by `WeightTransform.__init__`. Loading any LoRA adapter for a
    model whose `model_type` has a registered conversion mapping (MoE architectures that
    merge per-expert checkpoint weights into 3-D tensors, e.g. `nemotron_h`) therefore
    raises `TypeError: WeightConverter.__init__() got an unexpected keyword argument
    'distributed_operation'`.

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


def _load_and_merge_adapter(
    adapter_dir: str,
    torch_dtype: torch.dtype,
    explicit_vlm_base_model_id: Optional[str] = None,
):
    """Load, merge, and return the final model ready for saving.

    Handles three cases:
    1. Adapter trained on VLM (modality_type=image) → merge directly on VLM
    2. Adapter trained on CausalLM, base is VLM → merge CausalLM, transplant into VLM
    3. Adapter trained on CausalLM, base is text-only → merge directly on CausalLM
    """
    from peft import PeftConfig, PeftModel

    _patch_peft_weight_converter_compat()

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_id = peft_config.base_model_name_or_path
    trained_as_vlm = _was_trained_as_vlm(adapter_dir)
    vlm_base_model_id = _resolve_vlm_base_model_id(
        base_model_id, explicit_vlm_base_model_id
    )
    trc_kwargs = trust_remote_code_kwargs(base_model_id)

    if trained_as_vlm:
        if not vlm_base_model_id:
            raise RuntimeError(
                "Adapter keys indicate VLM training but no complete VLM base was found"
            )
        logger.info(
            f"Adapter trained on VLM: loading with {AutoModelForImageTextToText.__name__}"
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            vlm_base_model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **trust_remote_code_kwargs(vlm_base_model_id),
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        return model.merge_and_unload()

    logger.info("Adapter trained on CausalLM: merging with AutoPeftModelForCausalLM")
    causal_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **trc_kwargs,
    )
    merged_causal = causal_model.merge_and_unload()

    if vlm_base_model_id:
        merged_state = merged_causal.state_dict()
        del causal_model, merged_causal
        torch.cuda.empty_cache()
        return _transplant_into_vlm(merged_state, vlm_base_model_id, torch_dtype)

    return merged_causal


def _sanitize_sampling_params(generation_config) -> bool:
    """Reset sampling-only parameters left set while `do_sample` is not True.

    transformers >= 5 validates strictly inside ``GenerationConfig.save_pretrained``
    (``self.validate(strict=True)``, with no kwarg to relax it) and raises when a
    sampling-only parameter differs from its default while ``do_sample`` is not True.
    NVIDIA Nemotron-3-Nano ships ``{"temperature": 1.0, "top_p": 0.95}`` with no
    ``do_sample``, so saving a merged checkpoint dies with::

        ValueError: GenerationConfig is invalid:
        - `top_p`: `do_sample` is not set to `True`. However, `top_p` is set to `0.95` ...

    The error names both resolutions: set ``do_sample=True``, or unset the parameter. We
    unset, i.e. resolve toward greedy decoding, because a model fine-tuned to emit strict
    JSON should not sample by default - an unparameterised ``generate()`` on the exported
    checkpoint would otherwise produce temperature-1.0 top-p-0.95 output and malformed
    JSON. Callers that do want sampling still pass it per request, which is how both the
    endpoint and the Bedrock baseline in the evaluation notebook already work.

    Defaults are transformers' own (`temperature`/`top_p`/`typical_p` 1.0, `top_k` 50,
    `epsilon_cutoff`/`eta_cutoff` 0.0, `min_p`/`top_h` None). Returns True if anything
    changed.
    """
    if getattr(generation_config, "do_sample", None) is True:
        return False

    sampling_defaults = {
        "temperature": 1.0,
        "top_p": 1.0,
        "typical_p": 1.0,
        "top_k": 50,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "min_p": None,
        "top_h": None,
    }

    changed = False
    for attr, default in sampling_defaults.items():
        if not hasattr(generation_config, attr):
            continue
        current = getattr(generation_config, attr)
        if current is not None and current != default:
            logger.info(
                f"generation_config.{attr}: {current} -> {default} "
                "(sampling parameter set without do_sample=True; transformers refuses "
                "to save it)"
            )
            setattr(generation_config, attr, default)
            changed = True
    return changed


def _align_generation_config(tokenizer: AutoTokenizer, final_output_dir: str) -> None:
    """Point the exported generation_config at the tokenizer's eos/pad tokens.

    Runs on the saved directory rather than on the in-memory model because both merge
    paths reload the base model from the hub (`_merge_adapter_in_process` after the
    trainer has been deleted, `_merge_adapter_via_subprocess` in another process), so
    whatever `generation_config.json` they write comes from the base repo.

    Some repos declare an eos_token_id their chat template never emits (NVIDIA
    Nemotron-H declares `2` / `</s>` while its template ends turns with `<|im_end|>`).
    Left unaligned that ships into serving and the model generates past the end of its
    answer looking for a token it cannot produce.
    """
    if not os.path.exists(os.path.join(final_output_dir, "generation_config.json")):
        # Adapter-only exports have no generation_config; the base repo's applies.
        return

    generation_config = GenerationConfig.from_pretrained(final_output_dir)
    # The re-save below validates strictly too, so a base repo whose sampling params are
    # set without do_sample would abort here even when the merge itself succeeded.
    changed = _sanitize_sampling_params(generation_config)
    for attr, token_id in (
        ("eos_token_id", tokenizer.eos_token_id),
        ("pad_token_id", tokenizer.pad_token_id),
    ):
        current = getattr(generation_config, attr, None)
        if token_id is None:
            continue
        if isinstance(current, (list, tuple)):
            # `eos_token_id` may legitimately hold several stop tokens (Nemotron-3-Nano
            # 30B ships [2, 11] = </s> and <|im_end|>). Replacing the list with the
            # tokenizer's single id would drop the others, so add instead - dropping the
            # template's real turn terminator is the very failure this guards against.
            if token_id not in current:
                logger.info(
                    f"generation_config.{attr}: {list(current)} + [{token_id}] "
                    f"(tokenizer's token was missing from the list)"
                )
                setattr(generation_config, attr, list(current) + [token_id])
                changed = True
        elif current != token_id:
            logger.info(
                f"generation_config.{attr}: {current} -> {token_id} "
                f"(matching the tokenizer)"
            )
            setattr(generation_config, attr, token_id)
            changed = True

    if changed:
        generation_config.save_pretrained(final_output_dir)


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


def _merge_adapter_in_process(
    temp_dir: str,
    final_output_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    vlm_base_model_id: Optional[str] = None,
):
    """Merge LoRA adapter in the current process (for FSDP/DDP)."""
    with gpu_memory_manager():
        model = _load_and_merge_adapter(
            temp_dir,
            torch_dtype,
            explicit_vlm_base_model_id=vlm_base_model_id,
        )
        _coerce_tied_weights_keys(model)
        # Must run before save_pretrained: it writes generation_config.json, and
        # transformers validates it strictly there. The base repo's sampling params would
        # abort the save after training has already succeeded.
        if getattr(model, "generation_config", None) is not None:
            _sanitize_sampling_params(model.generation_config)
        model.save_pretrained(
            final_output_dir, safe_serialization=True, max_shard_size="2GB"
        )
        return model


def _merge_adapter_via_subprocess(
    temp_dir: str,
    final_output_dir: str,
    torch_dtype_str: str = "bfloat16",
    vlm_base_model_id: Optional[str] = None,
    trc_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Merge LoRA adapter in a clean subprocess to avoid DeepSpeed env conflicts.

    Auto-detects whether the base model is a VLM from the adapter config and
    loads with the correct auto class to preserve vision encoder weights.
    `trc_kwargs` carries `trust_remote_code` only when it has to be passed; an
    empty mapping lets Transformers resolve the implementation itself.
    """
    merge_script = textwrap.dedent(f"""\
        import glob
        import os
        import torch
        from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM
        from transformers import AutoConfig
        from transformers.core_model_loading import WeightConverter

        # peft 0.19.x passes distributed_operation / quantization_operation into
        # WeightConverter.__init__, which never accepted them (fixed on peft main).
        # Both are runtime fields defaulting to None, so forwarding them post-init
        # matches upstream. Without this, loading a LoRA adapter for a model whose
        # model_type has a registered conversion mapping (MoE 3-D expert weights,
        # e.g. nemotron_h) raises TypeError.
        _orig_wc_init = WeightConverter.__init__
        if not getattr(_orig_wc_init, "_peft_compat", False):
            def _wc_init(self, *a, distributed_operation=None, quantization_operation=None, **kw):
                _orig_wc_init(self, *a, **kw)
                self.distributed_operation = distributed_operation
                self.quantization_operation = quantization_operation
            _wc_init._peft_compat = True
            WeightConverter.__init__ = _wc_init

        adapter_dir = "{temp_dir}"
        output_dir = "{final_output_dir}"
        dtype = getattr(torch, "{torch_dtype_str}")
        trc = {trc_kwargs or {}!r}
        explicit_vlm_base_model_id = {vlm_base_model_id!r}

        peft_config = PeftConfig.from_pretrained(adapter_dir)
        base_model_id = peft_config.base_model_name_or_path

        # Check if adapter was trained on a VLM by inspecting weight keys
        trained_as_vlm = False
        try:
            sf_files = glob.glob(os.path.join(adapter_dir, "adapter_model*.safetensors"))
            if sf_files:
                from safetensors import safe_open
                with safe_open(sf_files[0], framework="pt") as sf:
                    trained_as_vlm = any("language_model" in k for k in sf.keys())
            else:
                bin_files = glob.glob(os.path.join(adapter_dir, "adapter_model*.bin"))
                if bin_files:
                    sd = torch.load(bin_files[0], map_location="cpu", weights_only=True)
                    trained_as_vlm = any("language_model" in k for k in sd.keys())
        except Exception as e:
            print(f"Warning: could not inspect adapter weights: {{e}}")

        # Check if base model is a VLM
        base_is_vlm = False
        try:
            from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
            config = AutoConfig.from_pretrained(base_model_id, **trc)
            base_is_vlm = config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
        except Exception:
            pass
        if explicit_vlm_base_model_id:
            try:
                config = AutoConfig.from_pretrained(explicit_vlm_base_model_id, **trc)
                if config.model_type not in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
                    raise ValueError(
                        f"Not an image-text model type: {{config.model_type}}"
                    )
            except Exception as e:
                raise ValueError(
                    f"Invalid vlm_base_model_id {{explicit_vlm_base_model_id}}: {{e}}"
                ) from e
            full_vlm_base_model_id = explicit_vlm_base_model_id
        elif base_is_vlm:
            full_vlm_base_model_id = base_model_id
        else:
            full_vlm_base_model_id = None

        try:
            from transformers import AutoModelForImageTextToText
            vlm_auto_cls = AutoModelForImageTextToText
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as vlm_auto_cls
            except ImportError:
                vlm_auto_cls = None

        if full_vlm_base_model_id and vlm_auto_cls is None:
            raise RuntimeError(
                "A VLM export was requested but this Transformers version has no "
                "image-text auto model class"
            )

        if trained_as_vlm and vlm_auto_cls:
            if not full_vlm_base_model_id:
                raise RuntimeError(
                    "Adapter was trained as a VLM but no complete VLM base was found"
                )
            # Case 1: adapter trained on VLM -> merge directly
            print(f"Adapter trained on VLM: loading with {{vlm_auto_cls.__name__}}")
            base_model = vlm_auto_cls.from_pretrained(
                full_vlm_base_model_id, torch_dtype=dtype,
                low_cpu_mem_usage=True, **trc,
            )
            model = PeftModel.from_pretrained(base_model, adapter_dir)
            model = model.merge_and_unload()
        else:
            # Merge as CausalLM first
            print("Merging adapter as CausalLM...")
            causal_model = AutoPeftModelForCausalLM.from_pretrained(
                adapter_dir, torch_dtype=dtype,
                low_cpu_mem_usage=True, **trc,
            )
            model = causal_model.merge_and_unload()

            if full_vlm_base_model_id and vlm_auto_cls:
                # Case 2: CausalLM adapter on VLM base -> transplant into full VLM
                print("Base is VLM: transplanting merged weights into full VLM...")
                merged_state = model.state_dict()
                del causal_model, model
                torch.cuda.empty_cache()

                vlm_model = vlm_auto_cls.from_pretrained(
                    full_vlm_base_model_id, torch_dtype=dtype,
                    low_cpu_mem_usage=True, **trc,
                )
                vlm_state = vlm_model.state_dict()

                # Find key prefix mapping: CausalLM "model." -> VLM "model.language_model."
                causal_lk = next((k for k in merged_state if ".layers.0." in k), None)
                vlm_lk = next((k for k in vlm_state if ".layers.0." in k and "language_model" in k), None)
                if not causal_lk or not vlm_lk:
                    raise RuntimeError(
                        "Cannot determine CausalLM-to-VLM key prefixes; export aborted"
                    )
                c_prefix = causal_lk.split("layers.0.")[0]
                v_prefix = vlm_lk.split("layers.0.")[0]
                print(f"Key mapping: '{{c_prefix}}*' -> '{{v_prefix}}*'")
                updated = 0
                updated_core = 0
                missing_core = []
                for ck, val in merged_state.items():
                    is_core = ck.startswith(c_prefix)
                    if is_core:
                        vk = v_prefix + ck[len(c_prefix):]
                    elif ck in vlm_state:
                        vk = ck
                    else:
                        vk = next((x for x in vlm_state if x.endswith(ck)), None)
                    if vk and vk in vlm_state:
                        if tuple(vlm_state[vk].shape) != tuple(val.shape):
                            raise RuntimeError(f"Shape mismatch: {{ck}} -> {{vk}}")
                        vlm_state[vk] = val
                        updated += 1
                        updated_core += int(is_core)
                    elif is_core:
                        missing_core.append(ck)
                if updated_core == 0 or missing_core:
                    raise RuntimeError(
                        f"Incomplete VLM transplant: {{updated_core}} core tensors "
                        f"updated, {{len(missing_core)}} missing"
                    )
                print(f"Transplanted {{updated}}/{{len(merged_state)}} weights")
                vlm_model.load_state_dict(vlm_state)
                model = vlm_model

        print("Saving merged model...")
        # Compat: transformers >=5.x calls .keys() on each module's _tied_weights_keys;
        # some remote-code models (e.g. Nemotron-H) declare it as a list. Coerce to a dict.
        for _m in model.modules():
            _tied = getattr(_m, "_tied_weights_keys", None)
            if isinstance(_tied, (list, tuple, set)):
                _m._tied_weights_keys = {{k: k for k in _tied}}

        # transformers >=5 validates generation_config strictly inside save_pretrained and
        # raises when a sampling-only param is set while do_sample is not True (NVIDIA
        # Nemotron-3-Nano 4B ships top_p=0.95 with no do_sample). Reset to the defaults,
        # i.e. greedy - see _sanitize_sampling_params above for why that direction.
        _gc = getattr(model, "generation_config", None)
        if _gc is not None and getattr(_gc, "do_sample", None) is not True:
            for _attr, _default in (
                ("temperature", 1.0), ("top_p", 1.0), ("typical_p", 1.0),
                ("top_k", 50), ("epsilon_cutoff", 0.0), ("eta_cutoff", 0.0),
                ("min_p", None), ("top_h", None),
            ):
                if hasattr(_gc, _attr):
                    _cur = getattr(_gc, _attr)
                    if _cur is not None and _cur != _default:
                        print(f"generation_config.{{_attr}}: {{_cur}} -> {{_default}}")
                        setattr(_gc, _attr, _default)

        model.save_pretrained(
            output_dir,
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


def _save_artifacts_on_main(
    tokenizer: AutoTokenizer,
    processor,
    final_output_dir: str,
    model_id: str = None,
) -> None:
    """Save tokenizer and processor to the output directory.

    If processor is None but the saved model is a VLM (has a config with a
    model_type registered for image-text-to-text), loads and saves the processor
    from the base model so the output is complete for multi-modal inference.
    """
    tokenizer.save_pretrained(final_output_dir)
    _align_generation_config(tokenizer, final_output_dir)
    if processor is not None:
        processor.save_pretrained(final_output_dir)
        if (
            hasattr(processor, "image_processor")
            and processor.image_processor is not None
        ):
            processor.image_processor.save_pretrained(final_output_dir)
        if (
            hasattr(processor, "video_processor")
            and processor.video_processor is not None
        ):
            processor.video_processor.save_pretrained(final_output_dir)
    elif model_id and _is_vlm_from_config(model_id):
        # No processor provided but base model is a VLM — load processor from base model
        try:
            logger.info(
                f"No processor provided but base model is a VLM. "
                f"Loading processor from: {model_id}"
            )
            base_processor = AutoProcessor.from_pretrained(
                model_id, **trust_remote_code_kwargs(model_id)
            )
            base_processor.save_pretrained(final_output_dir)
            # Also save sub-processors as separate files for compatibility (e.g. Ollama)
            if (
                hasattr(base_processor, "image_processor")
                and base_processor.image_processor is not None
            ):
                base_processor.image_processor.save_pretrained(final_output_dir)
            if (
                hasattr(base_processor, "video_processor")
                and base_processor.video_processor is not None
            ):
                base_processor.video_processor.save_pretrained(final_output_dir)
        except Exception as e:
            raise RuntimeError(
                f"Could not auto-save processor for VLM {model_id}: {e}"
            ) from e


def _is_visual_weight_key(key: str) -> bool:
    return (
        key.startswith(("visual.", "vision_model.", "vision_tower."))
        or ".visual." in key
        or ".vision_model." in key
        or ".vision_tower." in key
    )


def _validate_vlm_export(final_output_dir: str) -> None:
    """Fail unless an exported directory is a complete standalone VLM."""
    config_path = os.path.join(final_output_dir, "config.json")
    if not os.path.isfile(config_path):
        raise RuntimeError("VLM export is missing config.json")
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config.get("vision_config"), dict):
        raise RuntimeError("VLM export config.json has no vision_config")
    if str(config.get("model_type", "")).endswith("_text"):
        raise RuntimeError("VLM export has a text-only model_type")

    index_path = os.path.join(final_output_dir, "model.safetensors.index.json")
    weight_keys = []
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as index_file:
            weight_map = json.load(index_file).get("weight_map", {})
        weight_keys = list(weight_map)
        missing_shards = sorted(
            {
                filename
                for filename in weight_map.values()
                if not os.path.isfile(os.path.join(final_output_dir, filename))
            }
        )
        if missing_shards:
            raise RuntimeError(
                f"VLM export references missing shards: {missing_shards}"
            )
    else:
        import glob
        from safetensors import safe_open

        shard_paths = glob.glob(os.path.join(final_output_dir, "*.safetensors"))
        if not shard_paths:
            raise RuntimeError("VLM export contains no safetensors weights")
        for shard_path in shard_paths:
            with safe_open(shard_path, framework="pt") as shard:
                weight_keys.extend(shard.keys())

    visual_count = sum(_is_visual_weight_key(key) for key in weight_keys)
    if visual_count == 0:
        raise RuntimeError("VLM export contains no visual/vision weight tensors")
    required_processor_files = ["preprocessor_config.json"]
    if config.get("model_type") == "qwen3_5":
        required_processor_files.append("video_preprocessor_config.json")
    missing_processor_files = [
        filename
        for filename in required_processor_files
        if not os.path.isfile(os.path.join(final_output_dir, filename))
    ]
    if missing_processor_files:
        raise RuntimeError(
            "VLM export is missing required processor files: "
            f"{missing_processor_files}"
        )
    logger.info(
        "Validated standalone VLM export: %d tensors, %d visual tensors",
        len(weight_keys),
        visual_count,
    )


def _detect_distributed_strategy(trainer: DPOTrainer) -> Tuple[bool, bool]:
    """Detect whether DeepSpeed or FSDP is active."""
    use_deepspeed = (
        hasattr(trainer.accelerator.state, "deepspeed_plugin")
        and trainer.accelerator.state.deepspeed_plugin is not None
    )
    use_fsdp = trainer.is_fsdp_enabled
    return use_deepspeed, use_fsdp


def save_model(
    trainer: DPOTrainer,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    processor,
    script_args: ScriptArguments,
    training_args: "DPOConfig",
    accelerator: Accelerator,
    mlflow_enabled: bool,
    final_output_dir: str,
) -> None:
    """Save the trained model with proper DeepSpeed ZeRO-3 handling and online merging."""
    logger.info("STARTING MODEL SAVE PROCESS")
    vlm_source_model_id = _resolve_vlm_base_model_id(
        script_args.model_id, script_args.vlm_base_model_id
    )

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
                    vlm_base_model_id=script_args.vlm_base_model_id,
                    trc_kwargs=trust_remote_code_for(script_args, training_args),
                )
                _save_artifacts_on_main(
                    tokenizer, processor, final_output_dir, vlm_source_model_id
                )
                if vlm_source_model_id:
                    _validate_vlm_export(final_output_dir)
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
                    vlm_base_model_id=script_args.vlm_base_model_id,
                )
                _save_artifacts_on_main(
                    tokenizer, processor, final_output_dir, vlm_source_model_id
                )
                if vlm_source_model_id:
                    _validate_vlm_export(final_output_dir)
                if mlflow_enabled:
                    register_model_in_mlflow(merged_model, tokenizer, script_args)

        accelerator.wait_for_everyone()

    else:
        # Covers both PEFT without merge and non-PEFT models
        # A full (non-adapter) save writes generation_config.json, which transformers
        # validates strictly - so sanitize before the save, not in the
        # _align_generation_config that only runs afterwards.
        generation_config = getattr(trainer.model, "generation_config", None)
        if generation_config is not None:
            _sanitize_sampling_params(generation_config)
        trainer.save_model(final_output_dir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            _save_artifacts_on_main(
                tokenizer, processor, final_output_dir, vlm_source_model_id
            )
            if not script_args.use_peft and vlm_source_model_id:
                _validate_vlm_export(final_output_dir)
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
            registered_model_name=f"model-{os.environ.get('MLFLOW_RUN_NAME', '').split('DPO-')[-1]}",
        )
    except Exception as e:
        logger.error(f"Error registering model in MLflow: {e}")
        raise


def _align_to_multiple(value: int, multiple: int = 64) -> int:
    """Round up to the next multiple for hardware efficiency."""
    return ((value + multiple - 1) // multiple) * multiple


def calculate_optimal_dpo_lengths(
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    deserialize_messages: bool = False,
    sample_size: int = 1000,
    percentile: float = 0.95,
    max_absolute_length: int = 32768,
) -> Optional[int]:
    """Calculate an optimal DPOConfig.max_length (prompt + completion) from the dataset.

    For each sampled row, renders ``prompt + chosen`` and ``prompt + rejected`` with
    apply_chat_template and keeps the longer of the two token counts. Returns the
    ``percentile`` length across samples, aligned up to a multiple of 64, or None if
    no lengths could be computed (the caller then leaves max_length unchanged).

    When ``deserialize_messages`` is True the dataset carries a lazy ``set_transform``
    that parses JSON-encoded fields on access, so iterating here already yields native
    message lists.
    """
    sample_indices = torch.randperm(len(dataset))[: min(sample_size, len(dataset))]
    sample_data = dataset.select(sample_indices)

    def _render(messages) -> int:
        if isinstance(messages, list):
            return len(tokenizer.apply_chat_template(messages, tokenize=True))
        return len(tokenizer.encode(str(messages), add_special_tokens=True))

    lengths = []
    errors = 0
    for sample in sample_data:
        try:
            prompt = sample.get("prompt")
            chosen = sample.get("chosen")
            rejected = sample.get("rejected")
            if prompt is None or chosen is None or rejected is None:
                errors += 1
                continue

            prompt_msgs = prompt if isinstance(prompt, list) else [prompt]
            chosen_msgs = chosen if isinstance(chosen, list) else [chosen]
            rejected_msgs = rejected if isinstance(rejected, list) else [rejected]

            len_chosen = _render(prompt_msgs + chosen_msgs)
            len_rejected = _render(prompt_msgs + rejected_msgs)
            lengths.append(max(len_chosen, len_rejected))
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning(f"DPO length calc error: {e}")

    valid = [length for length in lengths if length <= max_absolute_length]
    outliers = len(lengths) - len(valid)
    if not valid:
        logger.warning("Could not compute DPO lengths; leaving max_length unchanged")
        return None

    p_length = int(sorted(valid)[int(percentile * len(valid))])
    max_length = _align_to_multiple(p_length)
    logger.info(f"Analyzed {len(valid)} samples ({outliers} outliers, {errors} errors)")
    logger.info(f"Average length: {sum(valid) / len(valid):.1f}")
    logger.info(f"{percentile * 100}th percentile length: {p_length}")
    logger.info(f"Estimated max_length (aligned to 64): {max_length}")
    return max_length


def deserialize_conversations(dataset: Dataset) -> Dataset:
    """Deserialize JSON-encoded message fields for DPOTrainer conversational format.

    Uses set_transform for lazy deserialization to avoid Arrow schema conflicts
    when tool_calls have varying argument structures across samples. The raw data
    stays as JSON strings in Arrow; deserialization happens on-the-fly when
    DPOTrainer accesses each batch.
    """

    def _parse_field(value):
        # Deserialize only JSON-encoded *structures* (serialized message lists/dicts,
        # which start with '[' or '{'). Everything else is returned unchanged:
        #   - native list/dict            -> passthrough (idempotent)
        #   - standard-format plain text  -> passthrough (never raises)
        #   - bare scalars like "4"/"true"-> stay strings (not coerced to int/bool)
        if isinstance(value, str) and value.lstrip()[:1] in ("[", "{"):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        return value

    def transform(batch):
        if "prompt" not in batch:
            # Raw columns already stripped (e.g. remove_unused_columns=True ran after
            # DPOTrainer's own tokenization) — nothing left for this transform to do.
            return batch

        has_chosen = "chosen" in batch
        has_rejected = "rejected" in batch
        system = batch.get("system")
        batch_size = len(batch["prompt"])

        prompts, chosens, rejecteds = [], [], []
        for i in range(batch_size):
            prompt = _parse_field(batch["prompt"][i])

            # Conversational format: prompt is a list of message dicts, so prepend a
            # separate `system` column when present. Standard format: prompt is a
            # plain string and is left exactly as-is (no system merge, no iteration).
            if isinstance(prompt, list) and system and system[i]:
                prompt = [{"role": "system", "content": system[i]}] + prompt

            prompts.append(prompt)
            if has_chosen:
                chosens.append(_parse_field(batch["chosen"][i]))
            if has_rejected:
                rejecteds.append(_parse_field(batch["rejected"][i]))

        result = {"prompt": prompts}
        if has_chosen:
            result["chosen"] = chosens
        if has_rejected:
            result["rejected"] = rejecteds

        # Preserve extra fields (drop the now-merged system column)
        for key in batch:
            if key not in ("system", "prompt", "chosen", "rejected"):
                result[key] = batch[key]

        return result

    dataset.set_transform(transform)
    return dataset


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
    """Load training and test datasets.

    When deserialize_messages=True, parses JSON-serialized prompt/chosen/rejected
    fields back into message lists and merges system messages into prompt.
    DPOTrainer then handles apply_chat_template internally.
    """
    try:
        logger.info(f"Loading training dataset from {script_args.train_dataset_path}")
        train_ds = _load_dataset_auto(script_args.train_dataset_path)

        if script_args.deserialize_messages:
            logger.info("Deserializing JSON-encoded message fields")
            train_ds = deserialize_conversations(train_ds)

        test_ds = None
        if script_args.val_dataset_path:
            logger.info(f"Loading test dataset from {script_args.val_dataset_path}")
            test_ds = _load_dataset_auto(script_args.val_dataset_path)

            if script_args.deserialize_messages:
                logger.info("Deserializing val JSON-encoded message fields")
                test_ds = deserialize_conversations(test_ds)

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

    if script_args.vlm_base_model_id:
        _resolve_vlm_base_model_id(script_args.model_id, script_args.vlm_base_model_id)
        logger.info(
            "Validated explicit VLM export base: %s",
            script_args.vlm_base_model_id,
        )

    # Load model, tokenizer, and processor using centralized config
    # Processor is loaded for any VLM (auto-detected), not just when modality_type is "image".
    # modality_type controls the data pipeline; VLM detection controls model loading/merging.
    model = load_model(config_builder, script_args)
    tokenizer = load_tokenizer(script_args, training_args)
    is_vlm = script_args.modality_type == "image"
    processor = None
    if is_vlm:
        processor = load_processor(script_args, training_args)
        if processor is not None:
            logger.info(
                "Multi-modal mode: using processor as processing_class for DPOTrainer"
            )

    # Auto-calculate DPOConfig.max_length (prompt + completion) from the dataset when
    # requested and not set explicitly. Skipped for image modality, where sequences
    # include image tokens the tokenizer cannot measure here and max_length is forced
    # to None below anyway.
    #
    # The "not set explicitly" test compares against the DPOConfig default rather than
    # None: `max_length` defaults to 1024, never None, so a `... is None` guard here is
    # dead code - auto_calculate_lengths silently did nothing and every run trained at
    # 1024 tokens, truncating from the right and cutting the completion off multi-turn
    # samples. Mirrors the same check in sft/train.py.
    default_max_length = 1024  # DPOConfig.max_length default
    max_length_is_default = getattr(training_args, "max_length", None) in (
        None,
        default_max_length,
    )
    if script_args.auto_calculate_lengths and not is_vlm and max_length_is_default:
        logger.info("Auto-calculating optimal DPO max_length from dataset...")
        computed_max_length = calculate_optimal_dpo_lengths(
            tokenizer, train_ds, deserialize_messages=script_args.deserialize_messages
        )
        if computed_max_length is not None:
            training_args.max_length = computed_max_length
            logger.info(f"Set max_length={computed_max_length}")
    elif script_args.auto_calculate_lengths and not is_vlm:
        # Explicit value wins, but say so: silence here is what made the dead guard
        # above so hard to spot from the logs.
        logger.info(
            f"auto_calculate_lengths requested but max_length={training_args.max_length} "
            "was set explicitly; keeping the explicit value."
        )

    # Extract tools from dataset if available
    tools = extract_tools_from_dataset(train_ds)
    if tools:
        logger.info(f"Found {len(tools)} tools in dataset")
        training_args.tools = tools

    # Cast the frozen base model's params/buffers to a uniform dtype BEFORE LoRA is
    # applied, so PEFT's fp32 upcast of the adapter weights (see apply_lora_config /
    # get_peft_model) is preserved instead of being immediately overwritten.
    if script_args.cast_parameters_to_uniform_dtype:
        cast_parameters_to_uniform_dtype(model, config_builder.torch_dtype)

    # Apply PEFT before trainer (same as SFT) for FSDP compatibility
    if script_args.use_peft:
        model = apply_lora_config(model, script_args, is_vlm=is_vlm)

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

    # Initialize DPO trainer
    # Note: peft_config is NOT passed here — model is already wrapped with PEFT above.
    # DPOTrainer auto-detects PeftModel and uses adapter disabling for reference logits.
    # For VLMs with image data, set max_length=None to avoid truncating image tokens.
    # DPOTrainer auto-uses DataCollatorForVisionPreference when processor is passed.
    if processor is not None and script_args.modality_type == "image":
        training_args.max_length = None
        logger.info(
            "VLM + image data: set max_length=None to avoid truncating image tokens"
        )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=processor if processor is not None else tokenizer,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        callbacks=callbacks,
    )

    trainer = patch_dpo_trainer_dtype(trainer)

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
        processor,
        script_args,
        training_args,
        trainer.accelerator,
        mlflow_enabled,
        original_output_dir,
    )
    trainer.accelerator.wait_for_everyone()


def main() -> None:
    """Main function to parse arguments and start training."""
    parser = TrlParser((ScriptArguments, DPOConfig))
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
