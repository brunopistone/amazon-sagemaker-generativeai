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
import shutil
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
from trl import GRPOConfig, GRPOTrainer, TrlParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import WandbCallback
import contextlib
from typing import Any, Dict, List, Optional, Tuple
import wandb

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None

# Reward functions live in the `reward_function/` package beside this script, one
# per module, and register themselves by name (see reward_function/__init__.py).
# A script's own directory is normally sys.path[0] already, but `python -m` and
# some launchers set it differently, so add it explicitly: the import then works
# from any working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from reward_function import (  # noqa: E402
    CHAT_ROLES,
    TURN_YIELDING_ROLES,
    load_reward_functions,
    render_message_content,
    set_tool_call_format,
)

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
    # --- RLAIF: settings for the 'judge' / 'judge_http' reward functions ---------
    # All default to None, meaning "use the reward module's own default" (see
    # reward_function/_judge.py), so the defaults live in exactly one place. They
    # are ignored unless --reward_funcs selects a judge reward.
    judge_model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Judge model. For 'judge' (Bedrock) defaults to "
                "'anthropic.claude-opus-5'; for 'judge_http' there is no default "
                "and this is required."
            )
        },
    )
    judge_auth: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "How the 'judge_http' reward authenticates: 'bedrock_mantle' to use "
                "Amazon Bedrock Mantle with AWS credentials (endpoint derived from "
                "--judge_region, token auto-refreshed for the life of the run), or "
                "'api_key' for a static bearer token from --judge_api_key_env. "
                "Default api_key."
            )
        },
    )
    judge_region: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "AWS region for the 'judge' (Bedrock) reward and for --judge_auth "
                "bedrock_mantle. Default us-east-1."
            )
        },
    )
    judge_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "OpenAI-compatible base URL for the 'judge_http' reward, e.g. "
                "'http://judge-host:8000/v1'. Required unless --judge_auth is "
                "bedrock_mantle, which derives it from --judge_region."
            )
        },
    )
    judge_api_key_env: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name of the environment variable holding the judge API key, so the "
                "key stays out of the YAML config. Default JUDGE_API_KEY; ignored "
                "when the endpoint needs no key. Under --judge_auth bedrock_mantle "
                "this (or BEDROCK_API_KEY) short-circuits token minting, which lets "
                "the job run without an AWS credential chain."
            )
        },
    )
    judge_rubric_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "File containing the judging rubric, with '{prompt}' and "
                "'{completion}' placeholders (plus optional '{answer}' and "
                "'{scale}'). Defaults to a built-in general-quality rubric."
            )
        },
    )
    judge_score_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Upper bound of the rubric's score range; scores are divided by it "
                "to land in [0, 1]. Default 10."
            )
        },
    )
    judge_max_concurrency: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "In-flight judge calls per rank. The judge sees up to "
                "world_size x this, since rewards are computed per process. Default 8."
            )
        },
    )
    judge_retries: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Attempts per completion before giving up and returning no score for "
                "it (that sample then falls back to the other rewards). Default 3."
            )
        },
    )
    judge_timeout: Optional[float] = field(
        default=None,
        metadata={"help": "Per-call judge timeout in seconds. Default 120."},
    )
    judge_batch_timeout: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Ceiling, in seconds, on how long one training step may spend "
                "judging. Completions still unscored when it expires get no score "
                "for this reward, so the step always finishes. Sizing: a batch needs "
                "about ceil(completions_per_rank / judge_max_concurrency) x "
                "judge_timeout in the worst case. Default 900; set 0 for no deadline."
            )
        },
    )
    judge_preflight: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Make one throwaway judge call at startup and abort the job if it "
                "fails, so a bad API key, model id, base URL or token budget stops "
                "the run before the model loads instead of silently scoring every "
                "sample NaN for hours. Default true; costs one judge call."
            )
        },
    )
    judge_max_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Judge response cap. Must leave room for reasoning tokens on models "
                "that think before answering. Default 2048."
            )
        },
    )
    judge_effort: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Reasoning effort for the 'judge' (Bedrock Converse) reward: low, "
                "medium, high, xhigh or max, sent in Claude's shape "
                "(thinking.type=adaptive + output_config.effort). Off by default "
                "('none'): reasoning parameters are model-family-specific and "
                "mutually rejecting, and Claude with adaptive thinking plus an "
                "effort level can return its reasoning encrypted with the visible "
                "text empty, which makes every completion unscorable."
            )
        },
    )
    judge_structured_output: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Whether the 'judge' reward asks Bedrock to enforce the verdict's "
                "JSON schema server-side (Converse outputConfig): 'auto' (default) "
                "tries it and falls back to a prompt instruction with tolerant "
                "parsing if the model refuses, 'on' makes a refusal fatal, 'off' "
                "never sends it. Support is per-model: the Claude *-5 generation "
                "rejects it, 4.5/4.6 accept it."
            )
        },
    )
    judge_protocol: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "HTTP wire protocol for the 'judge_http' reward: chat_completions, "
                "responses, or anthropic. Inferred from the model id when unset "
                "(Claude -> anthropic, GPT-5.x -> responses, everything else -> "
                "chat_completions); the routes are not interchangeable and sending "
                "a model to the wrong one is a hard 4xx."
            )
        },
    )
    judge_log_critiques: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Log each judge critique into the completions table, which is how "
                "you audit for reward hacking. Default true."
            )
        },
    )
    length_reward_target: int = field(
        default=512,
        metadata={
            "help": (
                "Completion length in *characters* at which the built-in 'length' "
                "reward saturates at 1.0. Anything shorter is rewarded "
                "proportionally, so keep it in the same ballpark as the output "
                "length you actually want (roughly 4 characters per token)."
            )
        },
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
    vlm_base_model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Original full VLM used to restore the vision tower when GRPO trains "
                "the language model through AutoModelForCausalLM. Required when "
                "model_id is a text-only checkpoint derived from a multimodal model."
            )
        },
    )
    prompt_field: str = field(
        default="prompt",
        metadata={
            "help": (
                "Dataset column holding the prompt. Renamed to 'prompt' before "
                "training because GRPOTrainer reads that column name literally."
            )
        },
    )
    reward_funcs: Optional[str] = field(
        default="format,length",
        metadata={
            "help": (
                "Comma-separated reward functions: any module in "
                "reward_function/ (format, length, rouge, judge, judge_http) or a "
                "'module.path:function_name' import specification. The judge rewards "
                "score with an LLM (RLAIF) and are configured with --judge_*."
            )
        },
    )
    target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Target modules for LoRA"}
    )
    token: str = field(default=None, metadata={"help": "Hugging Face API token"})
    # NOTE: `trust_remote_code` deliberately does NOT live here. TRL 1.10.0 added
    # the same field to its configs (GRPOConfig/DPOConfig/SFTConfig), and TrlParser
    # builds one argparse namespace from both dataclasses, so defining it in both
    # aborts at startup with:
    #   argparse.ArgumentError: argument --trust_remote_code/--trust-remote-code:
    #   conflicting option strings
    # The YAML key still works - it binds to the trainer config instead - and
    # `trust_remote_code_for()` reads the value back off `script_args` after
    # `main()` copies it across. Same fix as `max_length` in the SFT script.
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


def trust_remote_code_for(
    script_args: ScriptArguments, training_args: Optional["GRPOConfig"] = None
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
        """Build complete model loading arguments.

        Deliberately does NOT pass `use_cache`. It is a *config* attribute, not a
        model `__init__` argument, and `from_pretrained` only diverts a kwarg into
        the config when the config already declares it at the top level:

        * flat configs (Llama, Mistral, dense Qwen) declare `use_cache`, so it was
          absorbed - and set `config.use_cache=False`. With no generation_config.json
          in the repo, `GenerationConfig.from_model_config` then inherits it, so
          every GRPO rollout generated WITHOUT a KV cache: quadratic instead of
          linear, over `num_generations` completions per prompt.
        * composite/VLM configs (Qwen3_5Config keeps `use_cache` in `text_config`)
          do not declare it at the top level, so it stayed in model_kwargs, reached
          `cls(config, **model_kwargs)` and raised
          `TypeError: __init__() got an unexpected keyword argument 'use_cache'`.

        Nothing needs to set it here: TRL's GRPOTrainer passes `use_cache=False`
        explicitly on each training forward pass, and generation must keep the cache
        enabled to be usably fast.
        """
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


def load_tokenizer(
    script_args: ScriptArguments, training_args: "GRPOConfig"
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


def _is_vlm_from_config(model_id: Optional[str]) -> bool:
    """Return whether ``model_id`` has an image-to-text model configuration."""
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
    """Resolve the full VLM that supplies vision weights during export."""
    if explicit_vlm_base_model_id:
        if not _is_vlm_from_config(explicit_vlm_base_model_id):
            raise ValueError(
                "vlm_base_model_id must reference a full VLM with a supported "
                f"image-text config, got: {explicit_vlm_base_model_id}"
            )
        return explicit_vlm_base_model_id
    if _is_vlm_from_config(adapter_base_model_id):
        return adapter_base_model_id
    return None


def _transplant_into_vlm(
    merged_causal_state: Dict[str, torch.Tensor],
    vlm_base_model_id: str,
    torch_dtype: torch.dtype,
):
    """Strictly transplant merged language weights into a complete VLM."""
    if AutoModelForImageTextToText is None:
        raise RuntimeError(
            "This Transformers version has no image-text auto model class"
        )

    logger.info(
        "Transplanting merged CausalLM weights into full VLM from %s",
        vlm_base_model_id,
    )
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        vlm_base_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **trust_remote_code_kwargs(vlm_base_model_id),
    )
    vlm_state = vlm_model.state_dict()

    causal_layer_key = next(
        (key for key in merged_causal_state if ".layers.0." in key), None
    )
    if causal_layer_key is None:
        raise RuntimeError("No transformer layer keys found in merged CausalLM state")
    causal_prefix = causal_layer_key.split("layers.0.")[0]

    vlm_layer_key = next(
        (key for key in vlm_state if ".layers.0." in key and "language_model" in key),
        None,
    )
    if vlm_layer_key is None:
        raise RuntimeError("No language_model layer keys found in the full VLM state")
    vlm_prefix = vlm_layer_key.split("layers.0.")[0]

    updated = 0
    core_count = 0
    missing_core = []
    shape_mismatches = []
    for causal_key, value in merged_causal_state.items():
        is_core = causal_key.startswith(causal_prefix)
        if is_core:
            core_count += 1
            vlm_key = vlm_prefix + causal_key[len(causal_prefix) :]
        elif causal_key in vlm_state:
            vlm_key = causal_key
        else:
            vlm_key = next((key for key in vlm_state if key.endswith(causal_key)), None)

        if not vlm_key or vlm_key not in vlm_state:
            if is_core:
                missing_core.append(causal_key)
            continue
        if tuple(vlm_state[vlm_key].shape) != tuple(value.shape):
            shape_mismatches.append(
                f"{causal_key} -> {vlm_key}: "
                f"{tuple(value.shape)} != {tuple(vlm_state[vlm_key].shape)}"
            )
            continue
        vlm_state[vlm_key] = value
        updated += 1

    if core_count == 0 or missing_core or shape_mismatches:
        details = []
        if missing_core:
            details.append(
                f"{len(missing_core)} unmapped core keys, e.g. {missing_core[:3]}"
            )
        if shape_mismatches:
            details.append(
                f"{len(shape_mismatches)} shape mismatches, e.g. "
                f"{shape_mismatches[:3]}"
            )
        raise RuntimeError(
            "Unsafe CausalLM-to-VLM transplant refused: " + "; ".join(details)
        )

    logger.info(
        "Transplanted %d/%d language-model tensors into VLM (%s* -> %s*)",
        updated,
        len(merged_causal_state),
        causal_prefix,
        vlm_prefix,
    )
    vlm_model.load_state_dict(vlm_state)
    return vlm_model


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


def _load_and_merge_adapter(
    adapter_dir: str,
    torch_dtype: torch.dtype,
    explicit_vlm_base_model_id: Optional[str] = None,
):
    """Merge the GRPO adapter and restore a full VLM when one is expected."""
    from peft import PeftConfig

    _patch_peft_weight_converter_compat()
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    adapter_base_model_id = peft_config.base_model_name_or_path
    vlm_base_model_id = _resolve_vlm_base_model_id(
        adapter_base_model_id, explicit_vlm_base_model_id
    )

    causal_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **trust_remote_code_kwargs(adapter_base_model_id),
    )
    merged_causal = causal_model.merge_and_unload()
    if not vlm_base_model_id:
        return merged_causal

    merged_state = merged_causal.state_dict()
    del causal_model, merged_causal
    torch.cuda.empty_cache()
    return _transplant_into_vlm(merged_state, vlm_base_model_id, torch_dtype)


def _merge_adapter_in_process(
    temp_dir: str,
    final_output_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    vlm_base_model_id: Optional[str] = None,
) -> AutoModelForCausalLM:
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
    """Merge LoRA in a clean process and preserve a full VLM when requested."""
    merge_script = textwrap.dedent(f"""\
        import torch
        from peft import AutoPeftModelForCausalLM, PeftConfig
        from transformers import AutoConfig
        from transformers.core_model_loading import WeightConverter

        _orig_wc_init = WeightConverter.__init__
        if not getattr(_orig_wc_init, "_peft_compat", False):
            def _wc_init(self, *a, distributed_operation=None, quantization_operation=None, **kw):
                _orig_wc_init(self, *a, **kw)
                self.distributed_operation = distributed_operation
                self.quantization_operation = quantization_operation
            _wc_init._peft_compat = True
            WeightConverter.__init__ = _wc_init

        adapter_dir = {temp_dir!r}
        output_dir = {final_output_dir!r}
        dtype = getattr(torch, {torch_dtype_str!r})
        trc = {trc_kwargs or {}!r}
        explicit_vlm_base_model_id = {vlm_base_model_id!r}

        peft_config = PeftConfig.from_pretrained(adapter_dir)
        adapter_base_model_id = peft_config.base_model_name_or_path

        def is_vlm(model_id):
            if not model_id:
                return False
            try:
                from transformers.models.auto.modeling_auto import (
                    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
                )
                config = AutoConfig.from_pretrained(model_id, **trc)
                return config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
            except Exception:
                return False

        if explicit_vlm_base_model_id:
            if not is_vlm(explicit_vlm_base_model_id):
                raise ValueError(
                    "vlm_base_model_id is not a supported full VLM: "
                    + explicit_vlm_base_model_id
                )
            full_vlm_base_model_id = explicit_vlm_base_model_id
        elif is_vlm(adapter_base_model_id):
            full_vlm_base_model_id = adapter_base_model_id
        else:
            full_vlm_base_model_id = None

        print("Loading adapter for merging...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **trc,
        )
        model = model.merge_and_unload()

        if full_vlm_base_model_id:
            try:
                from transformers import AutoModelForImageTextToText
                vlm_auto_cls = AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModelForVision2Seq as vlm_auto_cls

            merged_state = model.state_dict()
            del model
            torch.cuda.empty_cache()
            vlm_model = vlm_auto_cls.from_pretrained(
                full_vlm_base_model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **trc,
            )
            vlm_state = vlm_model.state_dict()
            causal_lk = next(
                (key for key in merged_state if ".layers.0." in key), None
            )
            vlm_lk = next(
                (
                    key for key in vlm_state
                    if ".layers.0." in key and "language_model" in key
                ),
                None,
            )
            if causal_lk is None or vlm_lk is None:
                raise RuntimeError(
                    "Cannot determine CausalLM-to-VLM key prefixes; export aborted"
                )
            causal_prefix = causal_lk.split("layers.0.")[0]
            vlm_prefix = vlm_lk.split("layers.0.")[0]
            core_count = 0
            updated = 0
            missing_core = []
            shape_mismatches = []
            for causal_key, value in merged_state.items():
                is_core = causal_key.startswith(causal_prefix)
                if is_core:
                    core_count += 1
                    vlm_key = vlm_prefix + causal_key[len(causal_prefix):]
                elif causal_key in vlm_state:
                    vlm_key = causal_key
                else:
                    vlm_key = next(
                        (key for key in vlm_state if key.endswith(causal_key)),
                        None,
                    )
                if not vlm_key or vlm_key not in vlm_state:
                    if is_core:
                        missing_core.append(causal_key)
                    continue
                if tuple(vlm_state[vlm_key].shape) != tuple(value.shape):
                    shape_mismatches.append(causal_key)
                    continue
                vlm_state[vlm_key] = value
                updated += 1
            if core_count == 0 or missing_core or shape_mismatches:
                raise RuntimeError(
                    "Unsafe CausalLM-to-VLM transplant refused: "
                    f"{{len(missing_core)}} unmapped core keys, "
                    f"{{len(shape_mismatches)}} shape mismatches"
                )
            print(
                f"Transplanted {{updated}}/{{len(merged_state)}} tensors "
                f"into {{full_vlm_base_model_id}}"
            )
            vlm_model.load_state_dict(vlm_state)
            model = vlm_model

        print("Saving merged model...")
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
    final_output_dir: str,
    vlm_base_model_id: Optional[str] = None,
) -> None:
    """Save tokenizer and, for standalone VLM exports, the full processor."""
    tokenizer.save_pretrained(final_output_dir)
    _align_generation_config(tokenizer, final_output_dir)
    if not vlm_base_model_id:
        return

    try:
        processor = AutoProcessor.from_pretrained(
            vlm_base_model_id,
            **trust_remote_code_kwargs(vlm_base_model_id),
        )
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
    except Exception as e:
        raise RuntimeError(
            f"Could not save processor from VLM base {vlm_base_model_id}: {e}"
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
    output_dir = os.path.abspath(final_output_dir)
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(config_path):
        raise RuntimeError(f"VLM export is missing config.json: {output_dir}")

    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config.get("vision_config"), dict):
        raise RuntimeError("VLM export config.json has no vision_config")
    if str(config.get("model_type", "")).endswith("_text"):
        raise RuntimeError(
            f"VLM export has text-only model_type={config.get('model_type')!r}"
        )

    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    weight_keys = []
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as index_file:
            index = json.load(index_file)
        weight_map = index.get("weight_map", {})
        weight_keys = list(weight_map)
        missing_shards = sorted(
            {
                filename
                for filename in weight_map.values()
                if not os.path.isfile(os.path.join(output_dir, filename))
            }
        )
        if missing_shards:
            raise RuntimeError(
                f"VLM export references missing shards: {missing_shards}"
            )
    else:
        from glob import glob
        from safetensors import safe_open

        shard_paths = glob(os.path.join(output_dir, "*.safetensors"))
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
        if not os.path.isfile(os.path.join(output_dir, filename))
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


def _drop_reference_adapter(output_dir: str) -> None:
    """Delete the auxiliary "ref" adapter GRPOTrainer saves next to "default".

    This script applies LoRA before constructing the trainer (required for FSDP), so
    GRPOTrainer receives a model that is already a ``PeftModel``. When ``beta != 0.0``
    it then adds a second adapter named ``ref``, a frozen copy of the initial weights
    used as the reference policy for the KL term. ``PeftModel.save_pretrained``
    persists *every* adapter it knows about: ``default`` (the trained policy) at the
    root of the output directory and ``ref`` in a ``ref/`` subdirectory.

    Only ``default`` is wanted in the exported artifact. The merge path already reads
    just the root adapter, so this mainly keeps a duplicate copy of the *untrained*
    weights out of the shipped model when the adapter is exported unmerged.
    """
    ref_dir = os.path.join(output_dir, "ref")
    if os.path.isdir(ref_dir):
        logger.info(f"Removing auxiliary GRPO reference adapter at {ref_dir}")
        shutil.rmtree(ref_dir, ignore_errors=True)


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
    training_args: "GRPOConfig",
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
                _drop_reference_adapter(temp_dir)
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
                    tokenizer, final_output_dir, vlm_source_model_id
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
                _drop_reference_adapter(temp_dir)
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
                    tokenizer, final_output_dir, vlm_source_model_id
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
            _drop_reference_adapter(final_output_dir)
            _save_artifacts_on_main(tokenizer, final_output_dir, vlm_source_model_id)
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
    sample_size: int = 1000,
    percentile: float = 0.95,
) -> int:
    """Calculate optimal max_completion_length for GRPO.

    Tokenizes prompts using apply_chat_template to match GRPOTrainer's internal
    tokenization and logs prompt length statistics. Returns only max_completion_length
    since GRPOConfig does not have a max_prompt_length parameter.

    Reads the ``prompt`` column: ``_normalize_prompt_for_grpo`` has already renamed
    any custom prompt field by the time this runs.

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
            prompt = sample.get("prompt", "")

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


def _looks_like_serialized_chat(prompt: str) -> bool:
    """True if a plain-string prompt is really a JSON-encoded message list.

    ``deserialize_messages`` left off on a JSON-string dataset fails silently: the
    lazy transform never parses anything, GRPOTrainer sees a ``str`` and skips the
    chat template, and the policy trains on the literal text
    ``[{"role": "user", ...}]``. Nothing downstream objects - a string prompt is a
    legitimate standard-format prompt - so this is the only place it can be caught.
    """
    text = prompt.lstrip()
    if text[:1] != "[":
        return False
    try:
        parsed = json.loads(text, strict=False)
    except (json.JSONDecodeError, ValueError):
        return False
    return (
        isinstance(parsed, list)
        and bool(parsed)
        and all(isinstance(m, dict) and "role" in m for m in parsed)
    )


def _validate_grpo_prompts(
    dataset: Dataset, split: str, sample_size: int = 256
) -> None:
    """Fail fast on prompt shapes that train silently but wrongly.

    GRPO gives no early feedback on a malformed prompt: generation succeeds, the
    reward comes back a plausible number, and the run spends its budget learning
    from a corrupted view of the task. Each check below maps to a defect that has
    actually reached a paid run:

      1. A prompt ending on an ``assistant`` message makes the chat template
         CONTINUE that turn rather than open a new one, so the policy is trained to
         extend its own half-written reply instead of taking the next action.
      2. A message that flattens to empty text is a hole in the context. Every
         reward function sees it, and the judge in particular then grades a
         trajectory with the actions missing.
      3. An unknown role is dropped silently by most chat templates, so the turn
         vanishes from the rendered prompt with no error anywhere.

    Raises on any of the three. Logs the turn-count distribution either way, so
    whether this is a single-turn or multi-turn dataset is visible in the training
    log instead of being reverse-engineered from a metrics post-mortem.

    Only ``sample_size`` rows are inspected, spread evenly across the split and
    always including the last row, rather than taken from the head: shape defects
    are systematic, and spreading also catches one confined to a tail shard that a
    head slice would never reach.
    """
    total = len(dataset)
    if total == 0:
        raise ValueError(f"{split} dataset is empty; GRPO has nothing to sample from.")

    if total <= sample_size:
        indices = list(range(total))
    else:
        # Evenly spaced *and* pinned to the final row. Truncating a strided range
        # to sample_size instead (``range(0, total, total // sample_size)[:n]``)
        # silently drops the tail - with total=1000, n=256 it stops at row 765.
        step = total / sample_size
        indices = sorted({int(i * step) for i in range(sample_size)} | {total - 1})

    turn_counts: Dict[int, int] = {}
    conversational = plain = 0
    bad_final_role: List[Tuple[int, str]] = []
    blank_messages: List[Tuple[int, int, str]] = []
    unknown_roles: List[Tuple[int, str]] = []
    undeserialized: List[int] = []

    for i in indices:
        prompt = dataset[i]["prompt"]

        if isinstance(prompt, str):
            plain += 1
            if not prompt.strip():
                blank_messages.append((i, 0, "<plain-text prompt>"))
            elif _looks_like_serialized_chat(prompt):
                undeserialized.append(i)
            continue
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(
                f"{split}[{i}]: prompt is {type(prompt).__name__}, expected a string "
                "(standard format) or a non-empty list of message dicts "
                "(conversational format). If the column holds JSON strings, set "
                "deserialize_messages: true."
            )

        conversational += 1
        turn_counts[len(prompt)] = turn_counts.get(len(prompt), 0) + 1

        for position, message in enumerate(prompt):
            if not isinstance(message, dict):
                raise ValueError(
                    f"{split}[{i}] message {position}: expected a dict with 'role' "
                    f"and 'content', got {type(message).__name__}."
                )
            role = message.get("role")
            if role not in CHAT_ROLES:
                unknown_roles.append((i, str(role)))
            if not render_message_content(message).strip():
                blank_messages.append((i, position, str(role)))

        final_role = prompt[-1].get("role")
        if final_role not in TURN_YIELDING_ROLES:
            bad_final_role.append((i, str(final_role)))

    if conversational and plain:
        logger.warning(
            f"{split}: mixed prompt formats in the sample ({conversational} "
            f"conversational, {plain} plain-text). This is legal - the chat template "
            "is applied per row - but usually means the dataset was built by two "
            "different code paths."
        )

    if conversational:
        spread = ", ".join(f"{k} msgs: {v}" for k, v in sorted(turn_counts.items()))
        multi = sum(v for k, v in turn_counts.items() if k > 2)
        logger.info(
            f"{split}: {conversational}/{len(indices)} sampled prompts are "
            f"conversational; {multi} are multi-turn (>2 messages). "
            f"Turn spread -> {spread}"
        )

    problems = []
    if undeserialized:
        shown = ", ".join(str(i) for i in undeserialized[:5])
        problems.append(
            f"{len(undeserialized)} prompt(s) are JSON-encoded message lists still "
            f"stored as strings (rows {shown}). Set deserialize_messages: true - "
            "left off, the chat template is skipped entirely and the model trains on "
            "the literal text '[{\"role\": ...}]'."
        )
    if bad_final_role:
        shown = ", ".join(f"row {i} ends on '{r}'" for i, r in bad_final_role[:5])
        problems.append(
            f"{len(bad_final_role)} prompt(s) do not end on a turn-yielding role "
            f"{TURN_YIELDING_ROLES}: {shown}. The model would continue that turn "
            "instead of starting its own; drop the trailing message when building "
            "the dataset."
        )
    if unknown_roles:
        shown = ", ".join(f"row {i}: {r!r}" for i, r in unknown_roles[:5])
        problems.append(
            f"{len(unknown_roles)} message(s) carry a role outside {CHAT_ROLES}: "
            f"{shown}. Most chat templates drop these without warning."
        )
    if blank_messages:
        shown = ", ".join(
            f"row {i} message {p} (role {r})" for i, p, r in blank_messages[:5]
        )
        problems.append(
            f"{len(blank_messages)} message(s) flatten to empty text: {shown}. Check "
            "for payload stored in a field the renderer does not read - an assistant "
            "turn's action lives in 'tool_calls' and its rationale in "
            "'reasoning_content', neither of which is 'content'."
        )

    if problems:
        raise ValueError(
            f"{split} dataset failed GRPO prompt validation "
            f"({len(indices)} of {total} rows sampled):\n  - " + "\n  - ".join(problems)
        )

    logger.info(f"{split}: prompt validation passed on {len(indices)} sampled rows.")


def _normalize_prompt_for_grpo(
    dataset: Dataset, script_args: ScriptArguments
) -> Dataset:
    """Shape the dataset the way GRPOTrainer expects it.

    GRPOTrainer reads a column literally named ``prompt`` (``[x["prompt"] for x in
    inputs]``) and never looks at a configurable field name, so we:
      1. rename a custom prompt field to ``prompt`` (cheap metadata rename, no
         re-serialization) — without this, a non-default ``--prompt_field`` fails
         with ``KeyError: 'prompt'`` at the first generation step;
      2. optionally parse JSON-encoded prompts lazily via ``set_transform`` when
         ``deserialize_messages=True`` (mirrors the SFT and DPO scripts).

    The deserialization is lazy rather than a ``dataset.map()`` so the raw JSON
    strings stay in Arrow: a ``map`` has to commit to one Arrow type for the column
    and blows up (``ArrowInvalid``) on datasets that mix conversational rows
    (``list<struct>``) with plain-text rows (``string``). Every other column is
    passed through untouched, since GRPO forwards them to the reward functions as
    keyword arguments (``GRPOConfig.remove_unused_columns`` defaults to ``False``).
    """
    if script_args.prompt_field != "prompt":
        logger.info(f"Renaming column '{script_args.prompt_field}' -> 'prompt'")
        dataset = dataset.rename_column(script_args.prompt_field, "prompt")

    if script_args.deserialize_messages:

        def _transform(batch):
            if "prompt" not in batch:
                return batch
            parsed = []
            for prompt in batch["prompt"]:
                # Only parse strings that look like a serialized structure ('[' or
                # '{'), so standard plain-text prompts (and bare scalars like "4")
                # are left as-is rather than raising or being coerced to non-string
                # types.
                if isinstance(prompt, str) and prompt.lstrip()[:1] in ("[", "{"):
                    try:
                        prompt = json.loads(prompt)
                    except (json.JSONDecodeError, ValueError):
                        pass  # Keep as plain string (standard format)
                parsed.append(prompt)
            batch["prompt"] = parsed
            return batch

        dataset.set_transform(_transform)

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
    """Load training and test datasets."""
    try:
        logger.info(f"Loading training dataset from {script_args.train_dataset_path}")
        train_ds = _load_dataset_auto(script_args.train_dataset_path)
        logger.info(
            f"Training dataset loaded: {len(train_ds)} samples, "
            f"columns: {train_ds.column_names}"
        )
        train_ds = _normalize_prompt_for_grpo(train_ds, script_args)
        _validate_grpo_prompts(train_ds, "train")

        test_ds = None
        if script_args.val_dataset_path:
            logger.info(f"Loading test dataset from {script_args.val_dataset_path}")
            test_ds = _load_dataset_auto(script_args.val_dataset_path)
            logger.info(
                f"Test dataset loaded: {len(test_ds)} samples, "
                f"columns: {test_ds.column_names}"
            )
            test_ds = _normalize_prompt_for_grpo(test_ds, script_args)
            _validate_grpo_prompts(test_ds, "val")

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

    # Load model and tokenizer using centralized config
    model = load_model(config_builder, script_args)
    tokenizer = load_tokenizer(script_args, training_args)

    # Rewards re-render historical tool calls; match the model's own syntax.
    tool_call_format = set_tool_call_format(tokenizer=tokenizer)
    logger.info(f"Reward tool-call rendering format: {tool_call_format}")

    # Auto-calculate lengths if enabled
    if script_args.auto_calculate_lengths:
        logger.info("Auto-calculating optimal lengths from dataset...")
        max_completion_length = calculate_optimal_grpo_lengths(tokenizer, train_ds)
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

    # Load reward functions. `script_args` is forwarded so a parameterised reward
    # can read its own config field off it (e.g. `length` reads
    # `length_reward_target`) without this call site growing an argument per reward.
    reward_funcs = load_reward_functions(script_args.reward_funcs, script_args)

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

    if script_args.use_peft and trainer.accelerator.is_main_process:
        # With beta != 0.0 GRPOTrainer adds a second adapter ("ref") holding a frozen
        # copy of the initial weights, and its params are counted here too — expect
        # roughly twice the trainable count of the policy adapter alone.
        trainer.model.print_trainable_parameters()

    if script_args.checkpoint_dir is not None:
        os.makedirs(script_args.checkpoint_dir, exist_ok=True)

        original_output_dir = training_args.output_dir
        training_args.output_dir = script_args.checkpoint_dir
    else:
        original_output_dir = training_args.output_dir

    if getattr(training_args, "log_completions", False):
        # GRPOTrainer.log() writes the completions table to
        # {output_dir}/completions/completions_NNNNN.parquet but never creates that
        # subdirectory, and pandas.to_parquet refuses a missing parent - so
        # log_completions=True dies on the FIRST logging step with "Cannot save file
        # into a non-existent directory". Created here rather than earlier because
        # output_dir has only just been finalised: with checkpoint_dir set it points
        # at the SageMaker checkpoint channel, not at output_dir from the config.
        # exist_ok makes this safe to run from every rank even though only the main
        # process writes the table.
        os.makedirs(
            os.path.join(training_args.output_dir, "completions"), exist_ok=True
        )

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
        training_args,
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
