# CLAUDE.md

Navigation guide for the `amazon-sagemaker-generativeai` repo. The authoritative content overview is the top-level `README.md` (incl. the Model Support Matrix); this file captures what isn't obvious from the README and helps you orient quickly.

## What this repo is

A collection of standalone, mostly-Jupyter examples showing Generative AI workflows on Amazon SageMaker. It is **not** a single codebase — each folder under a numbered prefix is an independent example or family of examples with its own dependencies. Don't expect shared utilities, a unified test suite, or cross-folder imports.

## Navigation map

| Folder                             | What lives here                                                                                                                                                                                                                         | Notes                                                                                     |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `0_model_customization_recipes/`   | Config-driven SFT (`supervised_finetuning/`) and preference optimization (`preference_optimization/`) recipes for 20+ FMs (Llama, Qwen, GPT-OSS, DeepSeek, Phi, Gemma)                                                                  | Largest curated catalog. Strategies: QLoRA, Spectrum, Full FT                             |
| `1._getting_started/`              | **Placeholder** — README literally says "placeholder"                                                                                                                                                                                   | Skip unless asked to populate it                                                          |
| `2_end_to_end_genai_on_sagemaker/` | End-to-end MLOps. `2_model_customization/` and `3_inference/` are placeholders; substance is in `4_mlops/` (SM Pipelines + Unified Studio + DataZone, Abalone XGBoost demo)                                                             | Two of the three subfolders are empty                                                     |
| `3_distributed_training/`          | Distributed training deep-dives. Subfolders: `models/`, `reinforcement-learning/` (DPO + GRPO), `nvidia-nemo/`, `diffusers/`, `spectrum_finetuning/`, `unsloth/`, `time-series-forecasting/`, `distributed_training_sm_unified_studio/` | Largest folder. Most diverse stack — DDP, FSDP/FSDP2, DeepSpeed, Ray, vLLM, NeMo, Unsloth |
| `4_rag/`                           | One example: `voyageai-embedding-RAG/` (Voyage AI + OpenSearch KNN + Claude 3 via Bedrock)                                                                                                                                              | Single notebook                                                                           |
| `5_agents/`                        | Agent frameworks: `deepseek_crewai_based_agent/`, `langgraph_model_context_protocol/`, `ml-models-as-agent-tools/`, `sagemaker-strands-agentcore/`, `sagemaker-with-strands-agents/`                                                    | CrewAI, LangGraph + MCP, Strands, Bedrock AgentCore                                       |
| `6_use_cases/`                     | Task/industry recipes: RAG chatbot, text-to-SQL, phishing classification, function-calling SFT+DPO, summarization, summarization-to-image, job governance                                                                               | Smaller and older models (Flan-T5, Falcon, CodeLlama)                                     |
| `7_inference/`                     | `post_training_quantization/` (GPTQ/AWQ); `sagemaker-genai-hosting-examples/` is empty                                                                                                                                                  | Quantization + benchmarking only                                                          |
| `llm-performance-evaluation/`      | `deepseek-r1-distilled/` benchmarking with Ray                                                                                                                                                                                          | Single example                                                                            |
| `x_archive/`                       | 18 legacy Llama-2-era examples                                                                                                                                                                                                          | **Don't use as reference for new work** unless explicitly asked                           |

## Key things to know before changing anything

- **Many README "Notebook" links point to OTHER aws-samples repos** (`awsome-distributed-training`, `sagemaker-distributed-training-workshop`, `generative-ai-on-amazon-sagemaker`, `multi-modal-examples-for-amazon-sagemaker`, `sagemaker-studio-foundation-models`, `sample-ray-on-amazon-sagemaker-training-jobs`). If a user asks about a notebook listed in the Model Support Matrix, check whether it actually lives here before searching — it may be in a sibling repo.
- **Notebooks are the primary artifact.** When asked to inspect or modify, use the Read tool (it handles `.ipynb`). Don't re-run cells; the user runs them in SageMaker Studio.
- **Each example pins its own dependencies** in a local `requirements.txt`, `pyproject.toml`, or notebook cell. There is no root `requirements.txt`. Don't try to install repo-wide deps.
- **Folder casing matters**: `distributed_training_sm_unified_studio` (lowercase `sm`), not `SM_Unified_Studio`. The README sometimes uses different casing than the filesystem.
- **The `1._getting_started/` folder name has a literal `.` in it** — quote paths or use it exactly when running shell commands.
- **GPU-bound notebooks**: most training notebooks target instances like `ml.g5.*`, `ml.p4d.*`, `ml.p5.*`. Don't suggest running them locally; suggest SageMaker Training Jobs or HyperPod.
- **Security posture**: recent commits (`5832ce9`, `3e539ed`) hardened or commented out files for security issues. If you see commented-out blocks with a security note, leave them alone unless the user explicitly asks to revisit.

## Common tasks → where to look

- "How do I fine-tune model X?" → top-level `README.md` Model Support Matrix first; then `0_model_customization_recipes/` (recipe-style) or `3_distributed_training/models/` (deep-dive style).
- "Add a new model recipe" → mirror an existing folder under `0_model_customization_recipes/supervised_finetuning/` (config + notebook + scripts).
- "How does the MLOps pipeline work?" → `2_end_to_end_genai_on_sagemaker/4_mlops/`.
- "RAG / retrieval question" → `4_rag/voyageai-embedding-RAG/` (production-style) or `6_use_cases/rag_including_chatbot/` (basic).
- "Build an agent" → `5_agents/`. For Bedrock AgentCore: `sagemaker-strands-agentcore/` or `ml-models-as-agent-tools/2-agentcore/`.
- "Quantize a model" → `7_inference/post_training_quantization/`.

## Workflow conventions

- Don't introduce a top-level `requirements.txt`, shared `utils/`, or cross-folder refactors — each example is intentionally self-contained.
- Don't create new top-level numbered folders without checking with the user; the numbering is a learning-path convention.
- When editing a notebook, preserve cell outputs only if the user asks (otherwise clear them, since outputs balloon diffs and may contain account IDs / S3 URIs).
- Don't run training jobs to "verify" changes — they cost money and take hours. State the change and let the user run it.
