"""Prepare models for the benchmarks."""
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_name_list = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
]

for model_name in model_name_list:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        logger.success(f"Prepared {model_name}")
    except Exception as e:
        logger.error(f"Failed to prepare {model_name}: {e}")
        logger.error(
            "Check whether you have logged in with your tokens and "
            f"whether you have right access to the model {model_name}."
        )
