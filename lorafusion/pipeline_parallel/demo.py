"""Entrypoint for demonstrating the pipeline parallel training."""

import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from lorafusion.pipeline_parallel.pipe_module import PipeModel
from lorafusion.utils.common import maybe_setup_distributed
from lorafusion.utils.hf import create_packed_dummy_inputs

# Setup distributed environment
maybe_setup_distributed()


def main() -> None:
    """Run the pipeline parallel demo."""
    # Initialize models for each pipeline stage
    pipe_models = []
    num_stages = 4

    logger.info(f"Creating {num_stages} pipeline stages")

    for i in range(num_stages):
        # Load model for current stage
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        # Create the pipeline model for this stage
        pipe_model = PipeModel.from_model(
            model,
            model.config,
            i,
            num_stages,
        )
        pipe_model.to("cuda")
        pipe_models.append(pipe_model)

    logger.info("Creating dummy inputs")

    # Create dummy inputs
    all_inputs = create_packed_dummy_inputs(
        hidden_size=model.config.hidden_size,
        seq_len_list=[1024, 1024],
        return_labels=True,
    )

    # Move inputs to GPU
    for input_name, input_value in all_inputs.items():
        if isinstance(input_value, torch.Tensor):
            all_inputs[input_name] = input_value.to("cuda")

    # Extract inputs
    input_ids = all_inputs["input_ids"]
    inputs_embeds = None

    # Keys that are shared across all pipeline stages
    shared_keys = [
        "position_ids",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
    ]
    shared_inputs = {k: v for k, v in all_inputs.items() if k in shared_keys}

    logger.info("Running forward pass through pipeline stages")

    # Run forward pass through all pipeline stages
    for i, pipe_model in enumerate(pipe_models):
        logger.info(f"Processing stage {i + 1}/{num_stages}")

        if pipe_model.is_first_stage:
            # First stage gets input_ids
            inputs_embeds = pipe_model(input_ids=input_ids, **shared_inputs)[0]
        elif not pipe_model.is_last_stage:
            # Middle stages get inputs_embeds from previous stage
            inputs_embeds = pipe_model(inputs_embeds=inputs_embeds, **shared_inputs)[0]
        else:
            # Last stage computes loss
            loss = pipe_model(
                inputs_embeds=inputs_embeds,
                labels=all_inputs["labels"],
                **shared_inputs,
            )
            logger.info(f"Final loss: {loss[0].item()}")


if __name__ == "__main__":
    main()
