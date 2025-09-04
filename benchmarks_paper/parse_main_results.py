"""Parse benchmark results from CSV and generate data for plotting."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# Constants
EIGHT_B_METHODS_COUNT = 2
THIRTY_TWO_B_SEVENTY_B_METHODS_COUNT = 4


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data and clean it."""
    df = pd.read_csv(csv_path)

    # Remove timestamp column
    if "timestamp" in df.columns:
        df = df.drop("timestamp", axis=1)

    # Remove rows with invalid throughput (OOM or non-numeric)
    df = df[df["throughput_per_gpu"] != "OOM"]
    df["throughput_per_gpu"] = pd.to_numeric(df["throughput_per_gpu"], errors="coerce")
    df = df.dropna(subset=["throughput_per_gpu"])

    # Convert throughput_per_gpu to overall throughput by multiplying by number of GPUs
    df["throughput_per_gpu"] = df["throughput_per_gpu"] * df["nproc_per_node"]

    # Add throughput column by multiplying by number of nodes
    df["throughput"] = df["throughput_per_gpu"] * df["nnodes"]

    logger.info(f"Loaded {len(df)} valid records from {csv_path}")
    return df


def get_model_key(model_name: str) -> str:
    """Get standardized model key."""
    if "8B" in model_name:
        return "8B"
    if "32B" in model_name:
        return "32B"
    if "70B" in model_name:
        return "70B"
    raise ValueError(f"Unknown model: {model_name}")


def get_dataset_key(dataset_name: str) -> str:
    """Get standardized dataset key."""
    dataset_mapping = {
        "xsum": "XSUM",
        "cnn_dailymail": "CNNDM",
        "wikisum": "WikiSum",
        "mix": "Mixed",
        "heterogeneous": "Het",
    }
    return dataset_mapping.get(dataset_name, dataset_name)


def get_method_key_8b(row: pd.Series) -> str:
    """Get method key for 8B model."""
    if not row["apply_fused_lora"]:
        return "baseline"
    return "lorafusion"


def get_method_key_32b_70b(row: pd.Series) -> str:
    """Get method key for 32B/70B models."""
    if row["mlora"]:
        return "mlora"
    if not row["apply_fused_lora"] and row["use_fsdp"]:
        return "fsdp"
    if (
        not row["apply_fused_lora"]
        and not row["use_fsdp"]
        and row["pp_size"] > 1
    ):
        return "pp"
    if row["apply_fused_lora"]:
        return "lorafusion"
    return "unknown"


def process_8b_data(df: pd.DataFrame) -> dict[str, list[float | None]]:
    """Process 8B model data."""
    model_df = df[df["model"] == "meta-llama/Llama-3.1-8B-Instruct"].copy()

    # Group by dataset and method, then average throughput
    grouped = (
        model_df.groupby(["dataset_name", "apply_fused_lora", "use_multi_lora"])[
            "throughput"
        ]
        .mean()
        .reset_index()
    )

    results = {}
    datasets = ["xsum", "cnn_dailymail", "wikisum", "mix", "heterogeneous"]

    for dataset in datasets:
        dataset_data = grouped[grouped["dataset_name"] == dataset]

        # Get baseline (apply_fused_lora=False)
        baseline_data = dataset_data[dataset_data["apply_fused_lora"] == False] # noqa: E712
        baseline_throughput = (
            baseline_data["throughput"].iloc[0]
            if len(baseline_data) > 0
            else None
        )

        # Get LoRAFusion (max of apply_fused_lora=True, regardless of use_multi_lora)
        lorafusion_data = dataset_data[dataset_data["apply_fused_lora"] == True] # noqa: E712
        lorafusion_throughput = (
            lorafusion_data["throughput"].max()
            if len(lorafusion_data) > 0
            else None
        )

        dataset_key = get_dataset_key(dataset)
        results[dataset_key] = [baseline_throughput, lorafusion_throughput]

    return results


def process_32b_70b_data(
    df: pd.DataFrame, model_name: str
) -> dict[str, list[float | None]]:
    """Process 32B/70B model data."""
    model_df = df[df["model"] == model_name].copy()

    # Add method key
    model_df["method_key"] = model_df.apply(get_method_key_32b_70b, axis=1)

    # Group by dataset and method, then average throughput
    grouped = (
        model_df.groupby(["dataset_name", "method_key"])["throughput"]
        .mean()
        .reset_index()
    )

    results = {}
    datasets = ["xsum", "cnn_dailymail", "wikisum", "mix", "heterogeneous"]
    methods = ["fsdp", "pp", "mlora", "lorafusion"]

    for dataset in datasets:
        dataset_data = grouped[grouped["dataset_name"] == dataset]
        dataset_results = []

        for method in methods:
            method_data = dataset_data[dataset_data["method_key"] == method]
            throughput = (
                method_data["throughput"].iloc[0]
                if len(method_data) > 0
                else None
            )
            dataset_results.append(throughput)

        dataset_key = get_dataset_key(dataset)
        results[dataset_key] = dataset_results

    return results


def compute_heterogeneous_averages(
    results: dict[str, list[float | None]], model_type: str
) -> list[float | None]:
    """Compute heterogeneous averages from other datasets."""
    if model_type == "8B":
        # For 8B, average baseline and lorafusion across xsum, cnndm, wikisum, mix
        datasets = ["XSUM", "CNNDM", "WikiSum", "Mixed"]
        baseline_values = []
        lorafusion_values = []

        for dataset in datasets:
            if dataset in results and len(results[dataset]) >= EIGHT_B_METHODS_COUNT:
                if results[dataset][0] is not None:
                    baseline_values.append(results[dataset][0])
                if results[dataset][1] is not None:
                    lorafusion_values.append(results[dataset][1])

        baseline_avg = (
            sum(baseline_values) / len(baseline_values) if baseline_values else None
        )
        lorafusion_avg = (
            sum(lorafusion_values) / len(lorafusion_values)
            if lorafusion_values
            else None
        )

        return [baseline_avg, lorafusion_avg]

    # 32B/70B
    # For 32B/70B, average fsdp and pp across xsum, cnndm, wikisum, mix
    datasets = ["XSUM", "CNNDM", "WikiSum", "Mixed"]
    # fsdp, pp, mlora, lorafusion
    method_averages = [[] for _ in range(THIRTY_TWO_B_SEVENTY_B_METHODS_COUNT)]

    for dataset in datasets:
        if (
            dataset in results
            and len(results[dataset]) >= THIRTY_TWO_B_SEVENTY_B_METHODS_COUNT
        ):
            for i in range(THIRTY_TWO_B_SEVENTY_B_METHODS_COUNT):
                if results[dataset][i] is not None:
                    method_averages[i].append(results[dataset][i])

    # Compute averages for each method
    final_averages = []
    for method_values in method_averages:
        if method_values:
            final_averages.append(sum(method_values) / len(method_values))
        else:
            final_averages.append(None)

    return final_averages


def convert_to_plot_format(
    results: dict[str, Any],
) -> dict[str, list[list[float | None]]]:
    """Convert results to the format expected by the plotting script."""
    # Order: 8B, 32B, 70B
    model_order = ["8B", "32B", "70B"]
    dataset_order = ["XSUM", "CNNDM", "WikiSum", "Mixed", "Het"]

    plot_data = {}

    for model in model_order:
        if model in results:
            model_data = []
            for dataset in dataset_order:
                if dataset in results[model]:
                    model_data.append(results[model][dataset])
                # Fill with None if dataset not found
                elif model == "8B":
                    model_data.append([None] * EIGHT_B_METHODS_COUNT)
                else:
                    model_data.append([None] * THIRTY_TWO_B_SEVENTY_B_METHODS_COUNT)
            plot_data[model] = model_data

    return plot_data


def main() -> None:
    """Main function to parse results and generate JSON output."""
    csv_path = "logs/benchmark/results.csv"

    if not Path(csv_path).exists():
        logger.error(f"CSV file not found: {csv_path}")
        return None

    # Load and clean data
    df = load_and_clean_data(csv_path)

    # Process each model
    results = {}

    # Process 8B model
    logger.info("Processing 8B model data...")
    results["8B"] = process_8b_data(df)

    # Process 32B model
    logger.info("Processing 32B model data...")
    results["32B"] = process_32b_70b_data(df, "Qwen/Qwen2.5-32B-Instruct")

    # Process 70B model
    logger.info("Processing 70B model data...")
    results["70B"] = process_32b_70b_data(df, "meta-llama/Llama-3.1-70B-Instruct")

    # Compute heterogeneous averages
    logger.info("Computing heterogeneous averages...")
    for model_type in ["8B", "32B", "70B"]:
        if model_type in results:
            results[model_type]["Het"] = compute_heterogeneous_averages(
                results[model_type], model_type
            )

    # Convert to plot format
    plot_data = convert_to_plot_format(results)

    # Save results
    output_path = "parsed_results.json"
    with Path(output_path).open("w") as f:
        json.dump(plot_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    logger.info("\n=== PARSED RESULTS SUMMARY ===")
    for model, data in plot_data.items():
        logger.info(f"\n{model} Model:")
        for i, dataset in enumerate(["XSUM", "CNNDM", "WikiSum", "Mixed", "Het"]):
            logger.info(f"  {dataset}: {data[i]}")

    return plot_data


if __name__ == "__main__":
    main()
