#!/usr/bin/env python3

"""
Purpose: End-to-end orchestrator to download a Hugging Face model, analyze embeddings and transformer matrices, and generate a summary report.

Inputs:
- model_id (str): Hugging Face repository id or URL (e.g., "Qwen/Qwen3-4B").
- revision (str | None): Optional git revision/tag/commit for the model snapshot.
- output_dir (str): Output directory for generated CSV, plots, and the markdown report.
- use_cpu (bool): Force CPU mode for analysis scripts if True.

Outputs:
- Downloads model under data/models/<sanitized_model_id>/
- Runs scripts/transformer_analysis.py and scripts/embedding_analysis.py
- Writes CSV: <output_dir>/qwen_matrix_analysis.csv (or model prefixed)
- Writes plots/JSON from embedding analysis into <output_dir>/
- Writes Markdown report: <output_dir>/report.md summarizing model info and analysis outputs
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple


def sanitize_model_id(model_id: str) -> str:
    """
    Purpose: Make a filesystem-friendly subdirectory name from a Hugging Face model id.

    Inputs:
    - model_id (str): Original model id, potentially with slashes.

    Outputs:
    - str: Sanitized id usable as a directory name.
    """
    return model_id.strip().replace("/", "-").replace(" ", "_")


def download_model(model_id: str, revision: str | None, target_root: Path) -> Path:
    """
    Purpose: Download a model snapshot from Hugging Face into a local directory if not present.

    Inputs:
    - model_id (str): Hugging Face repo id or URL.
    - revision (str | None): Optional revision/tag/commit.
    - target_root (Path): Root directory where models are stored.

    Outputs:
    - Path: Local model directory path.
    """
    target_root.mkdir(parents=True, exist_ok=True)
    local_dir = target_root / sanitize_model_id(model_id)

    if local_dir.exists() and any(local_dir.glob("*.safetensors")):
        print(f"Model already present at: {local_dir}")
        return local_dir

    print(f"Downloading model '{model_id}' to {local_dir} ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            max_workers=8,
        )
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

    return local_dir


def read_model_config(model_dir: Path) -> Dict:
    """
    Purpose: Read Hugging Face config.json if present to extract basic model info.

    Inputs:
    - model_dir (Path): Local model directory.

    Outputs:
    - Dict: Parsed JSON config or empty dict if not found.
    """
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            return {}
    return {}


def count_parameters_safetensors(model_dir: Path) -> int:
    """
    Purpose: Estimate total parameter count by summing numel across tensors in all .safetensors shards.

    Inputs:
    - model_dir (Path): Local model directory with .safetensors files.

    Outputs:
    - int: Total number of parameters (elements).
    """
    try:
        from safetensors import safe_open
    except Exception:
        return 0

    total_params = 0
    shard_files = [f for f in model_dir.glob("*.safetensors") if not f.name.startswith("._")]
    for shard in shard_files:
        try:
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    try:
                        t = f.get_tensor(key)
                        total_params += int(t.numel())
                    except Exception:
                        # Skip tensors that fail to load
                        continue
        except Exception:
            continue
    return total_params


def run_transformer_analysis(model_dir: Path, output_dir: Path, model_id: str, use_cpu: bool) -> Path:
    """
    Purpose: Invoke transformer matrix analysis script and return CSV path.

    Inputs:
    - model_dir (Path): Local model directory path.
    - output_dir (Path): Output directory for the CSV file.
    - model_id (str): Model id for metadata in CSV.
    - use_cpu (bool): If True, pass --no_gpu.

    Outputs:
    - Path: Path to generated CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "transformer_matrix_analysis.csv"
    if csv_path.exists():
        print(f"Reusing existing transformer CSV: {csv_path}")
        return csv_path
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "scripts" / "transformer_analysis.py"),
        "--model_path", str(model_dir),
        "--out", str(csv_path),
        "--model_id", model_id,
    ]
    if use_cpu:
        cmd.append("--no_gpu")

    print("Running transformer analysis ...")
    subprocess.run(cmd, check=True)
    return csv_path


def run_embedding_analysis(model_dir: Path, output_dir: Path, use_cpu: bool) -> Tuple[Path, Path, Path]:
    """
    Purpose: Invoke embedding analysis script and return key output artifact paths.

    Inputs:
    - model_dir (Path): Local model directory path.
    - output_dir (Path): Output directory for analysis artifacts.
    - use_cpu (bool): If True, force CPU mode.

    Outputs:
    - (anova_plot, clustering_json, representatives_json): Paths to generated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    anova_plot = output_dir / "anova_curve.png"
    clustering_json = output_dir / "clustering_results.json"
    representatives_json = output_dir / "cluster_representatives.json"
    if clustering_json.exists() and anova_plot.exists():
        print(f"Reusing existing embedding outputs in: {output_dir}")
        return anova_plot, clustering_json, representatives_json

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "scripts" / "embedding_analysis.py"),
        "--model_path", str(model_dir),
        "--output_dir", str(output_dir),
        "--max_k", "100",
        "--skip_representatives",
    ]
    if use_cpu:
        cmd.append("--use_cpu")

    print("Running embedding analysis ...")
    subprocess.run(cmd, check=True)

    return anova_plot, clustering_json, representatives_json


def human_readable_count(n: int) -> str:
    """
    Purpose: Format large integer counts into human-readable strings.

    Inputs:
    - n (int): Number to format.

    Outputs:
    - str: Formatted number with units.
    """
    if n <= 0:
        return "0"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.0f}P"


def write_report(report_path: Path, model_id: str, model_dir: Path, config: Dict, param_count: int,
                 transformer_csv: Path, anova_plot: Path, clustering_json: Path) -> None:
    """
    Purpose: Generate a concise Markdown report summarizing model info and analysis outputs.

    Inputs:
    - report_path (Path): Destination path for the markdown report.
    - model_id (str): Hugging Face model id.
    - model_dir (Path): Local model directory.
    - config (Dict): Parsed config.json dictionary.
    - param_count (int): Estimated total parameter count.
    - transformer_csv (Path): Path to transformer analysis CSV.
    - anova_plot (Path): Path to ANOVA plot image.
    - clustering_json (Path): Path to clustering summary JSON.

    Outputs:
    - None: Writes a markdown file to report_path.
    """
    vocab_size = config.get("vocab_size") or config.get("vocabulary_size")
    num_layers = config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers")
    hidden_size = config.get("hidden_size") or config.get("n_embd") or config.get("dim")
    num_attention_heads = config.get("num_attention_heads") or config.get("n_head")
    rope = config.get("rope_theta") or config.get("rope_scaling")

    lines = []
    lines.append(f"# Model Analysis Report\n")
    lines.append(f"- Model ID: `{model_id}`\n")
    lines.append(f"- Local Path: `{model_dir}`\n")
    if param_count:
        lines.append(f"- Parameter Count: {human_readable_count(param_count)} ({param_count:,})\n")
    if vocab_size is not None:
        lines.append(f"- Vocab Size: {vocab_size}\n")
    if num_layers is not None:
        lines.append(f"- Layers: {num_layers}\n")
    if hidden_size is not None:
        lines.append(f"- Hidden Size: {hidden_size}\n")
    if num_attention_heads is not None:
        lines.append(f"- Attention Heads: {num_attention_heads}\n")
    if rope is not None:
        lines.append(f"- RoPE: {rope}\n")

    # Read compact clustering summary for human-readable content
    clusters_table = None
    silhouette = None
    if clustering_json.exists():
        try:
            summary = json.loads(clustering_json.read_text())
            silhouette = summary.get("silhouette_score")
            clusters = summary.get("clusters", [])
            # Build a small table
            header = ["cluster_id", "size", "percentage", "avg_distance", "silhouette_score", "meaningful_text"]
            rows = []
            for c in clusters:
                rows.append([
                    c.get("cluster_id"),
                    c.get("size"),
                    f"{c.get('percentage', 0.0):.2f}",
                    f"{c.get('avg_distance', 0.0):.4f}" if c.get('avg_distance') is not None else "",
                    f"{c.get('silhouette_score', 0.0):.4f}" if c.get('silhouette_score') is not None else "",
                    (c.get("meaningful_text") or "")[:80],
                ])
            clusters_table = (header, rows)
        except Exception:
            pass

    lines.append("\n## Results\n")
    lines.append(f"- Transformer CSV: `{transformer_csv}`\n")
    if silhouette is not None:
        lines.append(f"- Embedding silhouette (non-singleton clusters): {silhouette:.4f}\n")
    if anova_plot.exists():
        lines.append(f"- Embedding ANOVA Plot: `{anova_plot}`\n")
    if clusters_table:
        lines.append("\n### Embedding Clusters (summary)\n")
        header, rows = clusters_table
        # Markdown table
        lines.append("| " + " | ".join(header) + " |\n")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |\n")
        for r in rows[:50]:  # cap rows for readability
            lines.append("| " + " | ".join(str(x) for x in r) + " |\n")
        if len(rows) > 50:
            lines.append(f"\n... ({len(rows)-50} more clusters truncated in report)\n")

    lines.append("\n## Notes\n")
    lines.append("- CSV is sorted by numeric layer_index; non-layer rows come last.\n")
    lines.append("- Ranks reported are full numerical ranks (no truncation).\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF model, run analyses, and generate a report")
    parser.add_argument("--model_id", required=True, help="Hugging Face model id or URL, e.g. Qwen/Qwen3-4B")
    parser.add_argument("--revision", default=None, help="Optional model revision/tag/commit")
    parser.add_argument("--output_dir", default="data/outputs", help="Directory to store analysis outputs (default per-model subdir)")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU mode for analysis")
    args = parser.parse_args()

    start = time.time()
    # If this script is moved to repo root, project_root becomes parent of this file
    project_root = Path(__file__).resolve().parent
    models_root = project_root / "data" / "models"
    # Default to model-specific subdir to keep outputs organized
    default_outputs = (args.output_dir.strip() == "data/outputs")
    output_root = project_root / ("data/outputs/" + sanitize_model_id(args.model_id) if default_outputs else args.output_dir)

    # 1) Download (or reuse) model
    model_dir = download_model(args.model_id, args.revision, models_root)

    # 2) Basic model info
    config = read_model_config(model_dir)
    param_count = count_parameters_safetensors(model_dir)

    # 3) Run analyses
    transformer_csv = run_transformer_analysis(model_dir, output_root, args.model_id, use_cpu=args.use_cpu)
    anova_plot, clustering_json, representatives_json = run_embedding_analysis(model_dir, output_root, use_cpu=args.use_cpu)

    # 4) Write report
    report_path = output_root / "report.md"
    write_report(report_path, args.model_id, model_dir, config, param_count, transformer_csv, anova_plot, clustering_json)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()


