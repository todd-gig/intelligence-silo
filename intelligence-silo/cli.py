"""Intelligence Silo CLI — entrypoint for running, testing, and packaging nodes."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

app = typer.Typer(name="intel-silo", help="Intelligence Silo — Society of Minds neural architecture")
console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def run(
    config: str = typer.Option("config/silo.yaml", help="Path to silo.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Start the intelligence node."""
    setup_logging(verbose)

    from core.node import IntelligenceNode
    node = IntelligenceNode(config_path=config)

    console.print(Panel.fit(
        f"[bold green]Intelligence Node: {node.node_name}[/]\n"
        f"ID: {node.node_id}\n"
        f"Device: {node.device}\n"
        f"Models: {node.matrix.total_params_human} total params\n"
        f"Memory: {node.memory.config.embedding_dim}d embeddings",
        title="[bold]Intelligence Silo[/]",
    ))

    asyncio.run(node.start())


@app.command()
def health(
    config: str = typer.Option("config/silo.yaml", help="Path to silo.yaml"),
):
    """Show node health report."""
    from core.node import IntelligenceNode
    node = IntelligenceNode(config_path=config)
    report = node.health()

    # Node info
    table = Table(title="Node Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for k, v in report["node"].items():
        table.add_row(str(k), str(v))
    console.print(table)

    # Models
    table = Table(title="SLM Matrix")
    table.add_column("Model", style="cyan")
    table.add_column("Params", style="yellow")
    table.add_column("Role", style="white")
    table.add_column("Calls", style="green")
    for name, info in report["matrix"]["models"].items():
        table.add_row(name, info["params"], info["role"][:50], str(info["calls"]))
    console.print(table)
    console.print(f"[bold]Total params:[/] {report['matrix']['total_params']}")

    # Memory
    mem = report.get("society", {}).get("memory", report.get("memory", {}))
    console.print(Panel(json.dumps(mem, indent=2, default=str), title="Memory Health"))


@app.command()
def test(
    config: str = typer.Option("config/silo.yaml", help="Path to silo.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a test think cycle with sample input."""
    setup_logging(verbose)

    from core.node import IntelligenceNode
    node = IntelligenceNode(config_path=config)

    # Create sample input
    sample_input = {
        "type": "decision",
        "title": "Evaluate new partnership opportunity",
        "domain": "strategic",
        "context": {
            "partner": "Example Corp",
            "revenue_potential": 50000,
            "strategic_alignment": 0.8,
            "trust_signals": ["previous_collaboration", "industry_leader"],
        },
    }

    # Create dummy token IDs for the SLM matrix
    input_ids = torch.randint(0, 8192, (1, 64))
    query_embedding = torch.randn(1, 384)

    console.print("[bold]Running test think cycle...[/]\n")

    result = node.process(sample_input, input_ids, query_embedding)

    # Display results
    table = Table(title="Think Cycle Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Consensus", str(result["consensus"]))
    table.add_row("Confidence", f"{result['confidence']:.2f}")
    table.add_row("Minds Activated", str(result["minds_activated"]))
    table.add_row("Cycle Time", f"{result['cycle_time_ms']:.1f}ms")
    console.print(table)

    if result["verdict"]:
        console.print(Panel(
            json.dumps(result["verdict"], indent=2, default=str),
            title="Verdict",
        ))

    # Test deliberation
    console.print("\n[bold]Running deliberation (multi-cycle)...[/]\n")
    delib_result = node.deliberate(sample_input, input_ids, query_embedding)

    table = Table(title="Deliberation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Consensus", str(delib_result["consensus"]))
    table.add_row("Confidence", f"{delib_result['confidence']:.2f}")
    table.add_row("Total Time", f"{delib_result['cycle_time_ms']:.1f}ms")
    console.print(table)

    # Memory health after test
    console.print(Panel(
        json.dumps(node.memory.health(), indent=2, default=str),
        title="Memory State After Test",
    ))


@app.command()
def package(
    name: str = typer.Option("intel-node", help="Executable name"),
    weights_dir: str = typer.Option(None, help="Path to model weights directory"),
    signing_key: str = typer.Option(None, help="HMAC signing key"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Package the node as a distributable executable."""
    setup_logging(verbose)

    from packaging.builder import NodeBuilder
    builder = NodeBuilder()

    output = builder.build(
        name=name,
        include_models=weights_dir is not None,
        signing_key=signing_key,
        model_weights_dir=Path(weights_dir) if weights_dir else None,
    )

    console.print(f"[bold green]Package built:[/] {output}")


@app.command()
def matrix_info(
    config: str = typer.Option("config/silo.yaml", help="Path to silo.yaml"),
):
    """Show detailed information about the SLM matrix."""
    from core.node import IntelligenceNode
    node = IntelligenceNode(config_path=config)

    table = Table(title="Small Language Model Matrix")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Type", style="yellow")
    table.add_column("Hidden", style="white", justify="right")
    table.add_column("Layers", style="white", justify="right")
    table.add_column("Heads", style="white", justify="right")
    table.add_column("Params", style="green", justify="right")
    table.add_column("Role", style="dim")

    for name, cfg in node.matrix.configs.items():
        model = node.matrix.models[name]
        table.add_row(
            name, cfg.model_type.value,
            str(cfg.hidden_dim), str(cfg.num_layers), str(cfg.num_heads),
            model.param_count_human(), cfg.role[:40],
        )

    console.print(table)
    console.print(f"\n[bold]Total parameters:[/] {node.matrix.total_params_human}")
    console.print(f"[bold]Device:[/] {node.device}")


if __name__ == "__main__":
    app()
