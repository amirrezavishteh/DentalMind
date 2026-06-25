"""DentalMind CLI (Phase 1: `infer`).

Later phases add: infer-batch, train {phase1..4}, eval, serve, data {...}.
Those subcommands are registered as stubs so `--help` shows the full surface.
"""
from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from config_loader import load_config
from schema.finding import ModalityType
from schema.patient import PatientContext

app = typer.Typer(add_completion=False, help="DentalMind dental X-ray AI pipeline.")
data_app = typer.Typer(help="Data utilities (later phases).")
train_app = typer.Typer(help="Training phases (later phases).")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")

console = Console()


@app.command()
def infer(
    input: str = typer.Option(..., "--input", help="Image file or directory."),
    output: str = typer.Option("./results/", "--output", help="Output directory."),
    modality: str = typer.Option("auto", "--modality", help="auto|bw|opg|pa|fms|cbct"),
    use_vlm: bool = typer.Option(False, "--use-vlm", help="Enable C4 Tier-2 VLM (stub)."),
    patient_age: Optional[int] = typer.Option(None, "--patient-age"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML."),
    device: Optional[str] = typer.Option(None, "--device", help="cpu|cuda (overrides config)."),
):
    """Run inference on a single image or folder."""
    from pipeline.inference_engine import InferenceEngine

    cfg = load_config(config)
    modality_norm = _normalize_modality(modality)

    engine = InferenceEngine(cfg, checkpoint_dir=checkpoint, device=device)

    pc = PatientContext(age=patient_age) if patient_age is not None else None
    result = engine.infer(
        input_path=input,
        modality=modality_norm,
        patient_context=pc,
        use_vlm=use_vlm,
        output_dir=output,
    )

    if result.error:
        console.print(f"[red]Inference error:[/red] {result.error}")
        raise typer.Exit(code=1)

    console.print(f"[green]Done[/green] in {result.processing_time_ms:.0f} ms")
    console.print(f"  modality: {result.modality.value}")
    console.print(f"  teeth affected: {result.total_teeth_affected}")
    console.print(f"  highest severity: {result.highest_severity.value}")
    console.print(f"  outputs written to: {output}")


_MODALITY_ALIASES = {
    "auto": "auto",
    "bw": ModalityType.BW.value,
    "opg": ModalityType.OPG.value,
    "pa": ModalityType.PA.value,
    "fms": ModalityType.FMS.value,
    "cbct": ModalityType.CBCT.value,
}


def _normalize_modality(m: str) -> str:
    key = m.lower().strip()
    if key in _MODALITY_ALIASES:
        return _MODALITY_ALIASES[key]
    # allow full enum values too
    try:
        return ModalityType(key).value
    except ValueError:
        raise typer.BadParameter(f"Unknown modality '{m}'. Use auto|bw|opg|pa|fms|cbct.")


@app.command("infer-batch")
def infer_batch(
    input_dir: str = typer.Option(..., "--input-dir"),
    output_dir: str = typer.Option(..., "--output-dir"),
    workers: int = typer.Option(4, "--workers"),
):
    """Batch inference over a folder of cases. (Phase 1: sequential per-image.)"""
    from pathlib import Path
    from pipeline.inference_engine import InferenceEngine

    cfg = load_config(None)
    engine = InferenceEngine(cfg)
    in_dir = Path(input_dir)
    cases = [p for p in in_dir.iterdir() if p.is_file()]
    for case in cases:
        out = Path(output_dir) / case.stem
        console.print(f"[cyan]infer[/cyan] {case.name}")
        engine.infer(str(case), modality="auto", output_dir=str(out))
    console.print(f"[green]Batch done[/green]: {len(cases)} cases -> {output_dir}")


def _not_implemented(name: str):
    console.print(f"[yellow]'{name}' is not implemented in Phase 1.[/yellow]")
    raise typer.Exit(code=2)


@train_app.command("phase1")
def train_phase1(data_dir: str = typer.Option(...), epochs: int = 100,
                 fast_dev_run: bool = typer.Option(False, "--fast-dev-run")):
    _not_implemented("train phase1")


@train_app.command("phase3")
def train_phase3(config: str = typer.Option(...), resume: bool = False,
                 fast_dev_run: bool = typer.Option(False, "--fast-dev-run")):
    _not_implemented("train phase3")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8080, checkpoint: Optional[str] = None):
    _not_implemented("serve")


@app.command()
def eval(checkpoint: str = typer.Option(...), test_dir: str = typer.Option(...),
         modality: str = "opg"):
    _not_implemented("eval")


@data_app.command("download-dentex")
def download_dentex(output: str = typer.Option("./data/dentex/")):
    _not_implemented("data download-dentex")


@data_app.command("download-weights")
def download_weights(output: str = typer.Option("./weights/")):
    _not_implemented("data download-weights")


if __name__ == "__main__":
    app()
