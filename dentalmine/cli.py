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
def train_phase1(
    data_dir: str = typer.Option("../data/2D", "--data-dir"),
    output: str = typer.Option("../weights/dentvfm_phase1.pt", "--output"),
    model: str = typer.Option("vit_large_patch14_dinov2", "--model"),
    img_size: int = typer.Option(518, "--img-size"),
    epochs: int = typer.Option(100, "--epochs"),
    batch_size: int = typer.Option(64, "--batch-size"),
    device: Optional[str] = typer.Option(None, "--device"),
    pretrained: bool = typer.Option(False, "--pretrained"),
    fast_dev_run: bool = typer.Option(False, "--fast-dev-run"),
):
    """Phase 1: self-supervised pretrain the DentVFM backbone on 2D images."""
    from training.phase1_pretrain import build_argparser, train
    ns = build_argparser().parse_args([])
    ns.data_dir, ns.output, ns.model, ns.img_size = data_dir, output, model, img_size
    ns.epochs, ns.batch_size, ns.device = epochs, batch_size, device
    ns.pretrained, ns.fast_dev_run = pretrained, fast_dev_run
    train(ns)


@train_app.command("phase2")
def train_phase2(
    cbct_dir: str = typer.Option("../data/3D/cbct", "--cbct-dir"),
    labels_dir: Optional[str] = typer.Option(None, "--labels-dir"),
    backbone_ckpt: Optional[str] = typer.Option(None, "--backbone-ckpt"),
    output: str = typer.Option("../weights/c1_adapter.pt", "--output"),
    epochs: int = typer.Option(50, "--epochs"),
    device: Optional[str] = typer.Option(None, "--device"),
    fast_dev_run: bool = typer.Option(False, "--fast-dev-run"),
):
    """Phase 2: train the C1 cross-slice adapter on CBCT (data-gated)."""
    from training.phase2_c1_adapter import build_argparser, train
    ns = build_argparser().parse_args([])
    ns.cbct_dir, ns.labels_dir, ns.backbone_ckpt = cbct_dir, labels_dir, backbone_ckpt
    ns.output, ns.epochs, ns.device, ns.fast_dev_run = output, epochs, device, fast_dev_run
    train(ns)


@train_app.command("phase3")
def train_phase3(
    data_root: str = typer.Option("../data/2D", "--data-root"),
    output: str = typer.Option("../weights/phase3_heads.pt", "--output"),
    backbone_ckpt: Optional[str] = typer.Option(None, "--backbone-ckpt"),
    model: str = typer.Option("vit_large_patch14_dinov2", "--model"),
    img_size: int = typer.Option(518, "--img-size"),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(64, "--batch-size"),
    device: Optional[str] = typer.Option(None, "--device"),
    fast_dev_run: bool = typer.Option(False, "--fast-dev-run"),
):
    """Phase 3: multitask heads on frozen DentVFM + Med-CLIP alignment loss."""
    from training.phase3_multitask import build_argparser, train
    ns = build_argparser().parse_args([])
    ns.data_root, ns.output, ns.backbone_ckpt = data_root, output, backbone_ckpt
    ns.model, ns.img_size, ns.epochs = model, img_size, epochs
    ns.batch_size, ns.device, ns.fast_dev_run = batch_size, device, fast_dev_run
    train(ns)


@train_app.command("phase4")
def train_phase4(
    cbct_dir: str = typer.Option("../data/3D/cbct", "--cbct-dir"),
    masks_dir: Optional[str] = typer.Option(None, "--masks-dir"),
    pseudo_output: str = typer.Option("../data/pseudo", "--pseudo-output"),
    fast_dev_run: bool = typer.Option(False, "--fast-dev-run"),
):
    """Phase 4: 3D->2D distillation to pseudo-labelled DRRs (data-gated)."""
    from training.phase4_distillation import build_argparser, train
    ns = build_argparser().parse_args([])
    ns.cbct_dir, ns.masks_dir, ns.pseudo_output = cbct_dir, masks_dir, pseudo_output
    ns.fast_dev_run = fast_dev_run
    train(ns)


@train_app.command("clip-backbone")
def train_clip_backbone(
    data_root: str = typer.Option("../data/2D", "--data-root"),
    output: str = typer.Option("../weights/medclip_dental_finetuned.pt", "--output"),
    epochs: int = typer.Option(10, "--epochs"),
    batch_size: int = typer.Option(64, "--batch-size"),
    lr: float = typer.Option(1e-5, "--lr"),
    workers: int = typer.Option(4, "--workers"),
    device: Optional[str] = typer.Option(None, "--device"),
    config: Optional[str] = typer.Option(None, "--config"),
):
    """Fine-tune the shared Med-CLIP image tower on 2D dental data (prototype loss)."""
    from argparse import Namespace
    from training.clip_finetune import train as run_clip_finetune

    run_clip_finetune(Namespace(
        data_root=data_root, output=output, epochs=epochs,
        batch_size=batch_size, lr=lr, workers=workers, device=device, config=config,
    ))


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
