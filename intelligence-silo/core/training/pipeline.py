"""Training pipeline — orchestrates data generation, training, evaluation, and promotion."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .data_generator import DecisionDataGenerator, TrainingConfig
from .trainer import SLMTrainer, TrainingResult
from .evaluator import ModelEvaluator, EvaluationReport
from .promotion import PromotionGate

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of a full training run across all 6 models."""
    n_synthetic: int
    n_historical: int
    model_results: dict[str, TrainingResult] = field(default_factory=dict)
    evaluation_reports: dict[str, EvaluationReport] = field(default_factory=dict)
    promoted: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Training complete: {len(self.model_results)} models"]
        lines.append(f"  Data: {self.n_synthetic} synthetic + {self.n_historical} historical")
        lines.append(f"  Promoted: {self.promoted or ['none']}")
        lines.append(f"  Failed: {self.failed or ['none']}")
        for name, r in self.model_results.items():
            acc = f"acc={r.val_accuracy:.3f}" if r.val_accuracy else f"val_loss={r.val_loss:.4f}"
            lines.append(f"  [{name}] epoch={r.best_epoch} {acc} promoted={r.promoted}")
        return "\n".join(lines)


class TrainingPipeline:
    """End-to-end pipeline: generate → train → evaluate → promote.

    Usage:
        pipeline = TrainingPipeline()
        result = pipeline.run(epochs=50)
        print(result.summary())

    Or target specific models:
        result = pipeline.run(models=["classifier", "trust_assessor"])
    """

    ALL_MODELS = ["classifier", "scorer", "trust_assessor",
                  "memory_encoder", "pattern_detector", "causal_predictor"]

    def __init__(
        self,
        checkpoint_dir: str = "data/checkpoints",
        journal_dir: str = "data/decision_journal",
        outcomes_file: str = "data/learning_loop.jsonl",
        config: TrainingConfig | None = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.config = config or TrainingConfig(checkpoint_dir=checkpoint_dir)
        self.generator = DecisionDataGenerator(
            journal_dir=journal_dir,
            outcomes_file=outcomes_file,
        )
        self.trainer = SLMTrainer(self.config)
        self.evaluator = ModelEvaluator()
        self.gate = PromotionGate()

    def run(
        self,
        epochs: int = 50,
        n_synthetic: int = 2000,
        models: list[str] | None = None,
        save_report: bool = True,
    ) -> PipelineResult:
        """Run the full training pipeline."""
        target_models = models or self.ALL_MODELS
        self.config.epochs = epochs

        # 1. Generate data
        logger.info("Generating training data (%d synthetic)...", n_synthetic)
        all_records = self.generator.generate(n_synthetic=n_synthetic)
        n_hist = sum(
            len([r for r in recs if r.source != "synthetic"])
            for recs in all_records.values()
        ) // max(1, len(all_records))

        result = PipelineResult(n_synthetic=n_synthetic, n_historical=n_hist)

        # 2. Train each model
        from ..models.matrix import SLMMatrix
        matrix = SLMMatrix()

        for model_name in target_models:
            records = all_records.get(model_name, [])
            if not records:
                logger.warning("No records for %s — skipping", model_name)
                result.failed.append(model_name)
                continue

            slm = matrix.models.get(model_name)
            if slm is None:
                logger.warning("Model %s not found in SLMMatrix", model_name)
                result.failed.append(model_name)
                continue

            logger.info("Training %s on %d records...", model_name, len(records))
            try:
                train_result = self.trainer.train(model_name, slm, records)
                result.model_results[model_name] = train_result

                # 3. Evaluate
                eval_report = self.evaluator.evaluate(model_name, slm, records)
                result.evaluation_reports[model_name] = eval_report

                # 4. Promotion gate
                passed = self.gate.check(model_name, train_result, eval_report)
                train_result.promoted = passed
                if passed:
                    result.promoted.append(model_name)
                    self._save_production_weights(model_name, train_result.checkpoint_path)
                else:
                    logger.warning(
                        "[%s] Did not pass promotion gate: val_loss=%.4f acc=%s",
                        model_name, train_result.val_loss, train_result.val_accuracy,
                    )

            except Exception as exc:
                logger.exception("Training failed for %s: %s", model_name, exc)
                result.failed.append(model_name)

        # 5. Save report
        if save_report:
            self._save_report(result)

        logger.info(result.summary())
        return result

    def _save_production_weights(self, model_name: str, checkpoint_path: str) -> None:
        """Copy best checkpoint to the production weights location."""
        import shutil
        src = Path(checkpoint_path)
        if not src.exists():
            return
        dest_dir = Path("data/weights/production")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{model_name}.pt"
        shutil.copy2(src, dest)
        logger.info("[%s] Production weights saved → %s", model_name, dest)

    def _save_report(self, result: PipelineResult) -> None:
        report_dir = Path(self.checkpoint_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "training_report.json"
        report = {
            "n_synthetic": result.n_synthetic,
            "n_historical": result.n_historical,
            "promoted": result.promoted,
            "failed": result.failed,
            "models": {
                name: {
                    "epochs_run": r.epochs_run,
                    "train_loss": r.train_loss,
                    "val_loss": r.val_loss,
                    "val_accuracy": r.val_accuracy,
                    "best_epoch": r.best_epoch,
                    "duration_seconds": r.duration_seconds,
                    "promoted": r.promoted,
                    "checkpoint_path": r.checkpoint_path,
                }
                for name, r in result.model_results.items()
            },
        }
        report_path.write_text(json.dumps(report, indent=2))
        logger.info("Training report saved → %s", report_path)
