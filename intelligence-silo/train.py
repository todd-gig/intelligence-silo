#!/usr/bin/env python3
"""Train the intelligence silo SLM matrix.

Usage:
    python train.py                          # full run, 50 epochs, all 6 models
    python train.py --epochs 100             # longer run
    python train.py --models classifier scorer  # target specific models
    python train.py --synthetic 5000         # more synthetic data
    python train.py --no-historical          # skip journal/outcome data
    python train.py --device cpu             # force CPU (default: auto-detect mps/cuda/cpu)
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="Train the intelligence silo SLM matrix")
    parser.add_argument("--status", action="store_true",
                        help="Show current training trigger status and exit")
    parser.add_argument("--force", action="store_true",
                        help="Force a retrain now regardless of threshold/cooldown")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--synthetic", type=int, default=2000, help="Synthetic samples per run (default: 2000)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to train (default: all 6)")
    parser.add_argument("--no-historical", action="store_true",
                        help="Skip loading journal/outcome historical data")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Compute device (default: auto)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--checkpoint-dir", default="data/checkpoints",
                        help="Checkpoint output directory")
    parser.add_argument("--journal-dir", default="data/decision_journal",
                        help="Decision journal directory")
    parser.add_argument("--outcomes-file", default="data/learning_loop.jsonl",
                        help="Learning loop outcomes JSONL file")
    args = parser.parse_args()

    # Ensure we're running from the intelligence-silo directory
    here = Path(__file__).parent
    sys.path.insert(0, str(here))

    from core.training.pipeline import TrainingPipeline
    from core.training.trainer import TrainingConfig
    from core.training.trigger import RetrainTrigger, TriggerConfig

    # ── Status mode ──────────────────────────────────────────────────────────
    if args.status:
        trigger = RetrainTrigger(
            config=TriggerConfig(
                checkpoint_dir=args.checkpoint_dir,
                state_file=str(Path(args.checkpoint_dir).parent / "training_state.json"),
            )
        )
        st = trigger.status()
        bar_width = 30
        filled = int(bar_width * st["progress_pct"] / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        print("\n── Intelligence Silo — Training Status ──────────────────────")
        print(f"  Decisions accumulated : {st['decisions_since_last_train']:>6}  /  {st['threshold']} threshold")
        print(f"  Progress              : [{bar}] {st['progress_pct']:.1f}%")
        print(f"  Until next trigger    : {st['decisions_until_trigger']} more decisions")
        if st["cooldown_remaining_seconds"] > 0:
            mins = st["cooldown_remaining_seconds"] // 60
            print(f"  Cooldown remaining    : {mins}m {st['cooldown_remaining_seconds'] % 60}s")
        print(f"  Training active       : {'YES — running now' if st['training_active'] else 'No'}")
        print(f"  Runs completed        : {st['runs_completed']}")
        print(f"  Total decisions ever  : {st['total_decisions_ever']}")
        if st["last_train_at"]:
            print(f"  Last run at           : {st['last_train_at']}")
            print(f"  Last promoted         : {', '.join(st['last_train_promoted']) or 'none'}")
            if st["last_train_accuracies"]:
                print("  Last accuracies       :")
                for m, acc in st["last_train_accuracies"].items():
                    print(f"    {m:<22} {acc:.1%}")
        if st["domain_distribution"]:
            print(f"  Domain distribution   : {st['domain_distribution']}")
        print("─────────────────────────────────────────────────────────────\n")
        sys.exit(0)

    # ── Force-trigger mode ───────────────────────────────────────────────────
    if args.force:
        trigger = RetrainTrigger(
            config=TriggerConfig(
                epochs=args.epochs,
                n_synthetic=args.synthetic,
                device=args.device,
                checkpoint_dir=args.checkpoint_dir,
                state_file=str(Path(args.checkpoint_dir).parent / "training_state.json"),
            ),
            journal_dir=args.journal_dir,
            outcomes_file=args.outcomes_file,
        )
        logger.info("Force-triggering retrain...")
        trigger.force_trigger()
        # Wait for the background thread to finish
        if trigger._training_thread:
            trigger._training_thread.join()
        print(trigger.status())
        sys.exit(0)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    pipeline = TrainingPipeline(
        checkpoint_dir=args.checkpoint_dir,
        journal_dir=args.journal_dir,
        outcomes_file=args.outcomes_file,
        config=config,
    )

    logger.info("Starting training: epochs=%d synthetic=%d models=%s device=%s",
                args.epochs, args.synthetic, args.models or "all", args.device)

    result = pipeline.run(
        epochs=args.epochs,
        n_synthetic=args.synthetic,
        models=args.models,
        save_report=True,
    )

    print("\n" + "="*60)
    print(result.summary())
    print("="*60)

    if result.failed:
        logger.warning("Some models failed to train: %s", result.failed)
        sys.exit(1)

    if not result.promoted:
        logger.warning("No models passed promotion gates — check data quality")
        sys.exit(2)

    logger.info("Done. %d/%d models promoted to production.", len(result.promoted), len(result.model_results))


if __name__ == "__main__":
    main()
