#!/usr/bin/env python3
"""
Main training entrypoint for Tinker API integration with Lean4 FIM + RLVR.

This script initializes all components and runs the CISPO training loop
for training large MoE models on Lean4 proof infilling tasks.

Usage:
    # Basic usage with config file
    python train_tinker_fim.py --config configs/tinker_training.yaml

    # Resume from checkpoint
    python train_tinker_fim.py --config configs/tinker_training.yaml --resume

    # Override specific settings
    python train_tinker_fim.py --config configs/tinker_training.yaml --max-steps 500

Environment Variables:
    TINKER_API_KEY: Required. Your Tinker API key (can be set in .env file).
    FIM_MODEL_NAME: Override model name from config.
    FIM_MAX_STEPS: Override max training steps.
    FIM_CHECKPOINT_DIR: Override checkpoint directory.
    FIM_LOG_DIR: Override log directory.
    FIM_DATASET_PATH: Override dataset path.
    WANDB_PROJECT: Enable W&B logging with this project name.

    Note: Environment variables can be set in a .env file in the project root.
    The script will automatically load it if python-dotenv is installed.

Requirements:
    - All requirements from requirements.md
    - Tinker API key (set via TINKER_API_KEY environment variable)
    - Dataset in Parquet format with theorem data
    - Lean4 verification environment (verification_env/)
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

# Load .env file before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads from .env in current directory
except ImportError:
    pass  # python-dotenv not installed, rely on shell environment

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train large MoE models on Lean4 FIM tasks using Tinker API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    
    # Resume from checkpoint
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from specific checkpoint name (e.g., checkpoint_100)",
    )
    
    # Override settings
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override log directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Override group size for CISPO",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    
    # Verification settings
    parser.add_argument(
        "--verification-env",
        type=str,
        default="verification_env",
        help="Path to Lean4 verification environment",
    )
    parser.add_argument(
        "--no-sorries",
        action="store_true",
        help="Fail verification when file contains sorry",
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    
    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without starting training",
    )
    
    return parser.parse_args()


def apply_cli_overrides(config, args: argparse.Namespace):
    """Apply command line overrides to configuration."""
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        config.log_dir = args.log_dir
    if args.dataset is not None:
        config.dataset_path = args.dataset
    if args.model is not None:
        config.model_name = args.model
    if args.group_size is not None:
        config.group_size = args.group_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    return config


async def main_async(args: argparse.Namespace) -> int:
    """
    Main async entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Import components
    from tinker_integration.config import ConfigManager, TrainingConfig
    from tinker_integration.client import create_training_client, TinkerAuthenticationError
    from tinker_integration.metrics import MetricsLogger
    from tinker_integration.checkpoint import CheckpointManager
    from tinker_integration.error_handler import ErrorHandler
    from tinker_integration.async_verifier import AsyncVerifier
    from tinker_integration.env_group_builder import CurriculumEnvGroupBuilder, TheoremDataset
    from tinker_integration.training_loop import CISPOTrainingLoop
    from fim_rlvr_lean4.curriculum import CurriculumManager
    from fim_rlvr_lean4.lean_verifier import LeanVerifier
    
    # Load configuration
    logger.info("Loading configuration...")
    config_manager = ConfigManager(args.config)
    
    try:
        config = config_manager.load()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Apply CLI overrides
    config = apply_cli_overrides(config, args)
    
    # Validate dataset path
    if not config.dataset_path:
        logger.error(
            "Dataset path is required. Set via:\n"
            "  - 'dataset_path' in config YAML\n"
            "  - FIM_DATASET_PATH environment variable\n"
            "  - --dataset command line argument"
        )
        return 1
    
    if not Path(config.dataset_path).exists():
        logger.error(f"Dataset not found: {config.dataset_path}")
        return 1
    
    # Validate verification environment
    verification_env = Path(args.verification_env)
    if not verification_env.exists():
        logger.error(f"Verification environment not found: {verification_env}")
        return 1
    
    # Log effective configuration
    logger.info("Effective configuration:")
    for key, value in config.to_dict(mask_sensitive=True).items():
        logger.info(f"  {key}: {value}")
    
    # Dry run - just validate and exit
    if args.dry_run:
        logger.info("Dry run complete - configuration is valid")
        return 0
    
    # Initialize components
    logger.info("Initializing components...")
    
    # 1. Metrics logger
    metrics = MetricsLogger(
        log_dir=config.log_dir,
        wandb_project=config.wandb_project,
        wandb_run_name=config.wandb_run_name,
    )
    
    # Log configuration to metrics
    config_manager.log_effective_config(metrics)
    
    # 2. Error handler
    error_handler = ErrorHandler(
        max_retries=3,
        base_delay=1.0,
        critical_threshold=10,
    )
    
    # 3. Lean verifier
    logger.info(f"Initializing Lean verifier from {verification_env}...")
    lean_verifier = LeanVerifier(
        str(verification_env),
        no_sorries=args.no_sorries,
    )
    
    # 4. Async verifier
    async_verifier = AsyncVerifier(
        lean_verifier=lean_verifier,
        max_concurrent=config.max_concurrent_verifications,
        timeout_seconds=config.verification_timeout,
        metrics_logger=metrics,
    )
    
    # 5. Curriculum manager
    curriculum = CurriculumManager(
        levels=config.curriculum_levels,
        window_size=config.window_size,
        promotion_threshold=config.promotion_threshold,
    )
    
    # 6. Dataset
    logger.info(f"Loading dataset from {config.dataset_path}...")
    try:
        dataset = TheoremDataset(config.dataset_path)
        logger.info(f"Loaded {len(dataset)} theorems")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1
    
    # 9. Training client - use Tinker SDK directly
    logger.info(f"Initializing Tinker training client for {config.model_name}...")
    import tinker
    
    try:
        # Create ServiceClient
        service_client = tinker.ServiceClient()
        logger.info("ServiceClient created")
        
        # Create TrainingClient with LoRA
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
        )
        logger.info(f"Training client initialized with model: {config.model_name}")
        
        # Get tokenizer from training client
        tokenizer = training_client.get_tokenizer()
        logger.info(f"Tokenizer loaded: {type(tokenizer).__name__}")
        
    except tinker.AuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to initialize training client: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 10. Environment group builder - now with real tokenizer
    env_builder = CurriculumEnvGroupBuilder(
        dataset=dataset,
        curriculum_manager=curriculum,
        verifier=lean_verifier,  # Use sync verifier, training loop handles async
        tokenizer=tokenizer,
        group_size=config.group_size,
        max_tokens=512,
    )
    
    # 11. Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        training_client=training_client,
        curriculum_manager=curriculum,
    )
    
    # Handle resume
    start_step = 0
    if args.resume or args.resume_from:
        checkpoint_name = args.resume_from
        if checkpoint_name is None:
            checkpoint_name = checkpoint_manager.get_latest_checkpoint()
        
        if checkpoint_name:
            logger.info(f"Resuming from checkpoint: {checkpoint_name}")
            try:
                start_step = await checkpoint_manager.load(checkpoint_name)
                logger.info(f"Resumed at step {start_step}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return 1
        else:
            logger.warning("No checkpoint found to resume from, starting fresh")
    
    # 12. Training loop
    training_loop = CISPOTrainingLoop(
        service_client=service_client,
        training_client=training_client,
        tokenizer=tokenizer,
        env_group_builder=env_builder,
        config=config,
        metrics_logger=metrics,
        checkpoint_manager=checkpoint_manager,
        error_handler=error_handler,
        start_step=start_step,
    )
    
    # Set up graceful shutdown
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force shutdown requested")
            sys.exit(1)
        logger.info("Shutdown requested, finishing current step...")
        shutdown_requested = True
        training_loop.stop()
    
    # Register signal handlers
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except (ValueError, OSError):
        # Signal handling may not work in all environments
        pass
    
    # Run training
    logger.info("Starting training...")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Max steps: {config.max_steps}")
    logger.info(f"  Group size: {config.group_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Checkpoint interval: {config.checkpoint_interval}")
    
    try:
        summary = await training_loop.train()
        
        # Log final summary
        logger.info("Training complete!")
        logger.info(f"  Final step: {summary['final_step']}")
        logger.info(f"  Total verifications: {summary['total_verifications']}")
        logger.info(f"  Overall pass rate: {summary['overall_pass_rate']:.2%}")
        logger.info(f"  Total tokens: {summary['total_tokens_generated']}")
        
        # Save final metrics summary
        metrics.save_summary()
        
        # Log error statistics
        error_stats = error_handler.get_stats()
        if error_stats['transient_errors'] > 0 or error_stats['verification_crashes'] > 0:
            logger.warning("Error statistics:")
            logger.warning(f"  Transient errors: {error_stats['transient_errors']}")
            logger.warning(f"  Verification crashes: {error_stats['verification_crashes']}")
            logger.warning(f"  Timeouts: {error_stats['timeouts']}")
            if error_stats['flagged_theorems']:
                logger.warning(f"  Flagged theorems: {error_stats['flagged_theorems']}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130  # Standard exit code for SIGINT
    except RuntimeError as e:
        if "Critical error threshold" in str(e):
            logger.error(f"Training stopped due to critical errors: {e}")
            return 2
        raise
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        return 1
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        async_verifier.shutdown()
        metrics.close()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Add file handler if log directory specified
    if args.log_dir or (args.config and Path(args.config).exists()):
        # Will be set up after config is loaded
        pass
    
    # Run async main
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
