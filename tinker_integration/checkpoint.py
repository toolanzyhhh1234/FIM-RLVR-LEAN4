"""
CheckpointManager for saving and loading training state to local filesystem.

Saves:
- LoRA weights (via Tinker API)
- CurriculumManager state (JSON)
- Training metadata (step count)
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from fim_rlvr_lean4.curriculum import CurriculumManager


class TrainingClientProtocol(Protocol):
    """Protocol for Tinker training client checkpoint operations."""
    
    async def save_state_async(self, name: str) -> None:
        """Save LoRA weights with given name."""
        ...
    
    async def load_state_with_optimizer_async(self, name: str) -> None:
        """Load LoRA weights and optimizer state."""
        ...


class CheckpointManager:
    """
    Manages checkpointing of training state to local filesystem.
    
    Checkpoint structure:
        checkpoint_dir/
            checkpoint_{step}/
                metadata.json      # step, timestamp, weights_name
                curriculum.json    # CurriculumManager state
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        training_client: Optional[TrainingClientProtocol] = None,
        curriculum_manager: Optional["CurriculumManager"] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            training_client: Tinker training client for LoRA weight save/load
            curriculum_manager: CurriculumManager instance to save/restore
            logger: Optional logger instance
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.client = training_client
        self.curriculum = curriculum_manager
        self.logger = logger or logging.getLogger(__name__)
    
    async def save(self, step: int, extra_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save checkpoint at given step.
        
        Args:
            step: Current training step
            extra_metadata: Optional additional metadata to save
            
        Returns:
            Path to the checkpoint directory
        """
        checkpoint_name = f"checkpoint_{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Saving checkpoint at step {step} to {checkpoint_path}")
        
        # 1. Save LoRA weights via Tinker (if client available)
        weights_name = checkpoint_name
        if self.client is not None:
            try:
                await self.client.save_state_async(weights_name)
                self.logger.debug(f"Saved LoRA weights: {weights_name}")
            except Exception as e:
                self.logger.error(f"Failed to save LoRA weights: {e}")
                raise
        
        # 2. Save curriculum state (if manager available)
        if self.curriculum is not None:
            curriculum_path = checkpoint_path / "curriculum.json"
            self.curriculum.save(str(curriculum_path))
            self.logger.debug(f"Saved curriculum state to {curriculum_path}")
        
        # 3. Save metadata
        metadata = {
            "step": step,
            "weights_name": weights_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "has_curriculum": self.curriculum is not None,
            "has_weights": self.client is not None,
        }
        if extra_metadata:
            metadata["extra"] = extra_metadata
            
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_name}")
        return str(checkpoint_path)
    
    async def load(self, checkpoint_name: str) -> int:
        """
        Load checkpoint and return the step count.
        
        Args:
            checkpoint_name: Name of checkpoint directory (e.g., "checkpoint_100")
            
        Returns:
            The training step from the checkpoint
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is corrupted
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # 1. Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Checkpoint corrupted: missing metadata.json in {checkpoint_path}")
            
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        step = metadata["step"]
        weights_name = metadata.get("weights_name", checkpoint_name)
        
        # 2. Load LoRA weights via Tinker (if client available and weights were saved)
        if self.client is not None and metadata.get("has_weights", True):
            try:
                await self.client.load_state_with_optimizer_async(weights_name)
                self.logger.debug(f"Loaded LoRA weights: {weights_name}")
            except Exception as e:
                self.logger.error(f"Failed to load LoRA weights: {e}")
                raise
        
        # 3. Load curriculum state (if manager available and curriculum was saved)
        curriculum_path = checkpoint_path / "curriculum.json"
        if self.curriculum is not None and curriculum_path.exists():
            self._load_curriculum_state(curriculum_path)
            self.logger.debug(f"Loaded curriculum state from {curriculum_path}")
        
        self.logger.info(f"Checkpoint loaded: step={step}")
        return step
    
    def _load_curriculum_state(self, curriculum_path: Path) -> None:
        """Load curriculum state into the existing manager."""
        # Import here to avoid circular imports
        from fim_rlvr_lean4.curriculum import CurriculumManager as CM
        
        loaded = CM.load(str(curriculum_path))
        # Copy state to existing manager
        self.curriculum.states = loaded.states
        self.curriculum.levels = loaded.levels
        self.curriculum.window_size = loaded.window_size
        self.curriculum.promotion_threshold = loaded.promotion_threshold
        self.curriculum.prob_current = loaded.prob_current
        self.curriculum.prob_review = loaded.prob_review
        self.curriculum.prob_challenge = loaded.prob_challenge
    
    def list_checkpoints(self) -> list:
        """
        List available checkpoints in the checkpoint directory.
        
        Returns:
            Sorted list of checkpoint names (oldest first)
        """
        checkpoints = []
        if not self.checkpoint_dir.exists():
            return checkpoints
            
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint_"):
                metadata_path = path / "metadata.json"
                if metadata_path.exists():
                    checkpoints.append(path.name)
        
        # Sort by step number
        return sorted(checkpoints, key=lambda x: int(x.split("_")[1]))
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the most recent checkpoint name.
        
        Returns:
            Checkpoint name or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None
    
    def get_checkpoint_metadata(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint directory
            
        Returns:
            Metadata dict or None if checkpoint doesn't exist
        """
        metadata_path = self.checkpoint_dir / checkpoint_name / "metadata.json"
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to delete
            
        Returns:
            True if deleted, False if not found
        """
        import shutil
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            return False
        
        shutil.rmtree(checkpoint_path)
        self.logger.info(f"Deleted checkpoint: {checkpoint_name}")
        return True
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3) -> int:
        """
        Remove old checkpoints, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            return 0
        
        to_delete = checkpoints[:-keep_last_n]
        deleted = 0
        
        for checkpoint_name in to_delete:
            if self.delete_checkpoint(checkpoint_name):
                deleted += 1
        
        self.logger.info(f"Cleaned up {deleted} old checkpoints, kept {keep_last_n}")
        return deleted
