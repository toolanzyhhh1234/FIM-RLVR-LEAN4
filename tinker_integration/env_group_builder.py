"""
Curriculum-Aware Environment Group Builder for Tinker API integration.

This module provides the CurriculumEnvGroupBuilder and TheoremDataset classes
for creating groups of Lean4FIMEnv instances with curriculum-aware sampling.

Requirements covered:
- 3.1: Implement Tinker's EnvGroupBuilder interface
- 3.2: Query existing CurriculumManager for appropriate mask ratio
- 3.3: Create G environments per theorem (configurable, default: 4)
- 3.4: Call CurriculumManager.update_outcome() with verification results
- 3.5: Support 70/20/10 sampling policy (current/review/challenge levels)
- 3.6: Accept dataset of theorems and sample according to curriculum state
- 7.1: Load theorems from Parquet files (existing format)
- 7.2: Extract theorem_id, prefix, suffix, middle fields
- 7.3: Support filtering by theorem difficulty or source
- 7.4: Integrate with CurriculumManager to track per-theorem state
- 7.5: Apply dynamic masking using existing apply_dynamic_mask() function
"""

import os
import re
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

# Import existing masking utility
from fim_rlvr_lean4.masking import apply_dynamic_mask


def _bool_env(name: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    val = os.environ.get(name, "")
    if val.strip() == "":
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


# Environment variables for filtering (matching Unsloth pipeline)
FIM_EXCLUDE_SORRY = _bool_env("FIM_EXCLUDE_SORRY", True)


def is_valid_lean_sample(text: str, exclude_sorry: bool = True) -> bool:
    """
    Check if a Lean code sample is valid for training.
    
    Filters out:
    - Samples shorter than 50 characters
    - Samples containing 'sorry' or 'admit' (if exclude_sorry=True)
    - Samples without 'theorem', 'lemma', or 'def' keywords
    
    This matches the filtering logic in train_gspo_fim_qwen3-vl-8b.py
    to ensure consistent dataset quality between Unsloth and Tinker pipelines.
    
    Args:
        text: The Lean code text to validate.
        exclude_sorry: If True, exclude samples with sorry/admit (default: True).
                      Controlled by FIM_EXCLUDE_SORRY env var.
    
    Returns:
        True if the sample is valid for training, False otherwise.
    """
    if not text or len(text.strip()) < 50:
        return False
    
    if exclude_sorry:
        if re.search(r"\bsorry\b", text) or re.search(r"\badmit\b", text):
            return False
    
    # Must contain theorem, lemma, or def
    return ("theorem" in text) or ("lemma" in text) or ("def" in text)

if TYPE_CHECKING:
    from fim_rlvr_lean4.curriculum import CurriculumManager
    from .async_verifier import AsyncVerifier
    from .lean_env import Lean4FIMEnv, StepResult
    from .prompt_formatter import FIMPromptFormatter


@dataclass
class TheoremRecord:
    """
    A single theorem record from the dataset.
    
    Attributes:
        theorem_id: Unique identifier for the theorem.
        full_code: Complete Lean4 source code (prefix + middle + suffix).
        prefix: Code before the proof body.
        suffix: Code after the proof body.
        middle: The proof body (ground truth).
        metadata: Optional additional metadata (difficulty, source, etc.).
    """
    theorem_id: str
    full_code: str
    prefix: str
    suffix: str
    middle: str
    metadata: Optional[Dict[str, Any]] = None


class TheoremDataset:
    """
    Dataset of theorems loaded from Parquet files.
    
    Provides efficient loading and sampling of theorems for RL training.
    Supports filtering by difficulty, source, or custom predicates.
    
    Requirements covered:
    - 7.1: Load theorems from Parquet files
    - 7.2: Extract theorem_id, prefix, suffix, middle fields
    - 7.3: Support filtering by theorem difficulty or source
    
    Example:
        >>> dataset = TheoremDataset("data/theorems.parquet")
        >>> theorem = dataset.sample_theorem()
        >>> print(theorem["theorem_id"], theorem["prefix"][:50])
        
        >>> # With filtering
        >>> easy_dataset = TheoremDataset(
        ...     "data/theorems.parquet",
        ...     filter_fn=lambda row: row.get("difficulty", 0) < 3
        ... )
    
    Attributes:
        df: Polars DataFrame containing theorem data.
        _theorem_ids: List of theorem IDs for fast random access.
        _id_column: Name of the theorem ID column.
    """
    
    # Common column name mappings for different dataset formats
    COLUMN_MAPPINGS = {
        "theorem_id": ["theorem_id", "id", "name", "theorem_name"],
        "prefix": ["prefix", "context", "statement"],
        "suffix": ["suffix", "after"],
        "middle": ["middle", "proof", "solution", "tactic"],
    }
    
    def __init__(
        self,
        parquet_path: str,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        id_column: Optional[str] = None,
        exclude_sorry: Optional[bool] = None,
        apply_lean_filter: Optional[bool] = None,
    ):
        """
        Initialize the dataset from a Parquet file.
        
        Args:
            parquet_path: Path to the Parquet file containing theorems.
            filter_fn: Optional filter function that takes a row dict and
                      returns True to include, False to exclude.
            id_column: Optional explicit column name for theorem IDs.
                      If None, auto-detects from common column names.
            exclude_sorry: If True, exclude samples with sorry/admit.
                          If None, uses FIM_EXCLUDE_SORRY env var (default: True).
            apply_lean_filter: If True, apply standard Lean filtering.
                              If None, auto-detects based on filename:
                              - Files with 'filtered' in name: skip filtering
                              - Other files: apply filtering
        
        Raises:
            FileNotFoundError: If parquet_path doesn't exist.
            ValueError: If required columns are missing or all samples filtered.
        """
        import polars as pl
        
        self.df = pl.read_parquet(parquet_path)
        self._original_len = len(self.df)
        
        # Detect column names
        self._column_map = self._detect_columns()
        self._id_column = id_column or self._column_map.get("theorem_id", "theorem_id")
        
        # Determine exclude_sorry setting
        if exclude_sorry is None:
            exclude_sorry = FIM_EXCLUDE_SORRY
        self._exclude_sorry = exclude_sorry
        
        # Auto-detect if filtering should be applied
        # Skip filtering for pre-filtered datasets (filename contains 'filtered')
        if apply_lean_filter is None:
            apply_lean_filter = 'filtered' not in parquet_path.lower()
        
        # Apply standard Lean filtering (matching Unsloth pipeline)
        if apply_lean_filter:
            self._apply_lean_filter(exclude_sorry)
        
        # Apply custom filter if provided
        if filter_fn is not None:
            rows = self.df.to_dicts()
            filtered_indices = [
                i for i, row in enumerate(rows)
                if filter_fn(row)
            ]
            self.df = self.df[filtered_indices]
        
        # Cache theorem IDs for fast random access
        if self._id_column in self.df.columns:
            self._theorem_ids = self.df[self._id_column].to_list()
        else:
            # Generate synthetic IDs if not present
            self._theorem_ids = [f"theorem_{i}" for i in range(len(self.df))]
        
        # Log filtering results
        filtered_count = self._original_len - len(self)
        if apply_lean_filter:
            print(f"TheoremDataset: {self._original_len} -> {len(self)} samples "
                  f"({filtered_count} filtered, exclude_sorry={exclude_sorry})")
        else:
            print(f"TheoremDataset: loaded {len(self)} pre-filtered samples")
        
        if len(self) == 0:
            raise ValueError(
                f"All {self._original_len} samples were filtered out. "
                "Check dataset format or filtering settings."
            )
    
    def _apply_lean_filter(self, exclude_sorry: bool):
        """
        Apply standard Lean code filtering.
        
        Filters out:
        - Samples shorter than 50 characters
        - Samples containing 'sorry' or 'admit' (if exclude_sorry=True)
        - Samples without 'theorem', 'lemma', or 'def' keywords
        
        This matches the filtering in train_gspo_fim_qwen3-vl-8b.py.
        """
        # Find the text column to filter on
        text_col = None
        for col_name in ["formal_ground_truth", "prompt", "full_code", "code"]:
            if col_name in self.df.columns:
                text_col = col_name
                break
        
        if text_col is None:
            # Try detected columns
            text_col = self._column_map.get("prefix")
            if text_col is None:
                print("Warning: Could not find text column for filtering, skipping Lean filter")
                return
        
        # Convert to list for filtering
        rows = self.df.to_dicts()
        filtered_indices = []
        
        for i, row in enumerate(rows):
            text = row.get(text_col, "")
            if is_valid_lean_sample(text, exclude_sorry=exclude_sorry):
                filtered_indices.append(i)
        
        self.df = self.df[filtered_indices]
    
    def _detect_columns(self) -> Dict[str, str]:
        """
        Auto-detect column names from common mappings.
        
        Returns:
            Dictionary mapping standard names to actual column names.
        """
        column_map = {}
        available_columns = set(self.df.columns)
        
        for standard_name, candidates in self.COLUMN_MAPPINGS.items():
            for candidate in candidates:
                if candidate in available_columns:
                    column_map[standard_name] = candidate
                    break
        
        return column_map
    
    def _get_column(self, row: Dict[str, Any], standard_name: str) -> Optional[str]:
        """
        Get column value using detected column mapping.
        
        Args:
            row: Row dictionary from DataFrame.
            standard_name: Standard column name (e.g., "prefix", "middle").
        
        Returns:
            Column value or None if not found.
        """
        actual_name = self._column_map.get(standard_name, standard_name)
        return row.get(actual_name)
    
    def sample_theorem(self) -> Dict[str, Any]:
        """
        Sample a random theorem from the dataset.
        
        Returns:
            Dictionary containing:
                - theorem_id: Unique identifier
                - full_code: Complete source (prefix + middle + suffix)
                - prefix: Code before the hole
                - suffix: Code after the hole
                - middle: Ground truth proof body
        
        Raises:
            RuntimeError: If dataset is empty.
        """
        if len(self._theorem_ids) == 0:
            raise RuntimeError("Cannot sample from empty dataset")
        
        idx = random.randint(0, len(self._theorem_ids) - 1)
        return self.get_theorem(idx)
    
    def get_theorem(self, idx: int) -> Dict[str, Any]:
        """
        Get a theorem by index.
        
        Args:
            idx: Index into the dataset.
        
        Returns:
            Dictionary with theorem data.
        
        Raises:
            IndexError: If idx is out of bounds.
        """
        if idx < 0 or idx >= len(self._theorem_ids):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self._theorem_ids)}")
        
        row = self.df.row(idx, named=True)
        
        # Get theorem ID
        theorem_id = row.get(self._id_column, f"theorem_{idx}")
        
        # Get full code from the appropriate column
        # NuminaMath-LEAN uses 'formal_ground_truth' which contains the complete code
        full_code = None
        for col_name in ["formal_ground_truth", "full_code", "code", "prompt"]:
            if col_name in row and row[col_name]:
                full_code = row[col_name]
                break
        
        if not full_code:
            full_code = ""
        
        # For this dataset, prefix/suffix/middle are not pre-computed
        # They will be computed by apply_dynamic_mask in the env_group_builder
        # We just return the full_code and let masking handle the rest
        return {
            "theorem_id": theorem_id,
            "full_code": full_code,
            "prefix": "",  # Will be computed by masking
            "suffix": "",  # Will be computed by masking
            "middle": "",  # Will be computed by masking
        }
    
    def get_theorem_by_id(self, theorem_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a theorem by its ID.
        
        Args:
            theorem_id: The theorem's unique identifier.
        
        Returns:
            Dictionary with theorem data, or None if not found.
        """
        try:
            idx = self._theorem_ids.index(theorem_id)
            return self.get_theorem(idx)
        except ValueError:
            return None
    
    def __len__(self) -> int:
        """Return the number of theorems in the dataset."""
        return len(self._theorem_ids)
    
    def __iter__(self):
        """Iterate over all theorems in the dataset."""
        for idx in range(len(self)):
            yield self.get_theorem(idx)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing:
                - total: Total number of theorems
                - original: Original count before filtering
                - filtered: Number filtered out
                - exclude_sorry: Whether sorry/admit filtering is enabled
                - columns: Available column names
        """
        return {
            "total": len(self),
            "original": self._original_len,
            "filtered": self._original_len - len(self),
            "exclude_sorry": self._exclude_sorry,
            "columns": list(self.df.columns),
            "detected_mapping": self._column_map,
        }



class CurriculumEnvGroupBuilder:
    """
    Creates groups of Lean4FIMEnv instances with curriculum-aware sampling.
    
    Implements Tinker's EnvGroupBuilder interface, integrating with the
    existing CurriculumManager for mastery-based hole-size progression.
    
    Requirements covered:
    - 3.1: Implement Tinker's EnvGroupBuilder interface
    - 3.2: Query CurriculumManager for appropriate mask ratio
    - 3.3: Create G environments per theorem (configurable, default: 4)
    - 3.4: Call CurriculumManager.update_outcome() with verification results
    - 3.5: Support 70/20/10 sampling policy (via CurriculumManager)
    - 3.6: Sample from dataset according to curriculum state
    - 7.4: Integrate with CurriculumManager for per-theorem state
    - 7.5: Apply dynamic masking using apply_dynamic_mask()
    
    The builder creates G environments for the same theorem, each with
    potentially different hole placements (due to random masking). This
    enables group-relative advantage computation in CISPO training.
    
    Example:
        >>> from fim_rlvr_lean4.curriculum import CurriculumManager
        >>> 
        >>> dataset = TheoremDataset("data/theorems.parquet")
        >>> curriculum = CurriculumManager()
        >>> builder = CurriculumEnvGroupBuilder(
        ...     dataset=dataset,
        ...     curriculum_manager=curriculum,
        ...     verifier=async_verifier,
        ...     tokenizer=tokenizer,
        ...     group_size=4,
        ... )
        >>> 
        >>> # Create environment group
        >>> envs = builder.make_envs()
        >>> assert len(envs) == 4
        >>> 
        >>> # After getting results from training
        >>> builder.update_outcomes(results)
    
    Attributes:
        dataset: TheoremDataset for sampling theorems.
        curriculum: CurriculumManager for mask ratio selection.
        verifier: AsyncVerifier for proof verification.
        tokenizer: Tokenizer for encoding/decoding.
        group_size: Number of environments per theorem.
        formatter: FIMPromptFormatter for prompt construction.
    """
    
    def __init__(
        self,
        dataset: TheoremDataset,
        curriculum_manager: "CurriculumManager",
        verifier: "AsyncVerifier",
        tokenizer: Any,
        group_size: int = 4,
        prompt_formatter: Optional["FIMPromptFormatter"] = None,
        max_tokens: int = 512,
    ):
        """
        Initialize the environment group builder.
        
        Args:
            dataset: TheoremDataset containing theorems to sample from.
            curriculum_manager: CurriculumManager for mask ratio selection
                               and outcome tracking.
            verifier: AsyncVerifier for Lean4 proof verification.
            tokenizer: Tokenizer with encode() and decode() methods.
            group_size: Number of environments to create per theorem (default: 4).
                       This determines the group size for CISPO advantage computation.
            prompt_formatter: Optional FIMPromptFormatter. If None, creates default.
            max_tokens: Maximum tokens for model generation (default: 512).
        
        Raises:
            ValueError: If group_size < 1.
        """
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        
        self.dataset = dataset
        self.curriculum = curriculum_manager
        self.verifier = verifier
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.max_tokens = max_tokens
        
        # Lazy import to avoid circular dependency
        if prompt_formatter is None:
            from .prompt_formatter import FIMPromptFormatter
            self.formatter = FIMPromptFormatter()
        else:
            self.formatter = prompt_formatter
        
        # Track current theorem for outcome updates
        self._current_theorem_id: Optional[str] = None
        self._current_theorem_data: Optional[Dict[str, Any]] = None
        
        # Statistics
        self._groups_created = 0
        self._outcomes_updated = 0
    
    def make_envs(self) -> List["Lean4FIMEnv"]:
        """
        Create a group of environments for the same theorem.
        
        Uses curriculum manager to determine mask ratio (Requirement 3.2),
        then applies dynamic masking to create G environments (Requirement 3.3).
        Each environment may have a different hole placement due to random
        masking, but all share the same theorem and mask ratio.
        
        Returns:
            List of Lean4FIMEnv instances, length equals group_size.
        
        Raises:
            RuntimeError: If dataset is empty.
        """
        # Import here to avoid circular dependency
        from .lean_env import Lean4FIMEnv
        
        # Sample a theorem from dataset (Requirement 3.6)
        theorem = self.dataset.sample_theorem()
        self._current_theorem_id = theorem["theorem_id"]
        self._current_theorem_data = theorem
        
        # Get mask ratio from curriculum (Requirement 3.2)
        # This uses the 70/20/10 policy internally (Requirement 3.5)
        mask_ratio = self.curriculum.get_mask_ratio(theorem["theorem_id"])
        
        # Create G environments with same theorem but potentially different holes
        # (Requirement 3.3)
        envs = []
        for _ in range(self.group_size):
            # Apply dynamic masking (Requirement 7.5)
            prefix, suffix, middle = apply_dynamic_mask(
                theorem["full_code"],
                mask_ratio
            )
            
            env = Lean4FIMEnv(
                prefix=prefix,
                suffix=suffix,
                ground_truth_middle=middle,
                tokenizer=self.tokenizer,
                verifier=self.verifier,
                max_tokens=self.max_tokens,
                prompt_formatter=self.formatter,
            )
            envs.append(env)
        
        self._groups_created += 1
        return envs
    
    def update_outcomes(self, results: List["StepResult"]):
        """
        Update curriculum based on verification outcomes.
        
        Called after trajectories complete to track mastery and potentially
        promote theorems to harder difficulty levels (Requirement 3.4).
        
        Args:
            results: List of StepResult from environment steps.
                    Each result contains a reward (1.0 for success, 0.0 for failure).
        
        Note:
            This method should be called after make_envs() and after all
            environments have been stepped. The results should correspond
            to the environments created in the most recent make_envs() call.
        """
        if self._current_theorem_id is None:
            return
        
        for result in results:
            # reward=1.0 means success, reward=0.0 means failure
            success = result.reward > 0.5
            self.curriculum.update_outcome(self._current_theorem_id, success)
            self._outcomes_updated += 1
    
    def update_outcomes_from_list(self, outcomes: List[bool]):
        """
        Update curriculum based on boolean outcomes.
        
        Convenience method that accepts a list of booleans instead of StepResults.
        
        Args:
            outcomes: List of booleans (True for success, False for failure).
        """
        if self._current_theorem_id is None:
            return
        
        for success in outcomes:
            self.curriculum.update_outcome(self._current_theorem_id, success)
            self._outcomes_updated += 1
    
    def get_current_theorem_id(self) -> Optional[str]:
        """
        Get the ID of the current theorem being trained on.
        
        Returns:
            Theorem ID string, or None if no theorem has been sampled yet.
        """
        return self._current_theorem_id
    
    def get_current_theorem_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the full data of the current theorem.
        
        Returns:
            Dictionary with theorem data, or None if no theorem has been sampled.
        """
        return self._current_theorem_data
    
    def get_current_mask_ratio(self) -> Optional[float]:
        """
        Get the mask ratio for the current theorem.
        
        Returns:
            Mask ratio (0.0 to 1.0), or None if no theorem has been sampled.
        """
        if self._current_theorem_id is None:
            return None
        return self.curriculum.get_mask_ratio(self._current_theorem_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get builder statistics.
        
        Returns:
            Dictionary containing:
                - groups_created: Number of environment groups created
                - outcomes_updated: Number of outcome updates processed
                - dataset_size: Number of theorems in dataset
                - group_size: Configured group size
        """
        return {
            "groups_created": self._groups_created,
            "outcomes_updated": self._outcomes_updated,
            "dataset_size": len(self.dataset),
            "group_size": self.group_size,
        }
    
    def reset_stats(self):
        """Reset builder statistics to zero."""
        self._groups_created = 0
        self._outcomes_updated = 0
