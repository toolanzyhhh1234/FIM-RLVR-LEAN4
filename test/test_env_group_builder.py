"""
Tests for CurriculumEnvGroupBuilder and TheoremDataset.

Tests the curriculum-aware environment group builder and theorem dataset
for Tinker API integration.
"""

import os
import tempfile
import pytest
from typing import Tuple, List, Dict, Any
from unittest.mock import Mock

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

from tinker_integration.env_group_builder import TheoremDataset, CurriculumEnvGroupBuilder
from tinker_integration.lean_env import StepResult
from fim_rlvr_lean4.curriculum import CurriculumManager


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def encode(self, text: str) -> List[int]:
        """Simple encoding: convert chars to ASCII codes."""
        return [ord(c) for c in text[:100]]  # Limit to 100 chars
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Simple decoding: convert ASCII codes back to chars."""
        return "".join(chr(t) for t in token_ids if 32 <= t < 127)


class MockVerifier:
    """Mock verifier for testing."""
    
    def __init__(self, success: bool = True):
        self.success = success
        self.call_count = 0
    
    def verify(self, full_code: str) -> Tuple[bool, str]:
        self.call_count += 1
        return self.success, "Mock output"


class TestTheoremDataset:
    """Tests for TheoremDataset class."""
    
    @pytest.fixture
    def sample_parquet(self, tmp_path):
        """Create a sample parquet file for testing."""
        import polars as pl
        
        data = {
            "theorem_id": ["thm_1", "thm_2", "thm_3"],
            "prefix": [
                "theorem foo : 1 + 1 = 2 := by\n  ",
                "theorem bar : 2 + 2 = 4 := by\n  ",
                "theorem baz : 3 + 3 = 6 := by\n  ",
            ],
            "suffix": ["\n  rfl", "\n  rfl", "\n  rfl"],
            "middle": ["norm_num", "norm_num", "norm_num"],
        }
        df = pl.DataFrame(data)
        
        parquet_path = tmp_path / "test_theorems.parquet"
        df.write_parquet(parquet_path)
        return str(parquet_path)
    
    def test_load_parquet(self, sample_parquet):
        """Test loading theorems from parquet file."""
        dataset = TheoremDataset(sample_parquet)
        
        assert len(dataset) == 3
        stats = dataset.get_stats()
        assert stats["total"] == 3
        assert stats["original"] == 3
        assert stats["filtered"] == 0
    
    def test_sample_theorem(self, sample_parquet):
        """Test sampling a random theorem."""
        dataset = TheoremDataset(sample_parquet)
        
        theorem = dataset.sample_theorem()
        
        assert "theorem_id" in theorem
        assert "full_code" in theorem
        assert "prefix" in theorem
        assert "suffix" in theorem
        assert "middle" in theorem
        assert theorem["theorem_id"] in ["thm_1", "thm_2", "thm_3"]
    
    def test_get_theorem_by_index(self, sample_parquet):
        """Test getting theorem by index."""
        dataset = TheoremDataset(sample_parquet)
        
        theorem = dataset.get_theorem(0)
        
        assert theorem["theorem_id"] == "thm_1"
        assert "foo" in theorem["prefix"]
    
    def test_get_theorem_by_id(self, sample_parquet):
        """Test getting theorem by ID."""
        dataset = TheoremDataset(sample_parquet)
        
        theorem = dataset.get_theorem_by_id("thm_2")
        
        assert theorem is not None
        assert theorem["theorem_id"] == "thm_2"
        assert "bar" in theorem["prefix"]
    
    def test_get_theorem_by_id_not_found(self, sample_parquet):
        """Test getting non-existent theorem by ID."""
        dataset = TheoremDataset(sample_parquet)
        
        theorem = dataset.get_theorem_by_id("nonexistent")
        
        assert theorem is None
    
    def test_filter_function(self, sample_parquet):
        """Test filtering theorems."""
        dataset = TheoremDataset(
            sample_parquet,
            filter_fn=lambda row: row.get("theorem_id") != "thm_2"
        )
        
        assert len(dataset) == 2
        stats = dataset.get_stats()
        assert stats["filtered"] == 1
    
    def test_full_code_reconstruction(self, sample_parquet):
        """Test that full_code is correctly reconstructed."""
        dataset = TheoremDataset(sample_parquet)
        
        theorem = dataset.get_theorem(0)
        
        expected_full = theorem["prefix"] + theorem["middle"] + theorem["suffix"]
        assert theorem["full_code"] == expected_full
    
    def test_iteration(self, sample_parquet):
        """Test iterating over dataset."""
        dataset = TheoremDataset(sample_parquet)
        
        theorems = list(dataset)
        
        assert len(theorems) == 3
        assert all("theorem_id" in t for t in theorems)
    
    def test_index_out_of_bounds(self, sample_parquet):
        """Test that out of bounds index raises error."""
        dataset = TheoremDataset(sample_parquet)
        
        with pytest.raises(IndexError):
            dataset.get_theorem(100)
    
    def test_empty_dataset_sample(self, tmp_path):
        """Test sampling from empty dataset raises error."""
        import polars as pl
        
        # Create empty parquet
        df = pl.DataFrame({
            "theorem_id": [],
            "prefix": [],
            "suffix": [],
            "middle": [],
        })
        parquet_path = tmp_path / "empty.parquet"
        df.write_parquet(parquet_path)
        
        dataset = TheoremDataset(str(parquet_path))
        
        with pytest.raises(RuntimeError, match="empty dataset"):
            dataset.sample_theorem()


class TestCurriculumEnvGroupBuilder:
    """Tests for CurriculumEnvGroupBuilder class."""
    
    @pytest.fixture
    def sample_parquet(self, tmp_path):
        """Create a sample parquet file for testing."""
        import polars as pl
        
        data = {
            "theorem_id": ["thm_1", "thm_2", "thm_3"],
            "prefix": [
                "theorem foo : 1 + 1 = 2 := by\n  ",
                "theorem bar : 2 + 2 = 4 := by\n  ",
                "theorem baz : 3 + 3 = 6 := by\n  ",
            ],
            "suffix": ["\n  rfl", "\n  rfl", "\n  rfl"],
            "middle": ["norm_num", "norm_num", "norm_num"],
        }
        df = pl.DataFrame(data)
        
        parquet_path = tmp_path / "test_theorems.parquet"
        df.write_parquet(parquet_path)
        return str(parquet_path)
    
    @pytest.fixture
    def builder(self, sample_parquet):
        """Create a CurriculumEnvGroupBuilder for testing."""
        dataset = TheoremDataset(sample_parquet)
        curriculum = CurriculumManager()
        verifier = MockVerifier()
        tokenizer = MockTokenizer()
        
        return CurriculumEnvGroupBuilder(
            dataset=dataset,
            curriculum_manager=curriculum,
            verifier=verifier,
            tokenizer=tokenizer,
            group_size=4,
        )
    
    def test_init(self, builder):
        """Test builder initialization."""
        assert builder.group_size == 4
        assert builder._current_theorem_id is None
    
    def test_init_invalid_group_size(self, sample_parquet):
        """Test that invalid group_size raises ValueError."""
        dataset = TheoremDataset(sample_parquet)
        curriculum = CurriculumManager()
        verifier = MockVerifier()
        tokenizer = MockTokenizer()
        
        with pytest.raises(ValueError, match="group_size must be >= 1"):
            CurriculumEnvGroupBuilder(
                dataset=dataset,
                curriculum_manager=curriculum,
                verifier=verifier,
                tokenizer=tokenizer,
                group_size=0,
            )
    
    def test_make_envs_returns_correct_count(self, builder):
        """Test that make_envs returns correct number of environments."""
        envs = builder.make_envs()
        
        assert len(envs) == 4
    
    def test_make_envs_sets_current_theorem(self, builder):
        """Test that make_envs sets current theorem ID."""
        assert builder.get_current_theorem_id() is None
        
        builder.make_envs()
        
        assert builder.get_current_theorem_id() is not None
        assert builder.get_current_theorem_id() in ["thm_1", "thm_2", "thm_3"]
    
    def test_make_envs_returns_lean4fim_envs(self, builder):
        """Test that make_envs returns Lean4FIMEnv instances."""
        from tinker_integration.lean_env import Lean4FIMEnv
        
        envs = builder.make_envs()
        
        assert all(isinstance(env, Lean4FIMEnv) for env in envs)
    
    def test_update_outcomes_success(self, builder):
        """Test updating outcomes with successful results."""
        builder.make_envs()
        theorem_id = builder.get_current_theorem_id()
        
        # Create mock results (all successful)
        results = [
            StepResult(reward=1.0, episode_done=True),
            StepResult(reward=1.0, episode_done=True),
            StepResult(reward=1.0, episode_done=True),
            StepResult(reward=1.0, episode_done=True),
        ]
        
        builder.update_outcomes(results)
        
        # Check that curriculum was updated
        state = builder.curriculum.states[theorem_id]
        assert len(state.history) == 4
        assert sum(state.history) == 4  # All successes
    
    def test_update_outcomes_mixed(self, builder):
        """Test updating outcomes with mixed results."""
        builder.make_envs()
        theorem_id = builder.get_current_theorem_id()
        
        # Create mock results (mixed)
        results = [
            StepResult(reward=1.0, episode_done=True),
            StepResult(reward=0.0, episode_done=True),
            StepResult(reward=1.0, episode_done=True),
            StepResult(reward=0.0, episode_done=True),
        ]
        
        builder.update_outcomes(results)
        
        # Check that curriculum was updated
        state = builder.curriculum.states[theorem_id]
        assert len(state.history) == 4
        assert sum(state.history) == 2  # 2 successes, 2 failures
    
    def test_update_outcomes_no_theorem(self, builder):
        """Test that update_outcomes does nothing if no theorem sampled."""
        # Don't call make_envs first
        results = [StepResult(reward=1.0, episode_done=True)]
        
        # Should not raise
        builder.update_outcomes(results)
    
    def test_get_stats(self, builder):
        """Test getting builder statistics."""
        stats = builder.get_stats()
        
        assert stats["groups_created"] == 0
        assert stats["outcomes_updated"] == 0
        assert stats["dataset_size"] == 3
        assert stats["group_size"] == 4
    
    def test_stats_increment(self, builder):
        """Test that statistics increment correctly."""
        builder.make_envs()
        results = [StepResult(reward=1.0, episode_done=True) for _ in range(4)]
        builder.update_outcomes(results)
        
        stats = builder.get_stats()
        
        assert stats["groups_created"] == 1
        assert stats["outcomes_updated"] == 4
    
    def test_reset_stats(self, builder):
        """Test resetting statistics."""
        builder.make_envs()
        builder.reset_stats()
        
        stats = builder.get_stats()
        
        assert stats["groups_created"] == 0
        assert stats["outcomes_updated"] == 0
    
    def test_get_current_theorem_data(self, builder):
        """Test getting current theorem data."""
        assert builder.get_current_theorem_data() is None
        
        builder.make_envs()
        
        data = builder.get_current_theorem_data()
        assert data is not None
        assert "theorem_id" in data
        assert "full_code" in data
    
    def test_get_current_mask_ratio(self, builder):
        """Test getting current mask ratio."""
        assert builder.get_current_mask_ratio() is None
        
        builder.make_envs()
        
        ratio = builder.get_current_mask_ratio()
        assert ratio is not None
        assert 0.0 <= ratio <= 1.0


class TestCurriculumIntegration:
    """Tests for curriculum integration with environment builder."""
    
    @pytest.fixture
    def sample_parquet(self, tmp_path):
        """Create a sample parquet file for testing."""
        import polars as pl
        
        data = {
            "theorem_id": ["thm_1"],
            "prefix": ["theorem foo : 1 + 1 = 2 := by\n  "],
            "suffix": ["\n  rfl"],
            "middle": ["norm_num"],
        }
        df = pl.DataFrame(data)
        
        parquet_path = tmp_path / "test_theorems.parquet"
        df.write_parquet(parquet_path)
        return str(parquet_path)
    
    def test_curriculum_promotion(self, sample_parquet):
        """Test that curriculum promotes theorem after enough successes."""
        dataset = TheoremDataset(sample_parquet)
        curriculum = CurriculumManager(
            promotion_threshold=3,  # Promote after 3 successes
            window_size=8,
        )
        verifier = MockVerifier()
        tokenizer = MockTokenizer()
        
        builder = CurriculumEnvGroupBuilder(
            dataset=dataset,
            curriculum_manager=curriculum,
            verifier=verifier,
            tokenizer=tokenizer,
            group_size=1,
        )
        
        # Initial level should be 0
        builder.make_envs()
        theorem_id = builder.get_current_theorem_id()
        assert curriculum.get_current_level_index(theorem_id) == 0
        
        # Simulate 3 successful outcomes
        for _ in range(3):
            builder.make_envs()
            results = [StepResult(reward=1.0, episode_done=True)]
            builder.update_outcomes(results)
        
        # Should have promoted to level 1
        assert curriculum.get_current_level_index(theorem_id) == 1
