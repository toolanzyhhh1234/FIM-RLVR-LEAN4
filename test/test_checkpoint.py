"""
Tests for CheckpointManager.

Tests checkpoint save/load functionality with local filesystem storage.
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from tinker_integration.checkpoint import CheckpointManager


class MockTrainingClient:
    """Mock Tinker training client for testing."""
    
    def __init__(self):
        self.saved_states = []
        self.loaded_states = []
    
    async def save_state_async(self, name: str) -> None:
        self.saved_states.append(name)
    
    async def load_state_with_optimizer_async(self, name: str) -> None:
        self.loaded_states.append(name)


class MockCurriculumManager:
    """Mock CurriculumManager for testing."""
    
    def __init__(self):
        self.levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.window_size = 8
        self.promotion_threshold = 5
        self.prob_current = 0.70
        self.prob_review = 0.20
        self.prob_challenge = 0.10
        self.states = {}
        self._saved_path = None
    
    def save(self, path: str) -> None:
        self._saved_path = path
        data = {
            "levels": self.levels,
            "window_size": self.window_size,
            "promotion_threshold": self.promotion_threshold,
            "prob_current": self.prob_current,
            "prob_review": self.prob_review,
            "prob_challenge": self.prob_challenge,
            "states": self.states,
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> "MockCurriculumManager":
        with open(path, "r") as f:
            data = json.load(f)
        mgr = cls()
        mgr.levels = data["levels"]
        mgr.window_size = data["window_size"]
        mgr.promotion_threshold = data["promotion_threshold"]
        mgr.prob_current = data["prob_current"]
        mgr.prob_review = data["prob_review"]
        mgr.prob_challenge = data["prob_challenge"]
        mgr.states = data["states"]
        return mgr


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_client():
    """Create a mock training client."""
    return MockTrainingClient()


@pytest.fixture
def mock_curriculum():
    """Create a mock curriculum manager."""
    return MockCurriculumManager()


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""
    
    def test_creates_checkpoint_directory(self, temp_checkpoint_dir):
        """Test that checkpoint directory is created on init."""
        checkpoint_dir = Path(temp_checkpoint_dir) / "new_checkpoints"
        assert not checkpoint_dir.exists()
        
        manager = CheckpointManager(str(checkpoint_dir))
        
        assert checkpoint_dir.exists()
    
    def test_accepts_existing_directory(self, temp_checkpoint_dir):
        """Test that existing directory is accepted."""
        manager = CheckpointManager(temp_checkpoint_dir)
        assert manager.checkpoint_dir == Path(temp_checkpoint_dir)
    
    def test_optional_client_and_curriculum(self, temp_checkpoint_dir):
        """Test that client and curriculum are optional."""
        manager = CheckpointManager(temp_checkpoint_dir)
        assert manager.client is None
        assert manager.curriculum is None


class TestCheckpointSave:
    """Tests for checkpoint saving."""
    
    @pytest.mark.asyncio
    async def test_save_creates_checkpoint_directory(self, temp_checkpoint_dir, mock_client, mock_curriculum):
        """Test that save creates checkpoint subdirectory."""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            training_client=mock_client,
            curriculum_manager=mock_curriculum,
        )
        
        await manager.save(step=100)
        
        checkpoint_path = Path(temp_checkpoint_dir) / "checkpoint_100"
        assert checkpoint_path.exists()
        assert checkpoint_path.is_dir()
    
    @pytest.mark.asyncio
    async def test_save_creates_metadata_file(self, temp_checkpoint_dir, mock_client, mock_curriculum):
        """Test that save creates metadata.json."""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            training_client=mock_client,
            curriculum_manager=mock_curriculum,
        )
        
        await manager.save(step=100)
        
        metadata_path = Path(temp_checkpoint_dir) / "checkpoint_100" / "metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["step"] == 100
        assert metadata["weights_name"] == "checkpoint_100"
        assert "timestamp" in metadata
        assert metadata["has_curriculum"] is True
        assert metadata["has_weights"] is True
    
    @pytest.mark.asyncio
    async def test_save_creates_curriculum_file(self, temp_checkpoint_dir, mock_client, mock_curriculum):
        """Test that save creates curriculum.json."""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            training_client=mock_client,
            curriculum_manager=mock_curriculum,
        )
        
        await manager.save(step=100)
        
        curriculum_path = Path(temp_checkpoint_dir) / "checkpoint_100" / "curriculum.json"
        assert curriculum_path.exists()
    
    @pytest.mark.asyncio
    async def test_save_calls_client_save_state(self, temp_checkpoint_dir, mock_client, mock_curriculum):
        """Test that save calls training client's save_state_async."""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            training_client=mock_client,
            curriculum_manager=mock_curriculum,
        )
        
        await manager.save(step=100)
        
        assert "checkpoint_100" in mock_client.saved_states
    
    @pytest.mark.asyncio
    async def test_save_without_client(self, temp_checkpoint_dir, mock_curriculum):
        """Test that save works without training client."""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            curriculum_manager=mock_curriculum,
        )
        
        path = await manager.save(step=100)
        
        assert Path(path).exists()
        metadata_path = Path(path) / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["has_weights"] is False
    
    @pytest.mark.asyncio
    async def test_save_without_curriculum(self, temp_checkpoint_dir, mock_client):
        """Test that save works without curriculum manager."""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            training_client=mock_client,
        )
        
        path = await manager.save(step=100)
        
        assert Path(path).exists()
        curriculum_path = Path(path) / "curriculum.json"
        assert not curriculum_path.exists()
    
    @pytest.mark.asyncio
    async def test_save_with_extra_metadata(self, temp_checkpoint_dir):
        """Test that extra metadata is saved."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        await manager.save(step=100, extra_metadata={"learning_rate": 0.001, "epoch": 5})
        
        metadata_path = Path(temp_checkpoint_dir) / "checkpoint_100" / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["extra"]["learning_rate"] == 0.001
        assert metadata["extra"]["epoch"] == 5
    
    @pytest.mark.asyncio
    async def test_save_returns_checkpoint_path(self, temp_checkpoint_dir):
        """Test that save returns the checkpoint path."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        path = await manager.save(step=100)
        
        expected = str(Path(temp_checkpoint_dir) / "checkpoint_100")
        assert path == expected


class TestCheckpointLoad:
    """Tests for checkpoint loading."""
    
    @pytest.mark.asyncio
    async def test_load_returns_step(self, temp_checkpoint_dir, mock_client, mock_curriculum):
        """Test that load returns the saved step."""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            training_client=mock_client,
            curriculum_manager=mock_curriculum,
        )
        
        await manager.save(step=100)
        step = await manager.load("checkpoint_100")
        
        assert step == 100
    
    @pytest.mark.asyncio
    async def test_load_calls_client_load_state(self, temp_checkpoint_dir, mock_client, mock_curriculum):
        """Test that load calls training client's load_state_with_optimizer_async."""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            training_client=mock_client,
            curriculum_manager=mock_curriculum,
        )
        
        await manager.save(step=100)
        await manager.load("checkpoint_100")
        
        assert "checkpoint_100" in mock_client.loaded_states
    
    @pytest.mark.asyncio
    async def test_load_raises_on_missing_checkpoint(self, temp_checkpoint_dir):
        """Test that load raises FileNotFoundError for missing checkpoint."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        with pytest.raises(FileNotFoundError):
            await manager.load("checkpoint_999")
    
    @pytest.mark.asyncio
    async def test_load_raises_on_corrupted_checkpoint(self, temp_checkpoint_dir):
        """Test that load raises ValueError for corrupted checkpoint."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Create checkpoint directory without metadata
        checkpoint_path = Path(temp_checkpoint_dir) / "checkpoint_100"
        checkpoint_path.mkdir()
        
        with pytest.raises(ValueError, match="corrupted"):
            await manager.load("checkpoint_100")


class TestCheckpointList:
    """Tests for listing checkpoints."""
    
    @pytest.mark.asyncio
    async def test_list_checkpoints_empty(self, temp_checkpoint_dir):
        """Test list_checkpoints returns empty list when no checkpoints."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        checkpoints = manager.list_checkpoints()
        
        assert checkpoints == []
    
    @pytest.mark.asyncio
    async def test_list_checkpoints_returns_sorted(self, temp_checkpoint_dir):
        """Test list_checkpoints returns checkpoints sorted by step."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Save in non-sequential order
        await manager.save(step=300)
        await manager.save(step=100)
        await manager.save(step=200)
        
        checkpoints = manager.list_checkpoints()
        
        assert checkpoints == ["checkpoint_100", "checkpoint_200", "checkpoint_300"]
    
    @pytest.mark.asyncio
    async def test_get_latest_checkpoint(self, temp_checkpoint_dir):
        """Test get_latest_checkpoint returns most recent."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        await manager.save(step=100)
        await manager.save(step=200)
        await manager.save(step=300)
        
        latest = manager.get_latest_checkpoint()
        
        assert latest == "checkpoint_300"
    
    def test_get_latest_checkpoint_none_when_empty(self, temp_checkpoint_dir):
        """Test get_latest_checkpoint returns None when no checkpoints."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        latest = manager.get_latest_checkpoint()
        
        assert latest is None


class TestCheckpointMetadata:
    """Tests for checkpoint metadata operations."""
    
    @pytest.mark.asyncio
    async def test_get_checkpoint_metadata(self, temp_checkpoint_dir):
        """Test get_checkpoint_metadata returns metadata dict."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        await manager.save(step=100, extra_metadata={"test": "value"})
        
        metadata = manager.get_checkpoint_metadata("checkpoint_100")
        
        assert metadata["step"] == 100
        assert metadata["extra"]["test"] == "value"
    
    def test_get_checkpoint_metadata_returns_none_for_missing(self, temp_checkpoint_dir):
        """Test get_checkpoint_metadata returns None for missing checkpoint."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        metadata = manager.get_checkpoint_metadata("checkpoint_999")
        
        assert metadata is None


class TestCheckpointCleanup:
    """Tests for checkpoint cleanup operations."""
    
    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, temp_checkpoint_dir):
        """Test delete_checkpoint removes checkpoint."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        await manager.save(step=100)
        assert manager.list_checkpoints() == ["checkpoint_100"]
        
        result = manager.delete_checkpoint("checkpoint_100")
        
        assert result is True
        assert manager.list_checkpoints() == []
    
    def test_delete_checkpoint_returns_false_for_missing(self, temp_checkpoint_dir):
        """Test delete_checkpoint returns False for missing checkpoint."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        result = manager.delete_checkpoint("checkpoint_999")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints(self, temp_checkpoint_dir):
        """Test cleanup_old_checkpoints keeps only recent N."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        for step in [100, 200, 300, 400, 500]:
            await manager.save(step=step)
        
        deleted = manager.cleanup_old_checkpoints(keep_last_n=2)
        
        assert deleted == 3
        assert manager.list_checkpoints() == ["checkpoint_400", "checkpoint_500"]
    
    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints_no_op_when_few(self, temp_checkpoint_dir):
        """Test cleanup_old_checkpoints does nothing when fewer than N checkpoints."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        await manager.save(step=100)
        await manager.save(step=200)
        
        deleted = manager.cleanup_old_checkpoints(keep_last_n=5)
        
        assert deleted == 0
        assert len(manager.list_checkpoints()) == 2


class TestCheckpointRoundTrip:
    """Tests for checkpoint save/load round-trip."""
    
    @pytest.mark.asyncio
    async def test_round_trip_preserves_step(self, temp_checkpoint_dir, mock_client, mock_curriculum):
        """Test that save/load round-trip preserves step count."""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            training_client=mock_client,
            curriculum_manager=mock_curriculum,
        )
        
        original_step = 12345
        await manager.save(step=original_step)
        loaded_step = await manager.load("checkpoint_12345")
        
        assert loaded_step == original_step
    
    @pytest.mark.asyncio
    async def test_round_trip_preserves_extra_metadata(self, temp_checkpoint_dir):
        """Test that save/load round-trip preserves extra metadata."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        extra = {"learning_rate": 0.001, "batch_size": 32, "notes": "test run"}
        await manager.save(step=100, extra_metadata=extra)
        
        metadata = manager.get_checkpoint_metadata("checkpoint_100")
        
        assert metadata["extra"] == extra
