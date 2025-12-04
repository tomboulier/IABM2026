"""Tests for TrainingTracker interface and implementations."""
from src.domain.interfaces.training_tracker import TrainingTracker


class TestTrainingTrackerInterface:
    """Tests for the TrainingTracker abstract interface."""

    def test_tracker_has_on_training_start(self):
        """TrainingTracker should have on_training_start method."""
        assert hasattr(TrainingTracker, "on_training_start")

    def test_tracker_has_on_epoch_start(self):
        """TrainingTracker should have on_epoch_start method."""
        assert hasattr(TrainingTracker, "on_epoch_start")

    def test_tracker_has_on_batch_end(self):
        """TrainingTracker should have on_batch_end method."""
        assert hasattr(TrainingTracker, "on_batch_end")

    def test_tracker_has_on_epoch_end(self):
        """TrainingTracker should have on_epoch_end method."""
        assert hasattr(TrainingTracker, "on_epoch_end")

    def test_tracker_has_on_training_end(self):
        """TrainingTracker should have on_training_end method."""
        assert hasattr(TrainingTracker, "on_training_end")

    def test_all_methods_are_abstract(self):
        """All tracker methods should be abstract."""
        expected_abstract = {
            "on_training_start",
            "on_epoch_start",
            "on_batch_end",
            "on_epoch_end",
            "on_training_end",
        }
        assert expected_abstract.issubset(TrainingTracker.__abstractmethods__)
