"""Tests for TrainingTracker interface and implementations."""
import io
import sys

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


class TestConsoleTracker:
    """Tests for ConsoleTracker implementation."""

    def test_console_tracker_implements_interface(self):
        """ConsoleTracker should implement TrainingTracker interface."""
        from src.infrastructure.tensorflow.observability import ConsoleTracker

        tracker = ConsoleTracker()
        assert isinstance(tracker, TrainingTracker)

    def test_console_tracker_can_complete_training_cycle(self):
        """ConsoleTracker should handle a complete training cycle without errors."""
        from src.infrastructure.tensorflow.observability import ConsoleTracker

        tracker = ConsoleTracker()

        # Simulate training cycle
        tracker.on_training_start(total_epochs=2, total_batches=10, dataset_name="TestDataset")

        for epoch in range(2):
            tracker.on_epoch_start(epoch)
            for batch in range(10):
                tracker.on_batch_end(epoch, batch, loss=0.5 - batch * 0.01)
            tracker.on_epoch_end(epoch, avg_loss=0.45)

        tracker.on_training_end()

    def test_silent_tracker_produces_no_output(self):
        """SilentTracker should produce no console output."""
        from src.infrastructure.tensorflow.observability import SilentTracker

        tracker = SilentTracker()

        # Capture stdout
        captured = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured
        sys.stderr = captured

        try:
            tracker.on_training_start(total_epochs=1, total_batches=5)
            tracker.on_epoch_start(0)
            for batch in range(5):
                tracker.on_batch_end(0, batch, loss=0.5)
            tracker.on_epoch_end(0, avg_loss=0.5)
            tracker.on_training_end()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        assert captured.getvalue() == "", "SilentTracker should produce no output"
