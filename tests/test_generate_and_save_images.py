"""Tests for GenerateAndSaveImages use-case."""
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np

from src.domain.interfaces.model import Model


class TestGenerateAndSaveImages:
    """Tests for the GenerateAndSaveImages use-case."""

    def test_use_case_exists(self):
        """
        The GenerateAndSaveImages use-case class should exist in the domain layer.
        """
        from src.domain.use_cases.generate_and_save_images import GenerateAndSaveImages

        assert GenerateAndSaveImages is not None

    def test_run_generates_images(self):
        """
        Running the use-case should generate the requested number of images.
        """
        from src.domain.use_cases.generate_and_save_images import GenerateAndSaveImages

        mock_model = MagicMock(spec=Model)
        # Return fake images (n, H, W, C) in [0, 1] range
        mock_model.generate_images.return_value = np.random.rand(5, 28, 28, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            use_case = GenerateAndSaveImages(
                model=mock_model,
                num_images=5,
                output_dir=tmpdir,
            )
            use_case.run()

            mock_model.generate_images.assert_called_once_with(5)

    def test_run_saves_images_to_output_dir(self):
        """
        Running the use-case should save images as PNG files in the output directory.
        """
        from src.domain.use_cases.generate_and_save_images import GenerateAndSaveImages

        mock_model = MagicMock(spec=Model)
        mock_model.generate_images.return_value = np.random.rand(3, 28, 28, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            use_case = GenerateAndSaveImages(
                model=mock_model,
                num_images=3,
                output_dir=tmpdir,
            )
            use_case.run()

            # Check that 3 PNG files were created
            files = os.listdir(tmpdir)
            png_files = [f for f in files if f.endswith(".png")]
            assert len(png_files) == 3

    def test_run_creates_output_dir_if_not_exists(self):
        """
        Running the use-case should create the output directory if it doesn't exist.
        """
        from src.domain.use_cases.generate_and_save_images import GenerateAndSaveImages

        mock_model = MagicMock(spec=Model)
        mock_model.generate_images.return_value = np.random.rand(1, 28, 28, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nested", "output")

            use_case = GenerateAndSaveImages(
                model=mock_model,
                num_images=1,
                output_dir=output_dir,
            )
            use_case.run()

            assert os.path.exists(output_dir)
            files = os.listdir(output_dir)
            assert len(files) == 1

    def test_run_loads_weights_if_path_provided(self):
        """
        If a weights path is provided, the use-case should load it before generating.
        """
        from src.domain.use_cases.generate_and_save_images import GenerateAndSaveImages

        mock_model = MagicMock(spec=Model)
        mock_model.generate_images.return_value = np.random.rand(1, 28, 28, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            use_case = GenerateAndSaveImages(
                model=mock_model,
                num_images=1,
                output_dir=tmpdir,
                weights_path="/path/to/weights.h5",
            )
            use_case.run()

            mock_model.load.assert_called_once_with("/path/to/weights.h5")

    def test_run_does_not_load_if_no_weights_path(self):
        """
        If no weights path is provided, the model should not be loaded.
        """
        from src.domain.use_cases.generate_and_save_images import GenerateAndSaveImages

        mock_model = MagicMock(spec=Model)
        mock_model.generate_images.return_value = np.random.rand(1, 28, 28, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            use_case = GenerateAndSaveImages(
                model=mock_model,
                num_images=1,
                output_dir=tmpdir,
            )
            use_case.run()

            mock_model.load.assert_not_called()

    def test_saved_images_are_valid_pngs(self):
        """
        The saved PNG files should be valid and loadable.
        """
        from src.domain.use_cases.generate_and_save_images import GenerateAndSaveImages
        from PIL import Image

        mock_model = MagicMock(spec=Model)
        mock_model.generate_images.return_value = np.random.rand(2, 32, 32, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            use_case = GenerateAndSaveImages(
                model=mock_model,
                num_images=2,
                output_dir=tmpdir,
            )
            use_case.run()

            files = os.listdir(tmpdir)
            for f in files:
                if f.endswith(".png"):
                    img = Image.open(os.path.join(tmpdir, f))
                    assert img.size == (32, 32)
                    assert img.mode == "RGB"
