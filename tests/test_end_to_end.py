"""True end-to-end test that runs main.py as a subprocess."""
import subprocess
import sys
from pathlib import Path


def test_main_with_test_configuration():
    """
    End-to-end test: Run main.py with test configuration file.

    This test verifies that the complete pipeline runs without errors
    when launched from command line with a test configuration.
    """
    # Get paths
    project_root = Path(__file__).parent.parent
    main_script = project_root / "main.py"
    test_config = project_root / "tests" / "fixtures" / "configuration_test_end_to_end.toml"

    # Verify files exist
    assert main_script.exists(), f"main.py not found at {main_script}"
    assert test_config.exists(), f"Test config not found at {test_config}"

    # Run main.py with test configuration
    result = subprocess.run(
        [sys.executable, str(main_script), "-c", str(test_config)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=300  # 5 minutes max
    )

    # Print output for debugging if test fails
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    # Assert successful execution
    assert result.returncode == 0, (
        f"main.py exited with code {result.returncode}\n"
        f"STDERR: {result.stderr}"
    )
