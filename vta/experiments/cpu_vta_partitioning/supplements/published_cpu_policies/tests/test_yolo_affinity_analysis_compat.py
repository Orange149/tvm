"""The read-only compatibility accepts exactly the real collection layout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parents[1]/'src')]
from run_yolo_affinity_check_analysis_compat import raw_scope_valid


def test_real_collected_directory_and_wrong_task_boot_or_generation(tmp_path):
    expected = tmp_path/'raw'/'boot'/'task'
    assert raw_scope_valid(expected/'transfers'/'123'/'collected', expected)
    for path in (expected, expected/'transfers'/'bad'/'collected',
                 expected/'transfers'/'123'/'extra'/'collected',
                 expected.parent/'wrong-task'/'transfers'/'123'/'collected',
                 tmp_path/'raw'/'wrong-boot'/'task'/'transfers'/'123'/'collected'):
        assert not raw_scope_valid(path, expected)


def test_symlink_escape_rejected(tmp_path):
    expected = tmp_path/'raw'/'boot'/'task'
    expected.mkdir(parents=True)
    outside = tmp_path/'outside'; outside.mkdir()
    (expected/'transfers').symlink_to(outside, target_is_directory=True)
    assert not raw_scope_valid(expected/'transfers'/'123'/'collected', expected)
