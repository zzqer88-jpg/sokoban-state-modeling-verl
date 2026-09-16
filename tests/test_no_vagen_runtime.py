from pathlib import Path

from sokoban_state_modeling.scripts.preflight_training import check_no_vagen_runtime


def test_runtime_source_and_dependencies_do_not_import_vagen():
    project_root = Path(__file__).parents[1]
    check_no_vagen_runtime(project_root)
