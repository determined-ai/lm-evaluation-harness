"""
Tests for Determined-specific changes which were made.
"""


from lm_eval import tasks
from main import pattern_match


def test_pattern_match_single():
    task_names = pattern_match(["arc_easy"], tasks.ALL_TASKS)
    assert task_names
    assert len(task_names) == 1


def test_pattern_match_glob():
    task_names = pattern_match(["hendrycksTest-*"], tasks.ALL_TASKS)
    assert task_names
    assert len(task_names) == 57
