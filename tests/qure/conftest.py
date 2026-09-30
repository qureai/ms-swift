"""pytest collection guard for tests/qure (review BC-3).

These tests need this host's stage-6b checkpoint and training data. Where either is missing (another machine, CI),
the whole directory is left out of collection instead of failing, so `pytest tests/` elsewhere is unaffected. Each
test module also applies its process env (CUDA_VISIBLE_DEVICES='', IMAGE_MAX_TOKEN_NUM, QURE_PLUGIN_MODE, ...) only
around its own tests (module-scoped fixture, restored afterwards) -- never at import.

    SEM_TEST_CKPT / QURE_TEST_CKPT   checkpoint the tests use (default: stage-6b checkpoint-608)
    QURE_TESTS_FORCE=1               collect even when the checkpoint / data are missing
"""
import os

_CKPT = (os.environ.get('SEM_TEST_CKPT') or os.environ.get('QURE_TEST_CKPT')
         or '/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/output/'
         'Qwen3.5-2B_stage6b_alltools/v0-20260906-173237/checkpoint-608')
_DATA = '/fastdata4/sahil_work/VLM_qure/data_preparation/training_data'
_MISSING = [p for p in (_CKPT, _DATA) if not os.path.isdir(p)]

collect_ignore_glob = ['test_*.py'] if (_MISSING and os.environ.get('QURE_TESTS_FORCE') != '1') else []


def pytest_report_header(config):
    if collect_ignore_glob:
        return f'tests/qure: SKIPPED (missing {_MISSING}; QURE_TESTS_FORCE=1 to collect anyway)'
    return None
