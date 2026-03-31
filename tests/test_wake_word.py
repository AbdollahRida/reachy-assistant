"""
Tests for the custom wake word configuration in assistant.py.

All heavy dependencies (Porcupine, Whisper, Kokoro, etc.) are mocked via
sys.modules so these tests run without hardware, API keys, or model files.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, call


def _make_sys_mocks():
    """Return a dict of sys.modules mocks and the pvporcupine mock separately."""
    mock_pvp = MagicMock()
    mock_pvp.create.return_value = MagicMock(sample_rate=16000, frame_length=512)

    mocks = {
        'pyaudio': MagicMock(),
        'sounddevice': MagicMock(),
        'torch': MagicMock(),
        'emoji': MagicMock(),
        'numpy': MagicMock(),
        'faster_whisper': MagicMock(),
        'kokoro_onnx': MagicMock(),
        'silero_vad': MagicMock(),
        'anthropic': MagicMock(),
        'pvporcupine': mock_pvp,
        'reachy_mini': MagicMock(),
        'reachy_mini.utils': MagicMock(),
        'ddgs': MagicMock(),
        'dotenv': MagicMock(),  # prevent load_dotenv from reading .env
    }
    return mocks, mock_pvp


def _load_assistant(env_overrides: dict):
    """
    Import assistant.py fresh with mocked dependencies and the given env vars.
    WAKE_WORD_PATH and WAKE_WORD_NAME are stripped from the environment first
    so each test starts from a clean slate.
    """
    sys.modules.pop('assistant', None)
    mocks, mock_pvp = _make_sys_mocks()

    # Build a controlled env: keep the real OS env (needed for os.makedirs etc.)
    # but strip wake-word vars so tests don't bleed into each other.
    env = {k: v for k, v in os.environ.items()
           if k not in ('WAKE_WORD_PATH', 'WAKE_WORD_NAME')}
    env.update(env_overrides)

    import importlib
    from unittest.mock import patch
    with patch.dict(sys.modules, mocks), patch.dict(os.environ, env, clear=True):
        import assistant
    return assistant, mock_pvp


class TestWakeWordPorcupineInit(unittest.TestCase):

    def test_no_path_uses_builtin_keyword(self):
        """Without WAKE_WORD_PATH, Porcupine should use the built-in 'porcupine' keyword."""
        _, mock_pvp = _load_assistant({'PICOVOICE_API_KEY': 'test-key'})
        mock_pvp.create.assert_called_once_with(
            access_key='test-key',
            keywords=['porcupine']
        )

    def test_custom_path_uses_keyword_paths(self):
        """With WAKE_WORD_PATH set, Porcupine should use keyword_paths."""
        _, mock_pvp = _load_assistant({
            'PICOVOICE_API_KEY': 'test-key',
            'WAKE_WORD_PATH': '/path/to/hey_reachy.ppn',
        })
        mock_pvp.create.assert_called_once_with(
            access_key='test-key',
            keyword_paths=['/path/to/hey_reachy.ppn']
        )

    def test_custom_path_does_not_pass_keywords(self):
        """When a .ppn path is provided, the 'keywords' arg must not be passed."""
        _, mock_pvp = _load_assistant({
            'PICOVOICE_API_KEY': 'test-key',
            'WAKE_WORD_PATH': '/path/to/hey_reachy.ppn',
        })
        _, kwargs = mock_pvp.create.call_args
        self.assertNotIn('keywords', kwargs)

    def test_no_path_does_not_pass_keyword_paths(self):
        """Without a .ppn path, the 'keyword_paths' arg must not be passed."""
        _, mock_pvp = _load_assistant({'PICOVOICE_API_KEY': 'test-key'})
        _, kwargs = mock_pvp.create.call_args
        self.assertNotIn('keyword_paths', kwargs)


class TestWakeWordName(unittest.TestCase):

    def test_name_defaults_to_porcupine(self):
        """WAKE_WORD_NAME should default to 'Porcupine' when not set."""
        assistant, _ = _load_assistant({'PICOVOICE_API_KEY': 'test-key'})
        self.assertEqual(assistant.WAKE_WORD_NAME, 'Porcupine')

    def test_name_reads_from_env(self):
        """WAKE_WORD_NAME should reflect whatever is set in the environment."""
        assistant, _ = _load_assistant({
            'PICOVOICE_API_KEY': 'test-key',
            'WAKE_WORD_NAME': 'Hey Reachy',
        })
        self.assertEqual(assistant.WAKE_WORD_NAME, 'Hey Reachy')

    def test_name_independent_of_path(self):
        """WAKE_WORD_NAME and WAKE_WORD_PATH are independent config vars."""
        assistant, _ = _load_assistant({
            'PICOVOICE_API_KEY': 'test-key',
            'WAKE_WORD_PATH': '/path/to/hey_reachy.ppn',
            'WAKE_WORD_NAME': 'Hey Reachy',
        })
        self.assertEqual(assistant.WAKE_WORD_NAME, 'Hey Reachy')


if __name__ == '__main__':
    unittest.main()
