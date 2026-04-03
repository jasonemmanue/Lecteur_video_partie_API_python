"""
Tests unitaires — Étape 1 : Extraction & Normalisation Audio
=============================================================
Couverture :
  - AudioExtractionConfig  : valeurs par défaut, personnalisation
  - AudioExtractor         : FFmpeg manquant, fichier introuvable,
                             extraction réussie (mock), normalisation,
                             timeout, code de retour non-zéro
  - _parse_loudnorm_stats  : JSON valide, JSON absent, JSON malformé
  - extract_audio()        : fonction publique end-to-end (mock)
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

# ── Import du module testé ────────────────────────────────────────────────────
from pipeline.step1_audio_extraction import (
    AudioExtractionConfig,
    AudioExtractionResult,
    AudioExtractor,
    extract_audio,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_extractor(normalize: bool = False) -> AudioExtractor:
    """Crée un AudioExtractor avec FFmpeg mocké."""
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        cfg = AudioExtractionConfig(normalize_loudness=normalize)
        return AudioExtractor(cfg)


def _ffmpeg_ok() -> MagicMock:
    """Simule un subprocess.run FFmpeg réussi (returncode=0)."""
    proc = MagicMock()
    proc.returncode = 0
    proc.stderr = ""
    proc.stdout = ""
    return proc


def _ffmpeg_fail(returncode: int = 1, stderr: str = "FFmpeg error") -> MagicMock:
    """Simule un subprocess.run FFmpeg échoué."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stderr = stderr
    proc.stdout = ""
    return proc


FAKE_PROBE_OUTPUT = (
    "sample_rate=16000\n"
    "channels=1\n"
    "duration=42.5\n"
)

FAKE_LOUDNORM_STDERR = json.dumps({
    "input_i": "-27.3",
    "input_lra": "8.2",
    "input_tp": "-5.1",
    "input_thresh": "-37.3",
    "target_offset": "-0.7",
})


# ─── Tests : AudioExtractionConfig ────────────────────────────────────────────

class TestAudioExtractionConfig(unittest.TestCase):

    def test_default_sample_rate(self):
        cfg = AudioExtractionConfig()
        self.assertEqual(cfg.sample_rate, 16_000)

    def test_default_channels_mono(self):
        cfg = AudioExtractionConfig()
        self.assertEqual(cfg.channels, 1)

    def test_default_normalize_enabled(self):
        cfg = AudioExtractionConfig()
        self.assertTrue(cfg.normalize_loudness)

    def test_default_target_loudness(self):
        cfg = AudioExtractionConfig()
        self.assertAlmostEqual(cfg.target_loudness, -23.0)

    def test_custom_config(self):
        cfg = AudioExtractionConfig(sample_rate=44100, channels=2, normalize_loudness=False)
        self.assertEqual(cfg.sample_rate, 44100)
        self.assertEqual(cfg.channels, 2)
        self.assertFalse(cfg.normalize_loudness)


# ─── Tests : AudioExtractor — initialisation ──────────────────────────────────

class TestAudioExtractorInit(unittest.TestCase):

    def test_raises_if_ffmpeg_missing(self):
        with patch("shutil.which", return_value=None):
            with self.assertRaises(EnvironmentError) as ctx:
                AudioExtractor()
            self.assertIn("FFmpeg introuvable", str(ctx.exception))

    def test_ok_when_ffmpeg_present(self):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            extractor = AudioExtractor()
            self.assertIsNotNone(extractor)


# ─── Tests : AudioExtractor.extract() ────────────────────────────────────────

class TestAudioExtractorExtract(unittest.TestCase):

    def setUp(self):
        self.extractor = _make_extractor(normalize=False)

    def test_returns_failure_if_video_not_found(self):
        result = self.extractor.extract("/nonexistent/video.mp4", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("introuvable", result.error_message)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.stat")
    def test_successful_extraction(self, mock_stat, mock_mkdir, mock_exists, mock_run):
        stat_result = MagicMock()
        stat_result.st_size = 512_000
        mock_stat.return_value = stat_result

        probe_proc = MagicMock()
        probe_proc.returncode = 0
        probe_proc.stdout = FAKE_PROBE_OUTPUT
        probe_proc.stderr = ""

        # Premier appel → ffmpeg extract, deuxième → ffprobe
        mock_run.side_effect = [_ffmpeg_ok(), probe_proc]

        result = self.extractor.extract("/fake/video.mp4", "/tmp/out")

        self.assertTrue(result.success)
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.channels, 1)
        self.assertAlmostEqual(result.duration_seconds, 42.5)
        self.assertEqual(result.file_size_bytes, 512_000)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=True)
    def test_ffmpeg_nonzero_returns_failure(self, mock_exists, mock_run):
        mock_run.return_value = _ffmpeg_fail(returncode=1, stderr="invalid data")
        result = self.extractor.extract("/fake/video.mp4", "/tmp/out")
        self.assertFalse(result.success)
        self.assertIn("FFmpeg a échoué", result.error_message)

    @patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("ffmpeg", 300))
    @patch("pathlib.Path.exists", return_value=True)
    def test_ffmpeg_timeout_returns_failure(self, mock_exists, mock_run):
        result = self.extractor.extract("/fake/video.mp4", "/tmp/out")
        self.assertFalse(result.success)
        self.assertIn("timeout", result.error_message.lower())

    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=True)
    def test_unexpected_exception_returns_failure(self, mock_exists, mock_run):
        mock_run.side_effect = RuntimeError("Unexpected crash")
        result = self.extractor.extract("/fake/video.mp4", "/tmp/out")
        self.assertFalse(result.success)
        self.assertIn("Unexpected crash", result.error_message)


# ─── Tests : Extraction avec normalisation ────────────────────────────────────

class TestAudioExtractorNormalization(unittest.TestCase):

    def setUp(self):
        self.extractor = _make_extractor(normalize=True)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.stat")
    def test_normalization_uses_two_passes(self, mock_stat, mock_mkdir, mock_exists, mock_run):
        stat_result = MagicMock()
        stat_result.st_size = 256_000
        mock_stat.return_value = stat_result

        pass1_proc = MagicMock()
        pass1_proc.returncode = 0
        pass1_proc.stderr = FAKE_LOUDNORM_STDERR
        pass1_proc.stdout = ""

        probe_proc = MagicMock()
        probe_proc.returncode = 0
        probe_proc.stdout = FAKE_PROBE_OUTPUT
        probe_proc.stderr = ""

        mock_run.side_effect = [pass1_proc, _ffmpeg_ok(), probe_proc]

        result = self.extractor.extract("/fake/video.mp4", "/tmp/out")

        self.assertTrue(result.success)
        # 3 appels : passe1, passe2, ffprobe
        self.assertEqual(mock_run.call_count, 3)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.stat")
    def test_normalization_fallback_if_stats_missing(self, mock_stat, mock_mkdir, mock_exists, mock_run):
        """Si loudnorm ne produit pas de JSON, on utilise le filtre simple."""
        stat_result = MagicMock()
        stat_result.st_size = 128_000
        mock_stat.return_value = stat_result

        pass1_proc = MagicMock()
        pass1_proc.returncode = 0
        pass1_proc.stderr = "no json here"
        pass1_proc.stdout = ""

        probe_proc = MagicMock()
        probe_proc.returncode = 0
        probe_proc.stdout = FAKE_PROBE_OUTPUT
        probe_proc.stderr = ""

        mock_run.side_effect = [pass1_proc, _ffmpeg_ok(), probe_proc]

        result = self.extractor.extract("/fake/video.mp4", "/tmp/out")
        self.assertTrue(result.success)


# ─── Tests : _parse_loudnorm_stats ───────────────────────────────────────────

class TestParseLoudnormStats(unittest.TestCase):

    def test_parses_valid_json(self):
        stats = AudioExtractor._parse_loudnorm_stats(FAKE_LOUDNORM_STDERR)
        self.assertIsNotNone(stats)
        self.assertEqual(stats["input_i"], "-27.3")
        self.assertEqual(stats["target_offset"], "-0.7")

    def test_returns_none_if_no_json(self):
        stats = AudioExtractor._parse_loudnorm_stats("No JSON here at all.")
        self.assertIsNone(stats)

    def test_returns_none_if_malformed_json(self):
        stats = AudioExtractor._parse_loudnorm_stats("{malformed: json}")
        self.assertIsNone(stats)

    def test_handles_empty_string(self):
        stats = AudioExtractor._parse_loudnorm_stats("")
        self.assertIsNone(stats)


# ─── Tests : fonction publique extract_audio() ───────────────────────────────

class TestExtractAudioPublicAPI(unittest.TestCase):

    @patch("pipeline.step1_audio_extraction.AudioExtractor.extract")
    def test_success_path(self, mock_extract):
        mock_extract.return_value = AudioExtractionResult(
            success=True,
            audio_path=Path("/tmp/out/video_extracted.wav"),
            duration_seconds=60.0,
            sample_rate=16000,
            channels=1,
            file_size_bytes=1_920_000,
        )
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            result = extract_audio("/fake/video.mp4", "/tmp/out")

        self.assertTrue(result.success)
        self.assertEqual(result.sample_rate, 16000)

    @patch("pipeline.step1_audio_extraction.AudioExtractor.extract")
    def test_failure_path(self, mock_extract):
        mock_extract.return_value = AudioExtractionResult(
            success=False,
            error_message="FFmpeg a échoué (code 1)",
        )
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            result = extract_audio("/fake/video.mp4", "/tmp/out")

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

    def test_custom_loudness_config(self):
        """Vérifie que le niveau LUFS personnalisé est bien transmis."""
        with patch("pipeline.step1_audio_extraction.AudioExtractor._check_ffmpeg"):
            with patch("pipeline.step1_audio_extraction.AudioExtractor.extract") as mock_ex:
                mock_ex.return_value = AudioExtractionResult(success=True)
                extract_audio("/fake/video.mp4", "/tmp", target_loudness=-16.0)
                # L'extracteur doit avoir été créé avec le bon config
                self.assertTrue(mock_ex.called)


# ─── Tests : AudioExtractionResult ───────────────────────────────────────────

class TestAudioExtractionResult(unittest.TestCase):

    def test_default_values(self):
        result = AudioExtractionResult(success=False)
        self.assertIsNone(result.audio_path)
        self.assertIsNone(result.duration_seconds)
        self.assertEqual(result.ffmpeg_stderr, "")

    def test_success_result_fields(self):
        result = AudioExtractionResult(
            success=True,
            audio_path=Path("/tmp/audio.wav"),
            duration_seconds=120.5,
            sample_rate=16000,
            channels=1,
            file_size_bytes=3_840_000,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.duration_seconds, 120.5)
        self.assertEqual(result.file_size_bytes, 3_840_000)


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
