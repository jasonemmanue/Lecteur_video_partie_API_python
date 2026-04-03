"""
Tests unitaires — Étape 2 : Transcription Speech-to-Text (Whisper)
==================================================================
Couverture :
  - TranscriptionConfig    : valeurs par défaut, personnalisation
  - TranscriptionSegment   : duration, to_dict(), to_srt_block()
  - TranscriptionResult    : full_text, segment_count
  - WhisperTranscriber     : fichier manquant, import manquant,
                             transcription réussie (mock complet),
                             sauvegarde JSON, sauvegarde SRT,
                             détection de langue, gestion d'erreur
  - transcribe_audio()     : fonction publique end-to-end (mock)
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from pipeline.step2_transcription import (
    TranscriptionConfig,
    TranscriptionSegment,
    TranscriptionResult,
    WhisperTranscriber,
    transcribe_audio,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_segment(
    id: int = 0,
    start: float = 0.0,
    end: float = 3.5,
    text: str = "Hello world.",
    language: str = "en",
    confidence: float = 0.95,
    words: list | None = None,
) -> TranscriptionSegment:
    return TranscriptionSegment(
        id=id, start=start, end=end,
        text=text, language=language,
        confidence=confidence, words=words or [],
    )


def _make_whisper_segment(
    id=0, start=0.0, end=3.5,
    text=" Hello world.", words=None
) -> MagicMock:
    """Simule un segment retourné par faster-whisper."""
    seg = MagicMock()
    seg.id    = id
    seg.start = start
    seg.end   = end
    seg.text  = text
    seg.words = words or []
    return seg


def _make_whisper_info(language="en", probability=0.98, duration=42.0) -> MagicMock:
    info = MagicMock()
    info.language             = language
    info.language_probability = probability
    info.duration             = duration
    return info


def _make_transcriber() -> WhisperTranscriber:
    """Crée un WhisperTranscriber sans charger de vrai modèle."""
    t = WhisperTranscriber()
    t._model = MagicMock()  # modèle déjà "chargé"
    return t


# ─── Tests : TranscriptionConfig ─────────────────────────────────────────────

class TestTranscriptionConfig(unittest.TestCase):

    def test_default_model_size(self):
        cfg = TranscriptionConfig()
        self.assertEqual(cfg.model_size, "medium")

    def test_default_device_cpu(self):
        cfg = TranscriptionConfig()
        self.assertEqual(cfg.device, "cpu")

    def test_default_language_none(self):
        cfg = TranscriptionConfig()
        self.assertIsNone(cfg.language)

    def test_default_word_timestamps(self):
        cfg = TranscriptionConfig()
        self.assertTrue(cfg.word_timestamps)

    def test_default_vad_filter(self):
        cfg = TranscriptionConfig()
        self.assertTrue(cfg.vad_filter)

    def test_custom_config(self):
        cfg = TranscriptionConfig(
            model_size="large-v2",
            device="cuda",
            language="fr",
            word_timestamps=False,
        )
        self.assertEqual(cfg.model_size, "large-v2")
        self.assertEqual(cfg.device, "cuda")
        self.assertEqual(cfg.language, "fr")
        self.assertFalse(cfg.word_timestamps)


# ─── Tests : TranscriptionSegment ────────────────────────────────────────────

class TestTranscriptionSegment(unittest.TestCase):

    def test_duration_calculated_correctly(self):
        seg = _make_segment(start=1.0, end=4.5)
        self.assertAlmostEqual(seg.duration, 3.5)

    def test_duration_rounded_to_3_decimals(self):
        seg = _make_segment(start=0.001, end=3.456789)
        self.assertEqual(seg.duration, round(3.456789 - 0.001, 3))

    def test_to_dict_contains_required_keys(self):
        seg = _make_segment()
        d = seg.to_dict()
        for key in ("id", "start", "end", "text", "language", "confidence", "duration", "words"):
            self.assertIn(key, d)

    def test_to_dict_text_stripped(self):
        seg = _make_segment(text="  Hello world.  ")
        self.assertEqual(seg.to_dict()["text"], "Hello world.")

    def test_to_dict_confidence_rounded(self):
        seg = _make_segment(confidence=0.987654321)
        self.assertEqual(len(str(seg.to_dict()["confidence"]).split(".")[-1]), 4)

    def test_to_srt_block_format(self):
        seg = _make_segment(id=0, start=0.0, end=3.5, text="Hello world.")
        srt = seg.to_srt_block()
        lines = srt.strip().split("\n")
        self.assertEqual(lines[0], "1")              # numéro (id+1)
        self.assertIn("-->", lines[1])               # timecode
        self.assertEqual(lines[2], "Hello world.")   # texte

    def test_to_srt_timecode_format(self):
        seg = _make_segment(start=3661.5, end=3665.0)
        srt = seg.to_srt_block()
        timecode_line = srt.split("\n")[1]
        self.assertIn("01:01:01,500", timecode_line)

    def test_to_srt_block_id_incremented(self):
        seg = _make_segment(id=4)
        self.assertTrue(seg.to_srt_block().startswith("5\n"))


# ─── Tests : TranscriptionResult ─────────────────────────────────────────────

class TestTranscriptionResult(unittest.TestCase):

    def test_full_text_joins_segments(self):
        segs = [
            _make_segment(id=0, text="Hello"),
            _make_segment(id=1, text="world."),
        ]
        result = TranscriptionResult(success=True, segments=segs)
        self.assertEqual(result.full_text, "Hello world.")

    def test_full_text_strips_whitespace(self):
        segs = [_make_segment(text="  Hello  "), _make_segment(text="  world.  ")]
        result = TranscriptionResult(success=True, segments=segs)
        self.assertEqual(result.full_text, "Hello world.")

    def test_segment_count(self):
        segs = [_make_segment(id=i) for i in range(5)]
        result = TranscriptionResult(success=True, segments=segs)
        self.assertEqual(result.segment_count, 5)

    def test_empty_segments(self):
        result = TranscriptionResult(success=True)
        self.assertEqual(result.full_text, "")
        self.assertEqual(result.segment_count, 0)

    def test_failure_result_defaults(self):
        result = TranscriptionResult(success=False, error_message="boom")
        self.assertFalse(result.success)
        self.assertIsNone(result.json_path)
        self.assertIsNone(result.srt_path)


# ─── Tests : WhisperTranscriber — init & chargement modèle ───────────────────

class TestWhisperTranscriberInit(unittest.TestCase):

    def test_model_not_loaded_at_init(self):
        t = WhisperTranscriber()
        self.assertIsNone(t._model)

    def test_load_model_raises_if_faster_whisper_missing(self):
        t = WhisperTranscriber()
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with self.assertRaises(ImportError) as ctx:
                t._load_model()
            self.assertIn("faster-whisper", str(ctx.exception))

    def test_load_model_called_once(self):
        t = WhisperTranscriber()
        mock_model = MagicMock()
        mock_whisper_module = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            t._load_model()
            t._load_model()  # deuxième appel — ne doit pas recharger

        mock_whisper_module.WhisperModel.assert_called_once()


# ─── Tests : WhisperTranscriber.transcribe() ─────────────────────────────────

class TestWhisperTranscriberTranscribe(unittest.TestCase):

    def test_returns_failure_if_audio_not_found(self):
        t = _make_transcriber()
        result = t.transcribe("/nonexistent/audio.wav", "/tmp/out")
        self.assertFalse(result.success)
        self.assertIn("introuvable", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_returns_failure_on_unexpected_exception(self, mock_mkdir, mock_exists):
        t = _make_transcriber()
        t._model.transcribe.side_effect = RuntimeError("GPU crash")
        result = t.transcribe("/fake/audio.wav", "/tmp/out")
        self.assertFalse(result.success)
        self.assertIn("GPU crash", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_successful_transcription(self, mock_write, mock_mkdir, mock_exists):
        t = _make_transcriber()

        seg1 = _make_whisper_segment(id=0, start=0.0,  end=3.5,  text=" Hello world.")
        seg2 = _make_whisper_segment(id=1, start=3.5,  end=7.0,  text=" How are you?")
        info = _make_whisper_info(language="en", probability=0.98, duration=7.0)

        t._model.transcribe.return_value = (iter([seg1, seg2]), info)

        result = t.transcribe("/fake/audio.wav", "/tmp/out")

        self.assertTrue(result.success)
        self.assertEqual(result.language, "en")
        self.assertAlmostEqual(result.language_probability, 0.98)
        self.assertEqual(result.segment_count, 2)
        self.assertAlmostEqual(result.duration_seconds, 7.0)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_detects_french_language(self, mock_write, mock_mkdir, mock_exists):
        t = _make_transcriber()
        seg  = _make_whisper_segment(text=" Bonjour tout le monde.")
        info = _make_whisper_info(language="fr", probability=0.99)
        t._model.transcribe.return_value = (iter([seg]), info)

        result = t.transcribe("/fake/audio.wav", "/tmp/out")
        self.assertEqual(result.language, "fr")

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_json_and_srt_paths_set(self, mock_write, mock_mkdir, mock_exists):
        t = _make_transcriber()
        seg  = _make_whisper_segment()
        info = _make_whisper_info()
        t._model.transcribe.return_value = (iter([seg]), info)

        result = t.transcribe("/fake/audio.wav", "/tmp/out")

        self.assertIsNotNone(result.json_path)
        self.assertIsNotNone(result.srt_path)
        self.assertTrue(str(result.json_path).endswith("_transcript.json"))
        self.assertTrue(str(result.srt_path).endswith("_transcript.srt"))

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_write_text_called_twice(self, mock_write, mock_mkdir, mock_exists):
        """Vérifie que JSON et SRT sont bien écrits (2 appels write_text)."""
        t = _make_transcriber()
        seg  = _make_whisper_segment()
        info = _make_whisper_info()
        t._model.transcribe.return_value = (iter([seg]), info)

        t.transcribe("/fake/audio.wav", "/tmp/out")
        self.assertEqual(mock_write.call_count, 2)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_word_timestamps_included(self, mock_write, mock_mkdir, mock_exists):
        """Vérifie que les timestamps mot à mot sont bien inclus."""
        t = _make_transcriber()
        t.config.word_timestamps = True

        word = MagicMock()
        word.word        = "Hello"
        word.start       = 0.0
        word.end         = 0.5
        word.probability = 0.99

        seg  = _make_whisper_segment(words=[word])
        info = _make_whisper_info()
        t._model.transcribe.return_value = (iter([seg]), info)

        result = t.transcribe("/fake/audio.wav", "/tmp/out")
        self.assertTrue(result.success)
        self.assertEqual(len(result.segments[0].words), 1)
        self.assertEqual(result.segments[0].words[0]["word"], "Hello")


# ─── Tests : _save_json ───────────────────────────────────────────────────────

class TestSaveJson(unittest.TestCase):

    def test_json_structure(self):
        segs = [_make_segment(id=0, text="Hello.")]
        captured = {}

        def fake_write(self_path, content, **kwargs):
            captured["content"] = content

        with patch.object(Path, "write_text", fake_write):
            WhisperTranscriber._save_json(segs, "en", 0.98, 42.0, Path("/tmp/test.json"))

        data = json.loads(captured["content"])
        self.assertEqual(data["language"], "en")
        self.assertAlmostEqual(data["language_probability"], 0.98)
        self.assertEqual(data["segment_count"], 1)
        self.assertIn("segments", data)
        self.assertEqual(data["segments"][0]["text"], "Hello.")

    def test_json_utf8_encoding(self):
        """Les caractères non-ASCII doivent être préservés (ensure_ascii=False)."""
        segs = [_make_segment(text="Bonjour le monde, ça va ?")]
        captured = {}

        def fake_write(self_path, content, **kwargs):
            captured["content"] = content

        with patch.object(Path, "write_text", fake_write):
            WhisperTranscriber._save_json(segs, "fr", 0.99, 10.0, Path("/tmp/test.json"))

        self.assertIn("ça", captured["content"])


# ─── Tests : _save_srt ────────────────────────────────────────────────────────

class TestSaveSrt(unittest.TestCase):

    def test_srt_contains_all_segments(self):
        segs = [_make_segment(id=i, text=f"Segment {i}.") for i in range(3)]
        captured = {}

        def fake_write(self_path, content, **kwargs):
            captured["content"] = content

        with patch.object(Path, "write_text", fake_write):
            WhisperTranscriber._save_srt(segs, Path("/tmp/test.srt"))

        content = captured["content"]
        for i in range(3):
            self.assertIn(f"Segment {i}.", content)

    def test_srt_timecodes_present(self):
        segs = [_make_segment(start=0.0, end=3.5)]
        captured = {}

        def fake_write(self_path, content, **kwargs):
            captured["content"] = content

        with patch.object(Path, "write_text", fake_write):
            WhisperTranscriber._save_srt(segs, Path("/tmp/test.srt"))

        self.assertIn("-->", captured["content"])


# ─── Tests : fonction publique transcribe_audio() ────────────────────────────

class TestTranscribeAudioPublicAPI(unittest.TestCase):

    @patch("pipeline.step2_transcription.WhisperTranscriber.transcribe")
    def test_success_path(self, mock_transcribe):
        segs = [_make_segment(text="Test segment.")]
        mock_transcribe.return_value = TranscriptionResult(
            success=True,
            segments=segs,
            language="en",
            language_probability=0.97,
            duration_seconds=10.0,
            json_path=Path("/tmp/out/audio_transcript.json"),
            srt_path=Path("/tmp/out/audio_transcript.srt"),
        )
        result = transcribe_audio("/fake/audio.wav", "/tmp/out")
        self.assertTrue(result.success)
        self.assertEqual(result.language, "en")
        self.assertEqual(result.segment_count, 1)

    @patch("pipeline.step2_transcription.WhisperTranscriber.transcribe")
    def test_failure_path(self, mock_transcribe):
        mock_transcribe.return_value = TranscriptionResult(
            success=False,
            error_message="Modèle introuvable",
        )
        result = transcribe_audio("/fake/audio.wav", "/tmp/out")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

    @patch("pipeline.step2_transcription.WhisperTranscriber.transcribe")
    def test_custom_model_size_passed(self, mock_transcribe):
        mock_transcribe.return_value = TranscriptionResult(success=True)
        transcribe_audio("/fake/audio.wav", "/tmp/out", model_size="large-v2")
        mock_transcribe.assert_called_once()

    @patch("pipeline.step2_transcription.WhisperTranscriber.transcribe")
    def test_language_passed_to_config(self, mock_transcribe):
        mock_transcribe.return_value = TranscriptionResult(success=True)
        transcribe_audio("/fake/audio.wav", "/tmp/out", language="fr")
        mock_transcribe.assert_called_once()


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
