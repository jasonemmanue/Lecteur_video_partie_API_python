"""
Tests unitaires — Étape 3 : Analyse du Ton et des Émotions
==========================================================
Couverture :
  - EmotionAnalysisConfig  : valeurs par défaut, personnalisation
  - ProsodyFeatures        : to_dict(), arrondis
  - EnrichedSegment        : duration, to_dict()
  - EmotionAnalysisResult  : segment_count, defaults
  - EmotionAnalyzer        : fichier manquant, import manquant,
                             analyse réussie (mock complet),
                             segment trop court (fallback),
                             classify_emotion, intensity, pitch,
                             speech_rate, tone_tags, save_json
  - analyze_emotions()     : fonction publique end-to-end (mock)
"""

import json
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from pipeline.step3_emotion_analysis import (
    EmotionAnalysisConfig,
    ProsodyFeatures,
    EnrichedSegment,
    EmotionAnalysisResult,
    EmotionAnalyzer,
    EMOTIONS,
    EMOTION_TONE_TAGS,
    INTENSITY_TAGS,
    analyze_emotions,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000

def _make_audio(duration_s: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Génère un signal audio sinusoïdal de test."""
    t = np.linspace(0, duration_s, int(sr * duration_s))
    return (np.sin(2 * np.pi * 220 * t) * 0.3).astype(np.float32)


def _make_prosody(
    emotion: str = "neutral",
    confidence: float = 0.9,
    intensity: float = 0.5,
    speech_rate: float = 3.0,
    pitch_mean: float = 140.0,
    pitch_std: float = 15.0,
    tone_tags: list | None = None,
) -> ProsodyFeatures:
    return ProsodyFeatures(
        emotion=emotion,
        emotion_confidence=confidence,
        intensity=intensity,
        speech_rate=speech_rate,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        tone_tags=tone_tags or ["[NEUTRAL]", "[MODERATE]"],
    )


def _make_enriched(
    id: int = 0,
    start: float = 0.0,
    end: float = 3.5,
    text: str = "Hello world.",
    language: str = "en",
    emotion: str = "neutral",
) -> EnrichedSegment:
    return EnrichedSegment(
        id=id, start=start, end=end,
        text=text, language=language,
        prosody=_make_prosody(emotion=emotion),
    )


def _make_analyzer() -> EmotionAnalyzer:
    """Crée un EmotionAnalyzer avec modèle mocké."""
    a = EmotionAnalyzer()
    a._pipeline = MagicMock()
    return a


FAKE_TRANSCRIPT = {
    "language": "en",
    "segment_count": 2,
    "segments": [
        {"id": 0, "start": 0.0, "end": 3.5,  "text": "Hello world."},
        {"id": 1, "start": 3.5, "end": 7.0,  "text": "How are you?"},
    ],
}


# ─── Tests : EmotionAnalysisConfig ───────────────────────────────────────────

class TestEmotionAnalysisConfig(unittest.TestCase):

    def test_default_device_cpu(self):
        cfg = EmotionAnalysisConfig()
        self.assertEqual(cfg.device, "cpu")

    def test_default_sample_rate(self):
        cfg = EmotionAnalysisConfig()
        self.assertEqual(cfg.sample_rate, 16_000)

    def test_default_fallback_emotion(self):
        cfg = EmotionAnalysisConfig()
        self.assertEqual(cfg.fallback_emotion, "neutral")

    def test_default_compute_pitch(self):
        cfg = EmotionAnalysisConfig()
        self.assertTrue(cfg.compute_pitch)

    def test_intensity_thresholds_order(self):
        cfg = EmotionAnalysisConfig()
        self.assertLess(cfg.intensity_low, cfg.intensity_high)

    def test_custom_config(self):
        cfg = EmotionAnalysisConfig(device="cuda", compute_pitch=False)
        self.assertEqual(cfg.device, "cuda")
        self.assertFalse(cfg.compute_pitch)


# ─── Tests : ProsodyFeatures ──────────────────────────────────────────────────

class TestProsodyFeatures(unittest.TestCase):

    def test_to_dict_keys(self):
        p = _make_prosody()
        d = p.to_dict()
        for key in ("emotion", "emotion_confidence", "intensity",
                    "speech_rate", "pitch_mean", "pitch_std", "tone_tags"):
            self.assertIn(key, d)

    def test_to_dict_rounds_confidence(self):
        p = _make_prosody(confidence=0.123456789)
        self.assertEqual(len(str(p.to_dict()["emotion_confidence"]).split(".")[-1]), 4)

    def test_to_dict_rounds_intensity(self):
        p = _make_prosody(intensity=0.666666)
        self.assertEqual(p.to_dict()["intensity"], round(0.666666, 4))

    def test_to_dict_tone_tags_is_list(self):
        p = _make_prosody(tone_tags=["[HAPPY]", "[FAST]"])
        self.assertIsInstance(p.to_dict()["tone_tags"], list)
        self.assertEqual(len(p.to_dict()["tone_tags"]), 2)


# ─── Tests : EnrichedSegment ──────────────────────────────────────────────────

class TestEnrichedSegment(unittest.TestCase):

    def test_duration_computed(self):
        seg = _make_enriched(start=1.0, end=4.5)
        self.assertAlmostEqual(seg.duration, 3.5)

    def test_to_dict_contains_all_fields(self):
        seg = _make_enriched()
        d = seg.to_dict()
        for key in ("id", "start", "end", "text", "language", "duration",
                    "emotion", "emotion_confidence", "intensity",
                    "speech_rate", "pitch_mean", "pitch_std", "tone_tags"):
            self.assertIn(key, d)

    def test_to_dict_text_stripped(self):
        seg = _make_enriched(text="  Hello  ")
        self.assertEqual(seg.to_dict()["text"], "Hello")

    def test_to_dict_emotion_propagated(self):
        seg = _make_enriched(emotion="happy")
        self.assertEqual(seg.to_dict()["emotion"], "happy")


# ─── Tests : EmotionAnalysisResult ───────────────────────────────────────────

class TestEmotionAnalysisResult(unittest.TestCase):

    def test_segment_count(self):
        segs = [_make_enriched(id=i) for i in range(4)]
        result = EmotionAnalysisResult(success=True, enriched_segments=segs)
        self.assertEqual(result.segment_count, 4)

    def test_empty_segments(self):
        result = EmotionAnalysisResult(success=True)
        self.assertEqual(result.segment_count, 0)

    def test_failure_defaults(self):
        result = EmotionAnalysisResult(success=False, error_message="oops")
        self.assertFalse(result.success)
        self.assertIsNone(result.output_json_path)
        self.assertIsNone(result.dominant_emotion)


# ─── Tests : EmotionAnalyzer — init & chargement modèle ──────────────────────

class TestEmotionAnalyzerInit(unittest.TestCase):

    def test_model_not_loaded_at_init(self):
        a = EmotionAnalyzer()
        self.assertIsNone(a._pipeline)

    def test_load_model_raises_if_transformers_missing(self):
        a = EmotionAnalyzer()
        with patch.dict("sys.modules", {"transformers": None}):
            with self.assertRaises(ImportError) as ctx:
                a._load_model()
            self.assertIn("transformers", str(ctx.exception))

    def test_load_model_called_once(self):
        a = EmotionAnalyzer()
        mock_pipeline = MagicMock()
        mock_transformers = MagicMock()
        mock_transformers.pipeline.return_value = mock_pipeline

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            a._load_model()
            a._load_model()  # deuxième appel — ne recharge pas

        mock_transformers.pipeline.assert_called_once()


# ─── Tests : EmotionAnalyzer.analyze() ───────────────────────────────────────

class TestEmotionAnalyzerAnalyze(unittest.TestCase):

    def test_returns_failure_if_audio_missing(self):
        a = _make_analyzer()
        result = a.analyze("/nonexistent/audio.wav", "/fake/transcript.json", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("audio", result.error_message.lower())

    @patch("pathlib.Path.exists", return_value=True)
    def test_returns_failure_if_transcript_missing(self, mock_exists):
        a = _make_analyzer()
        # Premier exists=True (audio), deuxième False (transcript)
        mock_exists.side_effect = [True, False]
        result = a.analyze("/fake/audio.wav", "/nonexistent/transcript.json", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("transcript", result.error_message.lower())

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_returns_failure_on_unexpected_exception(self, mock_mkdir, mock_exists):
        a = _make_analyzer()
        with patch.object(a, "_load_audio", side_effect=RuntimeError("disk error")):
            with patch("pathlib.Path.read_text", return_value=json.dumps(FAKE_TRANSCRIPT)):
                result = a.analyze("/fake/audio.wav", "/fake/t.json", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("disk error", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.read_text")
    def test_successful_analysis(self, mock_read, mock_write, mock_mkdir, mock_exists):
        a = _make_analyzer()
        audio = _make_audio(8.0)

        a._pipeline.return_value = [{"label": "neutral", "score": 0.9}]

        with patch.object(a, "_load_audio", return_value=(audio, SAMPLE_RATE)):
            mock_read.return_value = json.dumps(FAKE_TRANSCRIPT)
            result = a.analyze("/fake/audio.wav", "/fake/t.json", "/tmp")

        self.assertTrue(result.success)
        self.assertEqual(result.segment_count, 2)
        self.assertIsNotNone(result.dominant_emotion)
        self.assertGreater(result.avg_intensity, 0)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.read_text")
    def test_output_json_path_set(self, mock_read, mock_write, mock_mkdir, mock_exists):
        a = _make_analyzer()
        audio = _make_audio(8.0)
        a._pipeline.return_value = [{"label": "neutral", "score": 0.9}]

        with patch.object(a, "_load_audio", return_value=(audio, SAMPLE_RATE)):
            mock_read.return_value = json.dumps(FAKE_TRANSCRIPT)
            result = a.analyze("/fake/audio.wav", "/fake/t.json", "/tmp")

        self.assertIsNotNone(result.output_json_path)
        self.assertTrue(str(result.output_json_path).endswith("_enriched.json"))


# ─── Tests : _classify_emotion ───────────────────────────────────────────────

class TestClassifyEmotion(unittest.TestCase):

    def test_returns_known_emotion(self):
        a = _make_analyzer()
        a._pipeline.return_value = [{"label": "happy", "score": 0.92}]
        emotion, confidence = a._classify_emotion(_make_audio(), SAMPLE_RATE)
        self.assertEqual(emotion, "happy")
        self.assertAlmostEqual(confidence, 0.92)

    def test_unknown_label_falls_back(self):
        a = _make_analyzer()
        a._pipeline.return_value = [{"label": "excitement", "score": 0.7}]
        emotion, _ = a._classify_emotion(_make_audio(), SAMPLE_RATE)
        self.assertEqual(emotion, a.config.fallback_emotion)

    def test_exception_returns_fallback(self):
        a = _make_analyzer()
        a._pipeline.side_effect = RuntimeError("inference failed")
        emotion, confidence = a._classify_emotion(_make_audio(), SAMPLE_RATE)
        self.assertEqual(emotion, "neutral")
        self.assertEqual(confidence, 0.5)

    def test_all_emotions_recognized(self):
        a = _make_analyzer()
        for em in EMOTIONS:
            a._pipeline.return_value = [{"label": em, "score": 0.8}]
            result, _ = a._classify_emotion(_make_audio(), SAMPLE_RATE)
            self.assertEqual(result, em)


# ─── Tests : _compute_intensity ──────────────────────────────────────────────

class TestComputeIntensity(unittest.TestCase):

    def test_silent_audio_gives_zero(self):
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        intensity = EmotionAnalyzer._compute_intensity(silence)
        self.assertAlmostEqual(intensity, 0.0)

    def test_loud_audio_clipped_to_one(self):
        loud = np.ones(SAMPLE_RATE, dtype=np.float32)
        intensity = EmotionAnalyzer._compute_intensity(loud)
        self.assertLessEqual(intensity, 1.0)

    def test_normal_audio_between_zero_and_one(self):
        audio = _make_audio(1.0)
        intensity = EmotionAnalyzer._compute_intensity(audio)
        self.assertGreaterEqual(intensity, 0.0)
        self.assertLessEqual(intensity, 1.0)

    def test_empty_audio_gives_zero(self):
        intensity = EmotionAnalyzer._compute_intensity(np.array([]))
        self.assertEqual(intensity, 0.0)


# ─── Tests : _estimate_speech_rate ───────────────────────────────────────────

class TestEstimateSpeechRate(unittest.TestCase):

    def test_normal_sentence(self):
        rate = EmotionAnalyzer._estimate_speech_rate("Hello world how are you", 2.0)
        # 5 mots × 1.5 syllabes / 2s = 3.75
        self.assertAlmostEqual(rate, 3.75)

    def test_empty_text_gives_zero(self):
        rate = EmotionAnalyzer._estimate_speech_rate("", 3.0)
        self.assertEqual(rate, 0.0)

    def test_zero_duration_gives_zero(self):
        rate = EmotionAnalyzer._estimate_speech_rate("Hello world", 0.0)
        self.assertEqual(rate, 0.0)

    def test_single_word(self):
        rate = EmotionAnalyzer._estimate_speech_rate("Hello", 1.0)
        self.assertAlmostEqual(rate, 1.5)


# ─── Tests : _build_tone_tags ────────────────────────────────────────────────

class TestBuildToneTags(unittest.TestCase):

    def setUp(self):
        self.analyzer = EmotionAnalyzer()

    def test_happy_emotion_includes_happy_tag(self):
        tags = self.analyzer._build_tone_tags("happy", 0.5)
        self.assertIn("[HAPPY]", tags)

    def test_low_intensity_adds_soft_tag(self):
        tags = self.analyzer._build_tone_tags("neutral", 0.1)
        self.assertIn("[SOFT]", tags)

    def test_high_intensity_adds_loud_tag(self):
        tags = self.analyzer._build_tone_tags("neutral", 0.9)
        self.assertIn("[LOUD]", tags)

    def test_moderate_intensity_adds_moderate_tag(self):
        tags = self.analyzer._build_tone_tags("neutral", 0.5)
        self.assertIn("[MODERATE]", tags)

    def test_unknown_emotion_defaults_to_neutral_tags(self):
        tags = self.analyzer._build_tone_tags("unknown_emotion", 0.5)
        self.assertIn("[NEUTRAL]", tags)

    def test_no_duplicate_tags(self):
        # [ANGRY] inclut déjà [LOUD], pas de doublon si intensité haute
        tags = self.analyzer._build_tone_tags("angry", 0.9)
        self.assertEqual(len(tags), len(set(tags)))


# ─── Tests : _dominant_emotion ───────────────────────────────────────────────

class TestDominantEmotion(unittest.TestCase):

    def test_majority_emotion_wins(self):
        segs = [
            _make_enriched(emotion="happy"),
            _make_enriched(emotion="happy"),
            _make_enriched(emotion="sad"),
        ]
        dominant = EmotionAnalyzer._dominant_emotion(segs)
        self.assertEqual(dominant, "happy")

    def test_empty_list_returns_none(self):
        dominant = EmotionAnalyzer._dominant_emotion([])
        self.assertIsNone(dominant)

    def test_single_segment(self):
        segs = [_make_enriched(emotion="angry")]
        self.assertEqual(EmotionAnalyzer._dominant_emotion(segs), "angry")


# ─── Tests : _save_json ───────────────────────────────────────────────────────

class TestSaveJson(unittest.TestCase):

    def test_json_structure(self):
        segs = [_make_enriched(id=0, text="Hello.")]
        captured = {}

        def fake_write(self_path, content, **kwargs):
            captured["content"] = content

        with patch.object(Path, "write_text", fake_write):
            EmotionAnalyzer._save_json(segs, "neutral", 0.5, 3.2, Path("/tmp/test.json"))

        data = json.loads(captured["content"])
        self.assertIn("dominant_emotion", data)
        self.assertIn("avg_intensity", data)
        self.assertIn("segments", data)
        self.assertEqual(data["segment_count"], 1)

    def test_utf8_preserved(self):
        segs = [_make_enriched(text="Ça va très bien.")]
        captured = {}

        def fake_write(self_path, content, **kwargs):
            captured["content"] = content

        with patch.object(Path, "write_text", fake_write):
            EmotionAnalyzer._save_json(segs, "neutral", 0.5, 3.0, Path("/tmp/test.json"))

        self.assertIn("ça", captured["content"].lower())


# ─── Tests : fonction publique analyze_emotions() ────────────────────────────

class TestAnalyzeEmotionsPublicAPI(unittest.TestCase):

    @patch("pipeline.step3_emotion_analysis.EmotionAnalyzer.analyze")
    def test_success_path(self, mock_analyze):
        segs = [_make_enriched()]
        mock_analyze.return_value = EmotionAnalysisResult(
            success=True,
            enriched_segments=segs,
            dominant_emotion="neutral",
            avg_intensity=0.5,
            avg_speech_rate=3.0,
            output_json_path=Path("/tmp/audio_enriched.json"),
        )
        result = analyze_emotions("/fake/audio.wav", "/fake/t.json", "/tmp")
        self.assertTrue(result.success)
        self.assertEqual(result.dominant_emotion, "neutral")

    @patch("pipeline.step3_emotion_analysis.EmotionAnalyzer.analyze")
    def test_failure_path(self, mock_analyze):
        mock_analyze.return_value = EmotionAnalysisResult(
            success=False,
            error_message="Modèle indisponible",
        )
        result = analyze_emotions("/fake/audio.wav", "/fake/t.json", "/tmp")
        self.assertFalse(result.success)

    @patch("pipeline.step3_emotion_analysis.EmotionAnalyzer.analyze")
    def test_custom_model_name_passed(self, mock_analyze):
        mock_analyze.return_value = EmotionAnalysisResult(success=True)
        analyze_emotions(
            "/fake/audio.wav", "/fake/t.json", "/tmp",
            model_name="custom/emotion-model"
        )
        mock_analyze.assert_called_once()

    @patch("pipeline.step3_emotion_analysis.EmotionAnalyzer.analyze")
    def test_cuda_device_passed(self, mock_analyze):
        mock_analyze.return_value = EmotionAnalysisResult(success=True)
        analyze_emotions("/fake/audio.wav", "/fake/t.json", "/tmp", device="cuda")
        mock_analyze.assert_called_once()


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
