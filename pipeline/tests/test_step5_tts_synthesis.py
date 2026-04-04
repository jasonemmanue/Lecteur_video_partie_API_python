"""
Tests unitaires — Étape 5 : Synthèse Vocale avec Clonage (XTTS-v2)
===================================================================
Couverture :
  - TTSConfig              : valeurs par défaut, personnalisation
  - SynthesizedSegment     : duration_diff, to_dict()
  - TTSResult              : segment_count, success_rate
  - XTTSSynthesizer        : fichier manquant, import manquant,
                             synthèse réussie (mock complet),
                             échec segment individuel,
                             time_stretch, resample, silence,
                             save_manifest, save_wav, load_audio
  - extract_speaker_sample : extraction réussie, audio trop court
  - synthesize_speech()    : fonction publique end-to-end (mock)
"""

import json
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from pipeline.step5_tts_synthesis import (
    TTSConfig,
    SynthesizedSegment,
    TTSResult,
    XTTSSynthesizer,
    extract_speaker_sample,
    synthesize_speech,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SR = 24_000  # taux natif XTTS-v2

FAKE_TRANSLATED_DATA = {
    "segment_count": 3,
    "source_language": "en",
    "target_language": "fr",
    "segments": [
        {
            "id": 0, "start": 0.0, "end": 3.5,
            "translated_text": "Bonjour et bienvenue.",
            "tts_prompt": "[NEUTRAL][MODERATE] Bonjour et bienvenue.",
            "emotion": "neutral", "speech_rate": 3.0, "duration": 3.5,
        },
        {
            "id": 1, "start": 3.5, "end": 7.0,
            "translated_text": "Comment allez-vous ?",
            "tts_prompt": "[HAPPY][FAST] Comment allez-vous ?",
            "emotion": "happy", "speech_rate": 3.5, "duration": 3.5,
        },
        {
            "id": 2, "start": 7.0, "end": 10.0,
            "translated_text": "Commençons.",
            "tts_prompt": "[NEUTRAL] Commençons.",
            "emotion": "neutral", "speech_rate": 2.8, "duration": 3.0,
        },
    ],
}


def _make_synth() -> XTTSSynthesizer:
    """Crée un XTTSSynthesizer avec modèle mocké."""
    s = XTTSSynthesizer()
    s._tts = MagicMock()
    return s


def _make_audio(duration_s: float = 1.0, sr: int = SR) -> np.ndarray:
    """Génère un signal sinusoïdal de test."""
    t = np.linspace(0, duration_s, int(sr * duration_s))
    return (np.sin(2 * np.pi * 220 * t) * 0.3).astype(np.float32)


def _make_synthesized_segment(
    id: int = 0,
    start: float = 0.0,
    end: float = 3.5,
    duration_synthesized: float = 3.5,
    duration_target: float = 3.5,
    speed_ratio: float = 1.0,
    success: bool = True,
    error: str | None = None,
) -> SynthesizedSegment:
    return SynthesizedSegment(
        id=id, start=start, end=end,
        tts_prompt="[NEUTRAL] Test.",
        audio_path=Path(f"/tmp/segments/seg_{id:03d}.wav"),
        duration_synthesized=duration_synthesized,
        duration_target=duration_target,
        speed_ratio=speed_ratio,
        success=success,
        error=error,
    )


# ─── Tests : TTSConfig ────────────────────────────────────────────────────────

class TestTTSConfig(unittest.TestCase):

    def test_default_model_name(self):
        cfg = TTSConfig()
        self.assertIn("xtts_v2", cfg.model_name)

    def test_default_device_cpu(self):
        cfg = TTSConfig()
        self.assertEqual(cfg.device, "cpu")

    def test_default_sample_rate(self):
        cfg = TTSConfig()
        self.assertEqual(cfg.sample_rate, 24_000)

    def test_default_output_sample_rate(self):
        cfg = TTSConfig()
        self.assertEqual(cfg.output_sample_rate, 16_000)

    def test_default_speaker_sample_duration(self):
        cfg = TTSConfig()
        self.assertGreaterEqual(cfg.speaker_sample_duration, 6.0)

    def test_default_save_segments(self):
        cfg = TTSConfig()
        self.assertTrue(cfg.save_segments)

    def test_custom_config(self):
        cfg = TTSConfig(device="cuda", speed=1.2, temperature=0.8)
        self.assertEqual(cfg.device, "cuda")
        self.assertAlmostEqual(cfg.speed, 1.2)
        self.assertAlmostEqual(cfg.temperature, 0.8)


# ─── Tests : SynthesizedSegment ──────────────────────────────────────────────

class TestSynthesizedSegment(unittest.TestCase):

    def test_duration_diff_positive(self):
        seg = _make_synthesized_segment(
            duration_synthesized=4.0, duration_target=3.5
        )
        self.assertAlmostEqual(seg.duration_diff, 0.5, places=3)

    def test_duration_diff_negative(self):
        seg = _make_synthesized_segment(
            duration_synthesized=3.0, duration_target=3.5
        )
        self.assertAlmostEqual(seg.duration_diff, -0.5, places=3)

    def test_duration_diff_zero(self):
        seg = _make_synthesized_segment(
            duration_synthesized=3.5, duration_target=3.5
        )
        self.assertAlmostEqual(seg.duration_diff, 0.0)

    def test_to_dict_keys(self):
        seg = _make_synthesized_segment()
        d   = seg.to_dict()
        for key in ("id", "start", "end", "tts_prompt", "audio_path",
                    "duration_synthesized", "duration_target",
                    "speed_ratio", "duration_diff", "success", "error"):
            self.assertIn(key, d)

    def test_to_dict_audio_path_string(self):
        seg = _make_synthesized_segment()
        self.assertIsInstance(seg.to_dict()["audio_path"], str)

    def test_to_dict_audio_path_none(self):
        seg = _make_synthesized_segment()
        seg.audio_path = None
        self.assertIsNone(seg.to_dict()["audio_path"])

    def test_to_dict_success_true(self):
        seg = _make_synthesized_segment(success=True)
        self.assertTrue(seg.to_dict()["success"])

    def test_to_dict_error_propagated(self):
        seg = _make_synthesized_segment(success=False, error="TTS crash")
        self.assertEqual(seg.to_dict()["error"], "TTS crash")

    def test_speed_ratio_rounded(self):
        seg = _make_synthesized_segment(speed_ratio=1.142857)
        self.assertEqual(seg.to_dict()["speed_ratio"], round(1.142857, 4))


# ─── Tests : TTSResult ───────────────────────────────────────────────────────

class TestTTSResult(unittest.TestCase):

    def test_segment_count(self):
        segs   = [_make_synthesized_segment(id=i) for i in range(4)]
        result = TTSResult(success=True, synthesized_segments=segs)
        self.assertEqual(result.segment_count, 4)

    def test_success_rate_all_ok(self):
        segs   = [_make_synthesized_segment(success=True) for _ in range(3)]
        result = TTSResult(success=True, synthesized_segments=segs)
        self.assertAlmostEqual(result.success_rate, 1.0)

    def test_success_rate_partial(self):
        segs = [
            _make_synthesized_segment(success=True),
            _make_synthesized_segment(success=True),
            _make_synthesized_segment(success=False),
        ]
        result = TTSResult(success=True, synthesized_segments=segs)
        self.assertAlmostEqual(result.success_rate, 2 / 3)

    def test_success_rate_empty(self):
        result = TTSResult(success=True)
        self.assertEqual(result.success_rate, 0.0)

    def test_failure_defaults(self):
        result = TTSResult(success=False, error_message="boom")
        self.assertFalse(result.success)
        self.assertIsNone(result.audio_path)
        self.assertIsNone(result.manifest_path)


# ─── Tests : XTTSSynthesizer — init & chargement modèle ──────────────────────

class TestXTTSSynthesizerInit(unittest.TestCase):

    def test_model_not_loaded_at_init(self):
        s = XTTSSynthesizer()
        self.assertIsNone(s._tts)

    def test_load_model_raises_if_tts_missing(self):
        s = XTTSSynthesizer()
        with patch.dict("sys.modules", {"TTS": None, "TTS.api": None}):
            with self.assertRaises(ImportError) as ctx:
                s._load_model()
            self.assertIn("TTS", str(ctx.exception))

    def test_load_model_called_once(self):
        s = XTTSSynthesizer()
        mock_tts_module = MagicMock()
        mock_tts_instance = MagicMock()
        mock_tts_module.TTS.return_value.to.return_value = mock_tts_instance

        with patch.dict("sys.modules", {
            "TTS": mock_tts_module,
            "TTS.api": mock_tts_module,
        }):
            s._load_model()
            s._load_model()  # deuxième appel — ne doit pas recharger

        mock_tts_module.TTS.assert_called_once()


# ─── Tests : XTTSSynthesizer.synthesize() ────────────────────────────────────

class TestXTTSSynthesizerSynthesize(unittest.TestCase):

    def test_returns_failure_if_json_missing(self):
        s = _make_synth()
        result = s.synthesize("/nonexistent/translated.json", "/fake/speaker.wav", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("JSON traduit", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    def test_returns_failure_if_speaker_missing(self, mock_exists):
        s = _make_synth()
        mock_exists.side_effect = [True, False]
        result = s.synthesize("/fake/translated.json", "/nonexistent/speaker.wav", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("référence", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_returns_failure_on_exception(self, mock_mkdir, mock_exists):
        s = _make_synth()
        with patch("pathlib.Path.read_text", side_effect=RuntimeError("disk error")):
            result = s.synthesize("/fake/t.json", "/fake/speaker.wav", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("disk error", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.read_text")
    def test_successful_synthesis(self, mock_read, mock_write, mock_mkdir, mock_exists):
        s = _make_synth()
        mock_read.return_value = json.dumps(FAKE_TRANSLATED_DATA)

        audio_chunk = _make_audio(3.5)

        with patch.object(XTTSSynthesizer, "_synthesize_segment") as mock_seg:
            def _seg_with_path(i):
                s = _make_synthesized_segment(id=i)
                s.audio_path = Path(f"/tmp/seg_{i}.wav")
                return s
            mock_seg.side_effect = [_seg_with_path(i) for i in range(3)]
            with patch.object(XTTSSynthesizer, "_load_audio_array", return_value=audio_chunk):
                with patch.object(XTTSSynthesizer, "_save_wav"):
                    with patch.object(XTTSSynthesizer, "_save_manifest"):
                        result = s.synthesize("/fake/t.json", "/fake/speaker.wav", "/tmp")

        self.assertTrue(result.success)
        self.assertEqual(result.segment_count, 3)
        self.assertGreater(result.total_duration_s, 0)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.read_text")
    def test_output_audio_path_ends_with_tts(
        self, mock_read, mock_write, mock_mkdir, mock_exists
    ):
        s = _make_synth()
        mock_read.return_value = json.dumps(FAKE_TRANSLATED_DATA)
        audio_chunk = _make_audio(1.0)

        with patch.object(XTTSSynthesizer, "_synthesize_segment") as mock_seg:
            mock_seg.side_effect = [
                _make_synthesized_segment(id=i) for i in range(3)
            ]
            with patch.object(XTTSSynthesizer, "_load_audio_array", return_value=audio_chunk):
                with patch.object(XTTSSynthesizer, "_save_wav"):
                    with patch.object(XTTSSynthesizer, "_save_manifest"):
                        result = s.synthesize("/fake/audio_translated.json", "/fake/s.wav", "/tmp")

        self.assertIsNotNone(result.audio_path)
        self.assertTrue(str(result.audio_path).endswith("_tts.wav"))


# ─── Tests : _synthesize_segment ─────────────────────────────────────────────

class TestSynthesizeSegment(unittest.TestCase):

    @patch("pathlib.Path.mkdir")
    def test_successful_segment(self, mock_mkdir):
        s = _make_synth()
        audio = _make_audio(3.5)

        with patch.object(XTTSSynthesizer, "_load_audio_array", return_value=audio):
            with patch.object(XTTSSynthesizer, "_save_wav"):
                seg = s._synthesize_segment(
                    text="Bonjour.",
                    speaker_wav="/fake/speaker.wav",
                    language="fr",
                    seg_id=0,
                    duration_target=3.5,
                    output_dir=Path("/tmp"),
                )

        self.assertTrue(seg.success)
        self.assertIsNone(seg.error)
        self.assertAlmostEqual(seg.duration_synthesized, 3.5, places=1)

    @patch("pathlib.Path.mkdir")
    def test_failed_segment_returns_error(self, mock_mkdir):
        s = _make_synth()
        s._tts.tts_to_file.side_effect = RuntimeError("CUDA out of memory")

        seg = s._synthesize_segment(
            text="Bonjour.",
            speaker_wav="/fake/speaker.wav",
            language="fr",
            seg_id=1,
            duration_target=3.0,
            output_dir=Path("/tmp"),
        )

        self.assertFalse(seg.success)
        self.assertIn("CUDA out of memory", seg.error)
        self.assertEqual(seg.duration_synthesized, 0.0)

    @patch("pathlib.Path.mkdir")
    def test_speed_ratio_applied_when_needed(self, mock_mkdir):
        """Si durée synthétisée ≠ cible >10%, time_stretch est appelé."""
        s = _make_synth()
        # Audio 5s pour une cible de 3.5s → ratio 1.43 > 10%
        audio = _make_audio(5.0)

        with patch.object(XTTSSynthesizer, "_load_audio_array", return_value=audio):
            with patch.object(XTTSSynthesizer, "_time_stretch", return_value=_make_audio(3.5)) as mock_stretch:
                with patch.object(XTTSSynthesizer, "_save_wav"):
                    s._synthesize_segment(
                        text="Long text.",
                        speaker_wav="/fake/s.wav",
                        language="fr",
                        seg_id=0,
                        duration_target=3.5,
                        output_dir=Path("/tmp"),
                    )
                mock_stretch.assert_called_once()

    @patch("pathlib.Path.mkdir")
    def test_no_stretch_within_tolerance(self, mock_mkdir):
        """Si écart < 10%, time_stretch n'est pas appelé."""
        s = _make_synth()
        audio = _make_audio(3.5)  # cible 3.5s → ratio 1.0 ≈ pas d'étirement

        with patch.object(XTTSSynthesizer, "_load_audio_array", return_value=audio):
            with patch.object(XTTSSynthesizer, "_time_stretch") as mock_stretch:
                with patch.object(XTTSSynthesizer, "_save_wav"):
                    s._synthesize_segment(
                        text="Short.",
                        speaker_wav="/fake/s.wav",
                        language="fr",
                        seg_id=0,
                        duration_target=3.5,
                        output_dir=Path("/tmp"),
                    )
                mock_stretch.assert_not_called()


# ─── Tests : _time_stretch ────────────────────────────────────────────────────

class TestTimeStretch(unittest.TestCase):

    @patch.dict("sys.modules", {"pyrubberband": None})
    def test_stretch_ratio_2_halves_length(self):
        """ratio=2 → audio 2x plus long → après stretch 1/2 longueur."""
        audio = _make_audio(2.0)
        # ratio=2 signifie que l'audio dure 2x la cible → compresser à 0.5
        stretched = XTTSSynthesizer._time_stretch(audio, 2.0)
        # Longueur cible ≈ len(audio) / ratio
        expected_len = int(len(audio) / 2.0)
        self.assertAlmostEqual(len(stretched), expected_len, delta=50)

    @patch.dict("sys.modules", {"pyrubberband": None})
    def test_stretch_ratio_05_doubles_length(self):
        """ratio=0.5 → audio trop court → étirer x2."""
        audio = _make_audio(1.0)
        stretched = XTTSSynthesizer._time_stretch(audio, 0.5)
        expected_len = int(len(audio) / 0.5)
        self.assertAlmostEqual(len(stretched), expected_len, delta=50)

    @patch.dict("sys.modules", {"pyrubberband": None})
    def test_output_is_float32(self):
        audio     = _make_audio(1.0)
        stretched = XTTSSynthesizer._time_stretch(audio, 1.2)
        self.assertEqual(stretched.dtype, np.float32)

    def test_ratio_one_preserves_length(self):
        audio     = _make_audio(1.0)
        stretched = XTTSSynthesizer._time_stretch(audio, 1.0)
        self.assertAlmostEqual(len(stretched), len(audio), delta=10)


# ─── Tests : _resample_if_needed ─────────────────────────────────────────────

class TestResampleIfNeeded(unittest.TestCase):

    def test_no_resample_when_same_rate(self):
        audio    = _make_audio(1.0, sr=16_000)
        result   = XTTSSynthesizer._resample_if_needed(audio, 16_000, 16_000)
        np.testing.assert_array_equal(result, audio)

    def test_downsampled_shorter(self):
        audio  = _make_audio(1.0, sr=24_000)
        result = XTTSSynthesizer._resample_if_needed(audio, 24_000, 16_000)
        expected = int(len(audio) * 16_000 / 24_000)
        self.assertAlmostEqual(len(result), expected, delta=10)

    def test_upsampled_longer(self):
        audio  = _make_audio(1.0, sr=16_000)
        result = XTTSSynthesizer._resample_if_needed(audio, 16_000, 24_000)
        self.assertGreater(len(result), len(audio))


# ─── Tests : _make_silence ────────────────────────────────────────────────────

class TestMakeSilence(unittest.TestCase):

    def test_silence_is_zeros(self):
        s       = XTTSSynthesizer()
        silence = s._make_silence(100)
        self.assertTrue(np.all(silence == 0.0))

    def test_silence_length_correct(self):
        s        = XTTSSynthesizer()
        silence  = s._make_silence(150)
        expected = int(24_000 * 0.15)
        self.assertEqual(len(silence), expected)

    def test_silence_dtype_float32(self):
        s = XTTSSynthesizer()
        self.assertEqual(s._make_silence(100).dtype, np.float32)


# ─── Tests : _save_manifest ───────────────────────────────────────────────────

class TestSaveManifest(unittest.TestCase):

    def test_manifest_structure(self):
        segs     = [_make_synthesized_segment(id=i) for i in range(2)]
        captured = {}

        def fake_write(self_path, content, **kwargs):
            captured["content"] = content

        with patch.object(Path, "write_text", fake_write):
            XTTSSynthesizer._save_manifest(segs, Path("/tmp/manifest.json"))

        data = json.loads(captured["content"])
        self.assertIn("segment_count", data)
        self.assertIn("success_count", data)
        self.assertIn("segments", data)
        self.assertEqual(data["segment_count"], 2)

    def test_success_count_correct(self):
        segs = [
            _make_synthesized_segment(success=True),
            _make_synthesized_segment(success=False),
            _make_synthesized_segment(success=True),
        ]
        captured = {}

        def fake_write(self_path, content, **kwargs):
            captured["content"] = content

        with patch.object(Path, "write_text", fake_write):
            XTTSSynthesizer._save_manifest(segs, Path("/tmp/manifest.json"))

        data = json.loads(captured["content"])
        self.assertEqual(data["success_count"], 2)


# ─── Tests : extract_speaker_sample ─────────────────────────────────────────

class TestExtractSpeakerSample(unittest.TestCase):

    def test_successful_extraction(self):
        audio = _make_audio(10.0, sr=16_000)

        mock_sf = MagicMock()
        mock_sf.read.return_value = (audio, 16_000)

        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            result = extract_speaker_sample(
                "/fake/audio.wav", "/tmp/sample.wav", duration_s=6.0
            )

        self.assertTrue(result)
        mock_sf.write.assert_called_once()

    def test_import_error_returns_false(self):
        with patch.dict("sys.modules", {"soundfile": None}):
            result = extract_speaker_sample(
                "/fake/audio.wav", "/tmp/sample.wav"
            )
        self.assertFalse(result)

    def test_stereo_converted_to_mono(self):
        stereo = np.random.rand(16_000 * 10, 2).astype(np.float32)
        mock_sf = MagicMock()
        mock_sf.read.return_value = (stereo, 16_000)

        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            extract_speaker_sample("/fake/audio.wav", "/tmp/sample.wav")

        # soundfile.write doit recevoir un tableau 1D (mono)
        write_call_args = mock_sf.write.call_args
        written_audio = write_call_args[0][1]
        self.assertEqual(written_audio.ndim, 1)

    def test_offset_respected(self):
        audio = _make_audio(20.0, sr=16_000)
        mock_sf = MagicMock()
        mock_sf.read.return_value = (audio, 16_000)

        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            extract_speaker_sample(
                "/fake/audio.wav", "/tmp/sample.wav",
                duration_s=6.0, offset_s=5.0,
            )

        write_call_args = mock_sf.write.call_args
        written_audio   = write_call_args[0][1]
        # 6s à 16kHz = 96000 échantillons (± 1%)
        self.assertAlmostEqual(len(written_audio), 96_000, delta=960)


# ─── Tests : fonction publique synthesize_speech() ───────────────────────────

class TestSynthesizeSpeechPublicAPI(unittest.TestCase):

    @patch("pipeline.step5_tts_synthesis.XTTSSynthesizer.synthesize")
    def test_success_path(self, mock_synth):
        segs = [_make_synthesized_segment(id=i) for i in range(3)]
        mock_synth.return_value = TTSResult(
            success=True,
            synthesized_segments=segs,
            audio_path=Path("/tmp/audio_tts.wav"),
            total_duration_s=10.5,
            processing_time_s=42.0,
            sample_rate=16_000,
        )
        result = synthesize_speech(
            "/fake/translated.json", "/fake/speaker.wav", "/tmp"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.segment_count, 3)
        self.assertAlmostEqual(result.success_rate, 1.0)

    @patch("pipeline.step5_tts_synthesis.XTTSSynthesizer.synthesize")
    def test_failure_path(self, mock_synth):
        mock_synth.return_value = TTSResult(
            success=False,
            error_message="Modèle XTTS-v2 indisponible",
        )
        result = synthesize_speech(
            "/fake/translated.json", "/fake/speaker.wav", "/tmp"
        )
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

    @patch("pipeline.step5_tts_synthesis.XTTSSynthesizer.synthesize")
    def test_cuda_device_passed(self, mock_synth):
        mock_synth.return_value = TTSResult(success=True)
        synthesize_speech(
            "/fake/t.json", "/fake/s.wav", "/tmp", device="cuda"
        )
        mock_synth.assert_called_once()

    @patch("pipeline.step5_tts_synthesis.XTTSSynthesizer.synthesize")
    def test_speed_passed_to_config(self, mock_synth):
        mock_synth.return_value = TTSResult(success=True)
        synthesize_speech(
            "/fake/t.json", "/fake/s.wav", "/tmp", speed=1.2
        )
        mock_synth.assert_called_once()

    @patch("pipeline.step5_tts_synthesis.XTTSSynthesizer.synthesize")
    def test_save_segments_false(self, mock_synth):
        mock_synth.return_value = TTSResult(success=True)
        synthesize_speech(
            "/fake/t.json", "/fake/s.wav", "/tmp", save_segments=False
        )
        mock_synth.assert_called_once()


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
