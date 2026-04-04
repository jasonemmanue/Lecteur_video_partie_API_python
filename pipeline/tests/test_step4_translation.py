"""
Tests unitaires — Étape 4 : Traduction NLP
==========================================
Couverture :
  - TranslationConfig      : valeurs par défaut, personnalisation
  - TranslatedSegment      : duration, to_dict(), tts_prompt
  - TranslationResult      : segment_count, full_translated_text
  - NLPTranslator          : fichier manquant, import manquant,
                             résolution du modèle, traduction réussie,
                             batch processing, fallback erreur batch,
                             construction tts_prompt, save_json
  - Helpers                : get_nllb_code, is_helsinki_supported
  - translate_segments()   : fonction publique end-to-end (mock)
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.step4_translation import (
    TranslationConfig,
    TranslatedSegment,
    TranslationResult,
    NLPTranslator,
    NLLB_LANG_MAP,
    HELSINKI_MODEL_MAP,
    get_nllb_code,
    is_helsinki_supported,
    translate_segments,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

FAKE_ENRICHED_DATA = {
    "dominant_emotion": "neutral",
    "avg_intensity": 0.5,
    "segment_count": 3,
    "segments": [
        {
            "id": 0, "start": 0.0, "end": 3.5,
            "text": "Hello and welcome.",
            "emotion": "neutral", "speech_rate": 3.0,
            "tone_tags": ["[NEUTRAL]", "[MODERATE]"],
        },
        {
            "id": 1, "start": 3.5, "end": 7.0,
            "text": "How are you today?",
            "emotion": "happy", "speech_rate": 3.5,
            "tone_tags": ["[HAPPY]", "[FAST]"],
        },
        {
            "id": 2, "start": 7.0, "end": 10.0,
            "text": "Let us begin.",
            "emotion": "neutral", "speech_rate": 2.8,
            "tone_tags": ["[NEUTRAL]"],
        },
    ],
}


def _make_translator(src: str = "en", tgt: str = "fr") -> NLPTranslator:
    """Crée un NLPTranslator avec modèle mocké."""
    cfg = TranslationConfig(source_language=src, target_language=tgt)
    t   = NLPTranslator(cfg)
    t._pipeline = MagicMock()
    return t


def _make_segment(
    id: int = 0,
    start: float = 0.0,
    end: float = 3.5,
    source_text: str = "Hello.",
    translated_text: str = "Bonjour.",
    tone_tags: list | None = None,
    tts_prompt: str = "[NEUTRAL] Bonjour.",
    emotion: str = "neutral",
    speech_rate: float = 3.0,
) -> TranslatedSegment:
    return TranslatedSegment(
        id=id, start=start, end=end,
        source_text=source_text,
        translated_text=translated_text,
        tone_tags=tone_tags or ["[NEUTRAL]"],
        tts_prompt=tts_prompt,
        emotion=emotion,
        speech_rate=speech_rate,
        language_src="en",
        language_tgt="fr",
    )


def _pipeline_output(texts: list[str], translations: list[str]) -> list[dict]:
    """Simule la sortie d'un pipeline HuggingFace."""
    return [{"translation_text": t} for t in translations]


# ─── Tests : TranslationConfig ───────────────────────────────────────────────

class TestTranslationConfig(unittest.TestCase):

    def test_default_model_type(self):
        cfg = TranslationConfig()
        self.assertEqual(cfg.model_type, "nllb")

    def test_default_device_cpu(self):
        cfg = TranslationConfig()
        self.assertEqual(cfg.device, "cpu")

    def test_default_source_language(self):
        cfg = TranslationConfig()
        self.assertEqual(cfg.source_language, "en")

    def test_default_target_language(self):
        cfg = TranslationConfig()
        self.assertEqual(cfg.target_language, "fr")

    def test_default_inject_tone_tags(self):
        cfg = TranslationConfig()
        self.assertTrue(cfg.inject_tone_tags)

    def test_default_batch_size(self):
        cfg = TranslationConfig()
        self.assertEqual(cfg.batch_size, 8)

    def test_custom_config(self):
        cfg = TranslationConfig(
            model_type="helsinki",
            source_language="fr",
            target_language="en",
            device="cuda",
            num_beams=2,
        )
        self.assertEqual(cfg.model_type, "helsinki")
        self.assertEqual(cfg.source_language, "fr")
        self.assertEqual(cfg.device, "cuda")


# ─── Tests : TranslatedSegment ───────────────────────────────────────────────

class TestTranslatedSegment(unittest.TestCase):

    def test_duration_computed(self):
        seg = _make_segment(start=1.0, end=4.5)
        self.assertAlmostEqual(seg.duration, 3.5)

    def test_to_dict_keys(self):
        seg = _make_segment()
        d   = seg.to_dict()
        for key in ("id", "start", "end", "source_text", "translated_text",
                    "tone_tags", "tts_prompt", "emotion", "speech_rate",
                    "duration", "language_src", "language_tgt"):
            self.assertIn(key, d)

    def test_to_dict_source_text_stripped(self):
        seg = _make_segment(source_text="  Hello.  ")
        self.assertEqual(seg.to_dict()["source_text"], "Hello.")

    def test_to_dict_translated_text_stripped(self):
        seg = _make_segment(translated_text="  Bonjour.  ")
        self.assertEqual(seg.to_dict()["translated_text"], "Bonjour.")

    def test_to_dict_speech_rate_rounded(self):
        seg = _make_segment(speech_rate=3.14159)
        self.assertEqual(seg.to_dict()["speech_rate"], round(3.14159, 2))

    def test_to_dict_tone_tags_list(self):
        seg = _make_segment(tone_tags=["[HAPPY]", "[FAST]"])
        self.assertIsInstance(seg.to_dict()["tone_tags"], list)


# ─── Tests : TranslationResult ───────────────────────────────────────────────

class TestTranslationResult(unittest.TestCase):

    def test_segment_count(self):
        segs   = [_make_segment(id=i) for i in range(3)]
        result = TranslationResult(success=True, translated_segments=segs)
        self.assertEqual(result.segment_count, 3)

    def test_full_translated_text(self):
        segs = [
            _make_segment(id=0, translated_text="Bonjour"),
            _make_segment(id=1, translated_text="le monde."),
        ]
        result = TranslationResult(success=True, translated_segments=segs)
        self.assertEqual(result.full_translated_text, "Bonjour le monde.")

    def test_full_translated_text_empty(self):
        result = TranslationResult(success=True)
        self.assertEqual(result.full_translated_text, "")

    def test_failure_defaults(self):
        result = TranslationResult(success=False, error_message="boom")
        self.assertFalse(result.success)
        self.assertIsNone(result.output_json_path)
        self.assertEqual(result.segment_count, 0)


# ─── Tests : NLPTranslator — résolution du modèle ────────────────────────────

class TestResolveModelName(unittest.TestCase):

    def test_nllb_returns_default_model(self):
        cfg = TranslationConfig(model_type="nllb")
        t   = NLPTranslator(cfg)
        self.assertIn("nllb", t._resolve_model_name())

    def test_helsinki_en_fr_resolved(self):
        cfg = TranslationConfig(
            model_type="helsinki",
            source_language="en",
            target_language="fr",
        )
        t = NLPTranslator(cfg)
        self.assertIn("Helsinki-NLP", t._resolve_model_name())
        self.assertIn("en-fr", t._resolve_model_name())

    def test_helsinki_unsupported_pair_raises(self):
        cfg = TranslationConfig(
            model_type="helsinki",
            source_language="sw",
            target_language="ja",
        )
        t = NLPTranslator(cfg)
        with self.assertRaises(ValueError) as ctx:
            t._resolve_model_name()
        self.assertIn("non supportée", str(ctx.exception))

    def test_custom_model_name_used(self):
        cfg = TranslationConfig(model_type="nllb", model_name="my-custom/model")
        t   = NLPTranslator(cfg)
        self.assertEqual(t._resolve_model_name(), "my-custom/model")


# ─── Tests : NLPTranslator — chargement modèle ───────────────────────────────

class TestLoadModel(unittest.TestCase):

    def test_raises_if_transformers_missing(self):
        t = NLPTranslator()
        with patch.dict("sys.modules", {"transformers": None}):
            with self.assertRaises(ImportError) as ctx:
                t._load_model()
            self.assertIn("transformers", str(ctx.exception))

    def test_model_loaded_once(self):
        t = NLPTranslator()
        mock_tf = MagicMock()
        mock_tf.pipeline.return_value = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_tf}):
            t._load_model()
            t._load_model()
        mock_tf.pipeline.assert_called_once()


# ─── Tests : NLPTranslator.translate() ───────────────────────────────────────

class TestNLPTranslatorTranslate(unittest.TestCase):

    def test_returns_failure_if_json_missing(self):
        t      = _make_translator()
        result = t.translate("/nonexistent/enriched.json", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("introuvable", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_returns_failure_on_exception(self, mock_mkdir, mock_exists):
        t = _make_translator()
        with patch("pathlib.Path.read_text", side_effect=RuntimeError("disk error")):
            result = t.translate("/fake/enriched.json", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("disk error", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.read_text")
    def test_successful_translation(self, mock_read, mock_write, mock_mkdir, mock_exists):
        t = _make_translator()
        mock_read.return_value = json.dumps(FAKE_ENRICHED_DATA)
        t._pipeline.return_value = [
            {"translation_text": "Bonjour et bienvenue."},
            {"translation_text": "Comment allez-vous ?"},
            {"translation_text": "Commençons."},
        ]
        result = t.translate("/fake/enriched.json", "/tmp")
        self.assertTrue(result.success)
        self.assertEqual(result.segment_count, 3)
        self.assertEqual(result.source_language, "en")
        self.assertEqual(result.target_language, "fr")

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.read_text")
    def test_output_json_path_ends_with_translated(
        self, mock_read, mock_write, mock_mkdir, mock_exists
    ):
        t = _make_translator()
        mock_read.return_value = json.dumps(FAKE_ENRICHED_DATA)
        t._pipeline.return_value = [{"translation_text": "T"}] * 3
        result = t.translate("/fake/audio_enriched.json", "/tmp")
        self.assertIsNotNone(result.output_json_path)
        self.assertTrue(str(result.output_json_path).endswith("_translated.json"))

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.read_text")
    def test_processing_time_recorded(
        self, mock_read, mock_write, mock_mkdir, mock_exists
    ):
        t = _make_translator()
        mock_read.return_value = json.dumps(FAKE_ENRICHED_DATA)
        t._pipeline.return_value = [{"translation_text": "T"}] * 3
        result = t.translate("/fake/enriched.json", "/tmp")
        self.assertGreaterEqual(result.processing_time_s, 0.0)


# ─── Tests : _translate_batch ────────────────────────────────────────────────

class TestTranslateBatch(unittest.TestCase):

    def test_returns_translations(self):
        t = _make_translator()
        t._pipeline.return_value = [
            {"translation_text": "Bonjour."},
            {"translation_text": "Au revoir."},
        ]
        results = t._translate_batch(["Hello.", "Goodbye."])
        self.assertEqual(results, ["Bonjour.", "Au revoir."])

    def test_empty_batch_returns_empty(self):
        t = _make_translator()
        results = t._translate_batch([])
        self.assertEqual(results, [])

    def test_pipeline_error_returns_source(self):
        """En cas d'erreur, le texte source est conservé (fallback)."""
        t = _make_translator()
        t._pipeline.side_effect = RuntimeError("inference failed")
        texts   = ["Hello.", "Goodbye."]
        results = t._translate_batch(texts)
        self.assertEqual(results, texts)

    def test_nllb_passes_tgt_lang(self):
        t = _make_translator(tgt="fr")
        t.config.model_type = "nllb"
        t._pipeline.return_value = [{"translation_text": "Bonjour."}]
        t._translate_batch(["Hello."])
        call_kwargs = t._pipeline.call_args[1]
        self.assertIn("tgt_lang", call_kwargs)
        self.assertEqual(call_kwargs["tgt_lang"], "fra_Latn")

    def test_handles_list_of_list_output(self):
        """Certains pipelines retournent [[{...}]] au lieu de [{...}]."""
        t = _make_translator()
        t._pipeline.return_value = [[{"translation_text": "Bonjour."}]]
        results = t._translate_batch(["Hello."])
        self.assertEqual(results, ["Bonjour."])


# ─── Tests : _build_tts_prompt ───────────────────────────────────────────────

class TestBuildTtsPrompt(unittest.TestCase):

    def setUp(self):
        self.translator = _make_translator()

    def test_tags_prepended_to_text(self):
        prompt = self.translator._build_tts_prompt(
            "Bonjour.", ["[NEUTRAL]", "[MODERATE]"]
        )
        self.assertTrue(prompt.startswith("[NEUTRAL][MODERATE]"))
        self.assertIn("Bonjour.", prompt)

    def test_no_tags_returns_text_only(self):
        prompt = self.translator._build_tts_prompt("Bonjour.", [])
        self.assertEqual(prompt, "Bonjour.")

    def test_inject_disabled_returns_text_only(self):
        self.translator.config.inject_tone_tags = False
        prompt = self.translator._build_tts_prompt(
            "Bonjour.", ["[HAPPY]"]
        )
        self.assertEqual(prompt, "Bonjour.")

    def test_text_stripped(self):
        prompt = self.translator._build_tts_prompt(
            "  Bonjour.  ", ["[NEUTRAL]"]
        )
        self.assertFalse(prompt.endswith("  "))
        self.assertIn("Bonjour.", prompt)

    def test_multiple_tags_concatenated(self):
        prompt = self.translator._build_tts_prompt(
            "Salut.", ["[HAPPY]", "[FAST]", "[HIGH_PITCH]"]
        )
        self.assertIn("[HAPPY][FAST][HIGH_PITCH]", prompt)


# ─── Tests : _save_json ───────────────────────────────────────────────────────

class TestSaveJson(unittest.TestCase):

    def test_json_structure(self):
        segs     = [_make_segment(id=0, translated_text="Bonjour.")]
        captured = {}

        def fake_write(self_path, content, **kwargs):
            captured["content"] = content

        with patch.object(Path, "write_text", fake_write):
            NLPTranslator._save_json(segs, Path("/tmp/test.json"))

        data = json.loads(captured["content"])
        self.assertIn("segment_count", data)
        self.assertIn("source_language", data)
        self.assertIn("target_language", data)
        self.assertIn("segments", data)
        self.assertEqual(data["segment_count"], 1)

    def test_utf8_preserved(self):
        segs     = [_make_segment(translated_text="Ça va très bien.")]
        captured = {}

        def fake_write(self_path, content, **kwargs):
            captured["content"] = content

        with patch.object(Path, "write_text", fake_write):
            NLPTranslator._save_json(segs, Path("/tmp/test.json"))

        self.assertIn("ça", captured["content"].lower())

    def test_empty_segments_handled(self):
        captured = {}

        def fake_write(self_path, content, **kwargs):
            captured["content"] = content

        with patch.object(Path, "write_text", fake_write):
            NLPTranslator._save_json([], Path("/tmp/test.json"))

        data = json.loads(captured["content"])
        self.assertEqual(data["segment_count"], 0)
        self.assertEqual(data["segments"], [])


# ─── Tests : Helpers publics ──────────────────────────────────────────────────

class TestHelpers(unittest.TestCase):

    def test_get_nllb_code_french(self):
        self.assertEqual(get_nllb_code("fr"), "fra_Latn")

    def test_get_nllb_code_arabic(self):
        self.assertEqual(get_nllb_code("ar"), "arb_Arab")

    def test_get_nllb_code_unknown_returns_english(self):
        self.assertEqual(get_nllb_code("xx"), "eng_Latn")

    def test_all_v1_languages_mapped(self):
        """Toutes les langues V1 du cahier des charges sont dans NLLB_LANG_MAP."""
        v1_langs = ["fr", "en", "es", "de", "pt", "ar", "zh", "ja"]
        for lang in v1_langs:
            self.assertIn(lang, NLLB_LANG_MAP, f"Langue manquante : {lang}")

    def test_is_helsinki_supported_en_fr(self):
        self.assertTrue(is_helsinki_supported("en", "fr"))

    def test_is_helsinki_supported_unsupported(self):
        self.assertFalse(is_helsinki_supported("sw", "ja"))

    def test_nllb_codes_unique(self):
        """Chaque langue a un code NLLB unique."""
        codes = list(NLLB_LANG_MAP.values())
        self.assertEqual(len(codes), len(set(codes)))


# ─── Tests : fonction publique translate_segments() ──────────────────────────

class TestTranslateSegmentsPublicAPI(unittest.TestCase):

    @patch("pipeline.step4_translation.NLPTranslator.translate")
    def test_success_path(self, mock_translate):
        segs = [_make_segment()]
        mock_translate.return_value = TranslationResult(
            success=True,
            translated_segments=segs,
            source_language="en",
            target_language="fr",
            model_used="facebook/nllb-200-distilled-600M",
            processing_time_s=1.2,
            output_json_path=Path("/tmp/audio_translated.json"),
        )
        result = translate_segments("/fake/enriched.json", "/tmp", "en", "fr")
        self.assertTrue(result.success)
        self.assertEqual(result.segment_count, 1)

    @patch("pipeline.step4_translation.NLPTranslator.translate")
    def test_failure_path(self, mock_translate):
        mock_translate.return_value = TranslationResult(
            success=False,
            error_message="Modèle indisponible",
        )
        result = translate_segments("/fake/enriched.json", "/tmp", "en", "fr")
        self.assertFalse(result.success)

    @patch("pipeline.step4_translation.NLPTranslator.translate")
    def test_custom_target_language(self, mock_translate):
        mock_translate.return_value = TranslationResult(success=True)
        translate_segments("/fake/enriched.json", "/tmp", "en", "de", "nllb")
        mock_translate.assert_called_once()

    @patch("pipeline.step4_translation.NLPTranslator.translate")
    def test_helsinki_model_type(self, mock_translate):
        mock_translate.return_value = TranslationResult(success=True)
        translate_segments(
            "/fake/enriched.json", "/tmp", "en", "fr", "helsinki"
        )
        mock_translate.assert_called_once()


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
