"""
LinguaPlay Pipeline — Étape 4 : Traduction NLP
===============================================
Traduit les segments transcrits vers la langue cible en injectant
les métadonnées de ton (étape 3) comme prompts de contrôle TTS.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Codes de langue NLLB-200 ────────────────────────────────────────────────

NLLB_LANG_MAP: dict[str, str] = {
    "fr":  "fra_Latn",
    "en":  "eng_Latn",
    "es":  "spa_Latn",
    "de":  "deu_Latn",
    "pt":  "por_Latn",
    "ar":  "arb_Arab",
    "zh":  "zho_Hans",
    "ja":  "jpn_Jpan",
    "sw":  "swh_Latn",
    "it":  "ita_Latn",
    "ru":  "rus_Cyrl",
    "ko":  "kor_Hang",
}

HELSINKI_MODEL_MAP: dict[tuple[str, str], str] = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "pt"): "Helsinki-NLP/opus-mt-en-pt",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
}

NLLB_DEFAULT = "facebook/nllb-200-distilled-600M"


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TranslationConfig:
    model_type: str        = "nllb"
    model_name: str        = NLLB_DEFAULT
    device: str            = "cpu"
    max_length: int        = 512
    num_beams: int         = 4
    batch_size: int        = 8
    inject_tone_tags: bool = True
    source_language: str   = "en"
    target_language: str   = "fr"


# ─── Modèles de données ───────────────────────────────────────────────────────

@dataclass
class TranslatedSegment:
    id: int
    start: float
    end: float
    source_text: str
    translated_text: str
    tone_tags: list[str]
    tts_prompt: str
    emotion: str
    speech_rate: float
    language_src: str
    language_tgt: str

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "start":            self.start,
            "end":              self.end,
            "source_text":      self.source_text.strip(),
            "translated_text":  self.translated_text.strip(),
            "tone_tags":        self.tone_tags,
            "tts_prompt":       self.tts_prompt,
            "emotion":          self.emotion,
            "speech_rate":      round(self.speech_rate, 2),
            "duration":         self.duration,
            "language_src":     self.language_src,
            "language_tgt":     self.language_tgt,
        }


@dataclass
class TranslationResult:
    success: bool
    translated_segments: list[TranslatedSegment] = field(default_factory=list)
    source_language: str                          = ""
    target_language: str                          = ""
    model_used: str                               = ""
    processing_time_s: float                      = 0.0
    output_json_path: Path | None                 = None
    error_message: str | None                     = None

    @property
    def segment_count(self) -> int:
        return len(self.translated_segments)

    @property
    def full_translated_text(self) -> str:
        return " ".join(s.translated_text.strip() for s in self.translated_segments)


# ─── Traducteur principal ─────────────────────────────────────────────────────

class NLPTranslator:

    def __init__(self, config: TranslationConfig | None = None):
        self.config        = config or TranslationConfig()
        self._tokenizer    = None
        self._model        = None
        self._loaded       = False
        self._tgt_lang_code = NLLB_LANG_MAP.get(self.config.target_language, "fra_Latn")

    # ── Chargement du modèle ──────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            model_name = self._resolve_model_name()
            logger.info(f"[Étape 4] Chargement du modèle : {model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            if self.config.model_type == "nllb":
                src_code = NLLB_LANG_MAP.get(self.config.source_language, "eng_Latn")
                self._tokenizer.src_lang = src_code
                self._tgt_lang_code      = NLLB_LANG_MAP.get(
                    self.config.target_language, "fra_Latn"
                )

            self._loaded = True
            logger.info("[Étape 4] Modèle chargé.")

        except ImportError:
            raise ImportError(
                "transformers non installé. "
                "Lancez : pip install transformers sentencepiece"
            )

    def _resolve_model_name(self) -> str:
        if self.config.model_type == "helsinki":
            pair  = (self.config.source_language, self.config.target_language)
            model = HELSINKI_MODEL_MAP.get(pair)
            if model is None:
                raise ValueError(
                    f"Paire non supportée par Helsinki-NLP : "
                    f"{self.config.source_language} → {self.config.target_language}. "
                    f"Utilisez model_type='nllb'."
                )
            return model
        return self.config.model_name

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def translate(
        self,
        enriched_json_path: str | Path,
        output_dir: str | Path,
    ) -> TranslationResult:
        enriched_json_path = Path(enriched_json_path)
        output_dir         = Path(output_dir)

        if not enriched_json_path.exists():
            return TranslationResult(
                success=False,
                error_message=f"Fichier JSON enrichi introuvable : {enriched_json_path}",
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._load_model()
            return self._run_translation(enriched_json_path, output_dir)
        except Exception as exc:
            logger.exception("[Étape 4] Erreur inattendue lors de la traduction")
            return TranslationResult(success=False, error_message=str(exc))

    # ── Pipeline de traduction ────────────────────────────────────────────────

    def _run_translation(
        self, enriched_json_path: Path, output_dir: Path
    ) -> TranslationResult:

        data     = json.loads(enriched_json_path.read_text(encoding="utf-8"))
        segments = data.get("segments", [])

        logger.info(
            f"[Étape 4] Traduction de {len(segments)} segments "
            f"({self.config.source_language} → {self.config.target_language})…"
        )

        t0         = time.perf_counter()
        translated: list[TranslatedSegment] = []

        for batch_start in range(0, len(segments), self.config.batch_size):
            batch        = segments[batch_start: batch_start + self.config.batch_size]
            texts        = [seg.get("text", "").strip() for seg in batch]
            translations = self._translate_batch(texts)

            for seg, translated_text in zip(batch, translations):
                tone_tags  = seg.get("tone_tags", ["[NEUTRAL]"])
                tts_prompt = self._build_tts_prompt(translated_text, tone_tags)
                translated.append(TranslatedSegment(
                    id=seg["id"],
                    start=seg["start"],
                    end=seg["end"],
                    source_text=seg.get("text", ""),
                    translated_text=translated_text,
                    tone_tags=tone_tags,
                    tts_prompt=tts_prompt,
                    emotion=seg.get("emotion", "neutral"),
                    speech_rate=seg.get("speech_rate", 0.0),
                    language_src=self.config.source_language,
                    language_tgt=self.config.target_language,
                ))

        elapsed    = time.perf_counter() - t0
        model_used = self._resolve_model_name()

        out_path = output_dir / (
            enriched_json_path.stem.replace("_enriched", "") + "_translated.json"
        )
        self._save_json(translated, out_path)

        logger.info(
            f"[Étape 4] ✓ {len(translated)} segments traduits en {elapsed:.1f}s "
            f"via {model_used}"
        )

        return TranslationResult(
            success=True,
            translated_segments=translated,
            source_language=self.config.source_language,
            target_language=self.config.target_language,
            model_used=model_used,
            processing_time_s=round(elapsed, 2),
            output_json_path=out_path,
        )

    # ── Traduction d'un batch ─────────────────────────────────────────────────

    def _translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        try:
            import torch

            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )

            forced_bos = self._tokenizer.convert_tokens_to_ids(self._tgt_lang_code)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    num_beams=self.config.num_beams,
                    max_length=self.config.max_length,
                )

            return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

        except Exception as exc:
            logger.warning(f"[Étape 4] Erreur batch : {exc} — texte original conservé")
            return texts

    # ── Construction du prompt TTS ────────────────────────────────────────────

    def _build_tts_prompt(self, text: str, tone_tags: list[str]) -> str:
        if not self.config.inject_tone_tags or not tone_tags:
            return text.strip()
        prefix = "".join(tone_tags)
        return f"{prefix} {text.strip()}"

    # ── Sauvegarde ────────────────────────────────────────────────────────────

    @staticmethod
    def _save_json(segments: list[TranslatedSegment], path: Path) -> None:
        payload = {
            "segment_count":   len(segments),
            "source_language": segments[0].language_src if segments else "",
            "target_language": segments[0].language_tgt if segments else "",
            "segments":        [s.to_dict() for s in segments],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ─── Helpers publics ──────────────────────────────────────────────────────────

def get_nllb_code(iso_code: str) -> str:
    return NLLB_LANG_MAP.get(iso_code, "eng_Latn")


def is_helsinki_supported(src: str, tgt: str) -> bool:
    return (src, tgt) in HELSINKI_MODEL_MAP


def translate_segments(
    enriched_json_path: str,
    output_dir: str,
    source_language: str = "en",
    target_language: str = "fr",
    model_type: str = "nllb",
    device: str = "cpu",
) -> TranslationResult:
    config     = TranslationConfig(
        model_type=model_type,
        source_language=source_language,
        target_language=target_language,
        device=device,
    )
    translator = NLPTranslator(config)
    result     = translator.translate(enriched_json_path, output_dir)

    if result.success:
        logger.info(
            f"[Étape 4] ✓ Traduction terminée — "
            f"{result.segment_count} segments  "
            f"temps={result.processing_time_s}s"
        )
    else:
        logger.error(f"[Étape 4] ✗ Échec : {result.error_message}")

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 4:
        print(
            "Usage: python step4_translation.py "
            "<enriched.json> <output_dir> <target_lang> [source_lang] [model_type]"
        )
        sys.exit(1)

    tgt   = sys.argv[3]
    src   = sys.argv[4] if len(sys.argv) > 4 else "en"
    mtype = sys.argv[5] if len(sys.argv) > 5 else "nllb"
    res   = translate_segments(sys.argv[1], sys.argv[2], src, tgt, mtype)
    sys.exit(0 if res.success else 1)