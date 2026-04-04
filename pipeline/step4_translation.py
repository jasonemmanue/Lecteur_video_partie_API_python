"""
LinguaPlay Pipeline — Étape 4 : Traduction NLP
===============================================
Traduit les segments transcrits vers la langue cible en injectant
les métadonnées de ton (étape 3) comme prompts de contrôle TTS.

Modèles supportés :
  - Helsinki-NLP/opus-mt-{src}-{tgt}  (léger, +200 paires)
  - facebook/nllb-200-distilled-600M   (multilingue, recommandé V1)
  - facebook/nllb-200-1.3B             (meilleure qualité, GPU)

Sortie : fichier JSON avec segments traduits + balises de ton :
    {
        "id": 0,
        "start": 0.0,
        "end": 3.5,
        "source_text": "Hello and welcome.",
        "translated_text": "Bonjour et bienvenue.",
        "tone_tags": ["[NEUTRAL]", "[MODERATE]"],
        "tts_prompt": "[NEUTRAL][MODERATE] Bonjour et bienvenue.",
        "emotion": "neutral",
        "speech_rate": 3.2
    }
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Codes de langue NLLB-200 ────────────────────────────────────────────────
# Mapping code ISO 639-1 → code NLLB flores200

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

# Modèles Helsinki-NLP disponibles par paire de langues
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

# Modèle NLLB par défaut (couvre toutes les langues V1)
NLLB_DEFAULT = "facebook/nllb-200-distilled-600M"


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TranslationConfig:
    """Paramètres de traduction NLP."""
    model_type: str          = "nllb"      # "nllb" | "helsinki"
    model_name: str          = NLLB_DEFAULT
    device: str              = "cpu"       # cpu | cuda
    max_length: int          = 512         # tokens max en sortie
    num_beams: int           = 4           # beam search
    batch_size: int          = 8           # segments traités en batch
    inject_tone_tags: bool   = True        # injecter balises TTS dans tts_prompt
    source_language: str     = "en"        # code ISO 639-1
    target_language: str     = "fr"        # code ISO 639-1


# ─── Modèles de données ───────────────────────────────────────────────────────

@dataclass
class TranslatedSegment:
    """Segment traduit avec métadonnées pour le TTS."""
    id: int
    start: float
    end: float
    source_text: str
    translated_text: str
    tone_tags: list[str]
    tts_prompt: str          # balises + texte traduit → entrée TTS étape 5
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
    """Résultat complet de la traduction."""
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
    """
    Traduit les segments enrichis vers la langue cible.

    Flux :
        enriched_transcript.json
            → [Helsinki-NLP ou NLLB-200]
            → segments traduits + tts_prompt
            → translated_transcript.json
    """

    def __init__(self, config: TranslationConfig | None = None):
        self.config    = config or TranslationConfig()
        self._pipeline = None   # chargé à la demande

    # ── Chargement du modèle ──────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            model_name = self._resolve_model_name()
            logger.info(f"[Étape 4] Chargement du modèle : {model_name}")

            kwargs: dict = dict(
                task="translation",
                model=model_name,
                device=0 if self.config.device == "cuda" else -1,
                max_length=self.config.max_length,
            )

            # NLLB nécessite les codes flores200
            if self.config.model_type == "nllb":
                src = NLLB_LANG_MAP.get(self.config.source_language, "eng_Latn")
                tgt = NLLB_LANG_MAP.get(self.config.target_language, "fra_Latn")
                kwargs["src_lang"] = src
                kwargs["tgt_lang"] = tgt

            self._pipeline = hf_pipeline(**kwargs)
            logger.info("[Étape 4] Modèle chargé.")

        except ImportError:
            raise ImportError(
                "transformers non installé. "
                "Lancez : pip install transformers sentencepiece"
            )

    def _resolve_model_name(self) -> str:
        """Résout le nom du modèle selon le type et la paire de langues."""
        if self.config.model_type == "helsinki":
            pair = (self.config.source_language, self.config.target_language)
            model = HELSINKI_MODEL_MAP.get(pair)
            if model is None:
                raise ValueError(
                    f"Paire de langues non supportée par Helsinki-NLP : "
                    f"{self.config.source_language} → {self.config.target_language}. "
                    f"Utilisez model_type='nllb' pour cette paire."
                )
            return model
        return self.config.model_name   # NLLB ou modèle personnalisé

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def translate(
        self,
        enriched_json_path: str | Path,
        output_dir: str | Path,
    ) -> TranslationResult:
        """
        Traduit les segments du JSON enrichi (sortie étape 3).

        Args:
            enriched_json_path : JSON enrichi avec émotions (étape 3)
            output_dir         : dossier de sortie

        Returns:
            TranslationResult avec segments traduits
        """
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
            return TranslationResult(
                success=False,
                error_message=str(exc),
            )

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

        t0 = time.perf_counter()
        translated: list[TranslatedSegment] = []

        # Traduction par batch
        for batch_start in range(0, len(segments), self.config.batch_size):
            batch = segments[batch_start: batch_start + self.config.batch_size]
            texts = [seg.get("text", "").strip() for seg in batch]

            # Appel au modèle
            translations = self._translate_batch(texts)

            for seg, translated_text in zip(batch, translations):
                tone_tags  = seg.get("tone_tags", ["[NEUTRAL]"])
                tts_prompt = self._build_tts_prompt(
                    translated_text, tone_tags
                )
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

            logger.debug(
                f"[Étape 4] Batch {batch_start // self.config.batch_size + 1} "
                f"→ {len(batch)} segments traduits"
            )

        elapsed = time.perf_counter() - t0
        model_used = self._resolve_model_name()

        # Sauvegarde
        out_path = output_dir / (enriched_json_path.stem.replace("_enriched", "") + "_translated.json")
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
        """Traduit un batch de textes via le pipeline HuggingFace."""
        if not texts:
            return []
        try:
            kwargs: dict = {}
            if self.config.model_type == "nllb":
                tgt_code = NLLB_LANG_MAP.get(self.config.target_language, "fra_Latn")
                kwargs["tgt_lang"] = tgt_code

            results = self._pipeline(texts, **kwargs)

            # Résultat : liste de dicts ou liste de listes de dicts
            translated = []
            for res in results:
                if isinstance(res, list):
                    translated.append(res[0].get("translation_text", ""))
                else:
                    translated.append(res.get("translation_text", ""))
            return translated

        except Exception as exc:
            logger.warning(f"[Étape 4] Erreur batch : {exc} — texte original conservé")
            return texts   # fallback : texte source non traduit

    # ── Construction du prompt TTS ────────────────────────────────────────────

    def _build_tts_prompt(self, text: str, tone_tags: list[str]) -> str:
        """
        Construit le prompt TTS en injectant les balises de ton.
        Ex : "[HAPPY][FAST] Bonjour et bienvenue !"
        """
        if not self.config.inject_tone_tags or not tone_tags:
            return text.strip()
        prefix = "".join(tone_tags)
        return f"{prefix} {text.strip()}"

    # ── Sauvegarde ────────────────────────────────────────────────────────────

    @staticmethod
    def _save_json(segments: list[TranslatedSegment], path: Path) -> None:
        payload = {
            "segment_count": len(segments),
            "source_language": segments[0].language_src if segments else "",
            "target_language": segments[0].language_tgt if segments else "",
            "segments": [s.to_dict() for s in segments],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ─── Helpers publics ──────────────────────────────────────────────────────────

def get_nllb_code(iso_code: str) -> str:
    """Retourne le code NLLB flores200 pour un code ISO 639-1."""
    return NLLB_LANG_MAP.get(iso_code, "eng_Latn")


def is_helsinki_supported(src: str, tgt: str) -> bool:
    """Vérifie si la paire de langues est supportée par Helsinki-NLP."""
    return (src, tgt) in HELSINKI_MODEL_MAP


# ─── Fonction publique ────────────────────────────────────────────────────────

def translate_segments(
    enriched_json_path: str,
    output_dir: str,
    source_language: str = "en",
    target_language: str = "fr",
    model_type: str = "nllb",
    device: str = "cpu",
) -> TranslationResult:
    """
    Fonction publique principale — appelée par l'orchestrateur.

    Args:
        enriched_json_path : JSON enrichi (sortie étape 3)
        output_dir         : dossier de sortie
        source_language    : code ISO 639-1 source (ex: "en")
        target_language    : code ISO 639-1 cible  (ex: "fr")
        model_type         : "nllb" | "helsinki"
        device             : cpu | cuda

    Returns:
        TranslationResult
    """
    config = TranslationConfig(
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

    tgt    = sys.argv[3]
    src    = sys.argv[4] if len(sys.argv) > 4 else "en"
    mtype  = sys.argv[5] if len(sys.argv) > 5 else "nllb"
    res    = translate_segments(sys.argv[1], sys.argv[2], src, tgt, mtype)
    sys.exit(0 if res.success else 1)
