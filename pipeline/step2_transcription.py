"""
LinguaPlay Pipeline — Étape 2 : Transcription Speech-to-Text (Whisper)
=======================================================================
Transcrit l'audio normalisé (produit par l'étape 1) en texte segmenté
avec horodatage, via faster-whisper (OpenAI Whisper optimisé CPU/GPU).

Sortie : fichier JSON de segments + fichier SRT brut
Format segment :
    {
        "id": 0,
        "start": 0.0,
        "end":   3.24,
        "text":  "Hello and welcome to this video.",
        "words": [...],          # optionnel, si word_timestamps=True
        "language": "en",
        "confidence": 0.97
    }
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TranscriptionConfig:
    """Paramètres de transcription Whisper."""
    model_size: str        = "medium"     # tiny|base|small|medium|large-v2
    device: str            = "cpu"        # cpu | cuda
    compute_type: str      = "int8"       # int8 (CPU) | float16 (GPU)
    language: str | None   = None         # None = détection auto
    word_timestamps: bool  = True         # horodatage mot à mot
    beam_size: int         = 5
    vad_filter: bool       = True         # Voice Activity Detection
    vad_threshold: float   = 0.5
    min_silence_ms: int    = 500          # ms de silence minimum entre segments
    task: str              = "transcribe" # transcribe | translate (→ anglais)


# ─── Modèles de données ───────────────────────────────────────────────────────

@dataclass
class TranscriptionSegment:
    """Un segment transcrit avec son horodatage."""
    id: int
    start: float
    end: float
    text: str
    language: str
    confidence: float
    words: list[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "start":      self.start,
            "end":        self.end,
            "text":       self.text.strip(),
            "language":   self.language,
            "confidence": round(self.confidence, 4),
            "duration":   self.duration,
            "words":      self.words,
        }

    def to_srt_block(self) -> str:
        """Convertit le segment en bloc SRT."""
        def fmt(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        return (
            f"{self.id + 1}\n"
            f"{fmt(self.start)} --> {fmt(self.end)}\n"
            f"{self.text.strip()}\n"
        )


@dataclass
class TranscriptionResult:
    """Résultat complet de la transcription."""
    success: bool
    segments: list[TranscriptionSegment]    = field(default_factory=list)
    language: str | None                    = None
    language_probability: float             = 0.0
    duration_seconds: float                 = 0.0
    json_path: Path | None                  = None
    srt_path: Path | None                   = None
    error_message: str | None               = None
    model_size: str                         = "medium"

    @property
    def full_text(self) -> str:
        """Texte complet reconstitué depuis les segments."""
        return " ".join(s.text.strip() for s in self.segments)

    @property
    def segment_count(self) -> int:
        return len(self.segments)


# ─── Transcripteur principal ──────────────────────────────────────────────────

class WhisperTranscriber:
    """
    Transcrit un fichier audio WAV via faster-whisper.

    Flux :
        audio.wav → [faster-whisper] → segments horodatés → JSON + SRT
    """

    def __init__(self, config: TranscriptionConfig | None = None):
        self.config = config or TranscriptionConfig()
        self._model = None  # chargé à la demande (lazy loading)

    def _load_model(self):
        """Charge le modèle Whisper (une seule fois)."""
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            logger.info(
                f"[Étape 2] Chargement du modèle Whisper "
                f"'{self.config.model_size}' sur {self.config.device}…"
            )
            self._model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            logger.info("[Étape 2] Modèle chargé.")
        except ImportError:
            raise ImportError(
                "faster-whisper non installé. "
                "Lancez : pip install faster-whisper"
            )

    def transcribe(
        self,
        audio_path: str | Path,
        output_dir: str | Path,
    ) -> TranscriptionResult:
        """
        Transcrit un fichier audio et écrit les sorties JSON + SRT.

        Args:
            audio_path : chemin vers le WAV normalisé (sortie étape 1)
            output_dir : dossier de sortie pour JSON et SRT

        Returns:
            TranscriptionResult avec segments, langue détectée, chemins
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)

        if not audio_path.exists():
            return TranscriptionResult(
                success=False,
                error_message=f"Fichier audio introuvable : {audio_path}",
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._load_model()
            return self._run_transcription(audio_path, output_dir)
        except Exception as exc:
            logger.exception("[Étape 2] Erreur inattendue lors de la transcription")
            return TranscriptionResult(
                success=False,
                error_message=str(exc),
            )

    def _run_transcription(
        self, audio_path: Path, output_dir: Path
    ) -> TranscriptionResult:
        """Exécute la transcription et sauvegarde les fichiers."""

        logger.info(f"[Étape 2] Transcription de : {audio_path.name}")

        # ── Transcription via faster-whisper ──────────────────────────────
        vad_params = {
            "vad_filter": self.config.vad_filter,
            "vad_parameters": {
                "threshold": self.config.vad_threshold,
                "min_silence_duration_ms": self.config.min_silence_ms,
            },
        }

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=self.config.language,
            task=self.config.task,
            beam_size=self.config.beam_size,
            word_timestamps=self.config.word_timestamps,
            **vad_params,
        )

        detected_lang = info.language
        lang_prob     = info.language_probability
        duration      = info.duration

        logger.info(
            f"[Étape 2] Langue détectée : {detected_lang} "
            f"(confiance {lang_prob:.2%}) — durée : {duration:.1f}s"
        )

        # ── Itération sur les segments ─────────────────────────────────────
        segments: list[TranscriptionSegment] = []
        for seg in segments_iter:
            words = []
            if self.config.word_timestamps and seg.words:
                words = [
                    {"word": w.word, "start": w.start, "end": w.end,
                     "probability": round(w.probability, 4)}
                    for w in seg.words
                ]

            avg_confidence = (
                sum(w["probability"] for w in words) / len(words)
                if words else lang_prob
            )

            segments.append(TranscriptionSegment(
                id=seg.id,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text,
                language=detected_lang,
                confidence=avg_confidence,
                words=words,
            ))

        logger.info(f"[Étape 2] {len(segments)} segments transcrits.")

        # ── Sauvegarde JSON ───────────────────────────────────────────────
        json_path = output_dir / (audio_path.stem + "_transcript.json")
        self._save_json(segments, detected_lang, lang_prob, duration, json_path)

        # ── Sauvegarde SRT ────────────────────────────────────────────────
        srt_path = output_dir / (audio_path.stem + "_transcript.srt")
        self._save_srt(segments, srt_path)

        logger.info(f"[Étape 2] ✓ JSON → {json_path.name}")
        logger.info(f"[Étape 2] ✓ SRT  → {srt_path.name}")

        return TranscriptionResult(
            success=True,
            segments=segments,
            language=detected_lang,
            language_probability=lang_prob,
            duration_seconds=duration,
            json_path=json_path,
            srt_path=srt_path,
            model_size=self.config.model_size,
        )

    # ── Sauvegarde ─────────────────────────────────────────────────────────────

    @staticmethod
    def _save_json(
        segments: list[TranscriptionSegment],
        language: str,
        lang_prob: float,
        duration: float,
        path: Path,
    ) -> None:
        payload = {
            "language":             language,
            "language_probability": round(lang_prob, 4),
            "duration_seconds":     round(duration, 3),
            "segment_count":        len(segments),
            "segments":             [s.to_dict() for s in segments],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _save_srt(segments: list[TranscriptionSegment], path: Path) -> None:
        srt_content = "\n".join(s.to_srt_block() for s in segments)
        path.write_text(srt_content, encoding="utf-8")


# ─── Fonction publique (appelée par l'orchestrateur) ─────────────────────────

def transcribe_audio(
    audio_path: str,
    output_dir: str,
    model_size: str = "medium",
    language: str | None = None,
    device: str = "cpu",
) -> TranscriptionResult:
    """
    Fonction publique principale — appelée par l'orchestrateur du pipeline.

    Args:
        audio_path  : chemin WAV normalisé (sortie étape 1)
        output_dir  : dossier de sortie
        model_size  : taille du modèle Whisper (tiny/base/small/medium/large-v2)
        language    : code langue ISO (None = détection auto)
        device      : cpu | cuda

    Returns:
        TranscriptionResult
    """
    config = TranscriptionConfig(
        model_size=model_size,
        language=language,
        device=device,
    )
    transcriber = WhisperTranscriber(config)
    result      = transcriber.transcribe(audio_path, output_dir)

    if result.success:
        logger.info(
            f"[Étape 2] ✓ Transcription terminée — "
            f"langue={result.language}  segments={result.segment_count}  "
            f"durée={result.duration_seconds:.1f}s"
        )
    else:
        logger.error(f"[Étape 2] ✗ Échec : {result.error_message}")

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 3:
        print("Usage: python step2_transcription.py <audio.wav> <output_dir> [model_size]")
        sys.exit(1)

    model = sys.argv[3] if len(sys.argv) > 3 else "medium"
    res   = transcribe_audio(sys.argv[1], sys.argv[2], model_size=model)
    sys.exit(0 if res.success else 1)
