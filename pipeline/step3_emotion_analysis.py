"""
LinguaPlay Pipeline — Étape 3 : Analyse du Ton et des Émotions
==============================================================
Analyse les segments audio pour extraire les marqueurs prosodiques
(émotion, intensité, débit) via wav2vec2 + classificateur d'émotions.

Ces métadonnées enrichissent le texte traduit (étape 4) avec des
balises de ton utilisées par le TTS (étape 5) pour reproduire
l'expressivité de l'orateur original.

Sortie : fichier JSON enrichi — chaque segment reçoit :
    {
        "id": 0,
        "start": 0.0,
        "end": 3.5,
        "text": "Hello world.",
        "emotion": "neutral",
        "emotion_confidence": 0.87,
        "intensity": 0.62,        # 0.0 (murmure) → 1.0 (cri)
        "speech_rate": 3.2,       # syllabes/seconde
        "pitch_mean": 142.5,      # Hz
        "pitch_std": 18.3,        # variabilité du pitch
        "tone_tags": ["[NEUTRAL]", "[MODERATE]"]
    }
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Émotions supportées ──────────────────────────────────────────────────────

EMOTIONS = [
    "neutral", "happy", "sad", "angry",
    "fearful", "disgusted", "surprised",
]

# Mapping émotion → balises TTS pour le clonage vocal (étape 5)
EMOTION_TONE_TAGS: dict[str, list[str]] = {
    "neutral":   ["[NEUTRAL]"],
    "happy":     ["[HAPPY]",    "[FAST]"],
    "sad":       ["[SAD]",      "[SLOW]"],
    "angry":     ["[ANGRY]",    "[LOUD]"],
    "fearful":   ["[FEARFUL]",  "[SOFT]"],
    "disgusted": ["[DISGUSTED]"],
    "surprised": ["[SURPRISED]","[HIGH_PITCH]"],
}

INTENSITY_TAGS: dict[str, str] = {
    "low":      "[SOFT]",
    "moderate": "[MODERATE]",
    "high":     "[LOUD]",
}


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class EmotionAnalysisConfig:
    """Paramètres d'analyse prosodique."""
    model_name: str      = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    device: str          = "cpu"
    sample_rate: int     = 16_000       # doit correspondre à l'étape 1
    min_segment_ms: int  = 500          # segments trop courts ignorés
    intensity_low: float = 0.35         # seuil bas intensité
    intensity_high: float = 0.65        # seuil haut intensité
    compute_pitch: bool  = True         # analyse F0 via librosa
    fallback_emotion: str = "neutral"   # émotion par défaut si échec


# ─── Modèles de données ───────────────────────────────────────────────────────

@dataclass
class ProsodyFeatures:
    """Caractéristiques prosodiques extraites d'un segment audio."""
    emotion: str
    emotion_confidence: float
    intensity: float            # 0.0 → 1.0 (RMS normalisé)
    speech_rate: float          # syllabes/seconde estimées
    pitch_mean: float           # Hz (0 si non calculé)
    pitch_std: float            # variabilité Hz
    tone_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "emotion":            self.emotion,
            "emotion_confidence": round(self.emotion_confidence, 4),
            "intensity":          round(self.intensity, 4),
            "speech_rate":        round(self.speech_rate, 2),
            "pitch_mean":         round(self.pitch_mean, 2),
            "pitch_std":          round(self.pitch_std, 2),
            "tone_tags":          self.tone_tags,
        }


@dataclass
class EnrichedSegment:
    """Segment de transcription enrichi avec les métadonnées prosodiques."""
    id: int
    start: float
    end: float
    text: str
    language: str
    prosody: ProsodyFeatures

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        return {
            "id":       self.id,
            "start":    self.start,
            "end":      self.end,
            "text":     self.text.strip(),
            "language": self.language,
            "duration": self.duration,
            **self.prosody.to_dict(),
        }


@dataclass
class EmotionAnalysisResult:
    """Résultat complet de l'analyse prosodique."""
    success: bool
    enriched_segments: list[EnrichedSegment] = field(default_factory=list)
    dominant_emotion: str | None             = None
    avg_intensity: float                     = 0.0
    avg_speech_rate: float                   = 0.0
    output_json_path: Path | None            = None
    error_message: str | None                = None

    @property
    def segment_count(self) -> int:
        return len(self.enriched_segments)


# ─── Analyseur principal ──────────────────────────────────────────────────────

class EmotionAnalyzer:
    """
    Analyse les marqueurs prosodiques de chaque segment audio.

    Flux :
        audio.wav + transcript.json
            → [wav2vec2 classifier] → émotions par segment
            → [librosa]             → pitch, intensité, débit
            → enriched_transcript.json
    """

    def __init__(self, config: EmotionAnalysisConfig | None = None):
        self.config    = config or EmotionAnalysisConfig()
        self._pipeline = None   # HuggingFace pipeline, chargé à la demande

    # ── Chargement du modèle ──────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"[Étape 3] Chargement du modèle : {self.config.model_name}")
            self._pipeline = hf_pipeline(
                "audio-classification",
                model=self.config.model_name,
                device=0 if self.config.device == "cuda" else -1,
            )
            logger.info("[Étape 3] Modèle chargé.")
        except ImportError:
            raise ImportError(
                "transformers non installé. "
                "Lancez : pip install transformers torch"
            )

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def analyze(
        self,
        audio_path: str | Path,
        transcript_json_path: str | Path,
        output_dir: str | Path,
    ) -> EmotionAnalysisResult:
        """
        Analyse les émotions et la prosodie de chaque segment.

        Args:
            audio_path           : WAV normalisé (sortie étape 1)
            transcript_json_path : JSON de transcription (sortie étape 2)
            output_dir           : dossier de sortie

        Returns:
            EmotionAnalysisResult avec segments enrichis
        """
        audio_path           = Path(audio_path)
        transcript_json_path = Path(transcript_json_path)
        output_dir           = Path(output_dir)

        # Validation
        if not audio_path.exists():
            return EmotionAnalysisResult(
                success=False,
                error_message=f"Fichier audio introuvable : {audio_path}",
            )
        if not transcript_json_path.exists():
            return EmotionAnalysisResult(
                success=False,
                error_message=f"Transcript JSON introuvable : {transcript_json_path}",
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._load_model()
            return self._run_analysis(audio_path, transcript_json_path, output_dir)
        except Exception as exc:
            logger.exception("[Étape 3] Erreur inattendue")
            return EmotionAnalysisResult(
                success=False,
                error_message=str(exc),
            )

    # ── Pipeline d'analyse ────────────────────────────────────────────────────

    def _run_analysis(
        self,
        audio_path: Path,
        transcript_json_path: Path,
        output_dir: Path,
    ) -> EmotionAnalysisResult:

        # Charger l'audio complet
        audio_array, sr = self._load_audio(audio_path)

        # Charger la transcription
        transcript = json.loads(transcript_json_path.read_text(encoding="utf-8"))
        segments   = transcript.get("segments", [])
        language   = transcript.get("language", "en")

        logger.info(f"[Étape 3] Analyse de {len(segments)} segments…")

        enriched: list[EnrichedSegment] = []
        for seg in segments:
            prosody = self._analyze_segment(audio_array, sr, seg)
            enriched.append(EnrichedSegment(
                id=seg["id"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                language=language,
                prosody=prosody,
            ))
            logger.debug(
                f"  Segment {seg['id']} → {prosody.emotion} "
                f"({prosody.emotion_confidence:.0%})"
            )

        # Statistiques globales
        dominant_emotion = self._dominant_emotion(enriched)
        avg_intensity    = float(np.mean([e.prosody.intensity for e in enriched])) if enriched else 0.0
        avg_speech_rate  = float(np.mean([e.prosody.speech_rate for e in enriched])) if enriched else 0.0

        # Sauvegarde
        out_path = output_dir / (audio_path.stem + "_enriched.json")
        self._save_json(enriched, dominant_emotion, avg_intensity, avg_speech_rate, out_path)

        logger.info(
            f"[Étape 3] ✓ {len(enriched)} segments enrichis — "
            f"émotion dominante : {dominant_emotion} — "
            f"intensité moy. : {avg_intensity:.2f}"
        )

        return EmotionAnalysisResult(
            success=True,
            enriched_segments=enriched,
            dominant_emotion=dominant_emotion,
            avg_intensity=avg_intensity,
            avg_speech_rate=avg_speech_rate,
            output_json_path=out_path,
        )

    # ── Analyse d'un segment ──────────────────────────────────────────────────

    def _analyze_segment(
        self, audio_array: np.ndarray, sr: int, seg: dict
    ) -> ProsodyFeatures:
        """Extrait les features prosodiques d'un segment."""

        start_idx = int(seg["start"] * sr)
        end_idx   = int(seg["end"]   * sr)
        segment_audio = audio_array[start_idx:end_idx]

        duration_ms = (seg["end"] - seg["start"]) * 1000

        # Segment trop court → fallback
        if duration_ms < self.config.min_segment_ms or len(segment_audio) == 0:
            return self._fallback_prosody(seg)

        # Émotion via wav2vec2
        emotion, confidence = self._classify_emotion(segment_audio, sr)

        # Intensité (RMS normalisé)
        intensity = self._compute_intensity(segment_audio)

        # Pitch via librosa
        pitch_mean, pitch_std = (0.0, 0.0)
        if self.config.compute_pitch:
            pitch_mean, pitch_std = self._compute_pitch(segment_audio, sr)

        # Débit de parole (estimation par longueur du texte)
        speech_rate = self._estimate_speech_rate(
            seg.get("text", ""), seg["end"] - seg["start"]
        )

        # Balises de ton
        tone_tags = self._build_tone_tags(emotion, intensity)

        return ProsodyFeatures(
            emotion=emotion,
            emotion_confidence=confidence,
            intensity=intensity,
            speech_rate=speech_rate,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            tone_tags=tone_tags,
        )

    # ── Classification d'émotion ──────────────────────────────────────────────

    def _classify_emotion(
        self, audio: np.ndarray, sr: int
    ) -> tuple[str, float]:
        """Classifie l'émotion via le pipeline HuggingFace."""
        try:
            inputs = {"array": audio.astype(np.float32), "sampling_rate": sr}
            preds  = self._pipeline(inputs, top_k=1)
            label  = preds[0]["label"].lower()
            score  = float(preds[0]["score"])
            # Normaliser le label vers nos émotions connues
            emotion = label if label in EMOTIONS else self.config.fallback_emotion
            return emotion, score
        except Exception:
            return self.config.fallback_emotion, 0.5

    # ── Intensité (RMS) ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_intensity(audio: np.ndarray) -> float:
        """Calcule l'intensité RMS normalisée entre 0 et 1."""
        if len(audio) == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        return float(np.clip(rms / 0.3, 0.0, 1.0))  # 0.3 = RMS max typique

    # ── Pitch (F0) via librosa ────────────────────────────────────────────────

    @staticmethod
    def _compute_pitch(audio: np.ndarray, sr: int) -> tuple[float, float]:
        """Estime la fréquence fondamentale F0 via librosa."""
        try:
            import librosa
            f0, _, _ = librosa.pyin(
                audio.astype(np.float32),
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            voiced = f0[~np.isnan(f0)]
            if len(voiced) == 0:
                return 0.0, 0.0
            return float(np.mean(voiced)), float(np.std(voiced))
        except ImportError:
            return 0.0, 0.0
        except Exception:
            return 0.0, 0.0

    # ── Débit de parole ───────────────────────────────────────────────────────

    @staticmethod
    def _estimate_speech_rate(text: str, duration_seconds: float) -> float:
        """Estime le débit en syllabes/seconde (approximation par mots)."""
        if duration_seconds <= 0:
            return 0.0
        words         = text.strip().split()
        word_count    = len(words)
        # Approximation : 1,5 syllabe par mot en moyenne
        syllable_est  = word_count * 1.5
        return round(syllable_est / duration_seconds, 2)

    # ── Balises de ton ────────────────────────────────────────────────────────

    def _build_tone_tags(self, emotion: str, intensity: float) -> list[str]:
        """Construit les balises TTS à partir de l'émotion et de l'intensité."""
        tags = list(EMOTION_TONE_TAGS.get(emotion, ["[NEUTRAL]"]))

        if intensity < self.config.intensity_low:
            intensity_label = "low"
        elif intensity > self.config.intensity_high:
            intensity_label = "high"
        else:
            intensity_label = "moderate"

        intensity_tag = INTENSITY_TAGS[intensity_label]
        if intensity_tag not in tags:
            tags.append(intensity_tag)

        return tags

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback_prosody(self, seg: dict) -> ProsodyFeatures:
        """Prosodie par défaut pour les segments trop courts."""
        return ProsodyFeatures(
            emotion=self.config.fallback_emotion,
            emotion_confidence=0.5,
            intensity=0.5,
            speech_rate=self._estimate_speech_rate(
                seg.get("text", ""), seg.get("end", 0) - seg.get("start", 0)
            ),
            pitch_mean=0.0,
            pitch_std=0.0,
            tone_tags=["[NEUTRAL]", "[MODERATE]"],
        )

    # ── Chargement audio ──────────────────────────────────────────────────────

    def _load_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """Charge le fichier WAV en array numpy."""
        try:
            import soundfile as sf
            audio, sr = sf.read(str(audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # stéréo → mono
            return audio, sr
        except ImportError:
            raise ImportError(
                "soundfile non installé. "
                "Lancez : pip install soundfile"
            )

    # ── Statistiques globales ─────────────────────────────────────────────────

    @staticmethod
    def _dominant_emotion(segments: list[EnrichedSegment]) -> str | None:
        if not segments:
            return None
        counts: dict[str, int] = {}
        for seg in segments:
            counts[seg.prosody.emotion] = counts.get(seg.prosody.emotion, 0) + 1
        return max(counts, key=lambda k: counts[k])

    # ── Sauvegarde JSON ───────────────────────────────────────────────────────

    @staticmethod
    def _save_json(
        segments: list[EnrichedSegment],
        dominant_emotion: str | None,
        avg_intensity: float,
        avg_speech_rate: float,
        path: Path,
    ) -> None:
        payload = {
            "dominant_emotion": dominant_emotion,
            "avg_intensity":    round(avg_intensity, 4),
            "avg_speech_rate":  round(avg_speech_rate, 2),
            "segment_count":    len(segments),
            "segments":         [s.to_dict() for s in segments],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ─── Fonction publique ────────────────────────────────────────────────────────

def analyze_emotions(
    audio_path: str,
    transcript_json_path: str,
    output_dir: str,
    model_name: str | None = None,
    device: str = "cpu",
) -> EmotionAnalysisResult:
    """
    Fonction publique principale — appelée par l'orchestrateur.

    Args:
        audio_path           : WAV normalisé (sortie étape 1)
        transcript_json_path : JSON de transcription (sortie étape 2)
        output_dir           : dossier de sortie
        model_name           : modèle HuggingFace (None = défaut)
        device               : cpu | cuda

    Returns:
        EmotionAnalysisResult
    """
    cfg = EmotionAnalysisConfig(device=device)
    if model_name:
        cfg.model_name = model_name

    analyzer = EmotionAnalyzer(cfg)
    result   = analyzer.analyze(audio_path, transcript_json_path, output_dir)

    if result.success:
        logger.info(
            f"[Étape 3] ✓ Analyse terminée — "
            f"émotion dominante={result.dominant_emotion}  "
            f"segments={result.segment_count}"
        )
    else:
        logger.error(f"[Étape 3] ✗ Échec : {result.error_message}")

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 4:
        print("Usage: python step3_emotion_analysis.py <audio.wav> <transcript.json> <output_dir>")
        sys.exit(1)

    res = analyze_emotions(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0 if res.success else 1)
