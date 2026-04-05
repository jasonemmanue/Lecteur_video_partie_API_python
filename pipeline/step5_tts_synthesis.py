"""
LinguaPlay Pipeline — Étape 5 : Synthèse Vocale (edge-tts)
===========================================================
Génère l'audio traduit via Microsoft edge-tts (compatible Python 3.14+).
Chaque segment est synthétisé individuellement puis assemblé.

Flux :
    translated_transcript.json
        → [edge-tts]
        → segments WAV synthétisés
        → audio_translated.wav (concat + ajustement durée)

Sortie :
    - audio_tts.wav          : piste audio complète traduite
    - segments/seg_000.wav   : segments individuels (debug)
    - tts_manifest.json      : métadonnées de synthèse par segment
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ─── Voix edge-tts par langue ─────────────────────────────────────────────────

VOICE_MAP: dict[str, str] = {
    "fr": "fr-FR-DeniseNeural",
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "de": "de-DE-KatjaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "pt": "pt-BR-FranciscaNeural",
}


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TTSConfig:
    """Paramètres de synthèse vocale."""
    model_name: str            = "edge-tts"
    device: str                = "cpu"
    speaker_sample_duration: float = 6.0
    temperature: float         = 0.65
    length_penalty: float      = 1.0
    repetition_penalty: float  = 10.0
    top_k: int                 = 50
    top_p: float               = 0.85
    speed: float               = 1.0
    sample_rate: int           = 24_000
    output_sample_rate: int    = 16_000
    silence_between_ms: int    = 150
    save_segments: bool        = True


# ─── Modèles de données ───────────────────────────────────────────────────────

@dataclass
class SynthesizedSegment:
    """Résultat de synthèse d'un segment."""
    id: int
    start: float
    end: float
    tts_prompt: str
    audio_path: Path | None
    duration_synthesized: float
    duration_target: float
    speed_ratio: float
    success: bool
    error: str | None = None

    @property
    def duration_diff(self) -> float:
        return round(self.duration_synthesized - self.duration_target, 3)

    def to_dict(self) -> dict:
        return {
            "id":                   self.id,
            "start":                self.start,
            "end":                  self.end,
            "tts_prompt":           self.tts_prompt,
            "audio_path":           str(self.audio_path) if self.audio_path else None,
            "duration_synthesized": round(self.duration_synthesized, 3),
            "duration_target":      round(self.duration_target, 3),
            "speed_ratio":          round(self.speed_ratio, 4),
            "duration_diff":        self.duration_diff,
            "success":              self.success,
            "error":                self.error,
        }


@dataclass
class TTSResult:
    """Résultat complet de la synthèse vocale."""
    success: bool
    audio_path: Path | None                        = None
    synthesized_segments: list[SynthesizedSegment] = field(default_factory=list)
    manifest_path: Path | None                     = None
    total_duration_s: float                        = 0.0
    processing_time_s: float                       = 0.0
    sample_rate: int                               = 16_000
    error_message: str | None                      = None

    @property
    def segment_count(self) -> int:
        return len(self.synthesized_segments)

    @property
    def success_rate(self) -> float:
        if not self.synthesized_segments:
            return 0.0
        ok = sum(1 for s in self.synthesized_segments if s.success)
        return ok / len(self.synthesized_segments)


# ─── Synthétiseur principal ───────────────────────────────────────────────────

class XTTSSynthesizer:
    """Synthèse vocale multilingue via edge-tts (compatible Python 3.14+)."""

    def __init__(self, config: TTSConfig | None = None):
        self.config     = config or TTSConfig()
        self._tts       = None
        self._loaded    = False
        self._edge_tts  = None

    # ── Chargement du modèle ──────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._loaded:
            return
        try:
            import edge_tts
            self._edge_tts = edge_tts
            self._loaded   = True
            logger.info("[Étape 5] edge-tts chargé.")
        except ImportError:
            raise ImportError(
                "edge-tts non installé. Lancez : pip install edge-tts"
            )

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def synthesize(
        self,
        translated_json_path: str | Path,
        speaker_audio_path: str | Path,
        output_dir: str | Path,
    ) -> TTSResult:
        translated_json_path = Path(translated_json_path)
        speaker_audio_path   = Path(speaker_audio_path)
        output_dir           = Path(output_dir)

        if not translated_json_path.exists():
            return TTSResult(
                success=False,
                error_message=f"JSON traduit introuvable : {translated_json_path}",
            )
        if not speaker_audio_path.exists():
            return TTSResult(
                success=False,
                error_message=f"Audio de référence introuvable : {speaker_audio_path}",
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_segments:
            (output_dir / "segments").mkdir(exist_ok=True)

        try:
            self._load_model()
            return self._run_synthesis(
                translated_json_path, speaker_audio_path, output_dir
            )
        except Exception as exc:
            logger.exception("[Étape 5] Erreur inattendue lors de la synthèse")
            return TTSResult(success=False, error_message=str(exc))

    # ── Pipeline de synthèse ──────────────────────────────────────────────────

    def _run_synthesis(
        self,
        translated_json_path: Path,
        speaker_audio_path: Path,
        output_dir: Path,
    ) -> TTSResult:

        data     = json.loads(translated_json_path.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
        tgt_lang = data.get("target_language", "en")

        logger.info(
            f"[Étape 5] Synthèse de {len(segments)} segments "
            f"en '{tgt_lang}' via edge-tts"
        )

        t0 = time.perf_counter()
        synthesized: list[SynthesizedSegment] = []
        audio_chunks: list[np.ndarray]        = []
        silence = self._make_silence(self.config.silence_between_ms)

        for seg in segments:
            seg_id       = seg["id"]
            tts_prompt   = seg.get("tts_prompt", seg.get("translated_text", ""))
            duration_tgt = seg.get("duration", seg["end"] - seg["start"])

            logger.debug(f"  Segment {seg_id} : {tts_prompt[:50]}…")

            synth_seg = self._synthesize_segment(
                text=tts_prompt,
                speaker_wav=str(speaker_audio_path),
                language=tgt_lang,
                seg_id=seg_id,
                duration_target=duration_tgt,
                output_dir=output_dir,
            )
            synthesized.append(synth_seg)

            if synth_seg.success and synth_seg.audio_path:
                chunk = self._load_audio_array(synth_seg.audio_path)
                audio_chunks.append(chunk)
                audio_chunks.append(silence)

        # Assemblage de la piste complète
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            out_sr     = self.config.output_sample_rate
            full_audio = self._resample_if_needed(
                full_audio, self.config.sample_rate, out_sr
            )
        else:
            full_audio = np.zeros(self.config.output_sample_rate, dtype=np.float32)
            out_sr     = self.config.output_sample_rate

        out_audio_path = output_dir / (
            translated_json_path.stem.replace("_translated", "") + "_tts.wav"
        )
        self._save_wav(full_audio, out_sr, out_audio_path)

        total_duration  = len(full_audio) / out_sr
        processing_time = time.perf_counter() - t0

        manifest_path = output_dir / (out_audio_path.stem + "_manifest.json")
        self._save_manifest(synthesized, manifest_path)

        logger.info(
            f"[Étape 5] ✓ {len(synthesized)} segments synthétisés — "
            f"durée={total_duration:.1f}s  "
            f"taux de succès={self._success_rate(synthesized):.0%}  "
            f"temps={processing_time:.1f}s"
        )

        return TTSResult(
            success=True,
            audio_path=out_audio_path,
            synthesized_segments=synthesized,
            manifest_path=manifest_path,
            total_duration_s=round(total_duration, 2),
            processing_time_s=round(processing_time, 2),
            sample_rate=out_sr,
        )

    # ── Synthèse d'un segment ─────────────────────────────────────────────────

    def _synthesize_segment(
        self,
        text: str,
        speaker_wav: str,
        language: str,
        seg_id: int,
        duration_target: float,
        output_dir: Path,
    ) -> SynthesizedSegment:
        """Synthétise un segment via edge-tts et l'ajuste à la durée cible."""

        seg_path = output_dir / "segments" / f"seg_{seg_id:03d}.wav"

        try:
            voice = VOICE_MAP.get(language, "en-US-JennyNeural")

            communicate = self._edge_tts.Communicate(text, voice)
            asyncio.run(communicate.save(str(seg_path)))

            # Mesure durée réelle
            audio_arr    = self._load_audio_array(seg_path)
            duration_syn = len(audio_arr) / self.config.sample_rate

            # Ajustement temporel si l'écart dépasse 10%
            speed_ratio = 1.0
            if duration_target > 0:
                ratio = duration_syn / duration_target
                if abs(ratio - 1.0) > 0.1:
                    speed_ratio = ratio
                    audio_arr   = self._time_stretch(audio_arr, ratio)
                    self._save_wav(audio_arr, self.config.sample_rate, seg_path)
                    duration_syn = len(audio_arr) / self.config.sample_rate

            return SynthesizedSegment(
                id=seg_id,
                start=0.0,
                end=duration_syn,
                tts_prompt=text,
                audio_path=seg_path if self.config.save_segments else None,
                duration_synthesized=duration_syn,
                duration_target=duration_target,
                speed_ratio=speed_ratio,
                success=True,
            )

        except Exception as exc:
            logger.warning(f"[Étape 5] Segment {seg_id} échoué : {exc}")
            return SynthesizedSegment(
                id=seg_id,
                start=0.0,
                end=0.0,
                tts_prompt=text,
                audio_path=None,
                duration_synthesized=0.0,
                duration_target=duration_target,
                speed_ratio=1.0,
                success=False,
                error=str(exc),
            )

    # ── Ajustement temporel ───────────────────────────────────────────────────

    @staticmethod
    def _time_stretch(audio: np.ndarray, ratio: float) -> np.ndarray:
        try:
            import pyrubberband as pyrb
            return pyrb.time_stretch(audio, 24_000, 1.0 / ratio).astype(np.float32)
        except ImportError:
            pass
        try:
            from scipy.signal import resample
            target_len = int(len(audio) / ratio)
            return resample(audio, target_len).astype(np.float32)
        except ImportError:
            target_len = int(len(audio) / ratio)
            indices    = np.linspace(0, len(audio) - 1, target_len)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    # ── Rééchantillonnage ─────────────────────────────────────────────────────

    @staticmethod
    def _resample_if_needed(
        audio: np.ndarray, src_sr: int, tgt_sr: int
    ) -> np.ndarray:
        if src_sr == tgt_sr:
            return audio
        try:
            import librosa
            return librosa.resample(audio, orig_sr=src_sr, target_sr=tgt_sr)
        except ImportError:
            from scipy.signal import resample
            target_len = int(len(audio) * tgt_sr / src_sr)
            return resample(audio, target_len).astype(np.float32)

    # ── Silence inter-segments ────────────────────────────────────────────────

    def _make_silence(self, duration_ms: int) -> np.ndarray:
        n_samples = int(self.config.sample_rate * duration_ms / 1000)
        return np.zeros(n_samples, dtype=np.float32)

    # ── Chargement / sauvegarde audio ─────────────────────────────────────────

    @staticmethod
    def _load_audio_array(path: Path) -> np.ndarray:
        try:
            import soundfile as sf
            audio, _ = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio
        except ImportError:
            raise ImportError("soundfile non installé : pip install soundfile")

    @staticmethod
    def _save_wav(audio: np.ndarray, sr: int, path: Path) -> None:
        try:
            import soundfile as sf
            sf.write(str(path), audio, sr)
        except ImportError:
            raise ImportError("soundfile non installé : pip install soundfile")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _success_rate(segments: list[SynthesizedSegment]) -> float:
        if not segments:
            return 0.0
        return sum(1 for s in segments if s.success) / len(segments)

    @staticmethod
    def _save_manifest(segments: list[SynthesizedSegment], path: Path) -> None:
        payload = {
            "segment_count": len(segments),
            "success_count": sum(1 for s in segments if s.success),
            "segments":      [s.to_dict() for s in segments],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ─── Helpers publics ──────────────────────────────────────────────────────────

def extract_speaker_sample(
    audio_path: str | Path,
    output_path: str | Path,
    duration_s: float = 6.0,
    offset_s: float = 0.0,
) -> bool:
    try:
        import soundfile as sf
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        start = int(offset_s * sr)
        end   = min(int((offset_s + duration_s) * sr), len(audio))

        if (end - start) < int(duration_s * sr * 0.9):
            logger.warning(
                f"[Étape 5] Échantillon trop court : "
                f"{(end - start) / sr:.1f}s < {duration_s}s requis"
            )

        sample = audio[start:end]
        sf.write(str(output_path), sample, sr)
        logger.info(
            f"[Étape 5] Sample extrait : {(end - start) / sr:.1f}s → {output_path}"
        )
        return True

    except Exception as exc:
        logger.error(f"[Étape 5] Extraction sample échouée : {exc}")
        return False


# ─── Fonction publique ────────────────────────────────────────────────────────

def synthesize_speech(
    translated_json_path: str,
    speaker_audio_path: str,
    output_dir: str,
    device: str = "cpu",
    speed: float = 1.0,
    save_segments: bool = True,
) -> TTSResult:
    config = TTSConfig(device=device, speed=speed, save_segments=save_segments)
    synth  = XTTSSynthesizer(config)
    result = synth.synthesize(translated_json_path, speaker_audio_path, output_dir)

    if result.success:
        logger.info(
            f"[Étape 5] ✓ Synthèse terminée — "
            f"segments={result.segment_count}  "
            f"durée={result.total_duration_s}s  "
            f"taux_succès={result.success_rate:.0%}"
        )
    else:
        logger.error(f"[Étape 5] ✗ Échec : {result.error_message}")

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 4:
        print(
            "Usage: python step5_tts_synthesis.py "
            "<translated.json> <speaker.wav> <output_dir> [device]"
        )
        sys.exit(1)

    device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    res    = synthesize_speech(sys.argv[1], sys.argv[2], sys.argv[3], device)
    sys.exit(0 if res.success else 1)