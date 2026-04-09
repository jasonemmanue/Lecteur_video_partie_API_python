"""
LinguaPlay Pipeline — Etape 5 : Synthese Vocale avec Clonage (XTTS-v2)
=======================================================================
Correction v2 :
  - Selection intelligente du meilleur passage de parole pour le sample
    du locuteur (evite les 10 premieres secondes si bruit/musique).
  - COQUI_TOS_AGREED=1 gere l'acceptation des termes sans interaction.
  - Telechargement des poids via hf-mirror.com (sans token HF).
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

XTTS_SUPPORTED_LANGS: set[str] = {
    "en", "fr", "es", "de", "it", "pt", "pl", "tr",
    "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi",
}

XTTS_LANG_MAP: dict[str, str] = {
    "zh": "zh-cn", "zh-cn": "zh-cn", "pt": "pt", "pt-br": "pt",
    "ar": "ar", "ko": "ko", "ja": "ja", "hi": "hi", "hu": "hu",
    "nl": "nl", "cs": "cs", "pl": "pl", "tr": "tr", "ru": "ru",
    "it": "it", "de": "de", "es": "es", "fr": "fr", "en": "en",
}

VOICE_MAP_FALLBACK: dict[str, str] = {
    "fr": "fr-FR-DeniseNeural", "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural", "de": "de-DE-KatjaNeural",
    "ar": "ar-SA-ZariyahNeural", "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural", "pt": "pt-BR-FranciscaNeural",
    "it": "it-IT-ElsaNeural", "ru": "ru-RU-SvetlanaNeural",
    "ko": "ko-KR-SunHiNeural", "sw": "sw-KE-ZuriNeural",
    "hi": "hi-IN-SwaraNeural", "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-ZofiaNeural", "tr": "tr-TR-EmelNeural",
    "hu": "hu-HU-NoemiNeural", "cs": "cs-CZ-VlastaNeural",
}

XTTS_SAMPLE_RATE    = 24_000
XTTS_REQUIRED_FILES = ["config.json", "model.pth", "vocab.json"]
XTTS_ALL_FILES      = ["config.json", "model.pth", "vocab.json",
                        "hash.md5", "speakers_xtts.pth"]
HF_MIRROR_BASE = "https://hf-mirror.com/coqui/XTTS-v2/resolve/main"
HF_DIRECT_BASE = "https://huggingface.co/coqui/XTTS-v2/resolve/main"


# ─── Acceptation des termes Coqui ────────────────────────────────────────────

def _accept_coqui_tos() -> None:
    """
    Pre-accepte les termes CPML de Coqui (usage non-commercial/academique).
    Supprime le prompt interactif [y/n] qui cause EOF dans Celery.
    Ref : https://coqui.ai/cpml.txt
    """
    os.environ["COQUI_TOS_AGREED"] = "1"

    try:
        tos_dir  = Path(os.environ.get("HOME", str(Path.home()))) / ".coqui"
        tos_file = tos_dir / "tos_agreed"
        tos_dir.mkdir(parents=True, exist_ok=True)
        if not tos_file.exists():
            tos_file.write_text("I agree to the terms of the non-commercial CPML.\n")
    except Exception:
        pass

    import builtins
    _orig = builtins.input

    def _auto_yes(prompt: str = "") -> str:
        p = prompt.lower()
        if any(k in p for k in ("agree", "license", "licence", "coqui", "cpml", "y/n", "[y/n]")):
            logger.info(f"[Etape 5 / XTTS-v2] Prompt licence intercepte -> 'y' auto.")
            return "y"
        return _orig(prompt)

    builtins.input = _auto_yes


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TTSConfig:
    model_name: str                = "xtts-v2"
    device: str                    = "cpu"
    speed: float                   = 1.0
    sample_rate: int               = XTTS_SAMPLE_RATE
    output_sample_rate: int        = 16_000
    silence_between_ms: int        = 150
    save_segments: bool            = True
    xtts_enabled: bool             = True
    xtts_temperature: float        = 0.7
    xtts_length_penalty: float     = 1.0
    xtts_repetition_penalty: float = 10.0
    xtts_top_k: int                = 50
    xtts_top_p: float              = 0.85
    # Duree cible du sample locuteur (en secondes)
    speaker_sample_duration: float = 10.0
    max_speed_ratio: float         = 2.0
    min_speed_ratio: float         = 0.5
    openvoice_tau: float           = 0.3
    openvoice_enabled: bool        = False
    cosyvoice_enabled: bool        = False


# ─── Modeles de donnees ───────────────────────────────────────────────────────

@dataclass
class SynthesizedSegment:
    id: int
    start: float
    end: float
    tts_prompt: str
    audio_path: Path | None
    duration_synthesized: float
    duration_target: float
    speed_ratio: float
    success: bool
    voice_cloned: bool = False
    error: str | None  = None

    @property
    def duration_diff(self) -> float:
        return round(self.duration_synthesized - self.duration_target, 3)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "start": self.start, "end": self.end,
            "tts_prompt": self.tts_prompt,
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "duration_synthesized": round(self.duration_synthesized, 3),
            "duration_target": round(self.duration_target, 3),
            "speed_ratio": round(self.speed_ratio, 4),
            "duration_diff": self.duration_diff,
            "success": self.success,
            "voice_cloned": self.voice_cloned,
            "error": self.error,
        }


@dataclass
class TTSResult:
    success: bool
    audio_path: Path | None                        = None
    synthesized_segments: list[SynthesizedSegment] = field(default_factory=list)
    manifest_path: Path | None                     = None
    total_duration_s: float                        = 0.0
    processing_time_s: float                       = 0.0
    sample_rate: int                               = 16_000
    voice_cloning_active: bool                     = False
    error_message: str | None                      = None

    @property
    def segment_count(self) -> int:
        return len(self.synthesized_segments)

    @property
    def success_rate(self) -> float:
        if not self.synthesized_segments:
            return 0.0
        return sum(1 for s in self.synthesized_segments if s.success) / len(self.synthesized_segments)

    @property
    def clone_rate(self) -> float:
        if not self.synthesized_segments:
            return 0.0
        return sum(1 for s in self.synthesized_segments if s.voice_cloned) / len(self.synthesized_segments)


# ─── Gestion des poids XTTS-v2 ───────────────────────────────────────────────

def _get_xtts_local_dir() -> Path:
    tts_home = os.environ.get("TTS_HOME", "")
    if tts_home:
        base = Path(tts_home)
    else:
        hf_home = os.environ.get("HF_HOME", "")
        base = Path(hf_home) / "tts" if hf_home else Path.home() / ".local" / "share" / "tts"
    return base / "tts_models--multilingual--multi-dataset--xtts_v2"


def _xtts_weights_available() -> bool:
    model_dir = _get_xtts_local_dir()
    for f in XTTS_REQUIRED_FILES:
        p = model_dir / f
        if not p.exists() or p.stat().st_size < 1000:
            return False
    return True


def _download_file_no_token(filename: str, dest: Path) -> bool:
    import urllib.request
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    for url, token in [
        (f"{HF_MIRROR_BASE}/{filename}", ""),
        (f"{HF_DIRECT_BASE}/{filename}", ""),
        (f"{HF_DIRECT_BASE}/{filename}", hf_token),
    ]:
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "linguaplay/1.0")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=1800) as r:
                data = r.read()
            if len(data) < 100:
                continue
            dest.write_bytes(data)
            source = "hf-mirror.com" if "hf-mirror" in url else "huggingface.co"
            logger.info(f"[XTTS DL] {filename} OK ({source})")
            return True
        except Exception:
            continue
    return False


def _ensure_xtts_weights() -> bool:
    if _xtts_weights_available():
        logger.info(f"[Etape 5 / XTTS-v2] Poids en cache : {_get_xtts_local_dir()}")
        return True

    model_dir = _get_xtts_local_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[Etape 5 / XTTS-v2] Telechargement des poids (hf-mirror.com)...")

    for filename in XTTS_ALL_FILES:
        dest = model_dir / filename
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        if not _download_file_no_token(filename, dest):
            logger.warning(f"  [ECHEC DL] {filename}")

    for f in XTTS_REQUIRED_FILES:
        p = model_dir / f
        if not p.exists() or p.stat().st_size < 1000:
            logger.error(
                f"[Etape 5 / XTTS-v2] Fichier critique manquant : {f}\n"
                f"Source : {HF_MIRROR_BASE}/{f}\n"
                f"Destination : {model_dir}"
            )
            return False
    return True


# ─── Selection intelligente du sample locuteur ───────────────────────────────

def _find_best_speaker_window(
    audio: np.ndarray,
    sr: int,
    window_s: float = 10.0,
    min_speech_ratio: float = 0.6,
) -> int:
    """
    Trouve le meilleur offset (en samples) pour extraire un sample
    de parole propre, en evitant les passages silencieux ou bruyants.

    Algorithme :
      1. Divise l'audio en fenetres de window_s secondes
      2. Calcule le ratio de parole de chaque fenetre (RMS dans bande vocale)
      3. Retourne l'offset de la fenetre avec le meilleur ratio
         ET avec le RMS le plus stable (variance faible = parole continue)

    Retourne l'offset en nombre de samples.
    """
    window_samples = int(window_s * sr)
    total_samples  = len(audio)

    if total_samples <= window_samples:
        return 0

    # Nombre de fenetres candidates (toutes les 2 secondes)
    step_samples = int(2.0 * sr)
    candidates   = range(0, total_samples - window_samples, step_samples)

    best_offset = 0
    best_score  = -1.0

    for offset in candidates:
        window = audio[offset: offset + window_samples]

        # RMS global de la fenetre
        rms = float(np.sqrt(np.mean(window ** 2)))

        # Trop silencieux (< -40 dB) ou trop fort (saturation)
        if rms < 0.005 or rms > 0.95:
            continue

        # Stabilite : ecart-type du RMS par blocs de 0.5s
        # Un score eleve = parole continue, faible = silence intercale
        block_size = int(0.5 * sr)
        blocks = [window[i:i + block_size]
                  for i in range(0, len(window) - block_size, block_size)]
        if not blocks:
            continue

        block_rms = np.array([np.sqrt(np.mean(b ** 2)) for b in blocks])
        active_blocks = np.sum(block_rms > 0.01)
        speech_ratio  = active_blocks / len(block_rms)

        if speech_ratio < min_speech_ratio:
            continue

        # Score : favorise les passages avec parole dense et niveau stable
        stability = 1.0 / (1.0 + float(np.std(block_rms)))
        score     = speech_ratio * stability * rms

        if score > best_score:
            best_score  = score
            best_offset = offset

    logger.info(
        f"[Etape 5] Meilleur passage de parole : "
        f"offset={best_offset / sr:.1f}s  score={best_score:.4f}"
    )
    return best_offset


# ─── Gestionnaire XTTS-v2 ────────────────────────────────────────────────────

class XTTSCloner:

    def __init__(self, device: str = "cpu"):
        self.device  = device
        self._model  = None
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return True

        _accept_coqui_tos()

        if not _ensure_xtts_weights():
            logger.error(
                "[Etape 5 / XTTS-v2] Poids non disponibles.\n"
                f"Destination : {_get_xtts_local_dir()}"
            )
            return False

        try:
            from TTS.api import TTS as CoquiTTS
            os.environ["TTS_HOME"] = str(_get_xtts_local_dir().parent)
            use_gpu     = self.device.startswith("cuda")
            self._model = CoquiTTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                gpu=use_gpu,
            )
            self._loaded = True
            logger.info("[Etape 5 / XTTS-v2] Modele charge.")
            return True
        except ImportError:
            logger.warning("[Etape 5 / XTTS-v2] TTS non installe (pip install TTS>=0.22.0).")
            return False
        except Exception as exc:
            logger.warning(f"[Etape 5 / XTTS-v2] Chargement echoue : {exc}")
            return False

    def synthesize(
        self, text: str, speaker_wav: Path, output_path: Path,
        language: str = "fr", temperature: float = 0.7,
        length_penalty: float = 1.0, repetition_penalty: float = 10.0,
        top_k: int = 50, top_p: float = 0.85,
    ) -> bool:
        if not self._loaded or self._model is None:
            return False
        try:
            self._model.tts_to_file(
                text=text,
                speaker_wav=str(speaker_wav),
                language=XTTS_LANG_MAP.get(language, language),
                file_path=str(output_path),
                split_sentences=False,
            )
            return output_path.exists() and output_path.stat().st_size > 1000
        except Exception as exc:
            logger.warning(f"[Etape 5 / XTTS-v2] Synthese echouee : {exc}")
            return False


# ─── Synthetiseur principal ───────────────────────────────────────────────────

class XTTSSynthesizer:

    def __init__(self, config: TTSConfig | None = None):
        self.config           = config or TTSConfig()
        self._xtts            = None
        self._edge_tts        = None
        self._xtts_active     = False
        self._edge_tts_loaded = False

    def _load_models(self) -> None:
        if self.config.xtts_enabled and not self._xtts_active:
            cloner = XTTSCloner(device=self.config.device)
            self._xtts_active = cloner.load()
            if self._xtts_active:
                self._xtts = cloner
            else:
                logger.warning("[Etape 5] XTTS-v2 indisponible — fallback edge-tts.")

        if not self._edge_tts_loaded:
            try:
                import edge_tts
                self._edge_tts        = edge_tts
                self._edge_tts_loaded = True
            except ImportError:
                logger.warning("[Etape 5] edge-tts non disponible.")

    def synthesize(self, translated_json_path: str | Path,
                   speaker_audio_path: str | Path, output_dir: str | Path) -> TTSResult:
        translated_json_path = Path(translated_json_path)
        speaker_audio_path   = Path(speaker_audio_path)
        output_dir           = Path(output_dir)

        if not translated_json_path.exists():
            return TTSResult(success=False, error_message=f"JSON introuvable : {translated_json_path}")
        if not speaker_audio_path.exists():
            return TTSResult(success=False, error_message=f"Audio introuvable : {speaker_audio_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_segments:
            (output_dir / "segments").mkdir(exist_ok=True)

        try:
            self._load_models()
            return self._run_synthesis(translated_json_path, speaker_audio_path, output_dir)
        except Exception as exc:
            logger.exception("[Etape 5] Erreur inattendue")
            return TTSResult(success=False, error_message=str(exc))

    def _run_synthesis(self, translated_json_path: Path,
                       speaker_audio_path: Path, output_dir: Path) -> TTSResult:
        data     = json.loads(translated_json_path.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
        tgt_lang = data.get("target_language", "en")

        xtts_lang_ok = tgt_lang in XTTS_SUPPORTED_LANGS
        strategy = (
            "XTTS-v2 zero-shot (clonage natif)"
            if (self._xtts_active and xtts_lang_ok) else "edge-tts fallback"
        )
        logger.info(f"[Etape 5] {len(segments)} segments en '{tgt_lang}' — {strategy}")

        t0           = time.perf_counter()
        synthesized: list[SynthesizedSegment] = []
        audio_chunks: list[np.ndarray]        = []
        silence      = self._make_silence(self.config.silence_between_ms)
        out_sr       = self.config.output_sample_rate

        for seg in segments:
            seg_id       = seg["id"]
            clean_text   = self._strip_tone_tags(
                seg.get("tts_prompt", seg.get("translated_text", ""))
            )
            duration_tgt = seg.get("duration", seg["end"] - seg["start"])

            if not clean_text.strip():
                continue

            synth_seg = self._synthesize_segment(
                text=clean_text, speaker_audio_path=speaker_audio_path,
                language=tgt_lang, seg_id=seg_id,
                duration_target=duration_tgt, output_dir=output_dir,
            )
            synthesized.append(synth_seg)

            if synth_seg.success and synth_seg.audio_path:
                chunk = self._load_audio_array(synth_seg.audio_path)
                chunk = self._resample_if_needed(chunk, self.config.sample_rate, out_sr)
                audio_chunks.append(chunk)
                audio_chunks.append(silence)

        if not audio_chunks:
            return TTSResult(success=False, error_message="Aucun segment produit.",
                             synthesized_segments=synthesized)

        full_audio = np.concatenate(audio_chunks).astype(np.float32)
        total_dur  = len(full_audio) / out_sr
        out_path   = output_dir / "audio_tts.wav"
        self._save_wav(full_audio, out_sr, out_path)

        manifest_path = output_dir / "tts_manifest.json"
        self._save_manifest(synthesized, manifest_path)

        clone_ok = sum(1 for s in synthesized if s.voice_cloned)
        elapsed  = time.perf_counter() - t0
        logger.info(
            f"[Etape 5] Termine — {len(synthesized)} segments  "
            f"clones={clone_ok}/{len(synthesized)}  duree={total_dur:.1f}s  temps={elapsed:.1f}s"
        )

        return TTSResult(
            success=True, audio_path=out_path,
            synthesized_segments=synthesized, manifest_path=manifest_path,
            total_duration_s=round(total_dur, 2), processing_time_s=round(elapsed, 2),
            sample_rate=out_sr,
            voice_cloning_active=(self._xtts_active and xtts_lang_ok),
        )

    def _synthesize_segment(self, text: str, speaker_audio_path: Path, language: str,
                             seg_id: int, duration_target: float,
                             output_dir: Path) -> SynthesizedSegment:
        seg_dir  = output_dir / "segments"
        out_path = seg_dir / f"seg_{seg_id:03d}.wav"

        try:
            voice_cloned = False
            use_xtts = (
                self._xtts_active and self._xtts is not None
                and language in XTTS_SUPPORTED_LANGS
            )

            if use_xtts:
                ok = self._xtts.synthesize(
                    text=text, speaker_wav=speaker_audio_path,
                    output_path=out_path, language=language,
                    temperature=self.config.xtts_temperature,
                    length_penalty=self.config.xtts_length_penalty,
                    repetition_penalty=self.config.xtts_repetition_penalty,
                    top_k=self.config.xtts_top_k, top_p=self.config.xtts_top_p,
                )
                if ok and out_path.exists():
                    voice_cloned            = True
                    self.config.sample_rate = XTTS_SAMPLE_RATE
                else:
                    logger.warning(f"  Segment {seg_id} : XTTS echoue -> edge-tts")
                    self._edge_tts_fallback(text, language, out_path)
            else:
                self._edge_tts_fallback(text, language, out_path)

            audio        = self._load_audio_array(out_path)
            duration_syn = len(audio) / self.config.sample_rate
            speed_ratio  = 1.0

            if duration_target > 0:
                ratio   = duration_syn / duration_target
                clamped = max(self.config.min_speed_ratio,
                              min(ratio, self.config.max_speed_ratio))
                if abs(clamped - 1.0) > 0.10:
                    audio        = self._time_stretch(audio, clamped)
                    self._save_wav(audio, self.config.sample_rate, out_path)
                    duration_syn = len(audio) / self.config.sample_rate
                    speed_ratio  = clamped

            return SynthesizedSegment(
                id=seg_id, start=0.0, end=duration_syn, tts_prompt=text,
                audio_path=out_path if self.config.save_segments else None,
                duration_synthesized=duration_syn, duration_target=duration_target,
                speed_ratio=speed_ratio, success=True, voice_cloned=voice_cloned,
            )

        except Exception as exc:
            logger.warning(f"[Etape 5] Segment {seg_id} echoue : {exc}")
            return SynthesizedSegment(
                id=seg_id, start=0.0, end=0.0, tts_prompt=text, audio_path=None,
                duration_synthesized=0.0, duration_target=duration_target,
                speed_ratio=1.0, success=False, voice_cloned=False, error=str(exc),
            )

    def _edge_tts_fallback(self, text: str, language: str, out_path: Path) -> None:
        if self._edge_tts is None:
            raise RuntimeError("Ni XTTS-v2 ni edge-tts disponibles.")
        asyncio.run(self._edge_tts.Communicate(
            text, VOICE_MAP_FALLBACK.get(language, "en-US-JennyNeural")
        ).save(str(out_path)))

    @staticmethod
    def _strip_tone_tags(text: str) -> str:
        import re
        return re.sub(r'\[[A-Z_]+\]\s*', '', text).strip()

    @staticmethod
    def _time_stretch(audio: np.ndarray, ratio: float) -> np.ndarray:
        try:
            import pyrubberband as pyrb
            return pyrb.time_stretch(audio, XTTS_SAMPLE_RATE, 1.0 / ratio).astype(np.float32)
        except ImportError:
            from scipy.signal import resample
            return resample(audio, int(len(audio) / ratio)).astype(np.float32)

    @staticmethod
    def _resample_if_needed(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
        if src_sr == tgt_sr:
            return audio
        try:
            import librosa
            return librosa.resample(audio, orig_sr=src_sr, target_sr=tgt_sr)
        except ImportError:
            from scipy.signal import resample
            return resample(audio, int(len(audio) * tgt_sr / src_sr)).astype(np.float32)

    def _make_silence(self, duration_ms: int) -> np.ndarray:
        return np.zeros(int(self.config.output_sample_rate * duration_ms / 1000),
                        dtype=np.float32)

    @staticmethod
    def _load_audio_array(path: Path) -> np.ndarray:
        import soundfile as sf
        audio, _ = sf.read(str(path), dtype="float32")
        return audio.mean(axis=1) if audio.ndim > 1 else audio

    @staticmethod
    def _save_wav(audio: np.ndarray, sr: int, path: Path) -> None:
        import soundfile as sf
        sf.write(str(path), audio, sr)

    @staticmethod
    def _save_manifest(segments: list[SynthesizedSegment], path: Path) -> None:
        path.write_text(json.dumps({
            "segment_count": len(segments),
            "success_count": sum(1 for s in segments if s.success),
            "cloned_count":  sum(1 for s in segments if s.voice_cloned),
            "segments":      [s.to_dict() for s in segments],
        }, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── Helpers publics ──────────────────────────────────────────────────────────

def extract_speaker_sample(
    audio_path: str | Path,
    output_path: str | Path,
    duration_s: float = 10.0,
    offset_s: float = 0.0,
) -> bool:
    """
    Extrait le meilleur passage de parole pour XTTS-v2.

    Si offset_s == 0.0 (valeur par defaut), cherche automatiquement
    le meilleur passage de parole dans tout l'audio (evite les debuts
    avec musique/generique qui degradent le clonage).

    Si offset_s > 0, utilise cet offset fixe (comportement precedent).

    Recommandations XTTS-v2 : 6-12s de parole claire, un seul locuteur,
    pas de musique de fond.
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        total_duration = len(audio) / sr

        if offset_s == 0.0 and total_duration > duration_s + 5.0:
            # Recherche automatique du meilleur passage
            best_offset_samples = _find_best_speaker_window(
                audio, sr, window_s=duration_s
            )
            start = best_offset_samples
        else:
            start = int(offset_s * sr)

        end    = min(start + int(duration_s * sr), len(audio))
        actual = (end - start) / sr

        if actual < 3.0:
            logger.warning(f"[Etape 5] Sample trop court ({actual:.1f}s < 3s min).")
        elif actual < 6.0:
            logger.warning(f"[Etape 5] Sample court ({actual:.1f}s) — ideal 6-12s.")

        sf.write(str(output_path), audio[start:end], sr)
        logger.info(
            f"[Etape 5] Sample locuteur : {actual:.1f}s "
            f"(debut a {start/sr:.1f}s) -> {output_path}"
        )
        return True

    except Exception as exc:
        logger.error(f"[Etape 5] Extraction sample echouee : {exc}")
        return False


def synthesize_speech(
    translated_json_path: str, speaker_audio_path: str, output_dir: str,
    device: str = "cpu", speed: float = 1.0, save_segments: bool = True,
    openvoice_enabled: bool = False, openvoice_tau: float = 0.3,
    cosyvoice_enabled: bool = False, xtts_enabled: bool = True,
) -> TTSResult:
    env_xtts = os.environ.get("XTTS_ENABLED", "").lower()
    if env_xtts in ("false", "0", "no"):
        xtts_enabled = False
    elif env_xtts in ("true", "1", "yes"):
        xtts_enabled = True

    config = TTSConfig(
        device=device, speed=speed, save_segments=save_segments,
        xtts_enabled=xtts_enabled, openvoice_tau=openvoice_tau,
    )
    result = XTTSSynthesizer(config).synthesize(
        translated_json_path, speaker_audio_path, output_dir
    )

    if result.success:
        engine = "XTTS-v2 (clonage)" if result.voice_cloning_active else "edge-tts (fallback)"
        logger.info(
            f"[Etape 5] Termine — moteur={engine}  "
            f"segments={result.segment_count}  clone={result.clone_rate:.0%}"
        )
    else:
        logger.error(f"[Etape 5] Echec : {result.error_message}")
    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 4:
        print("Usage: python step5_tts_synthesis.py <translated.json> <speaker.wav> <output_dir> [device]")
        sys.exit(1)
    res = synthesize_speech(sys.argv[1], sys.argv[2], sys.argv[3],
                             device=sys.argv[4] if len(sys.argv) > 4 else "cpu")
    sys.exit(0 if res.success else 1)