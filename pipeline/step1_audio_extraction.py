"""
LinguaPlay Pipeline — Etape 1 : Extraction & Normalisation Audio
================================================================
Extrait la piste audio d'une video source via FFmpeg et la normalise
en 16 kHz mono WAV, format optimal pour Whisper (STT).

Correction v2 :
  - Parsing loudnorm robuste : gere les variations de cles entre versions
    FFmpeg (input_i / input_I / input_integrated, etc.)
  - Fallback sur normalisation simple si le JSON loudnorm est mal forme
  - Timeout augmente pour les grosses videos (> 40 Mo)
"""

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class AudioExtractionConfig:
    sample_rate: int         = 16_000
    channels: int            = 1
    audio_codec: str         = "pcm_s16le"
    output_format: str       = "wav"
    ffmpeg_loglevel: str     = "error"
    normalize_loudness: bool = True
    target_loudness: float   = -23.0
    ffmpeg_bin: str          = "ffmpeg"
    # Timeout genereux pour les grosses videos
    timeout_seconds: int     = 600


# ─── Resultat ─────────────────────────────────────────────────────────────────

@dataclass
class AudioExtractionResult:
    success: bool
    audio_path: Path | None        = None
    duration_seconds: float | None = None
    sample_rate: int | None        = None
    channels: int | None           = None
    file_size_bytes: int | None    = None
    error_message: str | None      = None
    ffmpeg_stderr: str             = ""


# ─── Extracteur principal ──────────────────────────────────────────────────────

class AudioExtractor:

    def __init__(self, config: AudioExtractionConfig | None = None):
        self.config = config or AudioExtractionConfig()
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> None:
        if not shutil.which(self.config.ffmpeg_bin):
            raise EnvironmentError(
                f"FFmpeg introuvable : '{self.config.ffmpeg_bin}'. "
                "Installez FFmpeg et assurez-vous qu'il est dans le PATH."
            )

    def extract(self, video_path: str | Path, output_dir: str | Path) -> AudioExtractionResult:
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            return AudioExtractionResult(
                success=False,
                error_message=f"Fichier video introuvable : {video_path}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / (video_path.stem + "_extracted.wav")

        logger.info(f"[Etape 1] Extraction audio : {video_path.name}")

        try:
            if self.config.normalize_loudness:
                result = self._extract_with_normalization(video_path, audio_path)
            else:
                result = self._extract_raw(video_path, audio_path)

            if not result.success:
                return result

            metadata = self._probe_audio(audio_path)
            return AudioExtractionResult(
                success=True,
                audio_path=audio_path,
                duration_seconds=metadata.get("duration"),
                sample_rate=metadata.get("sample_rate"),
                channels=metadata.get("channels"),
                file_size_bytes=audio_path.stat().st_size,
            )

        except Exception as exc:
            logger.exception("[Etape 1] Erreur inattendue lors de l'extraction")
            return AudioExtractionResult(success=False, error_message=str(exc))

    # ── Extraction brute ──────────────────────────────────────────────────────

    def _extract_raw(self, video_path: Path, audio_path: Path) -> AudioExtractionResult:
        cmd = [
            self.config.ffmpeg_bin, "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", self.config.audio_codec,
            "-ar", str(self.config.sample_rate),
            "-ac", str(self.config.channels),
            "-loglevel", self.config.ffmpeg_loglevel,
            str(audio_path),
        ]
        return self._run_ffmpeg(cmd)

    # ── Extraction avec normalisation EBU R128 ────────────────────────────────

    def _extract_with_normalization(
        self, video_path: Path, audio_path: Path
    ) -> AudioExtractionResult:
        """
        Normalisation en 2 passes via loudnorm.
        Si la passe 1 echoue a produire des stats valides, on fait
        une normalisation simple en une seule passe (fallback robuste).
        """
        logger.info("[Etape 1] Passe 1 — Analyse loudness...")

        pass1_cmd = [
            self.config.ffmpeg_bin,
            "-i", str(video_path),
            "-af", f"loudnorm=I={self.config.target_loudness}:TP=-1.5:LRA=11:print_format=json",
            "-vn", "-sn", "-f", "null", "-",
            "-loglevel", "info",
        ]

        try:
            proc1 = subprocess.run(
                pass1_cmd, capture_output=True, text=True,
                timeout=self.config.timeout_seconds
            )
            loudnorm_stats = self._parse_loudnorm_stats_robust(proc1.stderr)
        except subprocess.TimeoutExpired:
            logger.warning("[Etape 1] Passe 1 timeout — fallback normalisation simple.")
            loudnorm_stats = None
        except Exception as exc:
            logger.warning(f"[Etape 1] Passe 1 echouee ({exc}) — fallback normalisation simple.")
            loudnorm_stats = None

        logger.info("[Etape 1] Passe 2 — Normalisation...")

        if loudnorm_stats:
            # Passe 2 avec les stats mesurees (meilleure qualite)
            af_filter = (
                f"loudnorm=I={self.config.target_loudness}:TP=-1.5:LRA=11"
                f":measured_I={loudnorm_stats['input_i']}"
                f":measured_LRA={loudnorm_stats['input_lra']}"
                f":measured_TP={loudnorm_stats['input_tp']}"
                f":measured_thresh={loudnorm_stats['input_thresh']}"
                f":offset={loudnorm_stats['target_offset']}"
                f":linear=true:print_format=none"
            )
            logger.info("[Etape 1] Normalisation 2 passes (stats mesures disponibles).")
        else:
            # Fallback : normalisation simple sans stats (moins precis mais robuste)
            af_filter = f"loudnorm=I={self.config.target_loudness}:TP=-1.5:LRA=11"
            logger.info("[Etape 1] Normalisation simple (fallback — stats non disponibles).")

        pass2_cmd = [
            self.config.ffmpeg_bin, "-y",
            "-i", str(video_path),
            "-af", af_filter,
            "-vn",
            "-acodec", self.config.audio_codec,
            "-ar", str(self.config.sample_rate),
            "-ac", str(self.config.channels),
            "-loglevel", self.config.ffmpeg_loglevel,
            str(audio_path),
        ]
        return self._run_ffmpeg(pass2_cmd)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_ffmpeg(self, cmd: list[str]) -> AudioExtractionResult:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.config.timeout_seconds
            )
            if proc.returncode != 0:
                return AudioExtractionResult(
                    success=False,
                    error_message=f"FFmpeg a echoue (code {proc.returncode})",
                    ffmpeg_stderr=proc.stderr,
                )
            return AudioExtractionResult(success=True, ffmpeg_stderr=proc.stderr)
        except subprocess.TimeoutExpired:
            return AudioExtractionResult(
                success=False,
                error_message=f"FFmpeg timeout (>{self.config.timeout_seconds}s)"
            )

    def _probe_audio(self, audio_path: Path) -> dict:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,duration",
            "-of", "default=noprint_wrappers=1",
            str(audio_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            metadata = {}
            for line in proc.stdout.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key == "sample_rate":
                        metadata["sample_rate"] = int(value)
                    elif key == "channels":
                        metadata["channels"] = int(value)
                    elif key == "duration":
                        try:
                            metadata["duration"] = float(value)
                        except ValueError:
                            pass
            return metadata
        except Exception:
            return {}

    @staticmethod
    def _parse_loudnorm_stats_robust(stderr: str) -> dict | None:
        """
        Parse les statistiques loudnorm depuis le stderr de FFmpeg.

        Gere les variations de cles entre versions de FFmpeg :
          - input_i     (FFmpeg < 4.x)
          - input_I     (certaines versions)
          - input_integrated (FFmpeg >= 5.x sur certaines builds)

        Retourne un dict normalise avec les cles attendues par la passe 2,
        ou None si le parsing echoue (dans ce cas on utilise le fallback).
        """
        # Extraire le bloc JSON de loudnorm depuis le stderr
        # Le JSON peut apparaitre seul ou embrique dans d'autres logs
        json_match = re.search(r'\{[^{}]*"input_[iI][^{}]*\}', stderr, re.DOTALL)
        if not json_match:
            # Tentative plus large
            json_match = re.search(r'\{.*?\}', stderr, re.DOTALL)
            if not json_match:
                logger.debug("[Etape 1] Aucun JSON loudnorm trouve dans stderr.")
                return None

        try:
            raw = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            logger.debug(f"[Etape 1] JSON loudnorm invalide : {exc}")
            return None

        # Normaliser les cles — FFmpeg utilise des variantes selon la version
        # On cherche chaque cle avec plusieurs noms possibles
        def get_key(d: dict, *candidates: str, default: str = "-70.0") -> str:
            for key in candidates:
                if key in d:
                    val = str(d[key])
                    # Verifier que la valeur est un nombre valide
                    try:
                        float(val)
                        return val
                    except (ValueError, TypeError):
                        continue
            return default

        try:
            stats = {
                # Integrated loudness
                "input_i": get_key(
                    raw,
                    "input_i", "input_I", "input_integrated",
                    "measured_I", "I",
                    default="-23.0"
                ),
                # Loudness range
                "input_lra": get_key(
                    raw,
                    "input_lra", "input_LRA", "input_loudness_range",
                    "LRA",
                    default="7.0"
                ),
                # True peak
                "input_tp": get_key(
                    raw,
                    "input_tp", "input_TP", "input_true_peak",
                    "TP",
                    default="-2.0"
                ),
                # Threshold
                "input_thresh": get_key(
                    raw,
                    "input_thresh", "input_threshold",
                    "threshold",
                    default="-32.0"
                ),
                # Target offset
                "target_offset": get_key(
                    raw,
                    "target_offset", "offset",
                    default="0.0"
                ),
            }

            logger.debug(
                f"[Etape 1] Stats loudnorm parsees : "
                f"I={stats['input_i']} LRA={stats['input_lra']} "
                f"TP={stats['input_tp']}"
            )
            return stats

        except Exception as exc:
            logger.warning(f"[Etape 1] Normalisation des cles loudnorm echouee : {exc}. "
                           "Fallback sur normalisation simple.")
            return None


# ─── Fonction publique ────────────────────────────────────────────────────────

def extract_audio(
    video_path: str,
    output_dir: str,
    normalize: bool = True,
    target_loudness: float = -23.0,
) -> AudioExtractionResult:
    config = AudioExtractionConfig(
        normalize_loudness=normalize,
        target_loudness=target_loudness,
    )
    extractor = AudioExtractor(config)
    result    = extractor.extract(video_path, output_dir)

    if result.success:
        duration = f"{result.duration_seconds:.1f}s" if result.duration_seconds else "N/A"
        size     = f"{result.file_size_bytes / 1024:.1f} KB" if result.file_size_bytes else "N/A"
        logger.info(
            f"[Etape 1] Audio extrait — "
            f"duree={duration}  taille={size}  path={result.audio_path}"
        )
    else:
        logger.error(f"[Etape 1] Echec : {result.error_message}")

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 3:
        print("Usage: python step1_audio_extraction.py <video_path> <output_dir>")
        sys.exit(1)
    res = extract_audio(sys.argv[1], sys.argv[2])
    sys.exit(0 if res.success else 1)