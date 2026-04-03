"""
LinguaPlay Pipeline — Étape 1 : Extraction & Normalisation Audio
================================================================
Extrait la piste audio d'une vidéo source via FFmpeg et la normalise
en 16 kHz mono WAV, format optimal pour Whisper (STT).
"""

import subprocess
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class AudioExtractionConfig:
    """Paramètres d'extraction et de normalisation audio."""
    sample_rate: int      = 16_000   # 16 kHz — requis par Whisper
    channels: int         = 1        # mono
    audio_codec: str      = "pcm_s16le"  # WAV 16-bit little-endian
    output_format: str    = "wav"
    ffmpeg_loglevel: str  = "error"  # silencieux sauf erreurs
    normalize_loudness: bool = True  # normalisation EBU R128
    target_loudness: float   = -23.0 # LUFS cible (standard broadcast)
    ffmpeg_bin: str       = "ffmpeg"


# ─── Résultat ─────────────────────────────────────────────────────────────────

@dataclass
class AudioExtractionResult:
    """Résultat retourné après extraction."""
    success: bool
    audio_path: Path | None          = None
    duration_seconds: float | None   = None
    sample_rate: int | None          = None
    channels: int | None             = None
    file_size_bytes: int | None      = None
    error_message: str | None        = None
    ffmpeg_stderr: str               = ""


# ─── Extracteur principal ──────────────────────────────────────────────────────

class AudioExtractor:
    """
    Extrait et normalise l'audio d'une vidéo source.

    Flux de traitement :
        video_input → [FFmpeg] → raw_audio → [loudnorm] → normalized_wav
    """

    def __init__(self, config: AudioExtractionConfig | None = None):
        self.config = config or AudioExtractionConfig()
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> None:
        """Vérifie que FFmpeg est disponible dans le PATH."""
        if not shutil.which(self.config.ffmpeg_bin):
            raise EnvironmentError(
                f"FFmpeg introuvable : '{self.config.ffmpeg_bin}'. "
                "Installez FFmpeg et assurez-vous qu'il est dans le PATH."
            )

    def extract(self, video_path: str | Path, output_dir: str | Path) -> AudioExtractionResult:
        """
        Extrait et normalise l'audio d'une vidéo.

        Args:
            video_path : chemin vers la vidéo source (MP4, MKV, AVI…)
            output_dir : dossier de sortie pour le fichier WAV

        Returns:
            AudioExtractionResult avec les métadonnées et le chemin du WAV
        """
        video_path  = Path(video_path)
        output_dir  = Path(output_dir)

        # Validation des entrées
        if not video_path.exists():
            return AudioExtractionResult(
                success=False,
                error_message=f"Fichier vidéo introuvable : {video_path}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Nom du fichier de sortie
        audio_filename = video_path.stem + "_extracted.wav"
        audio_path     = output_dir / audio_filename

        logger.info(f"[Étape 1] Extraction audio : {video_path.name}")

        try:
            if self.config.normalize_loudness:
                result = self._extract_with_normalization(video_path, audio_path)
            else:
                result = self._extract_raw(video_path, audio_path)

            if not result.success:
                return result

            # Récupérer les métadonnées du fichier produit
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
            logger.exception("[Étape 1] Erreur inattendue lors de l'extraction")
            return AudioExtractionResult(
                success=False,
                error_message=str(exc)
            )

    # ── Extraction brute (sans normalisation) ─────────────────────────────────

    def _extract_raw(self, video_path: Path, audio_path: Path) -> AudioExtractionResult:
        """Extraction simple sans filtre de loudness."""
        cmd = [
            self.config.ffmpeg_bin,
            "-y",                            # écraser si existant
            "-i", str(video_path),           # entrée
            "-vn",                           # pas de vidéo
            "-acodec", self.config.audio_codec,
            "-ar",     str(self.config.sample_rate),
            "-ac",     str(self.config.channels),
            "-loglevel", self.config.ffmpeg_loglevel,
            str(audio_path),
        ]
        return self._run_ffmpeg(cmd)

    # ── Extraction avec normalisation EBU R128 (2 passes) ─────────────────────

    def _extract_with_normalization(
        self, video_path: Path, audio_path: Path
    ) -> AudioExtractionResult:
        """
        Normalisation en 2 passes via le filtre loudnorm d'FFmpeg.
        Passe 1 : mesure les stats de loudness du fichier.
        Passe 2 : applique la normalisation avec les stats mesurées.
        """
        logger.info("[Étape 1] Passe 1 — Analyse loudness…")

        # Passe 1 : analyse (sortie /dev/null, on récupère le JSON stderr)
        pass1_cmd = [
            self.config.ffmpeg_bin,
            "-i", str(video_path),
            "-af", f"loudnorm=I={self.config.target_loudness}:TP=-1.5:LRA=11:print_format=json",
            "-vn", "-sn", "-f", "null", "-",
            "-loglevel", "info",  # nécessaire pour capturer le JSON
        ]
        proc1 = subprocess.run(
            pass1_cmd, capture_output=True, text=True
        )
        loudnorm_stats = self._parse_loudnorm_stats(proc1.stderr)

        # Passe 2 : normalisation avec les stats mesurées
        logger.info("[Étape 1] Passe 2 — Normalisation…")
        if loudnorm_stats:
            af_filter = (
                f"loudnorm=I={self.config.target_loudness}:TP=-1.5:LRA=11"
                f":measured_I={loudnorm_stats['input_i']}"
                f":measured_LRA={loudnorm_stats['input_lra']}"
                f":measured_TP={loudnorm_stats['input_tp']}"
                f":measured_thresh={loudnorm_stats['input_thresh']}"
                f":offset={loudnorm_stats['target_offset']}"
                f":linear=true:print_format=none"
            )
        else:
            # Fallback : normalisation simple si parsing échoue
            af_filter = f"loudnorm=I={self.config.target_loudness}:TP=-1.5:LRA=11"

        pass2_cmd = [
            self.config.ffmpeg_bin,
            "-y",
            "-i", str(video_path),
            "-af", af_filter,
            "-vn",
            "-acodec", self.config.audio_codec,
            "-ar",     str(self.config.sample_rate),
            "-ac",     str(self.config.channels),
            "-loglevel", self.config.ffmpeg_loglevel,
            str(audio_path),
        ]
        return self._run_ffmpeg(pass2_cmd)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_ffmpeg(self, cmd: list[str]) -> AudioExtractionResult:
        """Exécute une commande FFmpeg et retourne le résultat."""
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if proc.returncode != 0:
                return AudioExtractionResult(
                    success=False,
                    error_message=f"FFmpeg a échoué (code {proc.returncode})",
                    ffmpeg_stderr=proc.stderr,
                )
            return AudioExtractionResult(success=True, ffmpeg_stderr=proc.stderr)
        except subprocess.TimeoutExpired:
            return AudioExtractionResult(
                success=False,
                error_message="FFmpeg timeout (> 5 minutes)"
            )

    def _probe_audio(self, audio_path: Path) -> dict:
        """Utilise ffprobe pour récupérer les métadonnées du WAV produit."""
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
                        metadata["duration"] = float(value)
            return metadata
        except Exception:
            return {}

    @staticmethod
    def _parse_loudnorm_stats(stderr: str) -> dict | None:
        """Parse le JSON de loudnorm depuis le stderr de FFmpeg (passe 1)."""
        import json, re
        match = re.search(r'\{[^}]+\}', stderr, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None


# ─── Point d'entrée CLI ───────────────────────────────────────────────────────

def extract_audio(
    video_path: str,
    output_dir: str,
    normalize: bool = True,
    target_loudness: float = -23.0,
) -> AudioExtractionResult:
    """
    Fonction publique principale — appelée par l'orchestrateur du pipeline.

    Args:
        video_path      : chemin vidéo source
        output_dir      : dossier de sortie
        normalize       : activer la normalisation EBU R128
        target_loudness : niveau cible en LUFS

    Returns:
        AudioExtractionResult
    """
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
            f"[Étape 1] ✓ Audio extrait — "
            f"durée={duration}  taille={size}  path={result.audio_path}"
        )
    else:
        logger.error(f"[Étape 1] ✗ Échec : {result.error_message}")

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 3:
        print("Usage: python step1_audio_extraction.py <video_path> <output_dir>")
        sys.exit(1)

    res = extract_audio(sys.argv[1], sys.argv[2])
    sys.exit(0 if res.success else 1)
