"""
LinguaPlay Pipeline — Étape 6 : Synchronisation & Assemblage Vidéo
==================================================================
Aligne temporellement l'audio synthétisé (étape 5) avec la vidéo
originale via FFmpeg, puis assemble la vidéo finale traduite.
"""

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class SyncConfig:
    video_codec: str        = "libx264"
    audio_codec: str        = "aac"
    video_crf: int          = 23
    video_preset: str       = "medium"
    audio_bitrate: str      = "192k"
    output_format: str      = "mp4"
    max_atempo: float       = 2.0
    min_atempo: float       = 0.5
    embed_subtitles: bool   = False   # désactivé par défaut (cause des erreurs FFmpeg)
    ffmpeg_bin: str         = "ffmpeg"
    ffmpeg_loglevel: str    = "error"
    timeout_seconds: int    = 600


# ─── Modèles de données ───────────────────────────────────────────────────────

@dataclass
class SyncReport:
    video_duration_s: float
    audio_duration_s: float
    duration_diff_s: float
    atempo_filter: str
    adjustment_needed: bool
    segments_adjusted: int

    def to_dict(self) -> dict:
        return {
            "video_duration_s":  round(self.video_duration_s, 3),
            "audio_duration_s":  round(self.audio_duration_s, 3),
            "duration_diff_s":   round(self.duration_diff_s, 3),
            "atempo_filter":     self.atempo_filter,
            "adjustment_needed": self.adjustment_needed,
            "segments_adjusted": self.segments_adjusted,
        }


@dataclass
class AssemblyResult:
    success: bool
    output_video_path: Path | None  = None
    report: SyncReport | None       = None
    processing_time_s: float        = 0.0
    output_size_bytes: int          = 0
    ffmpeg_stderr: str              = ""
    error_message: str | None       = None

    @property
    def output_size_mb(self) -> float:
        return round(self.output_size_bytes / (1024 * 1024), 2)


# ─── Assembleur principal ─────────────────────────────────────────────────────

class VideoAssembler:

    def __init__(self, config: SyncConfig | None = None):
        self.config = config or SyncConfig()
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> None:
        if not shutil.which(self.config.ffmpeg_bin):
            raise EnvironmentError(
                f"FFmpeg introuvable : '{self.config.ffmpeg_bin}'. "
                "Installez FFmpeg et assurez-vous qu'il est dans le PATH."
            )

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def assemble(
        self,
        video_path: str | Path,
        audio_tts_path: str | Path,
        output_dir: str | Path,
        srt_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> AssemblyResult:
        video_path     = Path(video_path)
        audio_tts_path = Path(audio_tts_path)
        output_dir     = Path(output_dir)

        if not video_path.exists():
            return AssemblyResult(
                success=False,
                error_message=f"Vidéo source introuvable : {video_path}",
            )
        if not audio_tts_path.exists():
            return AssemblyResult(
                success=False,
                error_message=f"Audio TTS introuvable : {audio_tts_path}",
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._run_assembly(
                video_path, audio_tts_path, output_dir, srt_path, manifest_path
            )
        except Exception as exc:
            logger.exception("[Étape 6] Erreur inattendue lors de l'assemblage")
            return AssemblyResult(success=False, error_message=str(exc))

    # ── Pipeline d'assemblage ─────────────────────────────────────────────────

    def _run_assembly(
        self,
        video_path: Path,
        audio_tts_path: Path,
        output_dir: Path,
        srt_path: Path | None,
        manifest_path: Path | None,
    ) -> AssemblyResult:

        t0 = time.perf_counter()

        video_duration = self._probe_duration(video_path)
        audio_duration = self._probe_duration(audio_tts_path)

        logger.info(
            f"[Étape 6] Durée vidéo={video_duration:.2f}s  "
            f"audio TTS={audio_duration:.2f}s"
        )

        report = self._compute_sync_report(video_duration, audio_duration)

        logger.info(
            f"[Étape 6] Écart={report.duration_diff_s:+.2f}s  "
            f"filtre={report.atempo_filter}  "
            f"ajustement={'oui' if report.adjustment_needed else 'non'}"
        )

        stem     = video_path.stem
        out_path = output_dir / f"{stem}_translated.{self.config.output_format}"

        cmd    = self._build_ffmpeg_cmd(video_path, audio_tts_path, out_path, report)
        stderr = self._run_ffmpeg(cmd)

        if not out_path.exists():
            return AssemblyResult(
                success=False,
                error_message="FFmpeg n'a pas produit de fichier de sortie",
                ffmpeg_stderr=stderr,
            )

        elapsed   = time.perf_counter() - t0
        file_size = out_path.stat().st_size

        logger.info(
            f"[Étape 6] ✓ Vidéo assemblée → {out_path.name}  "
            f"taille={file_size / 1_048_576:.1f} MB  "
            f"temps={elapsed:.1f}s"
        )

        return AssemblyResult(
            success=True,
            output_video_path=out_path,
            report=report,
            processing_time_s=round(elapsed, 2),
            output_size_bytes=file_size,
            ffmpeg_stderr=stderr,
        )

    # ── Calcul du rapport de synchronisation ─────────────────────────────────

    def _compute_sync_report(
        self, video_duration: float, audio_duration: float
    ) -> SyncReport:
        diff  = audio_duration - video_duration
        ratio = audio_duration / video_duration if video_duration > 0 else 1.0

        if abs(ratio - 1.0) < 0.01:
            return SyncReport(
                video_duration_s=video_duration,
                audio_duration_s=audio_duration,
                duration_diff_s=diff,
                atempo_filter="",
                adjustment_needed=False,
                segments_adjusted=0,
            )

        atempo_filter = self._build_atempo_filter(ratio)

        return SyncReport(
            video_duration_s=video_duration,
            audio_duration_s=audio_duration,
            duration_diff_s=diff,
            atempo_filter=atempo_filter,
            adjustment_needed=True,
            segments_adjusted=1,
        )

    def _build_atempo_filter(self, ratio: float) -> str:
        r = max(self.config.min_atempo, min(ratio, 4.0))

        if self.config.min_atempo <= r <= self.config.max_atempo:
            return f"atempo={r:.6f}"
        elif r > self.config.max_atempo:
            r1 = self.config.max_atempo
            r2 = min(r / r1, self.config.max_atempo)
            return f"atempo={r1:.6f},atempo={r2:.6f}"
        else:
            r1 = self.config.min_atempo
            r2 = max(r / r1, self.config.min_atempo)
            return f"atempo={r1:.6f},atempo={r2:.6f}"

    # ── Construction de la commande FFmpeg ────────────────────────────────────

    def _build_ffmpeg_cmd(
        self,
        video_path: Path,
        audio_path: Path,
        out_path: Path,
        report: SyncReport,
    ) -> list[str]:
        """
        Commande FFmpeg simplifiée — sans sous-titres pour éviter
        les conflits d'options -map avec le SRT en entrée.
        """
        cmd = [
            self.config.ffmpeg_bin,
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
        ]

        # Filtre atempo si nécessaire
        if report.adjustment_needed and report.atempo_filter:
            cmd += ["-af", report.atempo_filter]

        cmd += ["-c:v", "copy"]
        cmd += ["-c:a", self.config.audio_codec]
        cmd += ["-b:a", self.config.audio_bitrate]
        cmd += ["-shortest"]
        cmd += ["-loglevel", self.config.ffmpeg_loglevel]
        cmd += [str(out_path)]

        return cmd

    # ── Exécution FFmpeg ──────────────────────────────────────────────────────

    def _run_ffmpeg(self, cmd: list[str]) -> str:
        logger.debug(f"[Étape 6] FFmpeg : {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg a échoué (code {proc.returncode}) : {proc.stderr}"
                )
            return proc.stderr
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"FFmpeg timeout (>{self.config.timeout_seconds}s)"
            )

    # ── Probe durée ───────────────────────────────────────────────────────────

    def _probe_duration(self, media_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            return float(proc.stdout.strip())
        except (ValueError, subprocess.TimeoutExpired):
            return 0.0


# ─── Helpers publics ──────────────────────────────────────────────────────────

def get_video_info(video_path: str | Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name"
        ":format=duration,size,bit_rate",
        "-of", "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data   = json.loads(proc.stdout)
        stream = data.get("streams", [{}])[0]
        fmt    = data.get("format", {})

        fps_raw = stream.get("r_frame_rate", "0/1")
        try:
            num, den = fps_raw.split("/")
            fps = round(int(num) / int(den), 2)
        except (ValueError, ZeroDivisionError):
            fps = 0.0

        return {
            "duration":    float(fmt.get("duration", 0)),
            "size_bytes":  int(fmt.get("size", 0)),
            "bitrate":     int(fmt.get("bit_rate", 0)),
            "width":       int(stream.get("width", 0)),
            "height":      int(stream.get("height", 0)),
            "fps":         fps,
            "codec_video": stream.get("codec_name", ""),
        }
    except Exception:
        return {}


def build_atempo_chain(
    ratio: float, min_val: float = 0.5, max_val: float = 2.0
) -> str:
    filters = []
    r = ratio
    while r > max_val:
        filters.append(f"atempo={max_val:.4f}")
        r /= max_val
    while r < min_val:
        filters.append(f"atempo={min_val:.4f}")
        r /= min_val
    filters.append(f"atempo={r:.4f}")
    return ",".join(filters)


# ─── Fonction publique ────────────────────────────────────────────────────────

def assemble_video(
    video_path: str,
    audio_tts_path: str,
    output_dir: str,
    srt_path: str | None = None,
    manifest_path: str | None = None,
    device: str = "cpu",
) -> AssemblyResult:
    config    = SyncConfig()
    assembler = VideoAssembler(config)
    result    = assembler.assemble(
        video_path, audio_tts_path, output_dir, srt_path, manifest_path
    )

    if result.success:
        logger.info(
            f"[Étape 6] ✓ Assemblage terminé — "
            f"taille={result.output_size_mb} MB  "
            f"temps={result.processing_time_s}s"
        )
    else:
        logger.error(f"[Étape 6] ✗ Échec : {result.error_message}")

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 4:
        print(
            "Usage: python step6_synchronization.py "
            "<video.mp4> <audio_tts.wav> <output_dir>"
        )
        sys.exit(1)

    res = assemble_video(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0 if res.success else 1)