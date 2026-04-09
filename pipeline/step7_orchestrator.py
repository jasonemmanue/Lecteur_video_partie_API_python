"""
LinguaPlay Pipeline — Etape 7 : Orchestrateur & Evaluation MOS
==============================================================
Orchestre les 6 etapes du pipeline de bout en bout et evalue la
qualite de la sortie via le score MOS (Mean Opinion Score).

Modifications v4 (XTTS-v2) :
  - PipelineConfig expose xtts_enabled (remplace cosyvoice_enabled)
  - cosyvoice_enabled / openvoice_enabled / openvoice_tau conserves
    pour compatibilite mais ignores
  - _step5_tts_synthesis transmet xtts_enabled a synthesize_speech
  - La duree du sample locuteur passe a 10s (optimale pour XTTS-v2)
  - Variable d'environnement : XTTS_ENABLED (true/false)
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ─── Etats du pipeline ────────────────────────────────────────────────────────

class PipelineStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    ERROR      = "error"


class PipelineStep(str, Enum):
    AUDIO_EXTRACTION = "audio_extraction"
    SPEECH_TO_TEXT   = "speech_to_text"
    EMOTION_ANALYSIS = "emotion_analysis"
    TRANSLATION      = "translation"
    TTS_SYNTHESIS    = "tts_synthesis"
    SYNCHRONIZATION  = "synchronization"
    FINALIZATION     = "finalization"


STEP_WEIGHTS: dict[PipelineStep, int] = {
    PipelineStep.AUDIO_EXTRACTION: 5,
    PipelineStep.SPEECH_TO_TEXT:   20,
    PipelineStep.EMOTION_ANALYSIS: 10,
    PipelineStep.TRANSLATION:      15,
    PipelineStep.TTS_SYNTHESIS:    30,
    PipelineStep.SYNCHRONIZATION:  15,
    PipelineStep.FINALIZATION:     5,
}

STEP_ORDER = list(PipelineStep)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    source_language: str    = "en"
    target_language: str    = "fr"
    whisper_model: str      = "medium"
    tts_device: str         = "cpu"
    translation_model: str  = "nllb"
    # 10s recommande pour XTTS-v2 (minimum 6s, ideal 8-12s)
    speaker_sample_s: float = 10.0
    keep_intermediates: bool = True
    output_format: str      = "mp4"

    # ── XTTS-v2 (moteur actif) ────────────────────────────────────────────────
    xtts_enabled: bool       = True

    # ── Compatibilite ascendante ──────────────────────────────────────────────
    # Ces champs ne sont plus utilises mais conserves pour ne pas casser
    # les appelants existants (workers Celery, tests, etc.)
    cosyvoice_enabled: bool  = False  # deprecie — ignore
    openvoice_enabled: bool  = False  # deprecie — ignore
    openvoice_tau: float     = 0.3    # deprecie — ignore

    def __post_init__(self) -> None:
        """Surcharge depuis variables d'environnement si presentes."""
        # XTTS_ENABLED est la variable principale
        env_xtts = os.environ.get("XTTS_ENABLED", "").lower()
        if env_xtts in ("false", "0", "no"):
            self.xtts_enabled = False
        elif env_xtts in ("true", "1", "yes"):
            self.xtts_enabled = True

        # Retrocompatibilite : si COSYVOICE_ENABLED=false et XTTS_ENABLED non defini
        # on desactive aussi XTTS (comportement conservateur)
        if not env_xtts:
            env_cv = os.environ.get("COSYVOICE_ENABLED", "").lower()
            if env_cv in ("false", "0", "no"):
                self.xtts_enabled = False

        # OPENVOICE_ENABLED=false ne change rien (deja False par defaut)


# ─── Modeles de donnees ───────────────────────────────────────────────────────

@dataclass
class StepResult:
    step: PipelineStep
    success: bool
    duration_s: float         = 0.0
    output_path: Path | None  = None
    error: str | None         = None

    def to_dict(self) -> dict:
        return {
            "step":        self.step.value,
            "success":     self.success,
            "duration_s":  round(self.duration_s, 2),
            "output_path": str(self.output_path) if self.output_path else None,
            "error":       self.error,
        }


@dataclass
class MOSEvaluation:
    mos_score: float
    wer_score: float
    sync_diff_s: float
    success_rate: float
    language_confidence: float
    clone_rate: float = 0.0
    details: dict = field(default_factory=dict)

    MOS_TARGET = 3.8
    WER_TARGET = 0.08

    @property
    def meets_mos_target(self) -> bool:
        return self.mos_score >= self.MOS_TARGET

    @property
    def meets_wer_target(self) -> bool:
        return self.wer_score <= self.WER_TARGET

    @property
    def overall_pass(self) -> bool:
        return self.meets_mos_target and self.meets_wer_target

    def to_dict(self) -> dict:
        return {
            "mos_score":           round(self.mos_score, 2),
            "wer_score":           round(self.wer_score, 4),
            "sync_diff_s":         round(self.sync_diff_s, 3),
            "success_rate":        round(self.success_rate, 4),
            "language_confidence": round(self.language_confidence, 4),
            "clone_rate":          round(self.clone_rate, 4),
            "meets_mos_target":    self.meets_mos_target,
            "meets_wer_target":    self.meets_wer_target,
            "overall_pass":        self.overall_pass,
            "details":             self.details,
        }


@dataclass
class PipelineResult:
    job_id: str
    status: PipelineStatus
    video_input: Path
    video_output: Path | None            = None
    step_results: list[StepResult]       = field(default_factory=list)
    mos_evaluation: MOSEvaluation | None = None
    current_step: PipelineStep | None    = None
    progress: int                        = 0
    total_duration_s: float              = 0.0
    error_message: str | None            = None

    def to_status_dict(self) -> dict:
        return {
            "job_id":           self.job_id,
            "status":           self.status.value,
            "progress":         self.progress,
            "current_step":     self.current_step.value if self.current_step else None,
            "total_duration_s": round(self.total_duration_s, 2),
            "error_message":    self.error_message,
            "output_path":      str(self.video_output) if self.video_output else None,
            "step_results":     [s.to_dict() for s in self.step_results],
            "mos_evaluation":   self.mos_evaluation.to_dict() if self.mos_evaluation else None,
        }


# ─── Orchestrateur principal ──────────────────────────────────────────────────

class PipelineOrchestrator:

    def __init__(
        self,
        config: PipelineConfig | None = None,
        progress_callback: Callable[[PipelineResult], None] | None = None,
    ):
        self.config            = config or PipelineConfig()
        self.progress_callback = progress_callback

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        job_id: str | None = None,
    ) -> PipelineResult:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        job_id     = job_id or str(uuid.uuid4())[:8]

        work_dir   = output_dir / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(
            job_id=job_id,
            status=PipelineStatus.PROCESSING,
            video_input=video_path,
        )

        ctx: dict = {
            "video_path": video_path,
            "work_dir":   work_dir,
        }

        pipeline_start = time.perf_counter()

        steps = [
            (PipelineStep.AUDIO_EXTRACTION, self._step1_audio_extraction),
            (PipelineStep.SPEECH_TO_TEXT,   self._step2_transcription),
            (PipelineStep.EMOTION_ANALYSIS, self._step3_emotion_analysis),
            (PipelineStep.TRANSLATION,      self._step4_translation),
            (PipelineStep.TTS_SYNTHESIS,    self._step5_tts_synthesis),
            (PipelineStep.SYNCHRONIZATION,  self._step6_synchronization),
            (PipelineStep.FINALIZATION,     self._step7_finalization),
        ]

        cumulative_weight = 0
        total_weight      = sum(STEP_WEIGHTS.values())

        for step_enum, step_fn in steps:
            result.current_step = step_enum
            result.progress     = int(cumulative_weight / total_weight * 100)

            logger.info(
                f"[Pipeline {job_id}] {step_enum.value} "
                f"(progression : {result.progress}%)"
            )

            if self.progress_callback:
                self.progress_callback(result)

            step_start = time.perf_counter()
            try:
                output_path = step_fn(ctx)
                step_dur    = time.perf_counter() - step_start
                result.step_results.append(StepResult(
                    step=step_enum,
                    success=True,
                    duration_s=step_dur,
                    output_path=output_path,
                ))
                logger.info(
                    f"[Pipeline {job_id}] OK {step_enum.value} "
                    f"en {step_dur:.1f}s"
                )

            except Exception as exc:
                step_dur = time.perf_counter() - step_start
                logger.exception(
                    f"[Pipeline {job_id}] ECHEC {step_enum.value} apres {step_dur:.1f}s"
                )
                result.step_results.append(StepResult(
                    step=step_enum,
                    success=False,
                    duration_s=step_dur,
                    error=str(exc),
                ))
                result.status        = PipelineStatus.ERROR
                result.error_message = f"Etape '{step_enum.value}' echouee : {exc}"
                result.total_duration_s = time.perf_counter() - pipeline_start
                if self.progress_callback:
                    self.progress_callback(result)
                return result

            cumulative_weight += STEP_WEIGHTS[step_enum]

        # Finalisation
        result.status           = PipelineStatus.DONE
        result.progress         = 100
        result.video_output     = ctx.get("video_output")
        result.total_duration_s = round(time.perf_counter() - pipeline_start, 2)
        result.mos_evaluation   = self._evaluate_mos(ctx, result)
        result.current_step     = None

        logger.info(
            f"[Pipeline {job_id}] Termine en {result.total_duration_s:.1f}s — "
            f"MOS={result.mos_evaluation.mos_score:.2f}  "
            f"clone_rate={result.mos_evaluation.clone_rate:.0%}"
        )

        if self.progress_callback:
            self.progress_callback(result)

        return result

    # ── Etapes du pipeline ────────────────────────────────────────────────────

    def _step1_audio_extraction(self, ctx: dict) -> Path:
        from pipeline.step1_audio_extraction import extract_audio
        result = extract_audio(
            str(ctx["video_path"]),
            str(ctx["work_dir"]),
        )
        if not result.success:
            raise RuntimeError(result.error_message)
        ctx["audio_path"] = result.audio_path
        return result.audio_path

    def _step2_transcription(self, ctx: dict) -> Path:
        from pipeline.step2_transcription import transcribe_audio
        result = transcribe_audio(
            str(ctx["audio_path"]),
            str(ctx["work_dir"]),
            model_size=self.config.whisper_model,
            language=self.config.source_language,
        )
        if not result.success:
            raise RuntimeError(result.error_message)
        ctx["transcript_json"]   = result.json_path
        ctx["transcript_srt"]    = result.srt_path
        ctx["detected_language"] = result.language
        ctx["lang_confidence"]   = result.language_probability
        return result.json_path

    def _step3_emotion_analysis(self, ctx: dict) -> Path:
        from pipeline.step3_emotion_analysis import analyze_emotions
        result = analyze_emotions(
            str(ctx["audio_path"]),
            str(ctx["transcript_json"]),
            str(ctx["work_dir"]),
        )
        if not result.success:
            raise RuntimeError(result.error_message)
        ctx["enriched_json"] = result.output_json_path
        return result.output_json_path

    def _step4_translation(self, ctx: dict) -> Path:
        from pipeline.step4_translation import translate_segments
        result = translate_segments(
            str(ctx["enriched_json"]),
            str(ctx["work_dir"]),
            source_language=self.config.source_language,
            target_language=self.config.target_language,
            model_type=self.config.translation_model,
        )
        if not result.success:
            raise RuntimeError(result.error_message)
        ctx["translated_json"] = result.output_json_path
        return result.output_json_path

    def _step5_tts_synthesis(self, ctx: dict) -> Path:
        from pipeline.step5_tts_synthesis import synthesize_speech, extract_speaker_sample

        # Extraction du sample locuteur (10s — ideal pour XTTS-v2)
        sample_path = ctx["work_dir"] / "speaker_sample.wav"
        extract_speaker_sample(
            str(ctx["audio_path"]),
            str(sample_path),
            duration_s=self.config.speaker_sample_s,
        )

        result = synthesize_speech(
            str(ctx["translated_json"]),
            str(sample_path),
            str(ctx["work_dir"]),
            device=self.config.tts_device,
            xtts_enabled=self.config.xtts_enabled,
            # Parametres de compatibilite — non utilises par step5
            openvoice_enabled=False,
            openvoice_tau=self.config.openvoice_tau,
            cosyvoice_enabled=False,
        )
        if not result.success:
            raise RuntimeError(result.error_message)

        ctx["audio_tts"]            = result.audio_path
        ctx["tts_manifest"]         = result.manifest_path
        ctx["tts_success_rate"]     = result.success_rate
        ctx["tts_clone_rate"]       = result.clone_rate
        ctx["voice_cloning_active"] = result.voice_cloning_active
        return result.audio_path

    def _step6_synchronization(self, ctx: dict) -> Path:
        from pipeline.step6_synchronization import assemble_video
        result = assemble_video(
            str(ctx["video_path"]),
            str(ctx["audio_tts"]),
            str(ctx["work_dir"]),
            srt_path=None,
            manifest_path=str(ctx.get("tts_manifest", "")),
        )
        if not result.success:
            raise RuntimeError(result.error_message)
        ctx["video_output"] = result.output_video_path
        ctx["sync_diff"]    = result.report.duration_diff_s if result.report else 0.0
        return result.output_video_path

    def _step7_finalization(self, ctx: dict) -> Path:
        import shutil
        video_out = ctx.get("video_output")
        if not video_out or not Path(video_out).exists():
            raise RuntimeError("Video de sortie introuvable apres assemblage")

        final_path = ctx["work_dir"].parent / Path(video_out).name
        shutil.copy2(str(video_out), str(final_path))
        ctx["video_output"] = final_path

        logger.info(f"[Pipeline] Video finale -> {final_path}")
        return final_path

    # ── Evaluation MOS ────────────────────────────────────────────────────────

    def _evaluate_mos(self, ctx: dict, result: PipelineResult) -> MOSEvaluation:
        tts_success_rate  = ctx.get("tts_success_rate", 1.0)
        clone_rate        = ctx.get("tts_clone_rate", 0.0)
        lang_confidence   = ctx.get("lang_confidence", 0.9)
        sync_diff         = abs(ctx.get("sync_diff", 0.0))
        steps_ok          = sum(1 for s in result.step_results if s.success)
        step_success_rate = steps_ok / len(result.step_results) if result.step_results else 0.0

        wer_estimated = max(0.0, 1.0 - lang_confidence)
        sync_score    = max(0.0, 1.0 - sync_diff / 5.0)
        clone_bonus   = clone_rate * 0.15

        composite = (
            0.30 * tts_success_rate +
            0.25 * lang_confidence  +
            0.25 * sync_score       +
            0.10 * step_success_rate +
            0.10 * clone_rate
        )
        mos = 1.0 + composite * 4.0 + clone_bonus

        return MOSEvaluation(
            mos_score=round(min(mos, 5.0), 2),
            wer_score=round(wer_estimated, 4),
            sync_diff_s=sync_diff,
            success_rate=round(step_success_rate, 4),
            language_confidence=round(lang_confidence, 4),
            clone_rate=round(clone_rate, 4),
            details={
                "tts_success_rate": tts_success_rate,
                "sync_score":       sync_score,
                "clone_bonus":      clone_bonus,
                "voice_engine":     "xtts-v2" if ctx.get("voice_cloning_active") else "edge-tts",
            },
        )


# ─── Fonction publique ────────────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    output_dir: str,
    source_language: str = "en",
    target_language: str = "fr",
    whisper_model: str = "medium",
    tts_device: str = "cpu",
    translation_model: str = "nllb",
    xtts_enabled: bool = True,
    job_id: str | None = None,
    progress_callback: Callable[[PipelineResult], None] | None = None,
) -> PipelineResult:
    """
    Lance le pipeline complet de traduction video.

    Args:
        video_path        : chemin vers la video source
        output_dir        : dossier de sortie
        source_language   : langue source (code ISO)
        target_language   : langue cible (code ISO)
        whisper_model     : modele Whisper (tiny/base/small/medium/large-v2)
        tts_device        : cpu | cuda
        translation_model : nllb | helsinki
        xtts_enabled      : activer XTTS-v2 pour le clonage vocal
        job_id            : identifiant unique du job (auto-genere si None)
        progress_callback : callback appele a chaque changement d'etape

    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        source_language=source_language,
        target_language=target_language,
        whisper_model=whisper_model,
        tts_device=tts_device,
        translation_model=translation_model,
        xtts_enabled=xtts_enabled,
    )
    orchestrator = PipelineOrchestrator(config, progress_callback)
    return orchestrator.run(video_path, output_dir, job_id)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 4:
        print(
            "Usage: python step7_orchestrator.py "
            "<video.mp4> <output_dir> <target_lang> [source_lang] [whisper_model]"
        )
        sys.exit(1)

    tgt    = sys.argv[3]
    src    = sys.argv[4] if len(sys.argv) > 4 else "en"
    model  = sys.argv[5] if len(sys.argv) > 5 else "medium"
    result = run_pipeline(
        sys.argv[1], sys.argv[2],
        source_language=src,
        target_language=tgt,
        whisper_model=model,
    )
    sys.exit(0 if result.status == PipelineStatus.DONE else 1)