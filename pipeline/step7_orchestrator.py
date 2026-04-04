"""
LinguaPlay Pipeline — Étape 7 : Orchestrateur & Évaluation MOS
==============================================================
Orchestre les 6 étapes du pipeline de bout en bout et évalue la
qualité de la sortie via le score MOS (Mean Opinion Score).

Flux complet :
    video.mp4
        → [Étape 1] audio_extracted.wav
        → [Étape 2] transcript.json + transcript.srt
        → [Étape 3] enriched.json
        → [Étape 4] translated.json
        → [Étape 5] audio_tts.wav + manifest.json
        → [Étape 6] video_translated.mp4
        → [Évaluation MOS] rapport de qualité

Rapport de progression compatible avec l'API FastAPI (étape F05) :
    {
        "job_id": "abc123",
        "status": "processing",
        "progress": 65,
        "current_step": "tts_synthesis",
        "estimated_remaining": 42
    }
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ─── États du pipeline ────────────────────────────────────────────────────────

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


# Poids de progression par étape (somme = 100)
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
    """Configuration globale du pipeline."""
    source_language: str    = "en"
    target_language: str    = "fr"
    whisper_model: str      = "medium"
    tts_device: str         = "cpu"
    translation_model: str  = "nllb"
    speaker_sample_s: float = 6.0
    keep_intermediates: bool = True   # conserver les fichiers intermédiaires
    output_format: str      = "mp4"


# ─── Modèles de données ───────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Résultat d'une étape individuelle."""
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
    """Évaluation de qualité MOS (Mean Opinion Score)."""
    mos_score: float              # 1.0 → 5.0
    wer_score: float              # Word Error Rate 0.0 → 1.0
    sync_diff_s: float            # écart sync audio-vidéo (secondes)
    success_rate: float           # taux de réussite des segments TTS
    language_confidence: float    # confiance détection langue Whisper
    details: dict = field(default_factory=dict)

    # Seuils du cahier des charges
    MOS_TARGET  = 3.8
    WER_TARGET  = 0.08

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
            "meets_mos_target":    self.meets_mos_target,
            "meets_wer_target":    self.meets_wer_target,
            "overall_pass":        self.overall_pass,
            "details":             self.details,
        }


@dataclass
class PipelineResult:
    """Résultat complet du pipeline."""
    job_id: str
    status: PipelineStatus
    video_input: Path
    video_output: Path | None         = None
    step_results: list[StepResult]    = field(default_factory=list)
    mos_evaluation: MOSEvaluation | None = None
    current_step: PipelineStep | None = None
    progress: int                     = 0
    total_duration_s: float           = 0.0
    error_message: str | None         = None

    def to_status_dict(self) -> dict:
        """Format compatible API FastAPI (F05)."""
        return {
            "job_id":              self.job_id,
            "status":              self.status.value,
            "progress":            self.progress,
            "current_step":        self.current_step.value if self.current_step else None,
            "total_duration_s":    round(self.total_duration_s, 2),
            "error_message":       self.error_message,
            "output_path":         str(self.video_output) if self.video_output else None,
            "step_results":        [s.to_dict() for s in self.step_results],
            "mos_evaluation":      self.mos_evaluation.to_dict() if self.mos_evaluation else None,
        }


# ─── Orchestrateur principal ──────────────────────────────────────────────────

class PipelineOrchestrator:
    """
    Orchestre les 6 étapes du pipeline LinguaPlay de bout en bout.

    Utilisation :
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run(video_path, output_dir, job_id)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        progress_callback: Callable[[PipelineResult], None] | None = None,
    ):
        self.config            = config or PipelineConfig()
        self.progress_callback = progress_callback  # appelé après chaque étape

    # ── Point d'entrée ────────────────────────────────────────────────────────

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        job_id: str | None = None,
    ) -> PipelineResult:
        """
        Lance le pipeline complet sur une vidéo.

        Args:
            video_path : vidéo source (MP4, MKV, AVI…)
            output_dir : dossier de sortie pour tous les fichiers
            job_id     : identifiant du job (généré si None)

        Returns:
            PipelineResult avec le statut complet et l'évaluation MOS
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        job_id     = job_id or str(uuid.uuid4())

        if not video_path.exists():
            return PipelineResult(
                job_id=job_id,
                status=PipelineStatus.ERROR,
                video_input=video_path,
                error_message=f"Vidéo introuvable : {video_path}",
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = output_dir / job_id
        work_dir.mkdir(exist_ok=True)

        result = PipelineResult(
            job_id=job_id,
            status=PipelineStatus.PROCESSING,
            video_input=video_path,
        )

        t0 = time.perf_counter()
        logger.info(f"[Pipeline] Job {job_id} démarré — {video_path.name}")

        # Contexte partagé entre les étapes
        ctx: dict = {"video_path": video_path, "work_dir": work_dir}

        steps = [
            (PipelineStep.AUDIO_EXTRACTION, self._step1_audio_extraction),
            (PipelineStep.SPEECH_TO_TEXT,   self._step2_transcription),
            (PipelineStep.EMOTION_ANALYSIS, self._step3_emotion_analysis),
            (PipelineStep.TRANSLATION,      self._step4_translation),
            (PipelineStep.TTS_SYNTHESIS,    self._step5_tts_synthesis),
            (PipelineStep.SYNCHRONIZATION,  self._step6_synchronization),
            (PipelineStep.FINALIZATION,     self._step7_finalization),
        ]

        for step_enum, step_fn in steps:
            result.current_step = step_enum
            result.progress     = self._compute_progress(step_enum)
            self._notify(result)

            step_result = self._run_step(step_enum, step_fn, ctx)
            result.step_results.append(step_result)

            if not step_result.success:
                result.status        = PipelineStatus.ERROR
                result.error_message = step_result.error
                result.total_duration_s = time.perf_counter() - t0
                logger.error(
                    f"[Pipeline] Job {job_id} échoué à l'étape "
                    f"'{step_enum.value}' : {step_result.error}"
                )
                self._notify(result)
                return result

        # Évaluation MOS
        result.current_step = PipelineStep.FINALIZATION
        result.progress     = 100
        result.mos_evaluation = self._evaluate_mos(ctx, result)
        result.video_output   = ctx.get("video_output")
        result.status         = PipelineStatus.DONE
        result.total_duration_s = time.perf_counter() - t0

        logger.info(
            f"[Pipeline] Job {job_id} terminé en {result.total_duration_s:.1f}s  "
            f"MOS={result.mos_evaluation.mos_score:.2f}  "
            f"WER={result.mos_evaluation.wer_score:.3f}"
        )
        self._notify(result)
        return result

    # ── Exécution sécurisée d'une étape ──────────────────────────────────────

    def _run_step(
        self,
        step: PipelineStep,
        fn: Callable,
        ctx: dict,
    ) -> StepResult:
        """Exécute une étape et capture les exceptions."""
        logger.info(f"[Pipeline] → Étape : {step.value}")
        t0 = time.perf_counter()
        try:
            output_path = fn(ctx)
            duration    = time.perf_counter() - t0
            logger.info(f"[Pipeline] ✓ {step.value} ({duration:.1f}s)")
            return StepResult(
                step=step, success=True,
                duration_s=duration, output_path=output_path,
            )
        except Exception as exc:
            duration = time.perf_counter() - t0
            logger.exception(f"[Pipeline] ✗ {step.value} échoué")
            return StepResult(
                step=step, success=False,
                duration_s=duration, error=str(exc),
            )

    # ── Étapes du pipeline ────────────────────────────────────────────────────

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
        # Extraire un échantillon de référence pour le clonage
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
        )
        if not result.success:
            raise RuntimeError(result.error_message)
        ctx["audio_tts"]      = result.audio_path
        ctx["tts_manifest"]   = result.manifest_path
        ctx["tts_success_rate"] = result.success_rate
        return result.audio_path

    def _step6_synchronization(self, ctx: dict) -> Path:
        from pipeline.step6_synchronization import assemble_video
        result = assemble_video(
            str(ctx["video_path"]),
            str(ctx["audio_tts"]),
            str(ctx["work_dir"]),
            srt_path=str(ctx.get("transcript_srt", "")),
            manifest_path=str(ctx.get("tts_manifest", "")),
        )
        if not result.success:
            raise RuntimeError(result.error_message)
        ctx["video_output"] = result.output_video_path
        ctx["sync_diff"]    = result.report.duration_diff_s if result.report else 0.0
        return result.output_video_path

    def _step7_finalization(self, ctx: dict) -> Path:
        """
        Copie la vidéo finale dans le dossier de sortie racine et
        nettoie les fichiers intermédiaires si demandé.
        """
        import shutil
        video_out = ctx.get("video_output")
        if not video_out or not Path(video_out).exists():
            raise RuntimeError("Vidéo de sortie introuvable après assemblage")

        # Copier vers output_dir racine
        final_path = ctx["work_dir"].parent / Path(video_out).name
        shutil.copy2(str(video_out), str(final_path))
        ctx["video_output"] = final_path

        logger.info(f"[Pipeline] Vidéo finale → {final_path}")
        return final_path

    # ── Évaluation MOS ────────────────────────────────────────────────────────

    def _evaluate_mos(self, ctx: dict, result: PipelineResult) -> MOSEvaluation:
        """
        Calcule le MOS estimé à partir des métriques disponibles.

        Le MOS réel (1-5) nécessite une évaluation humaine.
        Cette implémentation produit un MOS estimé automatiquement
        à partir de 4 indicateurs proxy :

          1. success_rate TTS  → segments synthétisés avec succès
          2. lang_confidence   → confiance de la détection de langue
          3. sync_diff         → écart de synchronisation audio-vidéo
          4. step_success_rate → proportion d'étapes réussies

        Formule :
            MOS_est = 1 + 4 × score_composite  (intervalle [1, 5])
        """
        # Collecte des métriques
        tts_success_rate  = ctx.get("tts_success_rate", 1.0)
        lang_confidence   = ctx.get("lang_confidence", 0.9)
        sync_diff         = abs(ctx.get("sync_diff", 0.0))
        steps_ok          = sum(1 for s in result.step_results if s.success)
        step_success_rate = steps_ok / len(result.step_results) if result.step_results else 0.0

        # WER estimé à partir de la confiance Whisper
        wer_estimated = max(0.0, 1.0 - lang_confidence)

        # Score sync : 1.0 si diff=0, décroît linéairement jusqu'à 0 pour diff≥5s
        sync_score = max(0.0, 1.0 - sync_diff / 5.0)

        # Score composite (moyenne pondérée)
        composite = (
            0.35 * tts_success_rate   +
            0.25 * lang_confidence    +
            0.25 * sync_score         +
            0.15 * step_success_rate
        )
        composite = max(0.0, min(1.0, composite))

        mos_score = 1.0 + 4.0 * composite

        return MOSEvaluation(
            mos_score=round(mos_score, 2),
            wer_score=round(wer_estimated, 4),
            sync_diff_s=round(sync_diff, 3),
            success_rate=round(tts_success_rate, 4),
            language_confidence=round(lang_confidence, 4),
            details={
                "tts_success_rate":  round(tts_success_rate, 4),
                "lang_confidence":   round(lang_confidence, 4),
                "sync_score":        round(sync_score, 4),
                "step_success_rate": round(step_success_rate, 4),
                "composite_score":   round(composite, 4),
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_progress(current_step: PipelineStep) -> int:
        """Calcule le pourcentage de progression basé sur les poids des étapes."""
        progress = 0
        for step in STEP_ORDER:
            if step == current_step:
                break
            progress += STEP_WEIGHTS.get(step, 0)
        return min(progress, 99)

    def _notify(self, result: PipelineResult) -> None:
        """Appelle le callback de progression si défini."""
        if self.progress_callback:
            try:
                self.progress_callback(result)
            except Exception:
                pass  # le callback ne doit pas bloquer le pipeline


# ─── Fonction publique ────────────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    output_dir: str,
    source_language: str = "en",
    target_language: str = "fr",
    whisper_model: str = "medium",
    device: str = "cpu",
    job_id: str | None = None,
    progress_callback: Callable | None = None,
) -> PipelineResult:
    """
    Fonction publique principale — point d'entrée du pipeline complet.

    Args:
        video_path        : chemin vers la vidéo source
        output_dir        : dossier de sortie
        source_language   : code ISO 639-1 langue source
        target_language   : code ISO 639-1 langue cible
        whisper_model     : taille du modèle Whisper
        device            : cpu | cuda
        job_id            : identifiant du job (optionnel)
        progress_callback : fonction appelée après chaque étape

    Returns:
        PipelineResult complet avec évaluation MOS
    """
    config = PipelineConfig(
        source_language=source_language,
        target_language=target_language,
        whisper_model=whisper_model,
        tts_device=device,
    )
    orchestrator = PipelineOrchestrator(config, progress_callback)
    return orchestrator.run(video_path, output_dir, job_id)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 4:
        print(
            "Usage: python step7_orchestrator.py "
            "<video.mp4> <output_dir> <target_lang> [source_lang] [model]"
        )
        sys.exit(1)

    tgt   = sys.argv[3]
    src   = sys.argv[4] if len(sys.argv) > 4 else "en"
    model = sys.argv[5] if len(sys.argv) > 5 else "medium"

    res = run_pipeline(sys.argv[1], sys.argv[2], src, tgt, model)
    print(json.dumps(res.to_status_dict(), indent=2, ensure_ascii=False))
    sys.exit(0 if res.status == PipelineStatus.DONE else 1)
