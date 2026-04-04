"""
LinguaPlay — Worker Celery
==========================
Exécute le pipeline de traduction en tâche asynchrone.
Celery + Redis comme broker et backend de résultats.
"""

import logging
import os
from celery import Celery

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

REDIS_URL    = os.getenv("REDIS_URL",    "redis://localhost:6379/0")
BROKER_URL   = os.getenv("BROKER_URL",  REDIS_URL)
BACKEND_URL  = os.getenv("BACKEND_URL", REDIS_URL)
OUTPUT_DIR   = os.getenv("OUTPUT_DIR",  "/tmp/linguaplay/outputs")

# ─── Application Celery ───────────────────────────────────────────────────────

celery_app = Celery(
    "linguaplay",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)

celery_app.conf.update(
    task_serializer         = "json",
    result_serializer       = "json",
    accept_content          = ["json"],
    timezone                = "UTC",
    enable_utc              = True,
    task_track_started      = True,
    task_acks_late          = True,
    worker_prefetch_multiplier = 1,
    task_routes             = {
        "workers.tasks.run_translation_pipeline": {"queue": "pipeline"},
    },
    result_expires          = 86_400,   # 24h
)


# ─── Tâche principale ─────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="workers.tasks.run_translation_pipeline",
    max_retries=2,
    default_retry_delay=30,
)
def run_translation_pipeline(
    self,
    job_id: str,
    video_path: str,
    target_lang: str,
    source_lang: str = "en",
    whisper_model: str = "medium",
) -> dict:
    """
    Tâche Celery — exécute le pipeline complet de traduction.

    Args:
        job_id       : identifiant unique du job
        video_path   : chemin de la vidéo source sur le serveur
        target_lang  : code ISO 639-1 de la langue cible
        source_lang  : code ISO 639-1 de la langue source
        whisper_model: taille du modèle Whisper

    Returns:
        dict compatible JobStatusResponse
    """
    import os
    from pathlib import Path

    logger.info(
        f"[Celery] Job {job_id} démarré — "
        f"{source_lang}→{target_lang}  vidéo={video_path}"
    )

    # Mise à jour du statut : processing
    self.update_state(
        state="PROGRESS",
        meta={
            "job_id":       job_id,
            "status":       "processing",
            "progress":     0,
            "current_step": "audio_extraction",
        },
    )

    def progress_callback(pipeline_result):
        """Transmet la progression à Celery."""
        self.update_state(
            state="PROGRESS",
            meta={
                "job_id":       job_id,
                "status":       "processing",
                "progress":     pipeline_result.progress,
                "current_step": pipeline_result.current_step.value
                                if pipeline_result.current_step else None,
            },
        )

    try:
        # Import différé pour éviter les chargements de modèles au démarrage
        from pipeline.step7_orchestrator import run_pipeline

        output_dir = Path(OUTPUT_DIR) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        result = run_pipeline(
            video_path=video_path,
            output_dir=str(output_dir),
            source_language=source_lang,
            target_language=target_lang,
            whisper_model=whisper_model,
            job_id=job_id,
            progress_callback=progress_callback,
        )

        status_dict = result.to_status_dict()

        # Construire l'URL de téléchargement si succès
        if result.video_output:
            status_dict["output_url"] = f"/download/{job_id}"

        logger.info(
            f"[Celery] Job {job_id} terminé — "
            f"status={status_dict['status']}"
        )
        return status_dict

    except Exception as exc:
        logger.exception(f"[Celery] Job {job_id} échoué")
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {
                "job_id":        job_id,
                "status":        "error",
                "progress":      0,
                "error_message": str(exc),
                "output_url":    None,
            }
