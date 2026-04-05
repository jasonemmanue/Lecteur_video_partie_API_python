"""
LinguaPlay API — Routes FastAPI
================================
POST   /upload              Upload d'une vidéo source
POST   /upload/url          Import depuis une URL
POST   /translate           Lancer une traduction (job Celery)
GET    /status/{job_id}     Statut du traitement
GET    /download/{job_id}   Télécharger la vidéo traduite
GET    /languages           Liste des langues supportées
DELETE /video/{video_id}    Supprimer une vidéo
"""

import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from api.schemas.models import (
    DeleteResponse,
    JobStatus,
    JobStatusResponse,
    LanguageInfo,
    LanguagesResponse,
    StepReport,
    TranslationRequest,
    TranslationResponse,
    UploadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ─── Configuration ────────────────────────────────────────────────────────────

UPLOAD_DIR  = Path(os.getenv("UPLOAD_DIR",  "C:/tmp/linguaplay/uploads"))
OUTPUT_DIR  = Path(os.getenv("OUTPUT_DIR",  "C:/tmp/linguaplay/outputs"))
MAX_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

SUPPORTED_LANGUAGES: list[LanguageInfo] = [
    LanguageInfo(code="fr", name="Français",        flag="🇫🇷"),
    LanguageInfo(code="en", name="English",         flag="🇬🇧"),
    LanguageInfo(code="es", name="Español",         flag="🇪🇸"),
    LanguageInfo(code="de", name="Deutsch",         flag="🇩🇪"),
    LanguageInfo(code="pt", name="Português",       flag="🇧🇷"),
    LanguageInfo(code="ar", name="العربية",          flag="🇸🇦"),
    LanguageInfo(code="zh", name="中文 (Mandarin)",  flag="🇨🇳"),
    LanguageInfo(code="ja", name="日本語",           flag="🇯🇵"),
]


# ─── Helper Celery optionnel ──────────────────────────────────────────────────

def _get_celery_task():
    """
    Import différé de la tâche Celery.
    Retourne None si Celery/Redis n'est pas disponible.
    L'API démarre normalement sans Celery — seul /translate sera indisponible.
    """
    try:
        from workers.tasks import run_translation_pipeline
        return run_translation_pipeline
    except Exception:
        return None


# ─── F01 — Upload vidéo ───────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Uploader une vidéo source",
)
async def upload_video(file: UploadFile = File(...)) -> UploadResponse:
    """Upload d'une vidéo source. Formats : MP4, MKV, AVI, MOV, WEBM (max 500 MB)."""

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Format non supporté : '{suffix}'. "
                   f"Formats acceptés : {sorted(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Fichier trop volumineux : {size_mb:.1f} MB > {MAX_SIZE_MB} MB",
        )

    video_id   = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{video_id}{suffix}"
    video_path.write_bytes(content)

    logger.info(
        f"[Upload] ✓ {video_id} — "
        f"{len(content)} bytes — {file.filename}"
    )

    return UploadResponse(
        video_id=video_id,
        filename=file.filename or "",
        size_bytes=len(content),
    )


# ─── F01 bis — Import depuis URL ─────────────────────────────────────────────

@router.post(
    "/upload/url",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Importer depuis une URL",
)
async def upload_from_url(payload: dict) -> UploadResponse:
    """Import depuis une URL YouTube ou lien direct (nécessite yt-dlp)."""

    url = payload.get("url", "").strip()
    if not url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL manquante.",
        )

    video_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{video_id}.mp4"

    try:
        import subprocess
        result = subprocess.run(
            ["yt-dlp", "-f", "mp4", "-o", str(out_path), url],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Impossible de télécharger : {result.stderr[:200]}",
            )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="yt-dlp non installé. Lancez : pip install yt-dlp",
        )
    except __import__("subprocess").TimeoutExpired:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Timeout lors du téléchargement.",
        )

    size_bytes = out_path.stat().st_size if out_path.exists() else 0
    logger.info(f"[Upload URL] ✓ {video_id} — {url[:60]}")

    return UploadResponse(
        video_id=video_id,
        filename=out_path.name,
        size_bytes=size_bytes,
    )


# ─── F03 — Lancer la traduction ───────────────────────────────────────────────

@router.post(
    "/translate",
    response_model=TranslationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Lancer une traduction",
)
async def start_translation(request: TranslationRequest) -> TranslationResponse:
    """Lance le pipeline asynchrone. Nécessite Celery + Redis."""

    video_files = list(UPLOAD_DIR.glob(f"{request.video_id}.*"))
    if not video_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vidéo '{request.video_id}' introuvable. "
                   "Uploadez d'abord via POST /upload.",
        )

    video_path = video_files[0]
    job_id     = str(uuid.uuid4())

    task = _get_celery_task()
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Worker indisponible. "
                   "Lancez Redis et le worker Celery pour les traductions.",
        )

    try:
        task.apply_async(
            kwargs={
                "job_id":        job_id,
                "video_path":    str(video_path),
                "target_lang":   request.target_lang,
                "source_lang":   request.source_lang,
                "whisper_model": request.whisper_model,
            },
            task_id=job_id,
            priority=request.priority,
        )
    except Exception as exc:
        logger.exception(f"[Translate] Échec soumission {job_id}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Impossible de lancer la tâche : {exc}",
        )

    logger.info(
        f"[Translate] Job {job_id} — "
        f"{request.source_lang}→{request.target_lang}"
    )

    return TranslationResponse(
        job_id=job_id,
        video_id=request.video_id,
        target_lang=request.target_lang,
    )


# ─── F05 — Statut ────────────────────────────────────────────────────────────

@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Statut du traitement",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Statut en temps réel d'un job de traduction."""

    task = _get_celery_task()
    if task is None:
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            progress=0,
            current_step="audio_extraction",
            error_message="Worker non disponible.",
        )

    try:
        async_result = task.AsyncResult(job_id)
        state = async_result.state  # ← déplacer DANS le try
    except Exception as exc:
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            progress=0,
            error_message="Redis non disponible — worker non démarré.",
    )

    if state == "PENDING":
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            progress=0,
            current_step="audio_extraction",
        )

    if state == "PROGRESS":
        meta = async_result.info or {}
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=meta.get("progress", 0),
            current_step=meta.get("current_step"),
        )

    if state == "SUCCESS":
        result       = async_result.result or {}
        step_results = [
            StepReport(**s) for s in result.get("step_results", [])
        ]
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.DONE,
            progress=100,
            output_url=result.get("output_url"),
            total_duration_s=result.get("total_duration_s", 0.0),
            step_results=step_results,
        )

    if state == "FAILURE":
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.ERROR,
            progress=0,
            error_message=str(async_result.info),
        )

    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        progress=0,
    )


# ─── Téléchargement ───────────────────────────────────────────────────────────

@router.get(
    "/download/{job_id}",
    summary="Télécharger la vidéo traduite",
)
async def download_video(job_id: str) -> FileResponse:
    """Télécharge la vidéo traduite une fois le job terminé."""

    job_dir     = OUTPUT_DIR / job_id
    video_files = (
        list(job_dir.glob("*_translated.mp4")) if job_dir.exists() else []
    )

    if not video_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vidéo traduite pour le job '{job_id}' introuvable.",
        )

    video_path = video_files[0]
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name,
        headers={
            "Content-Disposition": f'attachment; filename="{video_path.name}"'
        },
    )


# ─── GET /languages ───────────────────────────────────────────────────────────

@router.get(
    "/languages",
    response_model=LanguagesResponse,
    summary="Liste des langues supportées",
)
async def get_languages() -> LanguagesResponse:
    """Retourne les langues disponibles (V1)."""
    return LanguagesResponse(
        languages=SUPPORTED_LANGUAGES,
        total=len(SUPPORTED_LANGUAGES),
    )


# ─── DELETE /video/{video_id} ─────────────────────────────────────────────────

@router.delete(
    "/video/{video_id}",
    response_model=DeleteResponse,
    summary="Supprimer une vidéo",
)
async def delete_video(video_id: str) -> DeleteResponse:
    """Supprime une vidéo et tous ses fichiers associés."""

    deleted = False

    for video_file in UPLOAD_DIR.glob(f"{video_id}.*"):
        video_file.unlink(missing_ok=True)
        deleted = True
        logger.info(f"[Delete] {video_file.name}")

    job_output_dir = OUTPUT_DIR / video_id
    if job_output_dir.exists():
        shutil.rmtree(str(job_output_dir), ignore_errors=True)
        deleted = True

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vidéo '{video_id}' introuvable.",
        )

    return DeleteResponse(
        video_id=video_id,
        deleted=True,
        message="Vidéo et fichiers associés supprimés avec succès.",
    )