"""
LinguaPlay API — Schémas Pydantic
==================================
Modèles de validation des requêtes et réponses de l'API REST.
Compatible avec FastAPI et le format de statut du pipeline (étape 7).
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


# ─── Énumérations ─────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    ERROR      = "error"


class SupportedLanguage(str, Enum):
    FR = "fr"
    EN = "en"
    ES = "es"
    DE = "de"
    PT = "pt"
    AR = "ar"
    ZH = "zh"
    JA = "ja"


# ─── Requêtes ─────────────────────────────────────────────────────────────────

class TranslationRequest(BaseModel):
    """Corps de la requête POST /translate."""
    video_id: str = Field(..., min_length=1, description="ID de la vidéo uploadée")
    target_lang: str = Field(..., min_length=2, max_length=10, description="Code ISO 639-1 cible")
    source_lang: str = Field(default="en", min_length=2, max_length=10)
    whisper_model: str = Field(default="medium")
    priority: int = Field(default=0, ge=0, le=10)

    @field_validator("target_lang", "source_lang")
    @classmethod
    def validate_language(cls, v: str) -> str:
        supported = {lang.value for lang in SupportedLanguage}
        if v not in supported:
            raise ValueError(
                f"Langue '{v}' non supportée. "
                f"Langues disponibles : {sorted(supported)}"
            )
        return v


# ─── Réponses ─────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Réponse POST /upload."""
    video_id: str
    filename: str
    size_bytes: int
    message: str = "Vidéo uploadée avec succès"


class TranslationResponse(BaseModel):
    """Réponse POST /translate."""
    job_id: str
    video_id: str
    target_lang: str
    status: JobStatus = JobStatus.PENDING
    message: str = "Traduction lancée"


class MOSReport(BaseModel):
    """Rapport MOS inclus dans le statut d'un job terminé."""
    mos_score: float
    wer_score: float
    sync_diff_s: float
    success_rate: float
    language_confidence: float
    meets_mos_target: bool
    meets_wer_target: bool
    overall_pass: bool
    details: dict[str, Any] = Field(default_factory=dict)


class StepReport(BaseModel):
    """Rapport d'une étape individuelle du pipeline."""
    step: str
    success: bool
    duration_s: float
    output_path: str | None = None
    error: str | None = None


class JobStatusResponse(BaseModel):
    """Réponse GET /status/{job_id} — format F05 du cahier des charges."""
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    current_step: str | None = None
    estimated_remaining: int | None = None   # secondes
    error_message: str | None = None
    output_url: str | None = None            # URL de téléchargement si done
    total_duration_s: float = 0.0
    step_results: list[StepReport] = Field(default_factory=list)
    mos_evaluation: MOSReport | None = None


class LanguageInfo(BaseModel):
    """Informations sur une langue supportée."""
    code: str
    name: str
    flag: str
    available_v1: bool = True


class LanguagesResponse(BaseModel):
    """Réponse GET /languages."""
    languages: list[LanguageInfo]
    total: int


class DeleteResponse(BaseModel):
    """Réponse DELETE /video/{video_id}."""
    video_id: str
    deleted: bool
    message: str


class ErrorResponse(BaseModel):
    """Réponse d'erreur standard."""
    error: str
    detail: str | None = None
    status_code: int
