"""
LinguaPlay API — Application FastAPI principale
================================================
Configure l'application FastAPI avec :
  - Middleware JWT (authentification)
  - Rate limiting (10 req/heure par utilisateur)
  - CORS
  - Documentation Swagger/OpenAPI
  - Gestion des erreurs globale
"""

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.videos import router as videos_router

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

JWT_SECRET     = os.getenv("JWT_SECRET",  "change-me-in-production")
RATE_LIMIT     = int(os.getenv("RATE_LIMIT_PER_HOUR", "10"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


# ─── Rate Limiter (en mémoire — Redis en production) ─────────────────────────

class InMemoryRateLimiter:
    """
    Rate limiter simple en mémoire.
    En production : remplacer par slowapi + Redis.
    """
    def __init__(self, max_requests: int = 10, window_seconds: int = 3600):
        self.max_requests   = max_requests
        self.window_seconds = window_seconds
        self._store: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Vérifie si la clé (user_id ou IP) peut faire une requête.

        Returns:
            (allowed, remaining) — remaining = requêtes restantes
        """
        now       = time.time()
        window    = now - self.window_seconds
        requests  = [t for t in self._store[key] if t > window]
        self._store[key] = requests

        if len(requests) >= self.max_requests:
            return False, 0

        self._store[key].append(now)
        return True, self.max_requests - len(self._store[key])

    def reset(self, key: str) -> None:
        self._store.pop(key, None)


rate_limiter = InMemoryRateLimiter(max_requests=RATE_LIMIT)


# ─── Middleware JWT ───────────────────────────────────────────────────────────

class JWTMiddleware:
    """
    Middleware d'authentification JWT.
    Routes publiques exemptées : /, /docs, /openapi.json, /health, /languages.
    """
    PUBLIC_PATHS = {"/", "/docs", "/openapi.json", "/redoc", "/health", "/languages"}

    def __init__(self, app, secret: str):
        self.app    = app
        self.secret = secret

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        
        # Ignorer favicon.ico (ne pas exiger d'authentification)
        if path == "/favicon.ico":
            response = JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": "favicon not found"}
            )
            await response(scope, receive, send)
            return
        
        if path in self.PUBLIC_PATHS or path.startswith("/docs"):
            await self.app(scope, receive, send)
            return

        # Extraire le token JWT de l'en-tête Authorization
        headers = dict(scope.get("headers", []))
        auth    = headers.get(b"authorization", b"").decode("utf-8")

        if not auth.startswith("Bearer "):
            response = JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Token JWT manquant", "status_code": 401},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        token = auth[len("Bearer "):]
        user_id = self._verify_token(token)

        if user_id is None:
            response = JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Token JWT invalide ou expiré", "status_code": 401},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        # Injecter l'user_id dans le state de la requête
        scope["state"] = scope.get("state", {})
        scope["state"]["user_id"] = user_id

        # Rate limiting sur les routes de traduction
        if path == "/translate":
            allowed, remaining = rate_limiter.is_allowed(user_id)
            if not allowed:
                response = JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Trop de requêtes",
                        "detail": f"Limite : {RATE_LIMIT} traductions/heure",
                        "status_code": 429,
                    },
                    headers={"Retry-After": "3600"},
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)

    def _verify_token(self, token: str) -> str | None:
        """Vérifie et décode le JWT. Retourne l'user_id ou None."""
        try:
            import jwt
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
            return payload.get("sub")
        except Exception:
            return None


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[API] LinguaPlay API démarrée")
    yield
    logger.info("[API] LinguaPlay API arrêtée")


# ─── Application ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Factory de l'application FastAPI."""
    app = FastAPI(
        title="LinguaPlay API",
        description=(
            "API REST de traduction vidéo multilingue avec clonage vocal.\n\n"
            "**Pipeline** : FFmpeg → Whisper → wav2vec2 → NLLB-200 → XTTS-v2 → FFmpeg\n\n"
            "Authentification via JWT (header `Authorization: Bearer <token>`)."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Middleware JWT ────────────────────────────────────────────────────────
    app.add_middleware(JWTMiddleware, secret=JWT_SECRET)  # type: ignore[arg-type]

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(videos_router, tags=["Vidéos & Traduction"])

    # ── Homepage ──────────────────────────────────────────────────────────────
    @app.get("/", tags=["Système"], include_in_schema=False)
    async def homepage():
        return {
            "message": "Bienvenue sur LinguaPlay API",
            "status": "operational",
            "documentation": "http://127.0.0.1:8000/docs",
            "openapi": "http://127.0.0.1:8000/openapi.json",
            "endpoints": {
                "public": [
                    "GET /health - Santé du service",
                    "GET /languages - Langues supportées"
                ],
                "authenticated": [
                    "POST /upload - Uploader une vidéo",
                    "POST /translate - Lancer une traduction",
                    "GET /status/{job_id} - Statut du traitement",
                    "GET /download/{job_id} - Télécharger la vidéo traduite",
                    "DELETE /video/{video_id} - Supprimer une vidéo"
                ]
            },
            "authentication": "Bearer token JWT requis (sauf routes publiques)"
        }

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["Système"], include_in_schema=False)
    async def health():
        return {"status": "ok", "service": "LinguaPlay API"}

    # ── Gestionnaire d'erreurs global ─────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Erreur non gérée : {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Erreur interne du serveur",
                "detail": str(exc),
                "status_code": 500,
            },
        )

    return app


app = create_app()
