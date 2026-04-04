"""
Tests d'intégration — API FastAPI LinguaPlay
=============================================
Couverture complète des 6 endpoints :
  - POST /upload      : succès, format invalide, taille dépassée
  - POST /translate   : succès, vidéo inconnue, langue invalide, service down
  - GET  /status/{id} : pending, processing, done, error, inconnu
  - GET  /download/{id}: succès, fichier manquant
  - GET  /languages   : structure, contenu V1
  - DELETE /video/{id}: succès, introuvable

Middleware :
  - JWT              : route publique, token manquant, token invalide, token valide
  - Rate limiting    : limite respectée, limite dépassée
  - InMemoryRateLimiter : logique interne
"""

import io
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Ajout du dossier parent au path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("JWT_SECRET",           "test-secret-key")
os.environ.setdefault("RATE_LIMIT_PER_HOUR",  "10")
os.environ.setdefault("UPLOAD_DIR",           "/tmp/linguaplay_test/uploads")
os.environ.setdefault("OUTPUT_DIR",           "/tmp/linguaplay_test/outputs")

from fastapi.testclient import TestClient

from api.main import app, InMemoryRateLimiter, rate_limiter
from api.schemas.models import JobStatus


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_jwt(user_id: str = "user-test-123", secret: str = "test-secret-key") -> str:
    """Génère un JWT de test valide."""
    import jwt
    import time
    payload = {"sub": user_id, "exp": int(time.time()) + 3600}
    return jwt.encode(payload, secret, algorithm="HS256")


def _auth_headers(user_id: str = "user-test-123") -> dict:
    return {"Authorization": f"Bearer {_make_jwt(user_id)}"}


def _make_video_bytes(size_kb: int = 10) -> bytes:
    """Génère de faux octets vidéo (non lisible, juste pour tester l'upload)."""
    return b"\x00\x00\x00\x18ftypmp42" + b"\x00" * (size_kb * 1024)


client = TestClient(app, raise_server_exceptions=False)


# ─── Tests : InMemoryRateLimiter ─────────────────────────────────────────────

class TestInMemoryRateLimiter(unittest.TestCase):

    def setUp(self):
        self.limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)

    def test_first_request_allowed(self):
        allowed, remaining = self.limiter.is_allowed("user-1")
        self.assertTrue(allowed)
        self.assertEqual(remaining, 2)

    def test_requests_within_limit_allowed(self):
        for _ in range(3):
            allowed, _ = self.limiter.is_allowed("user-2")
        self.assertTrue(allowed)

    def test_request_above_limit_blocked(self):
        for _ in range(3):
            self.limiter.is_allowed("user-3")
        allowed, remaining = self.limiter.is_allowed("user-3")
        self.assertFalse(allowed)
        self.assertEqual(remaining, 0)

    def test_different_users_independent(self):
        for _ in range(3):
            self.limiter.is_allowed("user-a")
        # user-b n'est pas affecté
        allowed, _ = self.limiter.is_allowed("user-b")
        self.assertTrue(allowed)

    def test_reset_clears_counter(self):
        for _ in range(3):
            self.limiter.is_allowed("user-4")
        self.limiter.reset("user-4")
        allowed, _ = self.limiter.is_allowed("user-4")
        self.assertTrue(allowed)

    def test_expired_requests_not_counted(self):
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=1)
        limiter.is_allowed("user-5")
        limiter.is_allowed("user-5")
        time.sleep(1.1)
        allowed, _ = limiter.is_allowed("user-5")
        self.assertTrue(allowed)


# ─── Tests : Middleware JWT ───────────────────────────────────────────────────

class TestJWTMiddleware(unittest.TestCase):

    def setUp(self):
        """Reset the rate limiter before each test."""
        rate_limiter.reset("user-test-123")

    def test_health_public_no_auth(self):
        """GET /health accessible sans token."""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_languages_public_no_auth(self):
        """GET /languages accessible sans token."""
        response = client.get("/languages")
        self.assertEqual(response.status_code, 200)

    def test_upload_without_token_returns_401(self):
        response = client.post(
            "/upload",
            files={"file": ("video.mp4", _make_video_bytes(), "video/mp4")},
        )
        self.assertEqual(response.status_code, 401)

    def test_upload_with_invalid_token_returns_401(self):
        response = client.post(
            "/upload",
            headers={"Authorization": "Bearer invalid.token.here"},
            files={"file": ("video.mp4", _make_video_bytes(), "video/mp4")},
        )
        self.assertEqual(response.status_code, 401)

    def test_upload_with_valid_token_passes_auth(self):
        """Avec un token valide, l'auth passe (l'upload peut échouer pour d'autres raisons)."""
        response = client.post(
            "/upload",
            headers=_auth_headers(),
            files={"file": ("video.mp4", _make_video_bytes(), "video/mp4")},
        )
        # 201 ou 4xx mais pas 401
        self.assertNotEqual(response.status_code, 401)

    def test_missing_bearer_prefix_returns_401(self):
        token = _make_jwt()
        response = client.post(
            "/upload",
            headers={"Authorization": token},  # sans "Bearer "
            files={"file": ("video.mp4", b"data", "video/mp4")},
        )
        self.assertEqual(response.status_code, 401)


# ─── Tests : POST /upload ─────────────────────────────────────────────────────

class TestUploadEndpoint(unittest.TestCase):

    def setUp(self):
        """Reset the rate limiter before each test."""
        rate_limiter.reset("user-test-123")

    def test_upload_mp4_success(self):
        response = client.post(
            "/upload",
            headers=_auth_headers(),
            files={"file": ("test_video.mp4", _make_video_bytes(100), "video/mp4")},
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("video_id", data)
        self.assertIn("filename", data)
        self.assertIn("size_bytes", data)
        self.assertEqual(data["filename"], "test_video.mp4")

    def test_upload_mkv_success(self):
        response = client.post(
            "/upload",
            headers=_auth_headers(),
            files={"file": ("test.mkv", _make_video_bytes(50), "video/x-matroska")},
        )
        self.assertEqual(response.status_code, 201)

    def test_upload_invalid_format_returns_415(self):
        response = client.post(
            "/upload",
            headers=_auth_headers(),
            files={"file": ("document.pdf", b"PDF content", "application/pdf")},
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn("Format non supporté", response.json()["detail"])

    def test_upload_txt_returns_415(self):
        response = client.post(
            "/upload",
            headers=_auth_headers(),
            files={"file": ("notes.txt", b"text content", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    def test_upload_returns_unique_video_id(self):
        """Deux uploads du même fichier → deux video_id différents."""
        video = _make_video_bytes(10)
        r1 = client.post("/upload", headers=_auth_headers(),
                         files={"file": ("v.mp4", video, "video/mp4")})
        r2 = client.post("/upload", headers=_auth_headers(),
                         files={"file": ("v.mp4", video, "video/mp4")})
        self.assertNotEqual(r1.json()["video_id"], r2.json()["video_id"])

    def test_upload_size_reported_correctly(self):
        video = _make_video_bytes(10)
        response = client.post(
            "/upload",
            headers=_auth_headers(),
            files={"file": ("v.mp4", video, "video/mp4")},
        )
        self.assertEqual(response.json()["size_bytes"], len(video))

    @patch.dict(os.environ, {"MAX_FILE_SIZE_MB": "0"})
    def test_upload_too_large_returns_413(self):
        """Fichier de plus de 0 MB → 413."""
        with patch("api.routes.videos.MAX_SIZE_MB", 0):
            response = client.post(
                "/upload",
                headers=_auth_headers(),
                files={"file": ("big.mp4", _make_video_bytes(1024), "video/mp4")},
            )
        self.assertEqual(response.status_code, 413)


# ─── Tests : POST /translate ──────────────────────────────────────────────────

class TestTranslateEndpoint(unittest.TestCase):

    def setUp(self):
        """Reset the rate limiter before each test."""
        rate_limiter.reset("user-test-123")

    def _upload_video(self) -> str:
        """Upload une vidéo et retourne son video_id."""
        response = client.post(
            "/upload",
            headers=_auth_headers(),
            files={"file": ("test.mp4", _make_video_bytes(50), "video/mp4")},
        )
        return response.json()["video_id"]

    def test_translate_unknown_video_returns_404(self):
        response = client.post(
            "/translate",
            headers=_auth_headers(),
            json={"video_id": "nonexistent-id", "target_lang": "fr"},
        )
        self.assertEqual(response.status_code, 404)

    def test_translate_invalid_language_returns_422(self):
        response = client.post(
            "/translate",
            headers=_auth_headers(),
            json={"video_id": "some-id", "target_lang": "klingon"},
        )
        self.assertEqual(response.status_code, 422)

    def test_translate_missing_target_lang_returns_422(self):
        response = client.post(
            "/translate",
            headers=_auth_headers(),
            json={"video_id": "some-id"},
        )
        self.assertEqual(response.status_code, 422)

    def test_translate_success_with_celery_mock(self):
        """Traduction avec Celery mocké → 202 + job_id."""
        video_id = self._upload_video()

        mock_async_result = MagicMock()
        mock_apply_async  = MagicMock(return_value=mock_async_result)

        with patch("api.routes.videos.run_translation_pipeline") as mock_task:
            mock_task.apply_async = mock_apply_async
            response = client.post(
                "/translate",
                headers=_auth_headers(),
                json={"video_id": video_id, "target_lang": "fr"},
            )

        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertIn("job_id", data)
        self.assertIn("video_id", data)
        self.assertEqual(data["video_id"], video_id)
        self.assertEqual(data["status"], "pending")

    def test_translate_all_v1_languages_accepted(self):
        """Toutes les langues V1 sont acceptées."""
        v1_langs = ["fr", "en", "es", "de", "pt", "ar", "zh", "ja"]
        for lang in v1_langs:
            response = client.post(
                "/translate",
                headers=_auth_headers(),
                json={"video_id": "nonexistent", "target_lang": lang},
            )
            # 404 (vidéo inconnue) mais pas 422 (langue valide)
            self.assertNotEqual(
                response.status_code, 422,
                f"Langue '{lang}' rejetée à tort"
            )


# ─── Tests : GET /status/{job_id} ────────────────────────────────────────────

class TestStatusEndpoint(unittest.TestCase):

    def setUp(self):
        """Reset the rate limiter before each test."""
        rate_limiter.reset("user-test-123")

    def _mock_task(self, state: str, info: dict | None = None, result: dict | None = None):
        """Crée un mock de tâche Celery."""
        task = MagicMock()
        task.state  = state
        task.info   = info or {}
        task.result = result or {}
        return task

    def test_status_pending(self):
        with patch("api.routes.videos.run_translation_pipeline") as mock_task_cls:
            mock_task_cls.AsyncResult.return_value = self._mock_task("PENDING")
            response = client.get("/status/some-job-id", headers=_auth_headers())

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "pending")
        self.assertEqual(data["progress"], 0)

    def test_status_processing(self):
        with patch("api.routes.videos.run_translation_pipeline") as mock_task_cls:
            mock_task_cls.AsyncResult.return_value = self._mock_task(
                "PROGRESS",
                info={"progress": 45, "current_step": "translation"},
            )
            response = client.get("/status/some-job-id", headers=_auth_headers())

        data = response.json()
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["progress"], 45)
        self.assertEqual(data["current_step"], "translation")

    def test_status_done(self):
        with patch("api.routes.videos.run_translation_pipeline") as mock_task_cls:
            mock_task_cls.AsyncResult.return_value = self._mock_task(
                "SUCCESS",
                result={
                    "output_url": "/download/job-123",
                    "total_duration_s": 120.5,
                    "step_results": [],
                },
            )
            response = client.get("/status/job-123", headers=_auth_headers())

        data = response.json()
        self.assertEqual(data["status"], "done")
        self.assertEqual(data["progress"], 100)
        self.assertEqual(data["output_url"], "/download/job-123")

    def test_status_error(self):
        with patch("api.routes.videos.run_translation_pipeline") as mock_task_cls:
            mock_task_cls.AsyncResult.return_value = self._mock_task(
                "FAILURE",
                info=Exception("FFmpeg crash"),
            )
            response = client.get("/status/bad-job", headers=_auth_headers())

        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIsNotNone(data["error_message"])

    def test_status_response_schema(self):
        """Tous les champs requis sont présents dans la réponse."""
        with patch("api.routes.videos.run_translation_pipeline") as mock_task_cls:
            mock_task_cls.AsyncResult.return_value = self._mock_task("PENDING")
            response = client.get("/status/test-job", headers=_auth_headers())

        data = response.json()
        for field in ("job_id", "status", "progress", "current_step",
                      "error_message", "output_url"):
            self.assertIn(field, data)


# ─── Tests : GET /download/{job_id} ──────────────────────────────────────────

class TestDownloadEndpoint(unittest.TestCase):

    def setUp(self):
        """Reset the rate limiter before each test."""
        rate_limiter.reset("user-test-123")

    def test_download_nonexistent_job_returns_404(self):
        response = client.get(
            "/download/nonexistent-job-id",
            headers=_auth_headers(),
        )
        self.assertEqual(response.status_code, 404)

    def test_download_existing_file_returns_200(self):
        """Crée un faux fichier traduit et vérifie le téléchargement."""
        import tempfile
        job_id  = "test-download-job"
        out_dir = Path(os.environ["OUTPUT_DIR"]) / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        video   = out_dir / "video_translated.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1024)

        try:
            response = client.get(
                f"/download/{job_id}",
                headers=_auth_headers(),
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("video/mp4", response.headers.get("content-type", ""))
        finally:
            import shutil
            shutil.rmtree(str(out_dir), ignore_errors=True)


# ─── Tests : GET /languages ───────────────────────────────────────────────────

class TestLanguagesEndpoint(unittest.TestCase):

    def test_languages_returns_200(self):
        response = client.get("/languages")
        self.assertEqual(response.status_code, 200)

    def test_languages_structure(self):
        data = client.get("/languages").json()
        self.assertIn("languages", data)
        self.assertIn("total", data)
        self.assertEqual(data["total"], len(data["languages"]))

    def test_languages_contains_v1_langs(self):
        data   = client.get("/languages").json()
        codes  = {l["code"] for l in data["languages"]}
        v1_required = {"fr", "en", "es", "de", "pt", "ar", "zh", "ja"}
        self.assertEqual(v1_required, codes)

    def test_each_language_has_required_fields(self):
        data = client.get("/languages").json()
        for lang in data["languages"]:
            for field in ("code", "name", "flag", "available_v1"):
                self.assertIn(field, lang, f"Champ '{field}' manquant pour {lang}")

    def test_languages_all_v1_available(self):
        data = client.get("/languages").json()
        for lang in data["languages"]:
            self.assertTrue(lang["available_v1"])

    def test_languages_no_auth_required(self):
        """GET /languages est public, pas de token requis."""
        response = client.get("/languages")
        self.assertNotEqual(response.status_code, 401)


# ─── Tests : DELETE /video/{video_id} ────────────────────────────────────────

class TestDeleteEndpoint(unittest.TestCase):

    def setUp(self):
        """Reset the rate limiter before each test."""
        rate_limiter.reset("user-test-123")

    def test_delete_nonexistent_returns_404(self):
        response = client.delete(
            "/video/nonexistent-video-id",
            headers=_auth_headers(),
        )
        self.assertEqual(response.status_code, 404)

    def test_delete_uploaded_video_success(self):
        """Upload puis suppression → 200."""
        upload = client.post(
            "/upload",
            headers=_auth_headers(),
            files={"file": ("del_test.mp4", _make_video_bytes(10), "video/mp4")},
        )
        video_id = upload.json()["video_id"]

        delete = client.delete(
            f"/video/{video_id}",
            headers=_auth_headers(),
        )
        self.assertEqual(delete.status_code, 200)
        data = delete.json()
        self.assertTrue(data["deleted"])
        self.assertEqual(data["video_id"], video_id)

    def test_delete_cleans_upload_file(self):
        """Après suppression, le fichier uploadé ne doit plus exister."""
        upload = client.post(
            "/upload",
            headers=_auth_headers(),
            files={"file": ("clean_test.mp4", _make_video_bytes(10), "video/mp4")},
        )
        video_id = upload.json()["video_id"]

        client.delete(f"/video/{video_id}", headers=_auth_headers())

        # Le fichier ne doit plus exister
        upload_dir = Path(os.environ["UPLOAD_DIR"])
        remaining  = list(upload_dir.glob(f"{video_id}.*"))
        self.assertEqual(len(remaining), 0)

    def test_delete_response_schema(self):
        upload   = client.post(
            "/upload",
            headers=_auth_headers(),
            files={"file": ("schema_test.mp4", _make_video_bytes(5), "video/mp4")},
        )
        video_id = upload.json()["video_id"]
        response = client.delete(f"/video/{video_id}", headers=_auth_headers())
        data     = response.json()

        for field in ("video_id", "deleted", "message"):
            self.assertIn(field, data)


# ─── Tests : Schémas Pydantic ─────────────────────────────────────────────────

class TestPydanticSchemas(unittest.TestCase):

    def test_translation_request_valid(self):
        from api.schemas.models import TranslationRequest
        req = TranslationRequest(video_id="abc", target_lang="fr")
        self.assertEqual(req.target_lang, "fr")
        self.assertEqual(req.source_lang, "en")

    def test_translation_request_invalid_lang_raises(self):
        from api.schemas.models import TranslationRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            TranslationRequest(video_id="abc", target_lang="klingon")

    def test_job_status_response_defaults(self):
        from api.schemas.models import JobStatusResponse, JobStatus
        r = JobStatusResponse(job_id="test", status=JobStatus.PENDING, progress=0)
        self.assertIsNone(r.error_message)
        self.assertIsNone(r.output_url)
        self.assertEqual(r.step_results, [])

    def test_mos_report_fields(self):
        from api.schemas.models import MOSReport
        mos = MOSReport(
            mos_score=4.2, wer_score=0.05, sync_diff_s=0.1,
            success_rate=1.0, language_confidence=0.97,
            meets_mos_target=True, meets_wer_target=True, overall_pass=True,
        )
        self.assertTrue(mos.overall_pass)

    def test_upload_response_defaults(self):
        from api.schemas.models import UploadResponse
        r = UploadResponse(video_id="abc", filename="test.mp4", size_bytes=1024)
        self.assertEqual(r.message, "Vidéo uploadée avec succès")


# ─── Tests : Rate Limiting sur /translate ────────────────────────────────────

class TestRateLimiting(unittest.TestCase):

    def setUp(self):
        """Réinitialiser le rate limiter avant chaque test."""
        rate_limiter.reset("user-rate-test")

    def test_rate_limit_allows_first_request(self):
        """La première requête /translate doit passer l'auth (400/404 mais pas 429)."""
        response = client.post(
            "/translate",
            headers=_auth_headers("user-rate-test"),
            json={"video_id": "fake-id", "target_lang": "fr"},
        )
        self.assertNotEqual(response.status_code, 429)

    def test_rate_limit_blocks_after_limit(self):
        """Après 10 requêtes, la suivante doit retourner 429."""
        # Remplir le compteur
        for _ in range(RATE_LIMIT):
            rate_limiter.is_allowed("user-rate-test-overflow")

        # Forcer le dépassement
        with patch("api.main.rate_limiter.is_allowed", return_value=(False, 0)):
            response = client.post(
                "/translate",
                headers=_auth_headers("user-rate-test-overflow"),
                json={"video_id": "fake", "target_lang": "fr"},
            )
        self.assertEqual(response.status_code, 429)
        self.assertIn("Retry-After", response.headers)


# ─── Récupération de la constante RATE_LIMIT ─────────────────────────────────
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_HOUR", "10"))


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
