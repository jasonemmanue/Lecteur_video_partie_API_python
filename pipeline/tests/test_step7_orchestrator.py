"""
Tests unitaires — Étape 7 : Orchestrateur & Évaluation MOS
==========================================================
Couverture :
  - PipelineConfig       : valeurs par défaut, personnalisation
  - StepResult           : to_dict()
  - MOSEvaluation        : seuils, overall_pass, to_dict()
  - PipelineResult       : to_status_dict()
  - PipelineOrchestrator : vidéo manquante, échec étape intermédiaire,
                           pipeline complet (mock toutes étapes),
                           compute_progress, evaluate_mos,
                           callback de progression, job_id auto
  - run_pipeline()       : fonction publique end-to-end (mock)
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from pipeline.step7_orchestrator import (
    PipelineConfig,
    PipelineStatus,
    PipelineStep,
    StepResult,
    MOSEvaluation,
    PipelineResult,
    PipelineOrchestrator,
    STEP_WEIGHTS,
    STEP_ORDER,
    run_pipeline,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_step_result(
    step: PipelineStep = PipelineStep.AUDIO_EXTRACTION,
    success: bool = True,
    duration: float = 1.0,
    error: str | None = None,
) -> StepResult:
    return StepResult(
        step=step,
        success=success,
        duration_s=duration,
        output_path=Path("/tmp/output.wav") if success else None,
        error=error,
    )


def _make_mos(
    mos: float = 4.2,
    wer: float = 0.05,
    sync_diff: float = 0.1,
    success_rate: float = 1.0,
    lang_conf: float = 0.97,
) -> MOSEvaluation:
    return MOSEvaluation(
        mos_score=mos,
        wer_score=wer,
        sync_diff_s=sync_diff,
        success_rate=success_rate,
        language_confidence=lang_conf,
    )


def _make_orchestrator(callback=None) -> PipelineOrchestrator:
    return PipelineOrchestrator(PipelineConfig(), callback)


def _all_steps_ok() -> list[StepResult]:
    return [_make_step_result(step=s) for s in PipelineStep]


# ─── Tests : PipelineConfig ───────────────────────────────────────────────────

class TestPipelineConfig(unittest.TestCase):

    def test_default_source_language(self):
        self.assertEqual(PipelineConfig().source_language, "en")

    def test_default_target_language(self):
        self.assertEqual(PipelineConfig().target_language, "fr")

    def test_default_whisper_model(self):
        self.assertEqual(PipelineConfig().whisper_model, "medium")

    def test_default_tts_device(self):
        self.assertEqual(PipelineConfig().tts_device, "cpu")

    def test_default_speaker_sample(self):
        self.assertGreaterEqual(PipelineConfig().speaker_sample_s, 6.0)

    def test_default_keep_intermediates(self):
        self.assertTrue(PipelineConfig().keep_intermediates)

    def test_custom_config(self):
        cfg = PipelineConfig(
            source_language="fr",
            target_language="en",
            whisper_model="large-v2",
            tts_device="cuda",
        )
        self.assertEqual(cfg.source_language, "fr")
        self.assertEqual(cfg.tts_device, "cuda")
        self.assertEqual(cfg.whisper_model, "large-v2")


# ─── Tests : StepResult ───────────────────────────────────────────────────────

class TestStepResult(unittest.TestCase):

    def test_to_dict_keys(self):
        sr = _make_step_result()
        d  = sr.to_dict()
        for key in ("step", "success", "duration_s", "output_path", "error"):
            self.assertIn(key, d)

    def test_to_dict_step_is_string(self):
        sr = _make_step_result(step=PipelineStep.TRANSLATION)
        self.assertEqual(sr.to_dict()["step"], "translation")

    def test_to_dict_output_path_string(self):
        sr = _make_step_result(success=True)
        self.assertIsInstance(sr.to_dict()["output_path"], str)

    def test_to_dict_output_path_none_when_failed(self):
        sr = _make_step_result(success=False, error="crash")
        self.assertIsNone(sr.to_dict()["output_path"])

    def test_to_dict_duration_rounded(self):
        sr = _make_step_result(duration=3.14159)
        self.assertEqual(sr.to_dict()["duration_s"], round(3.14159, 2))

    def test_to_dict_error_propagated(self):
        sr = _make_step_result(success=False, error="GPU OOM")
        self.assertEqual(sr.to_dict()["error"], "GPU OOM")


# ─── Tests : MOSEvaluation ────────────────────────────────────────────────────

class TestMOSEvaluation(unittest.TestCase):

    def test_meets_mos_target_above(self):
        mos = _make_mos(mos=4.0)
        self.assertTrue(mos.meets_mos_target)

    def test_meets_mos_target_below(self):
        mos = _make_mos(mos=3.5)
        self.assertFalse(mos.meets_mos_target)

    def test_meets_mos_target_exactly(self):
        mos = _make_mos(mos=3.8)
        self.assertTrue(mos.meets_mos_target)

    def test_meets_wer_target_below(self):
        mos = _make_mos(wer=0.05)
        self.assertTrue(mos.meets_wer_target)

    def test_meets_wer_target_above(self):
        mos = _make_mos(wer=0.15)
        self.assertFalse(mos.meets_wer_target)

    def test_meets_wer_target_exactly(self):
        mos = _make_mos(wer=0.08)
        self.assertTrue(mos.meets_wer_target)

    def test_overall_pass_both_ok(self):
        mos = _make_mos(mos=4.0, wer=0.05)
        self.assertTrue(mos.overall_pass)

    def test_overall_pass_mos_fails(self):
        mos = _make_mos(mos=3.0, wer=0.05)
        self.assertFalse(mos.overall_pass)

    def test_overall_pass_wer_fails(self):
        mos = _make_mos(mos=4.0, wer=0.20)
        self.assertFalse(mos.overall_pass)

    def test_to_dict_keys(self):
        mos = _make_mos()
        d   = mos.to_dict()
        for key in ("mos_score", "wer_score", "sync_diff_s", "success_rate",
                    "language_confidence", "meets_mos_target",
                    "meets_wer_target", "overall_pass", "details"):
            self.assertIn(key, d)

    def test_to_dict_rounded(self):
        mos = _make_mos(mos=4.123456, wer=0.054321)
        d   = mos.to_dict()
        self.assertEqual(d["mos_score"], round(4.123456, 2))
        self.assertEqual(d["wer_score"], round(0.054321, 4))

    def test_mos_range_valid(self):
        """MOS doit toujours être entre 1.0 et 5.0."""
        mos = _make_mos(mos=4.5)
        self.assertGreaterEqual(mos.mos_score, 1.0)
        self.assertLessEqual(mos.mos_score, 5.0)


# ─── Tests : PipelineResult ───────────────────────────────────────────────────

class TestPipelineResult(unittest.TestCase):

    def _make_result(self, status=PipelineStatus.DONE) -> PipelineResult:
        return PipelineResult(
            job_id="test-job-123",
            status=status,
            video_input=Path("/fake/video.mp4"),
            video_output=Path("/tmp/video_translated.mp4"),
            step_results=_all_steps_ok(),
            mos_evaluation=_make_mos(),
            current_step=PipelineStep.FINALIZATION,
            progress=100,
            total_duration_s=120.5,
        )

    def test_to_status_dict_keys(self):
        result = self._make_result()
        d = result.to_status_dict()
        for key in ("job_id", "status", "progress", "current_step",
                    "total_duration_s", "error_message",
                    "output_path", "step_results", "mos_evaluation"):
            self.assertIn(key, d)

    def test_to_status_dict_status_string(self):
        result = self._make_result(PipelineStatus.DONE)
        self.assertEqual(result.to_status_dict()["status"], "done")

    def test_to_status_dict_progress(self):
        result = self._make_result()
        self.assertEqual(result.to_status_dict()["progress"], 100)

    def test_to_status_dict_step_results_list(self):
        result = self._make_result()
        self.assertIsInstance(result.to_status_dict()["step_results"], list)

    def test_to_status_dict_mos_included(self):
        result = self._make_result()
        d = result.to_status_dict()
        self.assertIsNotNone(d["mos_evaluation"])
        self.assertIn("mos_score", d["mos_evaluation"])

    def test_to_status_dict_error_result(self):
        result = PipelineResult(
            job_id="err-job",
            status=PipelineStatus.ERROR,
            video_input=Path("/fake/video.mp4"),
            error_message="Step failed",
        )
        d = result.to_status_dict()
        self.assertEqual(d["status"], "error")
        self.assertEqual(d["error_message"], "Step failed")
        self.assertIsNone(d["output_path"])


# ─── Tests : STEP_WEIGHTS ────────────────────────────────────────────────────

class TestStepWeights(unittest.TestCase):

    def test_weights_sum_to_100(self):
        self.assertEqual(sum(STEP_WEIGHTS.values()), 100)

    def test_all_steps_have_weight(self):
        for step in PipelineStep:
            self.assertIn(step, STEP_WEIGHTS)

    def test_all_weights_positive(self):
        for w in STEP_WEIGHTS.values():
            self.assertGreater(w, 0)


# ─── Tests : _compute_progress ───────────────────────────────────────────────

class TestComputeProgress(unittest.TestCase):

    def test_first_step_progress_zero(self):
        p = PipelineOrchestrator._compute_progress(PipelineStep.AUDIO_EXTRACTION)
        self.assertEqual(p, 0)

    def test_second_step_progress_equals_first_weight(self):
        p = PipelineOrchestrator._compute_progress(PipelineStep.SPEECH_TO_TEXT)
        self.assertEqual(p, STEP_WEIGHTS[PipelineStep.AUDIO_EXTRACTION])

    def test_progress_increases_with_steps(self):
        prev = -1
        for step in STEP_ORDER:
            p = PipelineOrchestrator._compute_progress(step)
            self.assertGreater(p, prev)
            prev = p

    def test_progress_never_reaches_100(self):
        for step in PipelineStep:
            p = PipelineOrchestrator._compute_progress(step)
            self.assertLessEqual(p, 99)

    def test_progress_non_negative(self):
        for step in PipelineStep:
            self.assertGreaterEqual(
                PipelineOrchestrator._compute_progress(step), 0
            )


# ─── Tests : _evaluate_mos ───────────────────────────────────────────────────

class TestEvaluateMOS(unittest.TestCase):

    def setUp(self):
        self.orchestrator = _make_orchestrator()

    def _make_ctx(
        self,
        tts_rate=1.0,
        lang_conf=0.98,
        sync_diff=0.0,
    ) -> dict:
        return {
            "tts_success_rate": tts_rate,
            "lang_confidence":  lang_conf,
            "sync_diff":        sync_diff,
        }

    def _make_pipeline_result(self, n_steps_ok=7) -> PipelineResult:
        steps = [
            _make_step_result(success=(i < n_steps_ok))
            for i in range(7)
        ]
        return PipelineResult(
            job_id="test",
            status=PipelineStatus.DONE,
            video_input=Path("/fake/video.mp4"),
            step_results=steps,
        )

    def test_perfect_conditions_high_mos(self):
        ctx    = self._make_ctx(tts_rate=1.0, lang_conf=0.99, sync_diff=0.0)
        result = self._make_pipeline_result(7)
        mos    = self.orchestrator._evaluate_mos(ctx, result)
        self.assertGreaterEqual(mos.mos_score, 4.5)

    def test_poor_conditions_low_mos(self):
        ctx    = self._make_ctx(tts_rate=0.3, lang_conf=0.5, sync_diff=4.0)
        result = self._make_pipeline_result(3)
        mos    = self.orchestrator._evaluate_mos(ctx, result)
        self.assertLess(mos.mos_score, 3.5)

    def test_mos_always_in_range(self):
        for tts in [0.0, 0.5, 1.0]:
            for lang in [0.3, 0.7, 1.0]:
                ctx    = self._make_ctx(tts_rate=tts, lang_conf=lang)
                result = self._make_pipeline_result()
                mos    = self.orchestrator._evaluate_mos(ctx, result)
                self.assertGreaterEqual(mos.mos_score, 1.0)
                self.assertLessEqual(mos.mos_score, 5.0)

    def test_wer_inversely_related_to_confidence(self):
        ctx1 = self._make_ctx(lang_conf=0.99)
        ctx2 = self._make_ctx(lang_conf=0.70)
        r    = self._make_pipeline_result()
        mos1 = self.orchestrator._evaluate_mos(ctx1, r)
        mos2 = self.orchestrator._evaluate_mos(ctx2, r)
        self.assertLess(mos1.wer_score, mos2.wer_score)

    def test_sync_diff_impacts_mos(self):
        ctx_good = self._make_ctx(sync_diff=0.0)
        ctx_bad  = self._make_ctx(sync_diff=4.5)
        r        = self._make_pipeline_result()
        mos_good = self.orchestrator._evaluate_mos(ctx_good, r)
        mos_bad  = self.orchestrator._evaluate_mos(ctx_bad, r)
        self.assertGreater(mos_good.mos_score, mos_bad.mos_score)

    def test_empty_ctx_no_crash(self):
        ctx    = {}
        result = self._make_pipeline_result()
        mos    = self.orchestrator._evaluate_mos(ctx, result)
        self.assertIsNotNone(mos)
        self.assertGreaterEqual(mos.mos_score, 1.0)

    def test_details_populated(self):
        ctx    = self._make_ctx()
        result = self._make_pipeline_result()
        mos    = self.orchestrator._evaluate_mos(ctx, result)
        self.assertIn("composite_score", mos.details)
        self.assertIn("sync_score", mos.details)


# ─── Tests : PipelineOrchestrator.run() ──────────────────────────────────────

class TestPipelineOrchestratorRun(unittest.TestCase):

    def test_returns_error_if_video_missing(self):
        o      = _make_orchestrator()
        result = o.run("/nonexistent/video.mp4", "/tmp")
        self.assertEqual(result.status, PipelineStatus.ERROR)
        self.assertIn("introuvable", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_stops_on_first_step_failure(self, mock_mkdir, mock_exists):
        """Si l'étape 1 échoue, les étapes suivantes ne sont pas exécutées."""
        o = _make_orchestrator()
        with patch.object(o, "_step1_audio_extraction", side_effect=RuntimeError("FFmpeg crash")):
            result = o.run("/fake/video.mp4", "/tmp")

        self.assertEqual(result.status, PipelineStatus.ERROR)
        self.assertEqual(len(result.step_results), 1)
        self.assertFalse(result.step_results[0].success)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_error_message_propagated(self, mock_mkdir, mock_exists):
        o = _make_orchestrator()
        with patch.object(o, "_step1_audio_extraction", side_effect=RuntimeError("disk full")):
            result = o.run("/fake/video.mp4", "/tmp")
        self.assertIn("disk full", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_job_id_auto_generated(self, mock_mkdir, mock_exists):
        o = _make_orchestrator()
        with patch.object(o, "_step1_audio_extraction", side_effect=RuntimeError("fail")):
            result = o.run("/fake/video.mp4", "/tmp")
        self.assertIsNotNone(result.job_id)
        self.assertGreater(len(result.job_id), 0)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_custom_job_id_used(self, mock_mkdir, mock_exists):
        o = _make_orchestrator()
        with patch.object(o, "_step1_audio_extraction", side_effect=RuntimeError("fail")):
            result = o.run("/fake/video.mp4", "/tmp", job_id="my-custom-id")
        self.assertEqual(result.job_id, "my-custom-id")

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_full_pipeline_success(self, mock_mkdir, mock_exists):
        """Pipeline complet avec toutes les étapes mockées."""
        o = _make_orchestrator()

        fake_path = Path("/tmp/output")

        patches = [
            patch.object(o, "_step1_audio_extraction", return_value=fake_path),
            patch.object(o, "_step2_transcription",    return_value=fake_path),
            patch.object(o, "_step3_emotion_analysis", return_value=fake_path),
            patch.object(o, "_step4_translation",      return_value=fake_path),
            patch.object(o, "_step5_tts_synthesis",    return_value=fake_path),
            patch.object(o, "_step6_synchronization",  return_value=fake_path),
            patch.object(o, "_step7_finalization",     return_value=fake_path),
        ]

        with patches[0], patches[1], patches[2], patches[3], \
             patches[4], patches[5], patches[6]:
            result = o.run("/fake/video.mp4", "/tmp")

        self.assertEqual(result.status, PipelineStatus.DONE)
        self.assertEqual(len(result.step_results), 7)
        self.assertTrue(all(s.success for s in result.step_results))
        self.assertIsNotNone(result.mos_evaluation)
        self.assertEqual(result.progress, 100)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_progress_callback_called(self, mock_mkdir, mock_exists):
        """Le callback de progression est appelé après chaque étape."""
        calls = []
        def callback(r): calls.append(r.current_step)

        o = PipelineOrchestrator(PipelineConfig(), callback)
        with patch.object(o, "_step1_audio_extraction", side_effect=RuntimeError("stop")):
            o.run("/fake/video.mp4", "/tmp")

        self.assertGreater(len(calls), 0)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_total_duration_recorded(self, mock_mkdir, mock_exists):
        o = _make_orchestrator()
        with patch.object(o, "_step1_audio_extraction", side_effect=RuntimeError("fail")):
            result = o.run("/fake/video.mp4", "/tmp")
        self.assertGreaterEqual(result.total_duration_s, 0)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_step_failure_in_middle(self, mock_mkdir, mock_exists):
        """Échec à l'étape 3 → étapes 4, 5, 6, 7 non exécutées."""
        o = _make_orchestrator()
        fake_path = Path("/tmp/output")

        with patch.object(o, "_step1_audio_extraction", return_value=fake_path), \
             patch.object(o, "_step2_transcription", return_value=fake_path), \
             patch.object(o, "_step3_emotion_analysis", side_effect=RuntimeError("model OOM")):
            result = o.run("/fake/video.mp4", "/tmp")

        self.assertEqual(result.status, PipelineStatus.ERROR)
        self.assertEqual(len(result.step_results), 3)
        self.assertTrue(result.step_results[0].success)
        self.assertTrue(result.step_results[1].success)
        self.assertFalse(result.step_results[2].success)


# ─── Tests : _run_step ───────────────────────────────────────────────────────

class TestRunStep(unittest.TestCase):

    def test_success_step_result(self):
        o   = _make_orchestrator()
        ctx = {}
        result = o._run_step(
            PipelineStep.AUDIO_EXTRACTION,
            lambda c: Path("/tmp/audio.wav"),
            ctx,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.output_path.as_posix(), "/tmp/audio.wav")
        self.assertIsNone(result.error)

    def test_failure_step_result(self):
        o   = _make_orchestrator()
        ctx = {}
        result = o._run_step(
            PipelineStep.SPEECH_TO_TEXT,
            lambda c: (_ for _ in ()).throw(RuntimeError("Whisper OOM")),
            ctx,
        )
        self.assertFalse(result.success)
        self.assertIn("Whisper OOM", result.error)
        self.assertIsNone(result.output_path)

    def test_duration_recorded(self):
        o   = _make_orchestrator()
        ctx = {}
        result = o._run_step(
            PipelineStep.TRANSLATION,
            lambda c: Path("/tmp/translated.json"),
            ctx,
        )
        self.assertGreaterEqual(result.duration_s, 0)


# ─── Tests : fonction publique run_pipeline() ─────────────────────────────────

class TestRunPipelinePublicAPI(unittest.TestCase):

    @patch("pipeline.step7_orchestrator.PipelineOrchestrator.run")
    def test_success_path(self, mock_run):
        mock_run.return_value = PipelineResult(
            job_id="abc123",
            status=PipelineStatus.DONE,
            video_input=Path("/fake/video.mp4"),
            video_output=Path("/tmp/video_translated.mp4"),
            step_results=_all_steps_ok(),
            mos_evaluation=_make_mos(),
            progress=100,
            total_duration_s=120.0,
        )
        result = run_pipeline("/fake/video.mp4", "/tmp", "en", "fr")
        self.assertEqual(result.status, PipelineStatus.DONE)
        self.assertEqual(result.progress, 100)

    @patch("pipeline.step7_orchestrator.PipelineOrchestrator.run")
    def test_failure_path(self, mock_run):
        mock_run.return_value = PipelineResult(
            job_id="err123",
            status=PipelineStatus.ERROR,
            video_input=Path("/fake/video.mp4"),
            error_message="FFmpeg crash",
        )
        result = run_pipeline("/fake/video.mp4", "/tmp")
        self.assertEqual(result.status, PipelineStatus.ERROR)

    @patch("pipeline.step7_orchestrator.PipelineOrchestrator.run")
    def test_custom_params_passed(self, mock_run):
        mock_run.return_value = PipelineResult(
            job_id="test",
            status=PipelineStatus.DONE,
            video_input=Path("/fake/video.mp4"),
        )
        run_pipeline(
            "/fake/video.mp4", "/tmp",
            source_language="fr",
            target_language="de",
            whisper_model="large-v2",
            device="cuda",
        )
        mock_run.assert_called_once()

    @patch("pipeline.step7_orchestrator.PipelineOrchestrator.run")
    def test_to_status_dict_serializable(self, mock_run):
        """to_status_dict() doit produire un JSON valide."""
        mock_run.return_value = PipelineResult(
            job_id="json-test",
            status=PipelineStatus.DONE,
            video_input=Path("/fake/video.mp4"),
            video_output=Path("/tmp/output.mp4"),
            step_results=_all_steps_ok(),
            mos_evaluation=_make_mos(),
            progress=100,
        )
        result = run_pipeline("/fake/video.mp4", "/tmp")
        d      = result.to_status_dict()
        # Doit être sérialisable en JSON sans erreur
        serialized = json.dumps(d)
        self.assertIsInstance(serialized, str)
        self.assertIn("job_id", serialized)


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
