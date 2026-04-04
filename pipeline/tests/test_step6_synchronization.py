"""
Tests unitaires — Étape 6 : Synchronisation & Assemblage Vidéo
==============================================================
Couverture :
  - SyncConfig           : valeurs par défaut, personnalisation
  - SyncReport           : to_dict(), calculs
  - AssemblyResult       : output_size_mb, defaults
  - VideoAssembler       : ffmpeg manquant, fichier manquant,
                           assemblage réussi (mock complet),
                           compute_sync_report (no-adjust/adjust),
                           build_atempo_filter (normal/hors-plage),
                           build_ffmpeg_cmd (avec/sans SRT),
                           probe_duration, run_ffmpeg (ok/fail/timeout)
  - Helpers              : build_atempo_chain, get_video_info
  - assemble_video()     : fonction publique end-to-end (mock)
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from pipeline.step6_synchronization import (
    SyncConfig,
    SyncReport,
    AssemblyResult,
    VideoAssembler,
    build_atempo_chain,
    get_video_info,
    assemble_video,
)


# ─── Helpers de test ──────────────────────────────────────────────────────────

def _make_assembler() -> VideoAssembler:
    """Crée un VideoAssembler avec FFmpeg mocké."""
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        return VideoAssembler()


def _make_report(
    video_duration: float = 60.0,
    audio_duration: float = 60.0,
    diff: float = 0.0,
    atempo: str = "",
    adjustment: bool = False,
    segments: int = 0,
) -> SyncReport:
    return SyncReport(
        video_duration_s=video_duration,
        audio_duration_s=audio_duration,
        duration_diff_s=diff,
        atempo_filter=atempo,
        adjustment_needed=adjustment,
        segments_adjusted=segments,
    )


def _ffmpeg_proc(returncode: int = 0, stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stderr     = stderr
    proc.stdout     = ""
    return proc


# ─── Tests : SyncConfig ───────────────────────────────────────────────────────

class TestSyncConfig(unittest.TestCase):

    def test_default_video_codec(self):
        self.assertEqual(SyncConfig().video_codec, "libx264")

    def test_default_audio_codec(self):
        self.assertEqual(SyncConfig().audio_codec, "aac")

    def test_default_crf(self):
        self.assertEqual(SyncConfig().video_crf, 23)

    def test_default_audio_bitrate(self):
        self.assertEqual(SyncConfig().audio_bitrate, "192k")

    def test_default_max_atempo(self):
        self.assertAlmostEqual(SyncConfig().max_atempo, 2.0)

    def test_default_min_atempo(self):
        self.assertAlmostEqual(SyncConfig().min_atempo, 0.5)

    def test_default_embed_subtitles(self):
        self.assertTrue(SyncConfig().embed_subtitles)

    def test_default_timeout(self):
        self.assertEqual(SyncConfig().timeout_seconds, 600)

    def test_custom_config(self):
        cfg = SyncConfig(video_crf=18, audio_bitrate="320k", embed_subtitles=False)
        self.assertEqual(cfg.video_crf, 18)
        self.assertEqual(cfg.audio_bitrate, "320k")
        self.assertFalse(cfg.embed_subtitles)


# ─── Tests : SyncReport ───────────────────────────────────────────────────────

class TestSyncReport(unittest.TestCase):

    def test_to_dict_keys(self):
        report = _make_report()
        d = report.to_dict()
        for key in ("video_duration_s", "audio_duration_s", "duration_diff_s",
                    "atempo_filter", "adjustment_needed", "segments_adjusted"):
            self.assertIn(key, d)

    def test_to_dict_rounded(self):
        report = _make_report(video_duration=60.123456, audio_duration=61.987654)
        d = report.to_dict()
        self.assertEqual(d["video_duration_s"], round(60.123456, 3))
        self.assertEqual(d["audio_duration_s"], round(61.987654, 3))

    def test_to_dict_adjustment_needed_false(self):
        report = _make_report(adjustment=False)
        self.assertFalse(report.to_dict()["adjustment_needed"])

    def test_to_dict_atempo_filter_empty(self):
        report = _make_report(atempo="")
        self.assertEqual(report.to_dict()["atempo_filter"], "")


# ─── Tests : AssemblyResult ───────────────────────────────────────────────────

class TestAssemblyResult(unittest.TestCase):

    def test_output_size_mb(self):
        result = AssemblyResult(success=True, output_size_bytes=10_485_760)
        self.assertAlmostEqual(result.output_size_mb, 10.0)

    def test_output_size_mb_zero(self):
        result = AssemblyResult(success=True, output_size_bytes=0)
        self.assertEqual(result.output_size_mb, 0.0)

    def test_failure_defaults(self):
        result = AssemblyResult(success=False, error_message="oops")
        self.assertFalse(result.success)
        self.assertIsNone(result.output_video_path)
        self.assertIsNone(result.report)


# ─── Tests : VideoAssembler — init ───────────────────────────────────────────

class TestVideoAssemblerInit(unittest.TestCase):

    def test_raises_if_ffmpeg_missing(self):
        with patch("shutil.which", return_value=None):
            with self.assertRaises(EnvironmentError) as ctx:
                VideoAssembler()
            self.assertIn("FFmpeg introuvable", str(ctx.exception))

    def test_ok_when_ffmpeg_present(self):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            assembler = VideoAssembler()
            self.assertIsNotNone(assembler)


# ─── Tests : VideoAssembler.assemble() ───────────────────────────────────────

class TestVideoAssemblerAssemble(unittest.TestCase):

    def test_returns_failure_if_video_missing(self):
        a = _make_assembler()
        result = a.assemble("/nonexistent/video.mp4", "/fake/audio.wav", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("Vidéo source", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    def test_returns_failure_if_audio_missing(self, mock_exists):
        a = _make_assembler()
        mock_exists.side_effect = [True, False]
        result = a.assemble("/fake/video.mp4", "/nonexistent/audio.wav", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("Audio TTS", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_returns_failure_on_exception(self, mock_mkdir, mock_exists):
        a = _make_assembler()
        with patch.object(a, "_probe_duration", side_effect=RuntimeError("probe failed")):
            result = a.assemble("/fake/video.mp4", "/fake/audio.wav", "/tmp")
        self.assertFalse(result.success)
        self.assertIn("probe failed", result.error_message)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.stat")
    def test_successful_assembly(self, mock_stat, mock_mkdir, mock_exists):
        a = _make_assembler()
        mock_stat.return_value.st_size = 52_428_800  # 50 MB

        with patch.object(a, "_probe_duration", return_value=60.0):
            with patch.object(a, "_run_ffmpeg", return_value=""):
                # simuler que le fichier de sortie existe
                with patch("pathlib.Path.exists", return_value=True):
                    result = a.assemble(
                        "/fake/video.mp4", "/fake/audio.wav", "/tmp"
                    )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.output_video_path)
        self.assertIsNotNone(result.report)
        self.assertGreaterEqual(result.processing_time_s, 0)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_returns_failure_if_output_not_created(self, mock_mkdir, mock_exists):
        """Si FFmpeg ne crée pas le fichier de sortie → échec."""
        a = _make_assembler()
        with patch.object(a, "_probe_duration", return_value=60.0):
            with patch.object(a, "_run_ffmpeg", return_value=""):
                # output n'existe pas après FFmpeg
                with patch("pathlib.Path.exists", side_effect=[True, True, False]):
                    result = a.assemble(
                        "/fake/video.mp4", "/fake/audio.wav", "/tmp"
                    )
        self.assertFalse(result.success)
        self.assertIn("fichier de sortie", result.error_message)


# ─── Tests : _compute_sync_report ────────────────────────────────────────────

class TestComputeSyncReport(unittest.TestCase):

    def setUp(self):
        self.assembler = _make_assembler()

    def test_no_adjustment_when_same_duration(self):
        report = self.assembler._compute_sync_report(60.0, 60.0)
        self.assertFalse(report.adjustment_needed)
        self.assertEqual(report.atempo_filter, "")

    def test_no_adjustment_within_1_percent(self):
        # 60s vs 60.5s → ratio 1.008 < 1%
        report = self.assembler._compute_sync_report(60.0, 60.5)
        self.assertFalse(report.adjustment_needed)

    def test_adjustment_needed_when_audio_longer(self):
        # audio 20% plus long
        report = self.assembler._compute_sync_report(60.0, 72.0)
        self.assertTrue(report.adjustment_needed)
        self.assertIn("atempo", report.atempo_filter)

    def test_adjustment_needed_when_audio_shorter(self):
        # audio 20% plus court
        report = self.assembler._compute_sync_report(60.0, 48.0)
        self.assertTrue(report.adjustment_needed)
        self.assertIn("atempo", report.atempo_filter)

    def test_duration_diff_computed(self):
        report = self.assembler._compute_sync_report(60.0, 65.0)
        self.assertAlmostEqual(report.duration_diff_s, 5.0)

    def test_zero_video_duration(self):
        """Pas de division par zéro si vidéo vide."""
        report = self.assembler._compute_sync_report(0.0, 60.0)
        self.assertFalse(report.adjustment_needed)


# ─── Tests : _build_atempo_filter ────────────────────────────────────────────

class TestBuildAtempoFilter(unittest.TestCase):

    def setUp(self):
        self.assembler = _make_assembler()

    def test_ratio_within_range(self):
        f = self.assembler._build_atempo_filter(1.2)
        self.assertIn("atempo=1.2", f)
        self.assertEqual(f.count("atempo"), 1)

    def test_ratio_exactly_max(self):
        f = self.assembler._build_atempo_filter(2.0)
        self.assertIn("atempo=2.0", f)

    def test_ratio_exactly_min(self):
        f = self.assembler._build_atempo_filter(0.5)
        self.assertIn("atempo=0.5", f)

    def test_ratio_above_max_uses_two_filters(self):
        """ratio=3.0 > 2.0 → deux filtres atempo enchaînés."""
        f = self.assembler._build_atempo_filter(3.0)
        self.assertEqual(f.count("atempo"), 2)

    def test_ratio_below_min_uses_two_filters(self):
        """ratio=0.3 < 0.5 → filtre atempo composé."""
        f = self.assembler._build_atempo_filter(0.3)
        self.assertIn("atempo", f)

    def test_filter_format_valid(self):
        """Le filtre doit commencer par 'atempo='."""
        f = self.assembler._build_atempo_filter(1.5)
        self.assertTrue(f.startswith("atempo="))


# ─── Tests : _build_ffmpeg_cmd ────────────────────────────────────────────────

class TestBuildFfmpegCmd(unittest.TestCase):

    def setUp(self):
        self.assembler = _make_assembler()
        self.report_no_adj = _make_report(adjustment=False)
        self.report_adj    = _make_report(adjustment=True, atempo="atempo=1.200000")

    def test_cmd_contains_input_files(self):
        cmd = self.assembler._build_ffmpeg_cmd(
            Path("/fake/video.mp4"),
            Path("/fake/audio.wav"),
            Path("/tmp/output.mp4"),
            self.report_no_adj,
            None,
        )
        cmd_str = ' '.join(cmd).replace('\\', '/')
        self.assertIn("/fake/video.mp4", cmd_str)
        self.assertIn("/fake/audio.wav", cmd_str)

    def test_cmd_maps_video_and_audio(self):
        cmd = self.assembler._build_ffmpeg_cmd(
            Path("/fake/video.mp4"),
            Path("/fake/audio.wav"),
            Path("/tmp/output.mp4"),
            self.report_no_adj,
            None,
        )
        self.assertIn("0:v:0", cmd)
        self.assertIn("1:a:0", cmd)

    def test_cmd_includes_atempo_when_adjustment(self):
        cmd = self.assembler._build_ffmpeg_cmd(
            Path("/fake/video.mp4"),
            Path("/fake/audio.wav"),
            Path("/tmp/output.mp4"),
            self.report_adj,
            None,
        )
        self.assertIn("-af", cmd)
        self.assertIn("atempo=1.200000", cmd)

    def test_cmd_no_atempo_when_no_adjustment(self):
        cmd = self.assembler._build_ffmpeg_cmd(
            Path("/fake/video.mp4"),
            Path("/fake/audio.wav"),
            Path("/tmp/output.mp4"),
            self.report_no_adj,
            None,
        )
        self.assertNotIn("-af", cmd)

    def test_cmd_contains_shortest_flag(self):
        cmd = self.assembler._build_ffmpeg_cmd(
            Path("/fake/video.mp4"),
            Path("/fake/audio.wav"),
            Path("/tmp/output.mp4"),
            self.report_no_adj,
            None,
        )
        self.assertIn("-shortest", cmd)

    def test_cmd_includes_srt_when_provided(self):
        with patch("pathlib.Path.exists", return_value=True):
            cmd = self.assembler._build_ffmpeg_cmd(
                Path("/fake/video.mp4"),
                Path("/fake/audio.wav"),
                Path("/tmp/output.mp4"),
                self.report_no_adj,
                Path("/fake/subtitles.srt"),
            )
        cmd_str = ' '.join(cmd).replace('\\', '/')
        self.assertIn("/fake/subtitles.srt", cmd_str)
        self.assertIn("mov_text", cmd)

    def test_cmd_no_srt_when_not_provided(self):
        cmd = self.assembler._build_ffmpeg_cmd(
            Path("/fake/video.mp4"),
            Path("/fake/audio.wav"),
            Path("/tmp/output.mp4"),
            self.report_no_adj,
            None,
        )
        self.assertNotIn("mov_text", cmd)

    def test_cmd_output_path_last(self):
        cmd = self.assembler._build_ffmpeg_cmd(
            Path("/fake/video.mp4"),
            Path("/fake/audio.wav"),
            Path("/tmp/output.mp4"),
            self.report_no_adj,
            None,
        )
        output_normalized = cmd[-1].replace('\\', '/')
        self.assertEqual(output_normalized, "/tmp/output.mp4")


# ─── Tests : _run_ffmpeg ─────────────────────────────────────────────────────

class TestRunFfmpeg(unittest.TestCase):

    def setUp(self):
        self.assembler = _make_assembler()

    @patch("subprocess.run")
    def test_success_returns_stderr(self, mock_run):
        mock_run.return_value = _ffmpeg_proc(returncode=0, stderr="warnings only")
        result = self.assembler._run_ffmpeg(["ffmpeg", "-i", "test.mp4"])
        self.assertEqual(result, "warnings only")

    @patch("subprocess.run")
    def test_nonzero_raises_runtime_error(self, mock_run):
        mock_run.return_value = _ffmpeg_proc(returncode=1, stderr="codec error")
        with self.assertRaises(RuntimeError) as ctx:
            self.assembler._run_ffmpeg(["ffmpeg", "-i", "bad.mp4"])
        self.assertIn("FFmpeg a échoué", str(ctx.exception))

    @patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("ffmpeg", 600))
    def test_timeout_raises_runtime_error(self, mock_run):
        with self.assertRaises(RuntimeError) as ctx:
            self.assembler._run_ffmpeg(["ffmpeg", "-i", "long.mp4"])
        self.assertIn("timeout", str(ctx.exception).lower())


# ─── Tests : _probe_duration ─────────────────────────────────────────────────

class TestProbeDuration(unittest.TestCase):

    def setUp(self):
        self.assembler = _make_assembler()

    @patch("subprocess.run")
    def test_returns_float_duration(self, mock_run):
        proc = MagicMock()
        proc.stdout = "120.543\n"
        mock_run.return_value = proc
        duration = self.assembler._probe_duration(Path("/fake/video.mp4"))
        self.assertAlmostEqual(duration, 120.543)

    @patch("subprocess.run")
    def test_returns_zero_on_invalid_output(self, mock_run):
        proc = MagicMock()
        proc.stdout = "N/A\n"
        mock_run.return_value = proc
        duration = self.assembler._probe_duration(Path("/fake/video.mp4"))
        self.assertEqual(duration, 0.0)

    @patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("ffprobe", 30))
    def test_returns_zero_on_timeout(self, mock_run):
        duration = self.assembler._probe_duration(Path("/fake/video.mp4"))
        self.assertEqual(duration, 0.0)


# ─── Tests : build_atempo_chain ──────────────────────────────────────────────

class TestBuildAtempoChain(unittest.TestCase):

    def test_normal_ratio(self):
        chain = build_atempo_chain(1.5)
        self.assertIn("atempo=1.5", chain)

    def test_high_ratio_chained(self):
        """ratio=4.0 → plusieurs filtres atempo enchaînés."""
        chain = build_atempo_chain(4.0)
        self.assertGreaterEqual(chain.count("atempo"), 2)

    def test_low_ratio_chained(self):
        """ratio=0.25 → deux filtres atempo=0.5."""
        chain = build_atempo_chain(0.25)
        self.assertGreaterEqual(chain.count("atempo"), 2)

    def test_ratio_one_single_filter(self):
        chain = build_atempo_chain(1.0)
        self.assertEqual(chain.count("atempo"), 1)

    def test_output_is_comma_separated(self):
        chain = build_atempo_chain(3.0)
        if chain.count("atempo") > 1:
            self.assertIn(",", chain)


# ─── Tests : get_video_info ───────────────────────────────────────────────────

class TestGetVideoInfo(unittest.TestCase):

    @patch("subprocess.run")
    def test_returns_video_info(self, mock_run):
        fake_output = json.dumps({
            "streams": [
                {
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "codec_name": "h264",
                }
            ],
            "format": {
                "duration": "120.5",
                "size": "52428800",
                "bit_rate": "3500000",
            },
        })
        proc = MagicMock()
        proc.stdout     = fake_output
        proc.returncode = 0
        mock_run.return_value = proc

        info = get_video_info("/fake/video.mp4")
        self.assertEqual(info["width"], 1920)
        self.assertEqual(info["height"], 1080)
        self.assertAlmostEqual(info["fps"], 30.0)
        self.assertEqual(info["codec_video"], "h264")
        self.assertAlmostEqual(info["duration"], 120.5)

    @patch("subprocess.run", side_effect=Exception("ffprobe not found"))
    def test_returns_empty_on_error(self, mock_run):
        info = get_video_info("/fake/video.mp4")
        self.assertEqual(info, {})

    @patch("subprocess.run")
    def test_handles_fractional_fps(self, mock_run):
        fake_output = json.dumps({
            "streams": [{"r_frame_rate": "24000/1001", "codec_name": "h264"}],
            "format": {"duration": "60.0", "size": "0", "bit_rate": "0"},
        })
        proc = MagicMock()
        proc.stdout = fake_output
        mock_run.return_value = proc
        info = get_video_info("/fake/video.mp4")
        self.assertAlmostEqual(info["fps"], 23.98, places=1)


# ─── Tests : fonction publique assemble_video() ───────────────────────────────

class TestAssembleVideoPublicAPI(unittest.TestCase):

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("pipeline.step6_synchronization.VideoAssembler.assemble")
    def test_success_path(self, mock_assemble, mock_which):
        mock_assemble.return_value = AssemblyResult(
            success=True,
            output_video_path=Path("/tmp/video_translated.mp4"),
            report=_make_report(),
            processing_time_s=42.0,
            output_size_bytes=52_428_800,
        )
        result = assemble_video(
            "/fake/video.mp4", "/fake/audio.wav", "/tmp"
        )
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output_video_path)
        self.assertAlmostEqual(result.output_size_mb, 50.0)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("pipeline.step6_synchronization.VideoAssembler.assemble")
    def test_failure_path(self, mock_assemble, mock_which):
        mock_assemble.return_value = AssemblyResult(
            success=False,
            error_message="FFmpeg indisponible",
        )
        result = assemble_video(
            "/fake/video.mp4", "/fake/audio.wav", "/tmp"
        )
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("pipeline.step6_synchronization.VideoAssembler.assemble")
    def test_srt_path_passed(self, mock_assemble, mock_which):
        mock_assemble.return_value = AssemblyResult(success=True)
        assemble_video(
            "/fake/video.mp4", "/fake/audio.wav", "/tmp",
            srt_path="/fake/subtitles.srt",
        )
        mock_assemble.assert_called_once()
        args, kwargs = mock_assemble.call_args
        # srt_path = 4e arg positionnel (index 3)
        srt = args[3] if len(args) > 3 else kwargs.get("srt_path")
        self.assertIsNotNone(srt)
        self.assertIn("subtitles.srt", str(srt))

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("pipeline.step6_synchronization.VideoAssembler.assemble")
    def test_manifest_path_passed(self, mock_assemble, mock_which):
        mock_assemble.return_value = AssemblyResult(success=True)
        assemble_video(
            "/fake/video.mp4", "/fake/audio.wav", "/tmp",
            manifest_path="/fake/manifest.json",
        )
        mock_assemble.assert_called_once()


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
