"""Checks for pooled caching, matched launch budgets and tokenizer handoff."""
import json
import itertools
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_pc_recon_screen as screen
import test_room_view_pools as room_tests
from evaluate_pc_overnight import verify_frozen_reconstruction, point_metrics
from train_pc_unite import _apply_recon_freeze
from hy3dgen.shapegen.pc_recon_runtime import PooledSurfaceDataset
from hy3dgen.shapegen.pc_render_dataset import collate_surface_render
from hy3dgen.shapegen.vggt_context import CachedVGGTContextStore


class OvernightTests(unittest.TestCase):
    def test_cpu_pipeline_trains_hands_off_and_exports_five_clouds(self):
        fixture = room_tests.RoomViewPoolTests()
        fixture.setUp()
        try:
            root = fixture.root
            pools = root / "views.json"
            pools.write_text(json.dumps(fixture.pools))
            cache = CachedVGGTContextStore(root / "cache")
            for path in fixture.paths:
                for pair in itertools.permutations(fixture.pools[Path(path).name], 2):
                    cache.save_joint(path, pair, {"cache_kind": "joint", "view_indices": list(pair),
                        "pe_frame": "camera", "geometry_source": "vggt_depth",
                        "patch_tokens": torch.randn(4, 64), "camera_token": torch.randn(2, 64),
                        "patch_centers": torch.randn(4, 3), "patch_keep": torch.ones(4, dtype=torch.bool)})
            repo = Path(__file__).resolve().parents[1]
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
            def run(command):
                result = subprocess.run([sys.executable, *command], cwd=repo, env=env,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            common = ["--dataset", "internscenes", "--data_dir", str(root),
                "--room_view_pools_json", str(pools), "--views_per_sample", "2",
                "--view_sample_mode", "random", "--align_mode", "c_meanrms", "--cache_pooled_surfaces",
                "--no_experiment_manifest", "--pretrained_load", "none", "--device", "cpu", "--seed", "0",
                "--num_latents", "4", "--num_registers", "4", "--embed_dim", "8", "--width", "64",
                "--heads", "4", "--num_ge_layers", "1", "--num_decoder_layers", "1",
                "--num_points_per_anchor", "8", "--pc_size", "4", "--pc_sharpedge_size", "4",
                "--query_sample_mode", "weighted_fps", "--fps_query_cache_items", "24",
                "--chamfer_backend", "chunked", "--chamfer_chunk_size", "8", "--sinkhorn_iters", "2",
                "--no-representation_noising", "--batch_size", "2", "--num_steps", "2", "--log_interval", "1",
                "--ckpt_interval", "0", "--vis_interval", "0", "--no-save_optimizer"]
            recon, flow = root / "recon", root / "flow"
            run(["train_pc_unite.py", *common, "--recon_only", "--output_dir", str(recon)])
            eval_common = ["--data_dir", str(root), "--room_view_pools_json", str(pools), "--device", "cpu",
                           "--sample_steps", "3", "--gen_seeds", "0", "1", "--vggt_cache_root", str(cache.cache_root)]
            run(["evaluate_pc_overnight.py", *eval_common, "--recon_only", "--ckpt", str(recon / "ckpt_final.pt"),
                 "--output_dir", str(root / "recon_eval")])
            run(["train_pc_unite.py", *common, "--freeze_recon", "--resume_ckpt", str(recon / "ckpt_final.pt"),
                 "--reset_optimizer_on_resume", "--additional_steps", "2", "--lr_schedule", "constant",
                 "--vggt_joint_cache", "--vggt_cache_root", str(cache.cache_root), "--flow_steps_per_recon", "2",
                 "--output_dir", str(flow)])
            run(["evaluate_pc_overnight.py", *eval_common, "--ckpt", str(flow / "ckpt_final.pt"),
                 "--reference_metrics", str(root / "recon_eval/metrics.json"), "--output_dir", str(root / "eval")])
            metrics = json.loads((root / "eval/metrics.json").read_text())
            self.assertGreater(metrics["frozen_reconstruction_tensors_verified"], 0)
            self.assertEqual(len(metrics["rooms"]), 4)
            self.assertTrue(all(len(r["generation"]) == 2 for r in metrics["rooms"]))
            self.assertEqual(len(list((root / "eval").rglob("*.ply"))), 10)
            self.assertEqual(torch.load(flow / "ckpt_final.pt", weights_only=False)["step"], 4)
        finally:
            fixture.tearDown()

    def test_pbs_startup_creates_persistent_output_for_all_variants(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            text = (repo / "train_overnight.pbs").read_text().replace(
                "PROJECT_ROOT=/projects_vol/gp_chuanxia.zheng/nathan/HY3DGS", "PROJECT_ROOT=" + str(project))
            text = text.replace('eval "$(/usr/local/anaconda2025/bin/conda shell.bash hook)"', ":")
            stubs = "module() { :; }\nconda() { :; }\ngit() { echo revision; }\nnvidia-smi() { :; }\npython() { :; }\n"
            launcher = project / "test.pbs"
            launcher.write_text(stubs + text)
            (project / "run_internscenes_overnight.sh").write_text('test -s "$OUT/git_commit.txt"\ntouch "$OUT/reached"\n')
            for name in ("control", "local_ot", "weak_anchor", "density_k8"):
                env = dict(os.environ, PBS_JOBID="123.gaas", EXPERIMENT=name)
                result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=10)
                log = project / "runs/logs" / f"overnight_{name}_123.gaas.log"
                self.assertEqual(result.returncode, 0, log.read_text() if log.exists() else result.stderr)
                self.assertTrue((project / "runs/overnight" / f"{name}_123.gaas/reached").exists())

    def test_pooled_cache_preserves_random_sampling_and_normalized_surfaces(self):
        fixture = room_tests.RoomViewPoolTests()
        fixture.setUp()
        try:
            source = fixture.dataset()
            rng = random.getstate()
            pooled = PooledSurfaceDataset(source)
            self.assertEqual(random.getstate(), rng)
            self.assertEqual(len(pooled), 2)
            self.assertEqual(len(pooled.items), 24)
            for index in [0, 1] * 8:
                state = random.getstate()
                uncached = source[index]
                random.setstate(state)
                with patch.object(source, "_load_multi_view_sample", side_effect=AssertionError("disk read")):
                    cached = pooled[index]
                torch.testing.assert_close(cached["surface"], uncached["surface"], rtol=0, atol=0)
                torch.testing.assert_close(cached["view_indices"], uncached["view_indices"])
                self.assertNotIn("rgb_views", cached)
                batch = collate_surface_render([cached])
                self.assertEqual(batch["view_indices"].shape, (1, 2))
                cached["surface"].zero_()
                random.setstate(state)
                self.assertTrue(torch.equal(pooled[index]["surface"], uncached["surface"]))
            with self.assertRaises(ValueError):
                PooledSurfaceDataset(source, keep_context=True)
            class JointCache(room_tests.CacheProbe):
                def load_joint(self, path, views):
                    return {"pair": tuple(views), "patch_tokens": torch.zeros(2, 4)}
            conditioned = PooledSurfaceDataset(fixture.dataset(vggt_cache=JointCache(), use_joint_vggt_cache=True), keep_context=True)
            item = conditioned[0]
            self.assertEqual(tuple(item["view_indices"].tolist()), item["vggt_cache"]["pair"])
            self.assertIn("vggt_cache", collate_surface_render([item]))
        finally:
            fixture.tearDown()

    def test_flow_update_and_reload_preserve_entire_reconstruction(self):
        model = screen.ReconstructionScreenTests().small_model()
        model.eval()
        surface = torch.randn(1, 32, 10)
        torch.manual_seed(42)
        before = model.forward_tokenizer(surface)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "recon.pt"
            torch.save({"model": model.state_dict()}, ckpt)
            _apply_recon_freeze(model)
            model.train()
            model.ensure_tokenizer_ge_eval()
            self.assertFalse(model.tokenizer_ge.training)
            self.assertTrue(all(not p.requires_grad for p in model.tokenizer_ge.parameters()))
            live_before = {k: v.clone() for k, v in model.ge.state_dict().items()}
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            loss = model.forward_denoising(before[0].detach(), torch.randn(1, 3, 64))["flow/flow_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            self.assertTrue(any(not torch.equal(v, model.ge.state_dict()[k]) for k, v in live_before.items()))
            self.assertGreater(verify_frozen_reconstruction(model, ckpt), 0)
            model.eval()
            torch.manual_seed(42)
            after = model.forward_tokenizer(surface)
            for a, b in zip(before, after):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            reloaded = screen.ReconstructionScreenTests().small_model()
            reloaded.freeze_tokenizer_ge(replace=True)
            reloaded.load_state_dict(model.state_dict(), strict=True)
            _apply_recon_freeze(reloaded, preserve_tokenizer_ge=True, preserve_denoiser_pe=True)
            self.assertGreater(verify_frozen_reconstruction(reloaded, ckpt), 0)
            with torch.no_grad():
                reloaded.point_head.weight.add_(1)
            with self.assertRaises(RuntimeError):
                verify_frozen_reconstruction(reloaded, ckpt)

    def test_all_launchers_have_matched_budgets_and_sequential_handoff(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "source.pt"
            checkpoint.touch()
            for name, local, anc, k in (("control", "none", "0.1", "16"), ("local_ot", "ot", "0.1", "16"),
                                        ("weak_anchor", "ot", "0.01", "16"), ("density_k8", "ot", "0.01", "8")):
                out = str(Path(tmp) / name)
                env = dict(os.environ, EXPERIMENT=name, DRY_RUN="1", SOURCE_CKPT=str(checkpoint), OUT=out)
                proc = subprocess.run(["bash", "run_internscenes_overnight.sh"], cwd=repo, env=env,
                                      text=True, capture_output=True, timeout=10)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                recon, eval_recon, flow, eval_flow = [shlex.split(line) for line in proc.stdout.splitlines()]
                for command in (recon, flow):
                    for flag, expected in (("--local_loss", local), ("--lambda_anc", anc), ("--fps_density_k", k),
                                           ("--batch_size", "2"), ("--seed", "0"), ("--lr", "1e-5"), ("--precision", "fp32")):
                        self.assertEqual(command[command.index(flag) + 1], expected)
                    self.assertIn("--cache_pooled_surfaces", command)
                    self.assertNotIn("--max_items", command)
                self.assertEqual(recon[recon.index("--additional_steps") + 1], "8000")
                self.assertEqual(flow[flow.index("--additional_steps") + 1], "12000")
                self.assertEqual(recon[recon.index("--resume_ckpt") + 1], str(checkpoint))
                self.assertEqual(flow[flow.index("--resume_ckpt") + 1], out + "/recon/ckpt_final.pt")
                self.assertIn("--recon_only", recon)
                self.assertIn("--freeze_recon", flow)
                self.assertIn("--reference_metrics", eval_flow)

    def test_metrics_report_outliers_and_coverage_separately(self):
        gt = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
        pred = torch.tensor([[[0., 0., 0.], [2., 0., 0.]]])
        result = point_metrics(pred, None, gt, None)
        self.assertEqual(result["cd"], 1)
        self.assertEqual(result["precision_norm_0.01"], .5)
        self.assertEqual(result["coverage_norm_0.01"], .5)


if __name__ == "__main__":
    unittest.main()
