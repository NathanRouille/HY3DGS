"""Joint cold-start integration, all-pair evaluation and shared gradient checks."""
import itertools
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import test_pc_recon_screen as screen
import test_room_view_pools as room_tests
from hy3dgen.shapegen.pc_joint_eval import evaluate_joint_pairs
from hy3dgen.shapegen.pc_joint_runtime import make_lr_scheduler, shared_ge_gradient_metrics
from hy3dgen.shapegen.pc_recon_runtime import PooledSurfaceDataset
from hy3dgen.shapegen.vggt_context import CachedVGGTContextStore
from prepare_joint_pretrained import prepare


class JointTests(unittest.TestCase):
    def test_warmup_cosine_and_resume_preserve_learning_rates(self):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = make_lr_scheduler(optimizer, schedule="cosine", steps=20, warmup_steps=4, min_ratio=.1)
        rates = []
        for update in range(20):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step(); scheduler.step()
            if update == 9:
                opt_state, sched_state = optimizer.state_dict(), scheduler.state_dict()
        self.assertAlmostEqual(rates[0], 2.5e-5)
        self.assertAlmostEqual(rates[3], 1e-4)
        self.assertTrue(all(a >= b for a, b in zip(rates[4:], rates[5:])))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        resumed = torch.optim.AdamW([parameter], lr=1e-4)
        rs = make_lr_scheduler(resumed, schedule="cosine", steps=20, warmup_steps=4, min_ratio=.1)
        resumed.load_state_dict(opt_state); rs.load_state_dict(sched_state)
        # Adam's state_dict holds references, but the LR values in param_groups are copied.
        self.assertAlmostEqual(resumed.param_groups[0]["lr"], rates[10])
        for target in rates[10:]:
            self.assertAlmostEqual(resumed.param_groups[0]["lr"], target)
            resumed.step(); rs.step()

    def test_gradient_cosine_detects_conflict_without_consuming_graph_or_changing_grads(self):
        model = torch.nn.Module(); model.ge = torch.nn.Linear(2, 1, bias=False)
        loss = model.ge(torch.tensor([[1., 2.]])).sum()
        metrics = shared_ge_gradient_metrics(model, 2 * loss, -3 * loss)
        self.assertAlmostEqual(metrics["shared_ge/gradient_cosine"], -1, places=6)
        self.assertAlmostEqual(metrics["shared_ge/flow_to_recon_ratio"], 1.5, places=6)
        self.assertIsNone(model.ge.weight.grad)
        (2 * loss - 3 * loss).backward()
        torch.testing.assert_close(model.ge.weight.grad, torch.tensor([[-1., -2.]]))

    def test_preparation_persists_only_original_encoder_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "source.pt", Path(tmp) / "persistent/encoder.pt"
            tensors = {"encoder.input_proj.weight": torch.randn(2, 3),
                       "encoder.cross_attn.weight": torch.randn(3, 3), "decoder.weight": torch.randn(4, 4)}
            torch.save(tensors, source)
            prepare(dest, source)
            loaded = torch.load(dest, weights_only=True)
            self.assertNotIn("decoder.weight", loaded)
            for key in loaded:
                torch.testing.assert_close(loaded[key], tensors[key], rtol=0, atol=0)
            source.unlink()
            prepare(dest)  # Persistent weights work after the source disappears.

    def test_launchers_use_full_joint_cold_start_and_only_coverage_differs(self):
        repo = Path(__file__).resolve().parents[1]
        commands = []
        for name in ("coverage4", "coverage2"):
            result = subprocess.run(["bash", "run_internscenes_joint_from_scratch.sh"], cwd=repo,
                env=dict(os.environ, DRY_RUN="1", EXPERIMENT=name, OUT="/tmp/matched_joint"),
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(result.stdout)
            for forbidden in ("--recon_only", "--freeze_recon", "--resume_ckpt"):
                self.assertNotIn(forbidden, command)
            for flag, value in (("--view_sample_mode", "random"), ("--fps_density_k", "8"),
                ("--flow_steps_per_recon", "4"), ("--lambda_flow", "1"), ("--lambda_recon", "1"),
                ("--pretrained_load", "cross_attn"), ("--num_steps", "40000"),
                ("--local_loss", "ot"), ("--lambda_local", "0.1"), ("--precision", "fp32")):
                self.assertEqual(command[command.index(flag) + 1], value)
            commands.append(command)
        self.assertEqual([(a, b) for a, b in zip(*commands) if a != b], [("4", "2")])

    def test_real_joint_training_updates_both_paths_and_evaluates_all_pairs(self):
        fixture = room_tests.RoomViewPoolTests(); fixture.setUp()
        try:
            root = fixture.root; pools = root / "views.json"
            pools.write_text(json.dumps(fixture.pools))
            cache = CachedVGGTContextStore(root / "cache")
            for path in fixture.paths:
                for pair in itertools.permutations(fixture.pools[Path(path).name], 2):
                    cache.save_joint(path, pair, {"cache_kind": "joint", "view_indices": list(pair),
                        "pe_frame": "camera", "geometry_source": "vggt_depth",
                        "patch_tokens": torch.randn(4, 64), "camera_token": torch.randn(2, 64),
                        "patch_centers": torch.randn(4, 3), "patch_keep": torch.ones(4, dtype=torch.bool)})
            out = root / "joint"; repo = Path(__file__).resolve().parents[1]
            command = [sys.executable, "train_pc_unite.py", "--dataset", "internscenes", "--data_dir", str(root),
                "--room_view_pools_json", str(pools), "--views_per_sample", "2", "--view_sample_mode", "random",
                "--align_mode", "c_meanrms", "--cache_pooled_surfaces", "--vggt_cache_root", str(cache.cache_root),
                "--vggt_joint_cache", "--no_experiment_manifest", "--pretrained_load", "none", "--device", "cpu",
                "--seed", "0", "--num_latents", "4", "--num_registers", "4", "--embed_dim", "8",
                "--width", "64", "--heads", "4", "--num_ge_layers", "1", "--num_decoder_layers", "1",
                "--num_points_per_anchor", "8", "--pc_size", "4", "--pc_sharpedge_size", "4",
                "--query_sample_mode", "weighted_fps", "--fps_density_k", "8", "--fps_query_cache_items", "24",
                "--chamfer_backend", "chunked", "--chamfer_chunk_size", "8", "--sinkhorn_iters", "2",
                "--register_noise_mode", "fixed", "--modulation_recon_timestep_max", "0", "--no-representation_noising",
                "--flow_steps_per_recon", "4", "--batch_size", "2", "--num_steps", "3", "--log_interval", "1",
                "--log_grad_norms", "--grad_log_interval", "1", "--lr_warmup_steps", "1", "--lr_min_ratio", ".1",
                "--local_loss", "ot", "--lambda_local", ".1", "--local_patch_count", "4",
                "--ckpt_interval", "2", "--no-save_optimizer", "--save_latest_optimizer", "--vis_interval", "0",
                "--joint_eval_interval", "2", "--joint_eval_sample_steps", "3", "--output_dir", str(out)]
            result = subprocess.run(command, cwd=repo, env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"),
                capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            logs = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(any(row["flow/loss"] > 0 for row in logs), logs)
            self.assertTrue(all(np.isfinite(row["train/recon"]) for row in logs), logs)
            self.assertTrue(all(np.isfinite(row["shared_ge/gradient_cosine"]) for row in logs))
            intermediate = json.loads((out / "eval/step_0000002/metrics.json").read_text())
            final = json.loads((out / "eval/final/metrics.json").read_text())
            self.assertEqual(intermediate["pair_count"], 24)
            self.assertEqual(final["pair_count"], 24)
            self.assertEqual(final["gen_seeds"], [0, 1, 2])
            self.assertEqual(len(list((out / "eval/final").rglob("*.ply"))), 14)
            snapshot = torch.load(out / "ckpt_0000002.pt", weights_only=False)
            latest = torch.load(out / "ckpt_last_resume.pt", weights_only=False)
            final_ckpt = torch.load(out / "ckpt_final.pt", weights_only=False)
            self.assertNotIn("optimizer", snapshot); self.assertIn("optimizer", latest)
            self.assertIn("optimizer", final_ckpt)
            self.assertFalse(any(k.startswith("tokenizer_ge.") for k in final_ckpt["model"]))
            for prefix in ("ge.", "decoder.", "encoder."):
                self.assertTrue(any(not torch.equal(v, final_ckpt["model"][k])
                    for k, v in snapshot["model"].items() if k.startswith(prefix)))
        finally:
            fixture.tearDown()

    def test_evaluation_restores_rng_and_training_modes(self):
        from train_pc_unite import _build_weak_context
        from hy3dgen.shapegen.vggt_context import VGGTContextBuilder
        model = screen.ReconstructionScreenTests().small_model().train()
        model.register_noise_mode = "fixed"; model.modulation_recon_timestep_max = 0
        builder = VGGTContextBuilder(width=64).train(); builder.attach_fourier(model.fourier_embedder)
        surface = torch.randn(1, 32, 10)[0]
        item = {"surface": surface, "mesh_path": "/tmp/room", "view_idx": torch.tensor(1),
            "view_indices": torch.tensor([1, 2]),
            "vggt_cache": {"cache_kind": "joint", "view_indices": [1, 2], "pe_frame": "camera",
                "geometry_source": "vggt_depth", "patch_tokens": torch.randn(4, 64),
                "camera_token": torch.randn(2, 64), "patch_centers": torch.randn(4, 3),
                "patch_keep": torch.ones(4, dtype=torch.bool)}}
        torch_rng, py_rng, np_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        with tempfile.TemporaryDirectory() as tmp:
            evaluate_joint_pairs(model, builder, {("/tmp/room", (1, 2)): item}, _build_weak_context,
                output_dir=tmp, step=0, sample_steps=3)
        self.assertTrue(model.training and builder.training)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(py_rng, random.getstate())
        np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])


if __name__ == "__main__":
    unittest.main()
