"""Numerical and integration checks for the accelerated reconstruction screen."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from hy3dgen.shapegen.pc_losses import (PointCloudAELoss, nearest_neighbor_distances,
                                      pairwise_dist2, sinkhorn_matching_loss, _sinkhorn_transport_cost)
from hy3dgen.shapegen.pc_local_loss import LocalPatchLoss, patch_distance
from hy3dgen.shapegen.pc_recon_runtime import FixedSurfaceDataset, autocast_context, flow_training_output
from hy3dgen.shapegen.models.autoencoders.shape_pc_unite import ShapePCUnite
from hy3dgen.shapegen.models.autoencoders import attention_blocks

torch.set_num_threads(2)


class ReconstructionScreenTests(unittest.TestCase):
    def test_detached_sinkhorn_plan_optimization_preserves_historical_objective(self):
        torch.manual_seed(4)
        p = torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)
        q = torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)
        costs = []
        for a, b in ((p, q), (p, p), (q, q)):
            c = pairwise_dist2(a, b)
            _, plan = _sinkhorn_transport_cost(c, epsilon=.1, n_iters=5)
            costs.append((plan.detach() * c).sum((1,2)) * a.shape[1])
        old = (costs[0] - .5 * costs[1] - .5 * costs[2].detach()).mean()
        new = sinkhorn_matching_loss(p, q, epsilon=.1, n_iters=5)
        torch.testing.assert_close(old, new)
        for a, b in zip(torch.autograd.grad(old, (p, q), retain_graph=True),
                        torch.autograd.grad(new, (p, q))):
            torch.testing.assert_close(a, b)

    def test_chunked_loss_indices_and_both_coordinate_gradients(self):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(23)
            p = torch.randn(2, 39, 3, dtype=dtype, requires_grad=True)
            q = torch.randn(2, 31, 3, dtype=dtype, requires_grad=True)
            dense = nearest_neighbor_distances(p, q)
            chunked = nearest_neighbor_distances(p, q, backend="chunked", chunk_size=7)
            for a, b in zip(dense[2:], chunked[2:]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            losses = [d[0].mean() + 2 * d[1].mean() for d in (dense, chunked)]
            torch.testing.assert_close(*losses, rtol=2e-6, atol=2e-6)
            grads = [torch.autograd.grad(loss, (p, q), retain_graph=True) for loss in losses]
            for a, b in zip(*grads):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)

    def test_chunk_boundaries_preserve_duplicate_first_index_ties(self):
        p = torch.tensor([[[0., 0, 0], [2, 0, 0], [0, 0, 0], [2, 0, 0]]], requires_grad=True)
        q = torch.tensor([[[0., 0, 0], [2, 0, 0], [0, 0, 0]]])
        d = nearest_neighbor_distances(p, q, backend="chunked", chunk_size=2)
        self.assertEqual(d[2].tolist(), [[0, 1, 0, 1]])
        self.assertEqual(d[3].tolist(), [[0, 1, 0]])
        (d[0].mean() + d[1].mean()).backward()
        torch.testing.assert_close(p.grad, torch.zeros_like(p))

    def test_full_criterion_preserves_rgb_anchor_delta_losses_and_gradients(self):
        torch.manual_seed(17)
        p = torch.randn(2, 32, 3, dtype=torch.float64, requires_grad=True)
        rgb = torch.rand(2, 32, 3, dtype=torch.float64, requires_grad=True)
        q, color = torch.randn_like(p), torch.rand_like(rgb)
        anchors = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
        fps = torch.randn_like(anchors)
        settings = dict(lambda_rgb=10, lambda_anc=.1, lambda_anc_cd=10,
                        lambda_delta=.001, cd_pred2gt=1, cd_gt2pred=2, sinkhorn_iters=3)
        dense = PointCloudAELoss(**settings)
        chunked = PointCloudAELoss(**settings, chamfer_backend="chunked", chamfer_chunk_size=9)
        losses = [criterion(p, rgb, q, color, anchors, fps)[0] for criterion in (dense, chunked)]
        torch.testing.assert_close(*losses)
        grads = [torch.autograd.grad(loss, (p, rgb, anchors), retain_graph=True) for loss in losses]
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b)

    def test_local_ot_matches_bruteforce_optimum_and_has_finite_gradients(self):
        p = torch.tensor([[[0., 0, 0], [0, 0, 0], [3, 0, 0], [3, 0, 0]]], requires_grad=True)
        q = torch.tensor([[[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]])
        ot = patch_distance(p, q, "ot")
        brute = min(float((p[0] - q[0, list(order)]).square().sum(-1).mean())
                    for order in itertools.permutations(range(4)))
        self.assertAlmostEqual(float(ot), 2 * brute)
        self.assertGreater(float(ot), float(patch_distance(p, q, "cd")))
        ot.backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_local_targets_deduplicate_and_skip_unsupported_patches(self):
        fps = torch.tensor([[0., 0, 0], [2., 0, 0], [4., 0, 0], [6., 0, 0]])
        offsets = torch.tensor([[0., 0, 0], [.01, 0, 0], [0, .01, 0], [.01, .01, 0]])
        gt = (fps[:, None] + offsets).reshape(-1, 3)
        pred = gt[None].clone().requires_grad_(True)
        for mode in ("cd", "ot"):
            objective = LocalPatchLoss(mode, patch_count=4)
            loss, stats = objective(pred, fps[None], torch.cat([gt, gt])[None], fps[None])
            self.assertEqual(float(stats["local/used_patches"]), 4)
            self.assertAlmostEqual(float(loss), 0)
            target = objective.cache[0]["target"]
            for patch_ in target:
                self.assertEqual(len(torch.unique(patch_, dim=0)), 4)
            sparse = fps[None].clone()
            zero, info = objective(pred, fps[None], sparse, fps[None])
            self.assertEqual(float(info["local/used_patches"]), 0)
            self.assertEqual(float(zero), 0)

    def small_model(self):
        torch.manual_seed(0)
        model = ShapePCUnite(num_latents=4, num_registers=4, embed_dim=8, width=64,
                             heads=4, num_ge_layers=1, num_decoder_layers=1,
                             pc_size=16, pc_sharpedge_size=16, point_feats=7,
                             num_points_per_anchor=8, query_sample_mode="weighted_fps")
        model.representation_noising = False
        model.fp32_output_heads = True
        return model

    def test_query_cache_hits_and_invalidates_on_points_features_and_settings(self):
        model = self.small_model()
        encoder = model.encoder
        encoder.query_cache_max_items = 3
        surface = torch.randn(1, 32, 10)
        pc, feats = model.encoder_inputs(surface)
        with patch.object(attention_blocks, "knn_density_from_reference",
                          wraps=attention_blocks.knn_density_from_reference) as density:
            first = encoder.sample_points_and_latents(pc, feats)
            count = density.call_count
            second = encoder.sample_points_and_latents(pc, feats)
            self.assertEqual(density.call_count, count)
            torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
            changed = feats.clone(); changed[:, :, 3] += .2
            encoder.sample_points_and_latents(pc, changed)
            self.assertGreater(density.call_count, count)
            count = density.call_count
            shifted = pc.clone(); shifted[:, 0, 0] += .02
            encoder.sample_points_and_latents(shifted, feats)
            self.assertGreater(density.call_count, count)
            count = density.call_count
            encoder.fps_density_k += 1
            encoder.sample_points_and_latents(pc, feats)
            self.assertGreater(density.call_count, count)

    def test_small_actual_model_bf16_keeps_output_heads_fp32_and_backpropagates(self):
        model = self.small_model()
        surface = torch.randn(2, 32, 10)
        with autocast_context(torch.device("cpu"), "bf16"):
            _, _, xyz, rgb, anchors = model.forward_tokenizer(surface)
        self.assertEqual(xyz.dtype, torch.float32)
        self.assertEqual(rgb.dtype, torch.float32)
        self.assertEqual(anchors.dtype, torch.float32)
        (xyz.square().mean() + rgb.mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_reconstruction_path_never_calls_flow(self):
        class Bomb:
            def forward_denoising(self, *args, **kwargs):
                raise AssertionError("Flow was called")
        output = flow_training_output(Bomb(), torch.ones(2, 4, 8), None, recon_only=True)
        self.assertEqual(float(output["flow/flow_loss"]), 0)
        with self.assertRaisesRegex(AssertionError, "Flow was called"):
            flow_training_output(Bomb(), torch.ones(2, 4, 8), None)

    def test_fixed_surface_cache_preserves_data_and_drops_renders(self):
        class Source:
            seed = 0; view_sample_mode = "first"; mesh_paths = ["room"]; samples = [("room", 0)]
            def __len__(self): return 1
            def __getitem__(self, i):
                return {"surface": torch.ones(32, 10), "mesh_path": "room",
                        "view_idx": torch.tensor(0), "rgb": torch.ones(3, 4, 4)}
        cached = FixedSurfaceDataset(Source())
        sample = cached[0]
        self.assertNotIn("rgb", sample)
        sample["surface"].zero_()
        self.assertEqual(float(cached[0]["surface"].sum()), 320)

    def test_cpu_training_resume_and_reconstruction_evaluator_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); room = root / "pack/rooms/gen__bathroom__101"
            (room / "rgb").mkdir(parents=True); (room / "depth_float32").mkdir()
            rng = np.random.default_rng(5)
            np.savez(room / "surface.npz", xyz=rng.uniform(-.2, .2, (32, 3)).astype(np.float32),
                     normals=np.tile([0., 1., 0.], (32, 1)), sharp=np.array([0]*16+[1]*16),
                     rgb=rng.uniform(0, 1, (32, 3)).astype(np.float32))
            poses = []
            for view in (0, 1):
                Image.new("RGB", (4, 4), (128, 128, 128)).save(room / f"rgb/view_{view:03d}.png")
                np.savez(room / f"depth_float32/frame_{view:04d}_depth.npz", depth=np.ones((4,4), np.float32), mask=np.ones((4,4), bool))
                poses.append({"pose_id": view, "cam_matrix_opencv": np.eye(4)[:3].tolist()})
            (room / "camera_poses.json").write_text(json.dumps({"room": {"intrinsics": {"fx":4,"fy":4,"cx":2,"cy":2}, "poses": poses}}))
            views = root / "views.json"; views.write_text(json.dumps({"101": [0,1]}))
            repo = Path(__file__).resolve().parents[1]
            out = root / "run"
            command = [sys.executable, str(repo / "train_pc_unite.py"),
                       "--dataset", "internscenes", "--data_dir", str(root / "pack"),
                       "--room_view_pools_json", str(views), "--views_per_sample", "2",
                       "--view_sample_mode", "first", "--align_mode", "c_meanrms",
                       "--cache_fixed_surfaces", "--recon_only", "--no_experiment_manifest",
                       "--pretrained_load", "none", "--device", "cpu", "--seed", "0",
                       "--num_latents", "4", "--num_registers", "4", "--embed_dim", "8",
                       "--width", "64", "--heads", "4", "--num_ge_layers", "1",
                       "--num_decoder_layers", "1", "--num_points_per_anchor", "8",
                       "--pc_size", "16", "--pc_sharpedge_size", "16",
                       "--query_sample_mode", "weighted_fps", "--fps_query_cache_items", "1",
                       "--chamfer_backend", "chunked", "--chamfer_chunk_size", "8",
                       "--sinkhorn_iters", "2", "--no-representation_noising", "--batch_size", "1",
                       "--num_steps", "2", "--log_interval", "1", "--log_grad_norms",
                       "--grad_log_interval", "1", "--ckpt_interval", "0", "--vis_interval", "0",
                       "--no-save_optimizer", "--output_dir", str(out)]
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all(m["flow/loss"] == 0 for m in metrics))
            self.assertIn("grad_norm/cd_pred2gt", metrics[-1])
            checkpoint = torch.load(out / "ckpt_final.pt", map_location="cpu", weights_only=False)
            self.assertNotIn("optimizer", checkpoint)
            resumed = root / "resumed"
            command[-1] = str(resumed)
            command += ["--resume_ckpt", str(out / "ckpt_final.pt"), "--reset_optimizer_on_resume", "--additional_steps", "1", "--lr_schedule", "constant"]
            result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(torch.load(resumed / "ckpt_final.pt", map_location="cpu", weights_only=False)["step"], 3)
            evaluation = root / "eval"
            result = subprocess.run([sys.executable, str(repo / "evaluate_pc_recon_screen.py"),
                                     "--ckpt", str(resumed / "ckpt_final.pt"), "--data_dir", str(root / "pack"),
                                     "--room_view_pools_json", str(views), "--output_dir", str(evaluation), "--device", "cpu"],
                                    cwd=repo, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(list(evaluation.rglob("*.ply"))), 4)


if __name__ == "__main__":
    unittest.main()
