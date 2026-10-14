"""Owned target coverage, coordinate gradients, invariance and full joint I/O."""
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import test_room_view_pools as room_tests
from hy3dgen.shapegen.pc_assignment import ExactAssignment
from hy3dgen.shapegen.pc_balanced_loss import BalancedPatchLoss
from hy3dgen.shapegen.pc_patches import balanced_partition, patch_descriptors, validate_partition
from hy3dgen.shapegen.pc_losses import PointCloudAELoss
from hy3dgen.shapegen.models.autoencoders.shape_pc_unite import ShapePCUnite
from hy3dgen.shapegen.vggt_context import CachedVGGTContextStore
from evaluate_pc_unite import load_unite_model
from prepare_balanced_patches import choose_surface, mesh_surface


class BalancedTests(unittest.TestCase):
    def test_partition_owns_every_point_and_refinement_reduces_objective(self):
        rng = np.random.default_rng(1)
        # Non-power-of-two patch count is supported as long as N is divisible.
        xyz = rng.normal(size=(96, 3)).astype(np.float32)
        idx, queries, stats = balanced_partition(xyz, rng.random((96, 3)), rng.normal(size=(96, 3)))
        validate_partition(idx, queries, len(xyz))
        self.assertEqual(idx.shape, (6, 16))
        self.assertTrue(all(a >= b - 1e-10 for a, b in zip(stats['energy'], stats['energy'][1:])))
        again = balanced_partition(xyz, rng.random((96, 3)), rng.normal(size=(96, 3)), color_weight=0, normal_weight=0)
        twice = balanced_partition(xyz, np.zeros((96, 3)), np.zeros((96, 3)), color_weight=0, normal_weight=0)
        np.testing.assert_array_equal(again[0], twice[0])
        bad = idx.copy(); bad[0, 0] = bad[1, 0]
        with self.assertRaises(ValueError): validate_partition(bad, queries, len(xyz))

    def test_descriptors_follow_similarity_transform_and_stay_finite_on_tiny_patch(self):
        xyz = torch.randn(1, 32, 3)
        idx = torch.arange(32).reshape(1, 2, 16)
        descriptors = patch_descriptors(xyz, idx)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        changed = patch_descriptors(3 * xyz @ rotation.T + 7, idx)
        torch.testing.assert_close(changed[..., 0], descriptors[..., 0] + np.log(3))
        # Trace-normalized covariance retains its unit trace after rotation.
        torch.testing.assert_close(changed[..., [1, 4, 6]].sum(-1), torch.ones(1, 2))
        self.assertTrue(torch.isfinite(patch_descriptors(torch.zeros_like(xyz), idx)).all())

    def test_assignment_is_bijective_in_presence_of_nn_collision(self):
        costs = torch.tensor([[[0., 1., 5.], [0., 2., 3.], [4., 0., 1.]]])
        assignment = ExactAssignment('scipy')(costs)
        self.assertEqual(assignment.sort(-1).values.tolist(), [[0, 1, 2]])
        self.assertEqual(float(costs.gather(2, assignment[..., None]).sum()), 2.)

    def test_absolute_ot_detects_anchor_shift_and_reaches_both_anchor_and_delta(self):
        gt = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [10., 0., 0.], [11., 0., 0.]]])
        indices = torch.tensor([[[0, 1], [2, 3]]])
        queries = gt[:, [0, 2]]
        anchors = queries.clone().requires_grad_(True)
        offsets = torch.tensor([[[[0., 0., 0.], [1., 0., 0.]], [[0., 0., 0.], [1., 0., 0.]]]], requires_grad=True)
        fn = BalancedPatchLoss(2, 'scipy')
        pred = (anchors[:, :, None] + offsets).reshape(1, 4, 3)
        anc, local, stats = fn(pred, anchors, gt, queries, indices)
        self.assertEqual(float(anc + local), 0.)
        shifted = anchors + torch.tensor([0., 2., 0.])
        pred = (shifted[:, :, None] + offsets).reshape(1, 4, 3)
        anc, local, _ = fn(pred, shifted, gt, queries, indices)
        self.assertAlmostEqual(float(local), 8.)
        (anc + local).backward()
        self.assertGreater(float(anchors.grad.norm()), 0)
        self.assertGreater(float(offsets.grad.norm()), 0)
        self.assertEqual(float(stats['local/valid_fraction']), 1.)
        # A permutation of free predicted registers does not change set loss.
        p = pred.reshape(1, 2, 2, 3).flip(1).reshape(1, 4, 3)
        a2, l2, _ = fn(p, shifted.flip(1), gt, queries, indices)
        torch.testing.assert_close(a2 + l2, anc + local)

    def test_distinct_fallback_does_not_hide_sharp_deficit(self):
        data = np.zeros((12, 10), np.float32)
        data[:8, 0] = np.arange(8)
        data[8:, 0] = 20; data[8:, 6] = 1
        surface, stats = choose_surface(data, 8, 4, np.random.default_rng(1), allow_deficit=True)
        self.assertEqual(len(np.unique(surface[:, :3], axis=0)), 8)
        self.assertEqual(stats['sharp_deficit'], 3)
        with self.assertRaises(ValueError):
            choose_surface(data, 8, 4, np.random.default_rng(1), allow_deficit=False)

    def test_fp32_mesh_sampling_preserves_world_frame_and_intended_quota(self):
        import trimesh
        with tempfile.TemporaryDirectory() as temp:
            mesh = trimesh.creation.box(extents=[5, 2, 1])
            mesh.apply_translation([100, 10, -50])
            path = Path(temp) / 'scene_split.glb'; mesh.export(path)
            surface, stats = mesh_surface(path, 64, 32, 123)
            self.assertEqual(surface.dtype, np.float32)
            self.assertEqual(stats['n_sharp'], 32)
            self.assertEqual(stats['n_uniform'], 32)
            self.assertEqual(len(np.unique(surface[:, :3], axis=0)),64)
            self.assertTrue((surface[:, :3] >= mesh.bounds[0] - 1e-5).all())
            self.assertTrue((surface[:, :3] <= mesh.bounds[1] + 1e-5).all())

    def test_descriptor_branch_preserves_initial_weights_and_embeddings(self):
        settings = dict(num_latents=4,num_registers=4,embed_dim=8,width=64,heads=4,
            num_ge_layers=1,num_decoder_layers=1,pc_size=32,pc_sharpedge_size=32,point_feats=7,
            num_points_per_anchor=16,query_sample_mode='balanced_patches',register_noise_mode='fixed',
            modulation_recon_timestep_max=0)
        torch.manual_seed(3); plain = ShapePCUnite(**settings)
        torch.manual_seed(3); augmented = ShapePCUnite(**settings, patch_descriptor_mode='scale_cov')
        for name, tensor in plain.state_dict().items():
            torch.testing.assert_close(tensor, augmented.state_dict()[name], rtol=0, atol=0)
        surface = torch.randn(1, 64, 10)
        idx = torch.arange(64).reshape(1, 4, 16); queries = idx[:, :, 0]
        kwargs = dict(patch_indices=idx, query_indices=queries)
        torch.testing.assert_close(plain.encode(surface, **kwargs)[0], augmented.encode(surface, **kwargs)[0], rtol=0, atol=0)

    def test_launchers_use_matched_cold_starts_and_identical_gt(self):
        repo = Path(__file__).resolve().parents[1]
        for name in ('clean_fps', 'patch_geometry', 'patch_features', 'patch_full'):
            command = subprocess.run(['bash', 'run_internscenes_balanced_joint.sh'], cwd=repo,
                env=dict(os.environ, DRY_RUN='1', EXPERIMENT=name, OUT='/tmp/balanced_dryrun'),
                text=True, capture_output=True, timeout=10)
            self.assertEqual(command.returncode, 0, command.stderr)
            args = shlex.split(command.stdout)
            for forbidden in ('--resume_ckpt', '--freeze_recon', '--recon_only'):
                self.assertNotIn(forbidden, args)
            for flag, value in (('--num_steps', '40000'), ('--flow_steps_per_recon', '4'),
                                ('--view_sample_mode', 'random'), ('--cd_gt2pred', '2')):
                self.assertEqual(args[args.index(flag) + 1], value)
            if name != 'clean_fps':
                self.assertEqual(args[args.index('--lambda_delta') + 1], '0')
                self.assertEqual(args[args.index('--anchor_loss_mode') + 1], 'matched')

    def test_joint_training_checkpoint_reload_and_all_pair_eval_with_prepared_data(self):
        fixture = room_tests.RoomViewPoolTests(); fixture.setUp()
        try:
            root = fixture.root; prepared = root / 'prepared'; prepared.mkdir()
            views = root / 'views.json'; views.write_text(json.dumps(fixture.pools))
            rng = np.random.default_rng(5)
            for room in fixture.pools:
                dest = prepared / room; dest.mkdir()
                xyz = rng.normal(size=(64, 3)).astype(np.float32) * .05
                rgb, nrm = rng.random((64, 3)).astype(np.float32), rng.normal(size=(64, 3)).astype(np.float32)
                np.savez(dest / 'surface.npz', xyz=xyz, normals=nrm, rgb=rgb, sharp=np.array([0]*32+[1]*32))
                idx, q, _ = balanced_partition(xyz, rgb, nrm)
                np.savez(dest / 'patches_features.npz', indices=idx, queries=q)
            cache = CachedVGGTContextStore(root / 'cache')
            for path in fixture.paths:
                for pair in itertools.permutations(fixture.pools[Path(path).name], 2):
                    cache.save_joint(path, pair, dict(cache_kind='joint', view_indices=list(pair),
                        pe_frame='camera', geometry_source='vggt_depth', patch_tokens=torch.randn(4,64),
                        camera_token=torch.randn(2,64), patch_centers=torch.randn(4,3), patch_keep=torch.ones(4,dtype=torch.bool)))
            repo = Path(__file__).resolve().parents[1]; out = root / 'joint'
            args = [sys.executable,'train_pc_unite.py','--dataset','internscenes','--data_dir',str(root),
                '--room_view_pools_json',str(views),'--views_per_sample','2','--view_sample_mode','random',
                '--align_mode','c_meanrms','--cache_pooled_surfaces','--vggt_cache_root',str(cache.cache_root),
                '--vggt_joint_cache','--no_experiment_manifest','--pretrained_load','none','--device','cpu',
                '--prepared_surface_root',str(prepared),'--query_sample_mode','balanced_patches',
                '--patch_descriptor_mode','scale_cov','--anchor_loss_mode','matched','--assignment_backend','scipy',
                '--lambda_anc','0','--lambda_anc_cd','0','--lambda_delta','0','--local_loss','ot','--local_patch_count','4',
                '--sinkhorn_iters','2','--seed','0','--num_latents','4','--num_registers','4','--embed_dim','8',
                '--width','64','--heads','4','--num_ge_layers','1','--num_decoder_layers','1','--num_points_per_anchor','16',
                '--pc_size','32','--pc_sharpedge_size','32','--register_noise_mode','fixed',
                '--modulation_recon_timestep_max','0','--no-representation_noising','--flow_steps_per_recon','4',
                '--batch_size','2','--num_steps','4','--log_interval','1','--log_grad_norms','--grad_log_interval','1',
                '--lr_warmup_steps','1','--ckpt_interval','2','--vis_interval','0','--joint_eval_interval','2',
                '--joint_eval_sample_steps','3','--output_dir',str(out)]
            result = subprocess.run(args,cwd=repo,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout[-2000:]+result.stderr[-4000:])
            records = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),4)
            self.assertTrue(all(row['local/valid_fraction'] == 1 for row in records))
            saved = torch.load(out/'ckpt_final.pt',weights_only=False)
            self.assertGreater(saved['args']['lambda_matched_anchor'],0)
            self.assertGreater(float(saved['model']['encoder.patch_proj.weight'].norm()),0)
            model, builder, config = load_unite_model(out/'ckpt_final.pt',torch.device('cpu'))
            self.assertEqual(model.patch_descriptor_mode,'scale_cov')
            final = json.loads((out/'eval/final/metrics.json').read_text())
            self.assertEqual(final['pair_count'],24)
            self.assertEqual(final['gen_seeds'],[0,1,2])
            self.assertTrue((out/'eval/final'/next(iter(fixture.pools))/'gt.ply').exists())
        finally:
            fixture.tearDown()


if __name__ == '__main__': unittest.main()
