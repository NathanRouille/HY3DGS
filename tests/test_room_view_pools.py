"""Protect per-room sampling, cache coverage, and deterministic evaluation."""

import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from hy3dgen.shapegen.pc_render_dataset import (
    SurfaceRenderDataset, build_internscenes_render_dataset,
)
from hy3dgen.shapegen.internscenes_gt import InternScenesGTSource
from hy3dgen.shapegen.room_view_pools import normalize_room_view_pools


class CacheProbe:
    def __init__(self, missing=None):
        self.checked = []
        self.missing = missing

    def has_joint(self, path, views):
        key = (Path(path).name, tuple(views))
        self.checked.append(key)
        return key != self.missing


class RoomViewPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pools = {"gen__bathroom__101": [9, 1, 7, 3], "gen__bathroom__202": [4, 12, 16, 32]}
        self.paths = []
        for room_id, views in self.pools.items():
            room = self.root / "rooms" / room_id
            (room / "rgb").mkdir(parents=True)
            (room / "depth_float32").mkdir()
            self.paths.append(str(room))
            np.savez(room / "surface.npz", xyz=np.column_stack([np.linspace(-.1,.1,8),np.ones(8),np.zeros(8)]).astype(np.float32), normals=np.tile([0.,1.,0.],(8,1)), sharp=np.array([0]*4+[1]*4), rgb=np.ones((8,3),np.float32)*.5)
            poses = []
            for view in views:
                Image.new("RGB",(4,4),(128,128,128)).save(room / "rgb" / f"view_{view:03d}.png")
                np.savez(room / "depth_float32" / f"frame_{view:04d}_depth.npz", depth=np.ones((4,4),np.float32), mask=np.ones((4,4),bool))
                poses.append({"pose_id":view,"cam_matrix_opencv":np.eye(4)[:3].tolist()})
            (room / "camera_poses.json").write_text(json.dumps({"room":{"intrinsics":{"fx":4,"fy":4,"cx":2,"cy":2},"poses":poses}}))

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, **kwargs):
        return SurfaceRenderDataset(str(self.root),self.paths,view_indices=[0,1],room_view_pools=self.pools,views_per_sample=2,gt_source=InternScenesGTSource(),internscenes_pack=True,gobjaverse_normalization=False,pc_size=4,pc_sharpedge_size=4,seed=0,align_mode="c_meanrms",strict_load=True,**kwargs)

    def test_random_training_uses_only_each_rooms_ordered_pairs(self):
        dataset = self.dataset()
        rng = random.Random(42)
        with patch("hy3dgen.shapegen.pc_render_dataset.random.sample",rng.sample):
            for path in self.paths:
                pairs = {tuple(dataset._choose_views(path,i)) for i in range(512)}
                self.assertEqual(pairs,set(itertools.permutations(self.pools[Path(path).name],2)))

    def test_evaluation_preserves_pool_order_and_loads_each_room(self):
        dataset = self.dataset(view_sample_mode="first")
        for i,path in enumerate(self.paths):
            self.assertEqual(dataset[i]["view_indices"].tolist(),self.pools[Path(path).name][:2])
            self.assertEqual(dataset[i]["mesh_path"],path)

    def test_cache_checks_all_twelve_pairs_per_room_without_crossing_pools(self):
        cache = CacheProbe()
        self.dataset(vggt_cache=cache,use_joint_vggt_cache=True)
        expected = {(sid,pair) for sid,pool in self.pools.items() for pair in itertools.permutations(pool,2)}
        self.assertEqual(set(cache.checked),expected)
        self.assertEqual(len(cache.checked),24)

    def test_missing_cache_or_view_fails_instead_of_dropping_a_room(self):
        cache = CacheProbe(("gen__bathroom__202",(4,12)))
        with self.assertRaises(FileNotFoundError):
            self.dataset(vggt_cache=cache,use_joint_vggt_cache=True)
        (Path(self.paths[1])/"rgb/view_012.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset()

    def test_manifest_selects_rooms_without_a_split_file(self):
        dataset = build_internscenes_render_dataset(str(self.root),room_view_pools=self.pools,views_per_sample=2,pc_size=4,pc_sharpedge_size=4,view_sample_mode="first")
        self.assertEqual(dataset.mesh_paths,self.paths)

    def test_fixed_two_view_pool_preserves_the_cached_reference_camera(self):
        self.pools = {sid:pool[:2] for sid,pool in self.pools.items()}
        cache = CacheProbe()
        dataset = self.dataset(vggt_cache=cache,use_joint_vggt_cache=True)
        self.assertEqual(len(cache.checked),2)
        for path in self.paths:
            for i in range(10):
                self.assertEqual(dataset._choose_views(path,i),self.pools[Path(path).name])

    def test_unknown_room_is_rejected_before_training(self):
        with self.assertRaises(FileNotFoundError):
            build_internscenes_render_dataset(str(self.root),room_view_pools={"missing":[1,2]},views_per_sample=2)

    def test_validation_rejects_duplicate_fractional_or_invalid_view_ids(self):
        self.assertEqual(normalize_room_view_pools({"101":["009",1]}),{"gen__bathroom__101":[9,1]})
        for values in ([1,1],[1.5,2],[True,2],[-1,2],[]):
            with self.assertRaises(ValueError):
                normalize_room_view_pools({"101":values})


if __name__ == "__main__":
    unittest.main()
