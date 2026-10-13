"""Sanity launch comparisons, deterministic tokenizer and hole diagnostics."""
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

import test_pc_recon_screen as screen
import test_room_view_pools as room_tests
from hy3dgen.shapegen.pc_quality_metrics import reconstruction_metrics


class ReconSanityTests(unittest.TestCase):
    def test_metrics_detect_missing_regions_and_duplicate_clusters(self):
        cluster=np.array([[0.,0,0],[.001,0,0],[0,.001,0],[.001,.001,0]])
        gt=np.concatenate([cluster,cluster+[1,0,0]])
        exact,_,_=reconstruction_metrics(gt,gt,thresholds=(.01,),cell_size=.1)
        self.assertEqual(exact['fscore_norm_0.01'],1)
        self.assertEqual(exact['gt_cell_worst_decile_recall_norm_0.01'],1)
        pred=np.repeat(cluster,2,axis=0)
        bad,dp,dq=reconstruction_metrics(pred,gt,thresholds=(.01,),cell_size=.1)
        self.assertEqual(bad['precision_norm_0.01'],1)
        self.assertEqual(bad['coverage_norm_0.01'],.5)
        self.assertAlmostEqual(bad['fscore_norm_0.01'],2/3)
        self.assertEqual(bad['gt_cell_coverage_norm_0.01'],.5)
        self.assertEqual(bad['gt_cell_worst_decile_recall_norm_0.01'],0)
        self.assertEqual(bad['pred_near_duplicate_fraction'],1)
        self.assertGreater(bad['gt_distance_p99'],.9)
        self.assertEqual(len(dp),8);self.assertEqual(len(dq),8)

    def test_metrics_reject_nonfinite_clouds(self):
        with self.assertRaises(ValueError):
            reconstruction_metrics(np.array([[np.nan,0,0]]),np.zeros((1,3)))

    def test_tokenizer_is_independent_of_rng_with_fixed_noise_and_zero_time(self):
        model=screen.ReconstructionScreenTests().small_model().eval()
        model.register_noise_mode='fixed'
        model.modulation_recon_timestep_max=0
        surface=torch.randn(1,32,10)
        with torch.no_grad():
            torch.manual_seed(1);a=model.forward_tokenizer(surface,representation_phase=False)
            torch.manual_seed(99);b=model.forward_tokenizer(surface,representation_phase=False)
        for first,second in zip(a,b):
            torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_launch_variants_change_only_the_intended_settings(self):
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.pt';source.touch()
            commands={}
            for name in ('fixed_global','fixed_ot','random_global','fixed_coverage'):
                env=dict(os.environ,DRY_RUN='1',EXPERIMENT=name,SOURCE_CKPT=str(source),OUT=str(Path(tmp)/name))
                result=subprocess.run(['bash','run_internscenes_recon_sanity.sh'],cwd=repo,env=env,
                                      capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr)
                native,initial,train,final=[shlex.split(line) for line in result.stdout.splitlines()]
                self.assertNotIn('--register_noise_mode',native)
                self.assertEqual(initial[initial.index('--register_noise_mode')+1],'fixed')
                self.assertIn('--recon_only',train);self.assertNotIn('--freeze_recon',train)
                self.assertNotIn('--max_items',train)
                for flag,value in (('--resume_ckpt',str(source)),('--register_noise_mode','fixed'),
                    ('--modulation_recon_timestep_max','0'),('--fps_density_k','16'),('--lambda_anc','0.01'),
                    ('--lambda_delta','0.001'),('--additional_steps','8000'),('--precision','fp32')):
                    self.assertEqual(train[train.index(flag)+1],value)
                self.assertEqual(final[final.index('--pair_mode')+1],'paired')
                # Ignore the necessarily distinct output directory.
                train=train[:-2]
                commands[name]=train
            def difference(left,right):
                return [(a,b) for a,b in zip(commands[left],commands[right]) if a!=b]
            self.assertEqual(difference('fixed_global','fixed_ot'),[('none','ot')])
            self.assertEqual(difference('fixed_global','fixed_coverage'),[('2','4')])
            self.assertEqual(difference('fixed_global','random_global'),[
                ('first','random'),('--cache_fixed_surfaces','--cache_pooled_surfaces')])

    def test_pbs_creates_fresh_persistent_output_for_all_variants(self):
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp)
            text=(repo/'train_recon_sanity.pbs').read_text().replace(
                'PROJECT_ROOT=/projects_vol/gp_chuanxia.zheng/nathan/HY3DGS','PROJECT_ROOT='+str(project))
            text=text.replace('eval "$(/usr/local/anaconda2025/bin/conda shell.bash hook)"',':')
            stubs='module() { :; }\nconda() { :; }\ngit() { echo revision; }\nnvidia-smi() { :; }\npython() { :; }\n'
            launcher=project/'test.pbs';launcher.write_text(stubs+text)
            (project/'run_internscenes_recon_sanity.sh').write_text('test -s "$OUT/git_commit.txt"\ntouch "$OUT/reached"\n')
            for name in ('fixed_global','fixed_ot','random_global','fixed_coverage'):
                result=subprocess.run(['bash',str(launcher)],env=dict(os.environ,PBS_JOBID='123.gaas',EXPERIMENT=name),
                                      capture_output=True,text=True,timeout=15)
                log=project/'runs/logs'/f'recon_sanity_{name}_123.gaas.log'
                self.assertEqual(result.returncode,0,log.read_text() if log.exists() else result.stderr)
                self.assertTrue((project/'runs/recon_sanity'/f'{name}_123.gaas/reached').exists())

    def test_launcher_evaluates_baselines_intermediate_checkpoints_and_writes_summary(self):
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bindir=root/'bin';bindir.mkdir()
            source=root/'source.pt';source.touch()
            fake=bindir/'python'
            fake.write_text('#!'+sys.executable+'\n'+'''import json, pathlib, sys
args=sys.argv[1:]
if args[0]=='-':
    code=sys.stdin.read()
    if code.startswith('import csv'):
        sys.argv=args
        exec(compile(code,'summary','exec'))
elif args[0]=='train_pc_unite.py':
    out=pathlib.Path(args[args.index('--output_dir')+1]);out.mkdir(parents=True)
    (out/'args.json').write_text('{}')
    budget=int(args[args.index('--additional_steps')+1])
    for step in range(21000,20000+budget,1000): (out/f'ckpt_{step:07d}.pt').touch()
    (out/'ckpt_final.pt').touch()
elif args[0]=='evaluate_pc_recon_sanity.py':
    out=pathlib.Path(args[args.index('--output_dir')+1]);out.mkdir(parents=True)
    summary={'primary':{'cd':.1,'gt_distance_p99':.05}}
    if args[args.index('--pair_mode')+1]=='paired': summary['alternate']={'cd':.2,'gt_distance_p99':.1}
    (out/'metrics.json').write_text(json.dumps({'summary':summary,
        'checkpoint':args[args.index('--ckpt')+1], 'register_noise_mode':'fixed',
        'modulation_recon_timestep_max':0}))
else: raise SystemExit('Unexpected command: '+repr(args))
''')
            fake.chmod(0o755)
            for name in ('fixed_global','fixed_ot','random_global','fixed_coverage'):
                out=root/name
                env=dict(os.environ,EXPERIMENT=name,SOURCE_CKPT=str(source),OUT=str(out),
                         PATH=str(bindir)+os.pathsep+os.environ['PATH'],DRY_RUN='0')
                result=subprocess.run(['bash','run_internscenes_recon_sanity.sh'],cwd=repo,env=env,
                                      capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertTrue((out/'COMPLETE').exists())
                self.assertEqual(len(list(out.glob('eval_ckpt_*'))),7)
                self.assertEqual(len((out/'quality_summary.csv').read_text().splitlines()),12)
                self.assertTrue((out/'commands.txt').exists())

    def test_cpu_training_resume_and_evaluation_on_fixed_ten_example_recipe(self):
        fixture=room_tests.RoomViewPoolTests();fixture.setUp()
        try:
            root=fixture.root
            pools=root/'views.json';pools.write_text(json.dumps(fixture.pools))
            repo=Path(__file__).resolve().parents[1]
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
            def run(args):
                result=subprocess.run([sys.executable,*args],cwd=repo,env=env,capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            common=['train_pc_unite.py','--dataset','internscenes','--data_dir',str(root),
                '--room_view_pools_json',str(pools),'--views_per_sample','2','--view_sample_mode','first',
                '--align_mode','c_meanrms','--cache_fixed_surfaces','--recon_only','--no_experiment_manifest',
                '--pretrained_load','none','--device','cpu','--seed','0','--batch_size','2',
                '--num_latents','4','--num_registers','4','--embed_dim','8','--width','64','--heads','4',
                '--num_ge_layers','1','--num_decoder_layers','1','--num_points_per_anchor','8',
                '--pc_size','4','--pc_sharpedge_size','4','--query_sample_mode','weighted_fps',
                '--fps_query_cache_items','2','--chamfer_backend','chunked','--chamfer_chunk_size','8',
                '--sinkhorn_iters','2','--no-representation_noising','--register_noise_mode','fixed',
                '--modulation_recon_timestep_max','0','--num_steps','2','--log_interval','1',
                '--ckpt_interval','1','--vis_interval','0','--no-save_optimizer']
            run([*common,'--output_dir',str(root/'train')])
            source=root/'train/ckpt_final.pt'
            run([*common,'--resume_ckpt',str(source),'--reset_optimizer_on_resume','--additional_steps','1',
                 '--lr_schedule','constant','--output_dir',str(root/'resume')])
            ckpt=root/'resume/ckpt_final.pt'
            saved=torch.load(ckpt,weights_only=False)
            self.assertEqual(saved['step'],3)
            self.assertEqual(saved['args']['register_noise_mode'],'fixed')
            self.assertEqual(saved['args']['modulation_recon_timestep_max'],0)
            self.assertNotIn('optimizer',saved)
            run(['evaluate_pc_recon_sanity.py','--ckpt',str(ckpt),'--data_dir',str(root),
                 '--room_view_pools_json',str(pools),'--device','cpu','--output_dir',str(root/'eval')])
            metrics=json.loads((root/'eval/metrics.json').read_text())
            self.assertEqual(set(metrics['summary']),{'primary','alternate'})
            self.assertEqual(len(metrics['rooms']),4)
            self.assertIn('gt_distance_p99',metrics['summary']['primary'])
            self.assertIn('fscore_norm_0.02',metrics['summary']['primary'])
            self.assertIn('pred_near_duplicate_fraction',metrics['summary']['primary'])
            self.assertEqual(len(list((root/'eval').rglob('*.ply'))),12)
        finally:
            fixture.tearDown()


if __name__=='__main__':
    unittest.main()
