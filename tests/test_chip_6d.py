import math
"""Six-dimensional extension of native three-point + lower-joint CHIP."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from hydra import compose, initialize_config_dir
import torch

from mimic_lite.tasks.chip import ChipTracking, chip_command, chip_privileged, chip_metric
from mimic_lite.tasks.chip_math import ChipSchedule, point_wrench, wrist_compliance_shift
from active_adaptation.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat


class SixDTests(unittest.TestCase):
    def fixture(self):
        s = ChipSchedule(1, 'cpu', compliance_mode='wrist_6d',
            compliance_6d_max=[.02,.02,.02,.2,.2,.2], left_compliance_mode='rigid')
        s.compliance_6d.zero_()
        s.compliance_6d[:,1] = torch.tensor([.02,0,0,.2,0,0])
        s.force.zero_(); s.force[:,1] = torch.tensor([0.,5,0])
        s.couple.zero_(); s.couple[:,1] = torch.tensor([0.,2,0])
        # Wrist x = world y, anchor x = world y. Commands should shift local x.
        identity = torch.tensor([1.,0,0,0])
        yaw = torch.tensor([2**-.5,0,0,2**-.5])
        q = torch.stack([identity,yaw,identity,yaw])[None]
        nominal_p = torch.tensor([[[.1,0,0],[.2,0,0],[0,0,.3]]])
        nominal_q = identity.expand(1,3,4).clone()
        c = SimpleNamespace(chip=s, chip_body_ids=[0,1,2], anchor_body_idx_asset=3,
            chip_compliance_6d_scale=torch.tensor([50.,50.,50.,5.,5.,5.]),
            chip_compliance_scale=10., chip_leg_ids=list(range(12)), obs_current_step_index=0,
            ref_joint_pos_future_=torch.arange(12.).reshape(1,1,12),
            ref_joint_vel_future_=torch.arange(12.).reshape(1,1,12)*.1,
            ref_anchor_quat_future_w=q[:,3:4],
            asset=SimpleNamespace(data=SimpleNamespace(body_link_quat_w=q)),
            chip_reference=lambda **kw: (nominal_p,nominal_q))
        return c, nominal_p, nominal_q

    def term(self, cls, c, **kw):
        term = cls(**kw)
        term._initialize(SimpleNamespace(command_manager=c, num_envs=1, device='cpu'))
        return term

    def test_native_layout_frames_and_nominal_reference(self):
        c,p,q = self.fixture()
        before_p, before_q = p.clone(), q.clone()
        obs = self.term(chip_command,c).compute()
        self.assertEqual(obs.shape,(1,69))
        torch.testing.assert_close(obs[:,:12],c.ref_joint_pos_future_[:,0])
        torch.testing.assert_close(obs[:,12:24],c.ref_joint_vel_future_[:,0])
        dp = obs[:,24:33].reshape(1,3,3)-p
        torch.testing.assert_close(dp,torch.tensor([[[0.,0,0],[-.1,0,0],[0,0,0]]]),atol=1e-6,rtol=1e-5)
        dq = quat_mul(obs[:,33:45].reshape(1,3,4),quat_conjugate(q))
        dr = axis_angle_from_quat(dq)
        torch.testing.assert_close(dr,torch.tensor([[[0.,0,0],[-.4,0,0],[0,0,0]]]),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(obs[:,51:69],torch.tensor([[0.,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0]]))
        torch.testing.assert_close(p,before_p);torch.testing.assert_close(q,before_q)
        priv = self.term(chip_privileged,c).compute()
        self.assertEqual(priv.shape,(1,57))
        torch.testing.assert_close(priv[:,:9],p.flatten(1))
        torch.testing.assert_close(priv[:,48:57],c.chip.couple.flatten(1))
        c.chip.force.zero_()
        pure = self.term(chip_command,c).compute()
        torch.testing.assert_close(pure[:,24:33],p.flatten(1))
        c.chip.compliance_6d.zero_()
        rigid = self.term(chip_command,c).compute()
        torch.testing.assert_close(rigid[:,24:45],torch.cat([p.flatten(1),q.flatten(1)],-1))

    def test_rotation_equivariance_and_com_wrench(self):
        c,_,_ = self.fixture()
        before = self.term(chip_command,c).compute()
        g = torch.tensor([2**-.5,2**-.5,0,0])
        from active_adaptation.utils.math import quat_rotate
        quats = c.asset.data.body_link_quat_w
        c.asset.data.body_link_quat_w = quat_mul(g.expand_as(quats),quats)
        c.ref_anchor_quat_future_w = quat_mul(g.expand_as(c.ref_anchor_quat_future_w),c.ref_anchor_quat_future_w)
        c.chip.force = quat_rotate(g.expand(1,3,4),c.chip.force)
        c.chip.couple = quat_rotate(g.expand(1,3,4),c.chip.couple)
        torch.testing.assert_close(self.term(chip_command,c).compute(),before,atol=2e-6,rtol=1e-5)
        p=torch.tensor([[[.1,0,0]]]);f=torch.tensor([[[0.,5,0]]]);mu=torch.tensor([[[2.,0,0]]])
        _,torque=point_wrench(f,p,torch.zeros_like(p),mu)
        torch.testing.assert_close(torque,torch.tensor([[[2.,0,.5]]]))

    def test_structured_patterns_and_independent_groups(self):
        torch.manual_seed(456)
        probs=[.16,.16,.16,.16,.16,.20]
        for group in ('translation','rotation'):
            for k,mask in enumerate(([1,0,0],[0,1,0],[0,0,1],[1,1,1],[0,0,0])):
                weights=[0.]*6;weights[k]=1.
                kw={group+'_pattern_probabilities':weights}
                s=ChipSchedule(256,'cpu',compliance_mode='wrist_6d',left_compliance_mode='rigid',
                    compliance_6d_max=[.02,.02,.02,.2,.2,.2],rigid_probability=0.,**kw)
                sl=slice(0,3) if group=='translation' else slice(3,6)
                v=s.compliance_6d[:,1,sl]
                torch.testing.assert_close(v>0,torch.tensor(mask).bool().expand_as(v))
                if k==3:
                    torch.testing.assert_close(v[:,0],v[:,1]);torch.testing.assert_close(v[:,1],v[:,2])
                self.assertFalse(s.compliance_6d[:,0].any())
                s._sample_compliance(torch.empty(0,dtype=torch.long))
        s=ChipSchedule(8192,'cpu',compliance_mode='wrist_6d',rigid_probability=0.,
            translation_pattern_probabilities=[1/3,1/3,1/3,0,0,0],
            rotation_pattern_probabilities=[1/3,1/3,1/3,0,0,0])
        c=s.compliance_6d[:,1]
        counts=torch.bincount(c[:,:3].argmax(-1)*3+c[:,3:].argmax(-1),minlength=9).float()/8192
        self.assertTrue(((counts-1/9).abs()<.02).all())

    def test_axial_couple_and_reset(self):
        torch.manual_seed(5)
        s=ChipSchedule(1024,'cpu',compliance_mode='wrist_6d',torque_axis_probability=.8,
            wrench_event_probabilities=[0.,1.,0.],warmup_steps=0,ramp_steps=0,force_interval=0,force_duration=(8,9))
        for _ in range(4):s.prepare()
        s.refresh_couple_frame(torch.tensor([1.,0,0,0]).expand(1024,3,4))
        before=s.couple.clone()
        s.refresh_couple_frame(torch.tensor([2**-.5,0,0,2**-.5]).expand(1024,3,4))
        expect=torch.stack([-before[...,1],before[...,0],before[...,2]],-1)
        torch.testing.assert_close(s.couple[s.couple_axis_mask],expect[s.couple_axis_mask],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(s.couple[~s.couple_axis_mask],before[~s.couple_axis_mask])
        self.assertFalse(s.force.any());self.assertFalse(s.couple[:,2].any())
        s.reset(torch.tensor([0,1]));self.assertFalse(s.couple[:2].any())

    def test_small_torque_distribution_and_physical_angle_bound(self):
        torch.manual_seed(2026)
        n=16384
        s=ChipSchedule(n,'cpu',compliance_mode='wrist_6d',max_torque=2.,
            compliance_6d_max=[.02,.02,.02,.8,.8,.8],
            torque_amplitude_bands=[[.1,.6,.7],[.6,1.,.2],[1.,2.,.1]],
            angular_hindsight_limit=math.pi/4,torque_axis_probability=.8,
            wrench_event_probabilities=[0.,1.,0.],warmup_steps=0,ramp_steps=0,
            force_interval=0,force_duration=(8,9))
        # Rigid directions preserve the requested (unconditioned) mixture.
        s.compliance_6d.zero_()
        q=torch.randn(n,3,4);q=q/q.norm(dim=-1,keepdim=True)
        s.couple_envelope.fill_(1.)
        s.refresh_couple_frame(q)
        a=s.couple_effective_amplitude[:,:2].flatten()
        frequencies=torch.stack(((a<.6).float().mean(),((a>=.6)&(a<1.)).float().mean(),(a>=1.).float().mean()))
        torch.testing.assert_close(frequencies,torch.tensor([.7,.2,.1]),atol=.012,rtol=0)
        self.assertGreaterEqual(a.min().item(),.1-1e-6)
        self.assertLessEqual(a.max().item(),2.)
        # Anisotropic compliance and changing actual wrist frames must bound the
        # same physical couple used by hindsight, not just clip the target.
        from mimic_lite.tasks.chip_math import wrist_compliance_shift
        for c in ([.8,.8,.8],[.8,0.,0.],[0.,.8,.3]):
            s.compliance_6d[:,:,3:]=torch.tensor(c)
            q=torch.randn(n,3,4);q=q/q.norm(dim=-1,keepdim=True)
            s.refresh_couple_frame(q)
            shift=wrist_compliance_shift(s.couple[:,:2],s.compliance_6d[...,3:],q[:,:2])
            self.assertLessEqual(shift.norm(dim=-1).max().item(),math.pi/4+1e-6)
            self.assertLessEqual(s.couple.norm(dim=-1).max().item(),2.+1e-6)
        # prepare can change compliance independently; its output is bounded too.
        s.c_remaining.fill_(1)
        s.prepare()
        shift=wrist_compliance_shift(s.couple[:,:2],s.compliance_6d[...,3:],s.couple_frame[:,:2])
        self.assertLessEqual(shift.norm(dim=-1).max().item(),math.pi/4+1e-6)
        s.reset(torch.arange(8));self.assertFalse(s.couple[:8].any())

    def test_config_and_metrics(self):
        with initialize_config_dir(config_dir=str(Path(__file__).parents[1]/'cfg'),version_base=None):
            cfg=compose(config_name='train',overrides=['task=chip_6d'])
        self.assertEqual(cfg.task.command.chip.left_compliance_mode,'rigid')
        self.assertEqual(list(cfg.task.command.chip.compliance_6d_max),[.02,.02,.02,.8,.8,.8])
        self.assertEqual(list(cfg.task.command.chip.compliance_6d_scale),[50.,50.,50.,1.25,1.25,1.25])
        self.assertEqual(cfg.task.observation.policy.chip_history.history_length,10)
        c,_,_=self.fixture();c.chip_applied_couple=c.chip.couple.clone()
        self.assertAlmostEqual(self.term(chip_metric,c,weight=1.,quantity='right_wrist_torque')._compute().item(),2.)
        self.assertAlmostEqual(self.term(chip_metric,c,weight=1.,quantity='right_wrist_angular_compliance')._compute().item(),.2/3,places=6)


if __name__=='__main__':unittest.main()
