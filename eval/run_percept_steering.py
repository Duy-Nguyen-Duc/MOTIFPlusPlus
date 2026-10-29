"""Table-1 benchmark of a stage-3 checkpoint, alone (--steer 0) or with perception steering (--steer 1.0).

    python eval/run_percept_steering.py --ckpt runs/<run>/outputs_stage3_elora_geod/epoch_0060.pth \
        --name <run> --out eval_results/<run> [--steer 1.0] [--episodes 50] [--pairs 0 1] [--workers 3]

Protocol (MOTIF Table 1): six few-shot robot/task pairs x 50 episodes, environment seeds 2025-2074, policy seed 7,
SHA-256 request seeds, 4 denoising steps, the full 16-action chunk executed per prediction, CPU (lavapipe) renderer.
Steering: before a chunk is executed, the object is located from the robot's own cameras (colour threshold +
ray/plane intersection; wrist camera through its calibrated mount, else the third-view camera) and the chunk's
planned xy endpoint is servoed onto it (approach while the gripper is open, place target while it holds something,
contact line for push/pull, across the peg axis for LiftPeg). Nothing is trained. The driver stages the checkpoint,
runs the pairs in worker processes (this file with --pair) and writes <out>/<name>/<robot>__<task>.json per pair and
<out>/SUMMARY.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANISKILL = Path(os.environ.get('MANISKILL_ROOT', REPO.parent / 'ManiSkill'))   # ManiSkill checkout with the CPU renderer patch
VK = Path(os.environ.get('VK_FALLBACK_ROOT', '/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.tmp_motif_install/vkfallback/root'))
CALIB = REPO / 'assets/wrist_camera/wrist_cameras.json'

PAIRS = [
    ('panda_wristcam', 'PushCube-v1'), ('panda_wristcam', 'PlaceSphere-v1'),
    ('xarm6_robotiq_wristcam', 'PullCube-v1'), ('xarm6_robotiq_wristcam', 'LiftPegUpright-v1'),
    ('widowxai_wristcam', 'PickCube-v1'), ('widowxai_wristcam', 'StackCube-v1')]
ROBOTS = ['panda_wristcam', 'xarm6_robotiq_wristcam', 'widowxai_wristcam']
TASKS = ['PushCube-v1', 'PlaceSphere-v1', 'PullCube-v1', 'LiftPegUpright-v1', 'PickCube-v1', 'StackCube-v1']
ALL_PAIRS = [(r, t) for r in ROBOTS for t in TASKS]
SEED_BASE, POLICY_SEED, EXEC_HORIZON = 2025, 7, 16

STEER_GAIN = {'panda': 0.039, 'xarm6': 0.038, 'widowxai': 0.035}   # tcp metres per unit position command
STEER_RADIUS, STEER_STEPS = 0.30, 4
TARGETS = {   # colour rule, half size (m), shape of the steerable actors per task
    'PlaceSphere-v1': {'sphere': dict(color='blue', half=0.02, shape='sphere')},
    'StackCube-v1': {'cubeA': dict(color='red', half=0.02, shape='cube'), 'cubeB': dict(color='green', half=0.02, shape='cube')},
    'PickCube-v1': {'cube': dict(color='red', half=0.02, shape='cube')},
    'PushCube-v1': {'cube': dict(color='blue', half=0.02, shape='cube')},
    'PullCube-v1': {'cube': dict(color='blue', half=0.02, shape='cube')},
    'LiftPegUpright-v1': {'peg': dict(color='redblue', half=0.025, shape='peg')},
}
INTERACTIONS = {   # grasp: tcp over the object (then over the place target while holding); push: contact point behind it
    'PlaceSphere-v1': dict(grasp='sphere', place=None, mode='grasp'),
    'StackCube-v1': dict(grasp='cubeA', place='cubeB', mode='grasp'),
    'PickCube-v1': dict(grasp='cube', place=None, mode='grasp'),
    'LiftPegUpright-v1': dict(grasp='peg', place=None, mode='line'),
    'PushCube-v1': dict(grasp='cube', place=None, mode='push', dir=(1.0, 0.0), standoff=0.01),
    'PullCube-v1': dict(grasp='cube', place=None, mode='push', dir=(-1.0, 0.0), standoff=0.01),
}
TABLE_Z = 0.0
MIN_PIXELS = 12


# ----------------------------------------------------------------------------------------------------------------
# perception: training-free object estimate from the cameras and the tcp pose
# ----------------------------------------------------------------------------------------------------------------
def interaction_point(task, target, obj_pos):
    spec = INTERACTIONS[task]
    p = np.asarray(obj_pos, float).copy()
    if spec['mode'] == 'push' and target == spec['grasp']:
        d = np.asarray(spec['dir'], float)
        d = d / np.linalg.norm(d)
        p[:2] -= d * (TARGETS[task][target]['half'] + spec['standoff'])
    return p


def quat_wxyz_to_matrix(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def color_mask(rgb, color):
    r, g, b = [rgb[..., i].astype(np.int32) for i in range(3)]
    if color == 'red':
        return (r > 90) & (r > 2 * g) & (r > 2 * b)
    if color == 'green':
        return (g > 80) & (g > 1.6 * r) & (g > 1.6 * b)
    if color == 'blue':
        return (b > 60) & (b > 1.8 * r) & (b > 1.3 * g)
    if color == 'redblue':
        return color_mask(rgb, 'red') | color_mask(rgb, 'blue')
    raise ValueError(color)


def largest_component(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


class Camera:
    """Pinhole camera (OpenCV axes) with pose T_wc (camera -> world)."""

    def __init__(self, K, T_wc):
        self.K = np.asarray(K, float).reshape(3, 3)
        self.Kinv = np.linalg.inv(self.K)
        T = np.asarray(T_wc, float).reshape(4, 4)
        self.R, self.o = T[:3, :3], T[:3, 3]

    def ray(self, u, v):
        return self.R @ (self.Kinv @ np.array([u, v, 1.0]))

    def plane_hit(self, d, z):
        if d[2] > -1e-6:
            return None
        lam = (z - self.o[2]) / d[2]
        return float(lam) if lam > 0 else None


class ObjectLocator:
    """World position of a target from the wrist camera (calibrated mount x tcp pose) or the third-view camera;
    the last estimate is kept while the target is out of view."""

    def __init__(self, robot_uid, task, cams=('wrist', 'third')):
        self.robot, self.task, self.cams = robot_uid.split('_')[0], task, tuple(cams)
        self.targets = TARGETS[task]
        c = json.load(open(CALIB))[self.robot]
        self.K_wrist = np.asarray(c['intrinsic_cv'], float)
        self.T_tcp_cam = np.asarray(c['T_tcp_cam_cv'], float)
        self.memory = {}

    def reset(self):
        self.memory.clear()

    def views(self, obs):
        tcp = np.asarray(obs['extra']['tcp_pose'], float).reshape(-1)
        T_wt = np.eye(4)
        T_wt[:3, :3] = quat_wxyz_to_matrix(tcp[3:7])
        T_wt[:3, 3] = tcp[:3]
        out = []
        for name in self.cams:
            if name == 'wrist':
                rgb = np.asarray(obs['sensor_data']['hand_camera']['rgb'], np.uint8)
                if rgb.ndim == 4:
                    rgb = rgb[0]
                out.append((name, rgb, Camera(self.K_wrist, T_wt @ self.T_tcp_cam)))
            else:
                rgb = np.asarray(obs['sensor_data']['third_view_camera']['rgb'], np.uint8)
                if rgb.ndim == 4:
                    rgb = rgb[0]
                sp = obs['sensor_param']['third_view_camera']
                E = np.asarray(sp['extrinsic_cv'], float).reshape(-1)
                E = E.reshape(3, 4) if E.size == 12 else E.reshape(4, 4)[:3]
                T_cw = np.vstack([E, [0, 0, 0, 1]])
                K = np.asarray(sp['intrinsic_cv'], float).reshape(-1)[-9:]
                out.append((name, rgb, Camera(K, np.linalg.inv(T_cw))))
        return out

    def _detect(self, rgb, spec):
        mask = color_mask(rgb, spec['color'])
        if mask.sum() < MIN_PIXELS:
            return None
        mask = largest_component(mask)
        if mask.sum() < MIN_PIXELS:
            return None
        vs, us = np.nonzero(mask)
        return float(np.median(us)) + 0.5, float(np.median(vs)) + 0.5, mask, 1.0

    def _lift(self, rgb, cam, det, spec):
        """Detection (u, v, mask) -> world position of the object centre by ray / plane intersection."""
        u, v, mask, _ = det
        half, shape = spec['half'], spec['shape']
        d = cam.ray(u, v)
        n = d / np.linalg.norm(d)
        if shape == 'sphere':
            lam = cam.plane_hit(d, half)
            return (None, {}) if lam is None else (cam.o + lam * d, dict(z_hit=half))
        vs, us = np.nonzero(mask)
        if shape == 'cube':
            # top face = brightest, uniformly lit part of the silhouette; its centroid ray meets z = 2*half at the centre xy
            b = rgb[mask].astype(np.int32).sum(1)
            p5, p95 = np.percentile(b, [5, 95])
            face = np.zeros_like(mask)
            face[vs, us] = b >= p95 - 0.12 * (p95 - p5)
            face = largest_component(face)
            if face.sum() >= 8:
                fv, fu = np.nonzero(face)
                d_f = cam.ray(float(np.median(fu)) + 0.5, float(np.median(fv)) + 0.5)
                lam = cam.plane_hit(d_f, 2 * half)
                if lam is not None:
                    p = cam.o + lam * d_f
                    p[2] = half
                    return p, dict(z_hit=2 * half, model='topface', n_face=int(face.sum()))
        z_hit = half + half * abs(n[2])          # silhouette centroid: top face (steep view) .. centre (side view)
        lam = cam.plane_hit(d, z_hit)
        if lam is None:
            return None, {}
        centre = cam.o + lam * d
        if shape != 'peg':
            return centre, dict(z_hit=z_hit, model='centroid')
        # peg: also its long axis (world xy) = principal axis of the silhouette projected on the plane z = z_hit
        sel = np.random.default_rng(0).choice(len(us), size=min(400, len(us)), replace=False)
        pix = np.stack([us[sel] + 0.5, vs[sel] + 0.5, np.ones(len(sel))], -1)
        D = pix @ cam.Kinv.T @ cam.R.T
        ok = D[:, 2] < -1e-6
        lam_s = (z_hit - cam.o[2]) / np.where(ok, D[:, 2], -1.0)
        pts = (cam.o[None, :2] + lam_s[:, None] * D[:, :2])[ok & (lam_s > 0)]
        axis = None
        if len(pts) >= 10:
            P = pts - pts.mean(0)
            w, V = np.linalg.eigh(P.T @ P)
            axis = V[:, int(np.argmax(w))]
        return centre, dict(z_hit=z_hit, model='peg', axis=None if axis is None else axis.round(4).tolist())

    def locate(self, obs, target, near=0.10):
        """Wrist estimate when the object is close and well inside the image, else third view, else wrist, else memory."""
        spec = self.targets[target]
        tcp = np.asarray(obs['extra']['tcp_pose'], float).reshape(-1)[:3]
        cands = {}
        for name, rgb, cam in self.views(obs):
            det = self._detect(rgb, spec)
            if det is None:
                continue
            pos, extra = self._lift(rgb, cam, det, spec)
            if pos is None:
                continue
            m = det[2]
            border = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
            cands[name] = dict(pos=pos, source=name, px=(round(det[0], 1), round(det[1], 1)), n_px=int(m.sum()),
                               score=round(det[3], 3), border=border, **extra)
        pick = None
        w = cands.get('wrist')
        if w is not None and w['n_px'] >= 40 and np.linalg.norm((w['pos'] - tcp)[:2]) < near \
                and not (w['border'] and spec['shape'] != 'peg' and 'third' in cands):
            pick = w
        elif 'third' in cands:
            pick = cands['third']
        elif 'wrist' in cands:
            pick = cands['wrist']
        if pick is not None:
            self.memory[target] = pick['pos']
            return pick
        if target in self.memory:
            return dict(pos=self.memory[target], source='memory')
        return dict(pos=None, source='none')


def gripper_holding(state_now, state_open, closed_cmd, lo=0.12, hi=0.85):
    """Grasp flag without privileged information: closing was commanded and the measured opening settled between
    open and fully closed."""
    if not closed_cmd:
        return False
    closed = 1.0 - round(state_open)
    span = abs(closed - state_open)
    if span < 1e-3:
        return False
    frac = abs(state_now - state_open) / span
    return lo < frac < hi


class Steering:
    """Chunk-endpoint servo: while the gripper is open (place phase: closed) and the estimated gripper -> target xy
    offset r is within STEER_RADIUS, the chunk's planned xy endpoint is shifted by k * (r - planned displacement),
    spread over its first STEER_STEPS steps. Push/pull servo the contact line (and the approach while still behind
    the contact point); LiftPeg servoes only across the peg axis."""

    def __init__(self, robot, task, k):
        self.robot, self.task, self.k = robot, task, k
        self.spec = INTERACTIONS.get(task)
        self.loc = ObjectLocator(robot, task) if self.spec else None
        self.state, self.last = {}, {}

    def reset(self):
        self.state.clear()
        if self.loc is not None:
            self.loc.reset()

    def __call__(self, pos, g, obs, grip_state):
        self.last = {}
        if self.spec is None:
            return pos
        name, place = self.spec['grasp'], self.spec['place']
        push, line = self.spec['mode'] == 'push', self.spec['mode'] == 'line'
        st = self.state
        st.setdefault('grip_init', float(g[0, 0]))
        st.setdefault('grip_state_open', grip_state)
        held = gripper_holding(grip_state, st['grip_state_open'], st.get('closed_cmd', False))
        st['closed_cmd'] = bool(np.any(np.abs(g[:, 0] - st['grip_init']) >= 0.25))
        placing = bool(place and held)
        target = place if placing else name
        est = self.loc.locate(obs, target)
        tcp = np.asarray(obs['extra']['tcp_pose'], dtype=float).reshape(-1)[:3]
        self.last.update(target=target, phase='place' if placing else 'approach', held=held, source=est['source'])
        if est['pos'] is None:
            return pos
        r = interaction_point(self.task, target, est['pos']) - tcp
        axis = est.get('axis') if line else None
        self.last['r'] = r.round(4).tolist()
        if self.k == 0:
            return pos
        open_mask = np.abs(g[:, 0] - st['grip_init']) < 0.25
        if placing:
            open_mask = ~open_mask
        if push:
            open_mask = np.ones_like(open_mask)
        rxy = r[:2]
        Qa = None
        if line:
            if axis is None:
                return pos
            a = np.asarray(axis, float); a = a / np.linalg.norm(a)
            Qa = np.eye(2) - np.outer(a, a)
        reach = float(np.linalg.norm(Qa @ rxy)) if Qa is not None else float(np.linalg.norm(rxy))
        if not open_mask[0] or reach > STEER_RADIUS:
            return pos
        gain = STEER_GAIN[self.robot.split('_')[0]]
        H = EXEC_HORIZON
        planned = gain * pos[:H, :2].sum(0)
        corr = self.k * (rxy - planned)
        if Qa is not None:
            corr = self.k * (Qa @ (rxy - planned))
        if push:
            dv = np.asarray(self.spec['dir'], float); dv = dv / np.linalg.norm(dv)
            along = float(rxy @ dv)
            P = np.outer(dv, dv); Q = np.eye(2) - P
            corr = self.k * (Q @ (rxy - planned)) + (self.k * (P @ (rxy - planned)) if along > 0.005 else 0.0)
        n = min(STEER_STEPS, H)
        per_step = np.clip(corr / (n * gain), -0.5, 0.5)
        pos = pos.copy()
        for t in range(n):
            if open_mask[t]:
                pos[t, :2] += per_step
        self.last.update(planned_xy=planned.round(4).tolist(), corr_xy=corr.round(4).tolist(),
                         per_step=per_step.round(4).tolist(), steered_steps=int(open_mask[:n].sum()))
        return pos


# ----------------------------------------------------------------------------------------------------------------
# worker: one robot/task pair under the benchmark protocol
# ----------------------------------------------------------------------------------------------------------------
def request_seed(policy_seed, robot, task, episode_seed, replan):
    key = json.dumps([policy_seed, robot, task, episode_seed, replan], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2 ** 63 - 1)


def run_pair(args):
    sys.path[:0] = [str(REPO), str(MANISKILL)]
    import gymnasium as gym
    import torch
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
    from experiment.inference_wrapper import InferenceWrapper
    from eval_maniskill.eval_motif_maniskill import (_gripper_state, _robot_key, ACTION_GRIPPER_DIMS,
                                                      ROBOT_UID_TO_POLICY_UID, STEP_LENGTHS, TASK_INSTRUCTIONS)
    robot, task = args.pair
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'{robot}__{task}.json'
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite completed pair: {target}')
    episode_file = out / f'{robot}__{task}.episodes.jsonl'
    trace_file = out / f'{robot}__{task}.trace.jsonl'
    if episode_file.exists() or trace_file.exists():
        raise RuntimeError('Partial pair exists; inspect before resuming in a fresh directory')
    manifest = json.loads(Path(args.manifest).read_text())
    policy = InferenceWrapper(Path(args.checkpoint), device='cuda:0')
    policy.set_robot_uid(ROBOT_UID_TO_POLICY_UID[robot])
    head = policy.model.flow_matching_head
    assert head.num_inference_timesteps == 4 and head.action_horizon == EXEC_HORIZON
    env = CPUGymWrapper(gym.make(
        task, obs_mode='rgb', robot_uids=robot,
        sim_config=dict(control_freq=20, sim_freq=100),
        sensor_configs=dict(shader_pack='default', width=256, height=256),
        human_render_camera_configs=dict(shader_pack='default'),
        viewer_camera_configs=dict(shader_pack='default'), render_mode='rgb_array',
        control_mode='pd_ee_delta_pose'))
    env.unwrapped.print_sim_details()
    steer = Steering(robot, task, args.steer) if args.steer > 0 else None

    def predict(element, seed):
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(seed)
            return policy.get_action(element)

    def executable(chunk, obs):
        g = np.asarray(chunk['action.gripper']).reshape(EXEC_HORIZON, -1)
        assert g.shape[1] == ACTION_GRIPPER_DIMS[_robot_key(robot)]
        pos = np.asarray(chunk['action.position'])
        if steer is not None:
            pos = steer(np.asarray(pos, dtype=float), g, obs, float(_gripper_state(robot, obs['agent']['qpos'])[0]))
        return np.concatenate([pos, np.asarray(chunk['action.rotation']), g], axis=1)[:EXEC_HORIZON]

    episodes, replay_checked = [], False
    total_start = time.perf_counter()
    try:
        with episode_file.open('x') as journal, trace_file.open('x') as traces:
            for episode_index in range(args.episodes):
                seed = SEED_BASE + episode_index
                start = time.perf_counter()
                if steer is not None:
                    steer.reset()
                env.reset(seed=seed)
                env.action_space.seed(seed)
                env.step(env.action_space.sample())
                obs, info = env.reset(seed=seed)
                reset_s = time.perf_counter() - start
                infer_s = step_s = 0.0
                successes = terminated_success = False
                replan = 0
                actions = []
                for step in range(args.step_limit or STEP_LENGTHS[task]):
                    if not actions:
                        state = np.concatenate([obs['extra']['tcp_pose'],
                                                _gripper_state(robot, obs['agent']['qpos'])]).astype(np.float32)
                        element = {
                            'video.image': np.asarray(obs['sensor_data']['third_view_camera']['rgb'], dtype=np.uint8)[None],
                            'video.wrist_image': np.asarray(obs['sensor_data']['hand_camera']['rgb'], dtype=np.uint8)[None],
                            'state.position': state[None, :3], 'state.rotation': state[None, 3:7],
                            'state.gripper': state[None, 7:8],
                            'annotation.human.task_description': [TASK_INSTRUCTIONS[task]],
                        }
                        rs = request_seed(POLICY_SEED, robot, task, seed, replan)
                        tic = time.perf_counter()
                        pred = executable(predict(element, rs), obs)
                        infer_s += time.perf_counter() - tic
                        if not replay_checked:   # the per-request RNG makes every prediction exactly repeatable
                            if not np.array_equal(pred, executable(predict(element, rs), obs)):
                                raise RuntimeError('Per-request RNG replay is not exact')
                            replay_checked = True
                        traces.write(json.dumps(dict(episode_seed=seed, step=step, replan=replan, request_seed=rs,
                                                     state=state.tolist(), action=pred.tolist(),
                                                     steer=dict(steer.last) if steer is not None else None)) + '\n')
                        traces.flush()
                        actions = list(pred)
                        replan += 1
                    tic = time.perf_counter()
                    obs, reward, terminated, truncated, info = env.step(actions.pop(0))
                    step_s += time.perf_counter() - tic
                    if 'success' not in info:
                        raise RuntimeError('Environment did not provide explicit success')
                    successes = bool(np.asarray(info['success']).item())
                    terminated_success = bool(terminated)
                    if successes or terminated_success:
                        break
                    if (step + 1) % 100 == 0:
                        print(json.dumps(dict(event='progress', episode=episode_index, step=step + 1)), flush=True)
                row = dict(episode=episode_index, episode_seed=seed, policy_seed=POLICY_SEED,
                           success=successes, terminated=terminated_success, steps=step + 1, replans=replan,
                           seconds=time.perf_counter() - start, reset_seconds=reset_s,
                           inference_seconds=infer_s, env_step_seconds=step_s)
                episodes.append(row)
                journal.write(json.dumps(row) + '\n')
                journal.flush()
                print(json.dumps(row), flush=True)
    finally:
        env.close()
    n_succ = sum(x['success'] for x in episodes)
    result = dict(robot=robot, task=task, success_rate=n_succ / len(episodes), successes=n_succ,
                  num_per_task=len(episodes), few_shot=True, policy_seed=POLICY_SEED, seed_base=SEED_BASE,
                  request_seed_scheme='sha256-v1', checkpoint_sha256=manifest['checkpoint_sha256'],
                  replan_steps=EXEC_HORIZON, denoising_steps=4, step_budget=args.step_limit or STEP_LENGTHS[task],
                  steering=steering_spec(args.steer), renderer_icd=os.environ.get('VK_ICD_FILENAMES'),
                  success_termination_mismatches=sum(x['success'] != x['terminated'] for x in episodes),
                  total_seconds=time.perf_counter() - total_start, replay_exact=True)
    temp = target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(target)
    print(json.dumps(result, indent=2), flush=True)


def steering_spec(k):
    if k <= 0:
        return None
    return dict(source='percept', detector='color', depth='plane', cams='wrist,third', k=k, radius=STEER_RADIUS,
                steps=STEER_STEPS, place=True, tasks='all six (grasp: Sphere/Stack/Pick; line: LiftPeg; push: Push/Pull)')


# ----------------------------------------------------------------------------------------------------------------
# driver: stage the checkpoint, run the pairs in worker processes, summarise
# ----------------------------------------------------------------------------------------------------------------
def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def stage(name, ckpt, out):
    """Evaluator-compatible copy: config.json beside a symlinked checkpoint and its experiment_cfg."""
    staged = out / 'staged' / name
    staged.mkdir(parents=True, exist_ok=True)
    (staged / 'config.json').write_text(json.dumps(json.loads((ckpt.parent / 'config.json').read_text()), indent=2) + '\n')
    link = staged / ckpt.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(ckpt.resolve())
    expcfg = staged / 'experiment_cfg'
    if not expcfg.exists():
        expcfg.symlink_to(ckpt.parent / 'experiment_cfg', target_is_directory=True)
    return link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, help='stage-3 checkpoint (config.json and experiment_cfg beside it)')
    ap.add_argument('--name', help='run name: results in <out>/<name>/')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--steer', type=float, default=1.0, help='steering gain k; 0 = policy alone')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--pairs', nargs='+', type=int, default=None, help='indices into the pair list (default: all)')
    ap.add_argument('--all-pairs', action='store_true', help='all 18 robot x task pairs instead of the 6 transfer pairs')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--retries', type=int, default=6, help='re-run pairs that did not finish (renderer / quota incidents)')
    ap.add_argument('--wait', type=int, default=600, help='seconds between retries')
    ap.add_argument('--step-limit', type=int, default=None, help='smoke tests only')
    ap.add_argument('--pair', nargs=2, metavar=('ROBOT', 'TASK'), help='worker mode: run this pair in-process')
    ap.add_argument('--checkpoint', type=Path, help='worker mode: staged checkpoint')
    ap.add_argument('--manifest', type=Path, help='worker mode: manifest json')
    args = ap.parse_args()
    if args.pair:
        return run_pair(args)
    if not (args.ckpt and args.name and args.out):
        ap.error('--ckpt, --name and --out are required')
    out = args.out if args.out.is_absolute() else REPO / args.out
    ckpt = args.ckpt if args.ckpt.is_absolute() else REPO / args.ckpt
    plist = ALL_PAIRS if args.all_pairs else PAIRS
    pairs = [plist[i] for i in (args.pairs if args.pairs is not None else range(len(plist)))]
    xdg = out / 'xdg'
    xdg.mkdir(mode=0o700, parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', LP_NUM_THREADS='8', PYTHONPATH=f'{REPO}:{MANISKILL}',
               VK_ICD_FILENAMES=str(VK / 'lvp_icd.json'), MS_RENDER_BACKEND='pci:0', XDG_RUNTIME_DIR=str(xdg),
               LD_LIBRARY_PATH=f'{VK}/usr/lib/x86_64-linux-gnu:{VK}/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', ''))
    digest = sha256(ckpt)
    staged = stage(args.name, ckpt, out)
    manifest = out / f'{args.name}_manifest.json'
    manifest.write_text(json.dumps({
        'checkpoint': str(staged), 'source_checkpoint': str(ckpt), 'checkpoint_sha256': digest,
        'code_manifest_sha256': 'live-repo+run_percept_steering.py', 'experiment': args.name,
        'execution_protocol': {'replan_steps': EXEC_HORIZON, 'denoising_steps': 4, 'seed_base': SEED_BASE,
                               'policy_seed': POLICY_SEED, 'steering': steering_spec(args.steer)}}, indent=2) + '\n')
    dest = out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)

    def run(job):
        robot, task = job
        if (dest / f'{robot}__{task}.json').exists():
            return f'{robot}/{task}: already done'
        cmd = [sys.executable, str(Path(__file__).resolve()), '--pair', robot, task, '--checkpoint', str(staged),
               '--manifest', str(manifest), '--out', str(dest), '--episodes', str(args.episodes), '--steer', str(args.steer)]
        if args.step_limit:
            cmd += ['--step-limit', str(args.step_limit)]
        t0 = time.time()
        with (out / 'logs' / f'{args.name}__{robot}__{task}.log').open('a') as f:
            r = subprocess.run(cmd, env=env, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        return f'{robot}/{task}: rc={r.returncode} {(time.time() - t0) / 60:.1f} min'

    t0 = time.time()
    for attempt in range(1, args.retries + 1):
        pending = [p for p in pairs if not (dest / f'{p[0]}__{p[1]}.json').exists()]
        if not pending:
            break
        if attempt > 1:
            part = out / f'partial_{attempt - 1}'
            part.mkdir(exist_ok=True)
            for robot, task in pending:
                for f in dest.glob(f'{robot}__{task}.*'):
                    shutil.move(str(f), part / f.name)
            print(f'{len(pending)} pair(s) unfinished; retry {attempt} in {args.wait} s', flush=True)
            time.sleep(args.wait)
        with ThreadPoolExecutor(args.workers) as ex:
            for line in ex.map(run, pending):
                print(line, flush=True)
    rows = []
    for robot, task in pairs:
        p = dest / f'{robot}__{task}.json'
        if not p.exists():
            raise SystemExit(f'missing {p}')
        r = json.loads(p.read_text())
        rows.append(dict(robot=robot, task=task, episodes=r['num_per_task'], successes=r['successes'],
                         split='transfer' if (robot, task) in PAIRS else 'seen'))
    summary = dict(name=args.name, checkpoint=str(ckpt), sha256=digest, episodes=sum(r['episodes'] for r in rows),
                   successes=sum(r['successes'] for r in rows), rows=rows, minutes=round((time.time() - t0) / 60, 1),
                   steering=steering_spec(args.steer))
    for split in ('transfer', 'seen'):
        sub = [r for r in rows if r['split'] == split]
        if sub:
            summary[f'{split}_episodes'] = sum(r['episodes'] for r in sub)
            summary[f'{split}_successes'] = sum(r['successes'] for r in sub)
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    per_task = ', '.join('%s=%d' % (r['task'], r['successes']) for r in rows)
    print(f"{args.name}: {summary['successes']}/{summary['episodes']} ({per_task}) in {summary['minutes']} min")


if __name__ == '__main__':
    sys.exit(main())
