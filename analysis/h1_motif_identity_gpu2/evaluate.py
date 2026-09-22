"""One F1 transfer pair: independent request RNG, direct GPU inference, CPU renderer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from experiment.inference_wrapper import InferenceWrapper
from eval_maniskill.eval_motif_maniskill import (
    _gripper_state, _robot_key, ACTION_GRIPPER_DIMS, ROBOT_UID_TO_POLICY_UID,
    STEP_LENGTHS, TASK_INSTRUCTIONS,
)


def request_seed(policy_seed, robot, task, episode_seed, replan):
    key = json.dumps([policy_seed, robot, task, episode_seed, replan], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63-1)


def diagnostic_value(value):
    """Serialize simulator diagnostics without adding them to policy inputs."""
    if isinstance(value, dict):
        return {str(k): diagnostic_value(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [diagnostic_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {'unsupported_type': type(value).__name__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--robot', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--replan-steps', type=int, choices=range(1, 17), default=5,
                    help='Actions executed per 16-action prediction; F1 reproduction uses 16')
    ap.add_argument('--seed-base', type=int, default=2025)
    ap.add_argument('--save-video', action='store_true')
    ap.add_argument('--policy-seed', type=int, default=7)
    ap.add_argument('--step-limit', type=int, default=None, help='Smoke test only')
    ap.add_argument('--diagnostics', action='store_true',
                    help='Record privileged actor state and step info for failure analysis only')
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f'{args.robot}__{args.task}.json'
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite completed pair: {target}')
    manifest = json.loads(args.manifest.read_text())
    policy = InferenceWrapper(args.checkpoint, device='cuda:0')
    policy.set_robot_uid(ROBOT_UID_TO_POLICY_UID[args.robot])
    head = policy.model.flow_matching_head
    assert head.config.motif_position_mode == 'slot_time'
    assert head.config.motif_num_slots == 3 and head.config.motif_tokens_per_slot == 16
    assert head.num_inference_timesteps == 4 and head.action_horizon == 16
    env = CPUGymWrapper(gym.make(
        args.task, obs_mode='rgb', robot_uids=args.robot,
        sim_config=dict(control_freq=20, sim_freq=100),
        sensor_configs=dict(shader_pack='default', width=256, height=256),
        human_render_camera_configs=dict(shader_pack='default'),
        viewer_camera_configs=dict(shader_pack='default'), render_mode='rgb_array',
        control_mode='pd_ee_delta_pose'))
    env.unwrapped.print_sim_details()
    episode_file = args.out / f'{args.robot}__{args.task}.episodes.jsonl'
    trace_file = args.out / f'{args.robot}__{args.task}.trace.jsonl'
    # A separate output directory is required for any retry, preventing mixed streams.
    if episode_file.exists() or trace_file.exists():
        raise RuntimeError('Partial pair exists; inspect before resuming in a fresh directory')
    episodes, effect = [], None
    video_writer = None
    total_start = time.perf_counter()
    def predict(element, seed):
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(seed)
            return policy.get_action(element)
    def executable(chunk):
        g = np.asarray(chunk['action.gripper']).reshape(16, -1)
        assert g.shape[1] == ACTION_GRIPPER_DIMS[_robot_key(args.robot)]
        return np.concatenate([np.asarray(chunk['action.position']),
                               np.asarray(chunk['action.rotation']), g], axis=1)[:args.replan_steps]
    def record_diagnostics(traces, seed, step, info, terminal=False):
        if not args.diagnostics:
            return
        state = env.unwrapped.get_state_dict()
        traces.write(json.dumps(dict(event='diagnostics', episode_seed=seed, step=step,
                                     terminal=terminal, info=diagnostic_value(info),
                                     actors=diagnostic_value(state.get('actors', {}))))+'\n')
        traces.flush()
    try:
        with episode_file.open('x') as journal, trace_file.open('x') as traces:
            for episode_index in range(args.episodes):
                seed = args.seed_base + episode_index
                start = time.perf_counter()
                env.reset(seed=seed)
                env.action_space.seed(seed)
                env.step(env.action_space.sample())
                obs, info = env.reset(seed=seed)
                if args.save_video:
                    import imageio.v2 as imageio
                    video_writer = imageio.get_writer(str(args.out / f'{args.robot}__{args.task}__{seed}.mp4'), fps=20)
                    video_writer.append_data(np.concatenate([
                        np.asarray(obs['sensor_data'][c]['rgb'], dtype=np.uint8)
                        for c in ['third_view_camera', 'hand_camera']], axis=1))
                reset_s = time.perf_counter()-start
                infer_s = step_s = 0.0
                successes = terminated_success = False
                replan = 0
                actions = []
                record_diagnostics(traces, seed, 0, info)
                for step in range(args.step_limit or STEP_LENGTHS[args.task]):
                    if not actions:
                        state = np.concatenate([obs['extra']['tcp_pose'],
                                                _gripper_state(args.robot, obs['agent']['qpos'])]).astype(np.float32)
                        element = {
                            'video.image':np.asarray(obs['sensor_data']['third_view_camera']['rgb'],dtype=np.uint8)[None],
                            'video.wrist_image':np.asarray(obs['sensor_data']['hand_camera']['rgb'],dtype=np.uint8)[None],
                            'state.position':state[None,:3], 'state.rotation':state[None,3:7],
                            'state.gripper':state[None,7:8],
                            'annotation.human.task_description':[TASK_INSTRUCTIONS[args.task]],
                        }
                        rs = request_seed(args.policy_seed,args.robot,args.task,seed,replan)
                        tic = time.perf_counter()
                        pred = executable(predict(element,rs))
                        infer_s += time.perf_counter()-tic
                        if effect is None:
                            # Trained-checkpoint test: permute contents, keep labels fixed.
                            perm = torch.arange(48,device='cuda:0').reshape(3,16).flip(1).flatten()
                            hook = head.latent_action_encoder.register_forward_pre_hook(
                                lambda module, inp: (inp[0][:,perm],))
                            try:
                                shuffled = executable(predict(element,rs))
                            finally:
                                hook.remove()
                            repeated = executable(predict(element,rs))
                            if not np.array_equal(pred,repeated):
                                raise RuntimeError('Per-request RNG replay is not exact')
                            effect = dict(temporal_permutation_executed_action_l2=float(np.linalg.norm(pred-shuffled)),
                                          temporal_permutation_max_abs=float(np.max(np.abs(pred-shuffled))),
                                          request_replay_exact=True)
                        traces.write(json.dumps(dict(episode_seed=seed,step=step,replan=replan,
                                                     request_seed=rs,state=state.tolist(),action=pred.tolist()))+'\n')
                        traces.flush()
                        actions = list(pred)
                        replan += 1
                    tic = time.perf_counter()
                    obs, reward, terminated, truncated, info = env.step(actions.pop(0))
                    step_s += time.perf_counter()-tic
                    if video_writer is not None:
                        video_writer.append_data(np.concatenate([
                            np.asarray(obs['sensor_data'][c]['rgb'], dtype=np.uint8)
                            for c in ['third_view_camera', 'hand_camera']], axis=1))
                    if 'success' not in info:
                        raise RuntimeError('Environment did not provide explicit success')
                    successes = bool(np.asarray(info['success']).item())
                    terminated_success = bool(terminated)
                    if args.diagnostics:
                        # Preserve per-step contact flags; actor poses are sampled
                        # at replanning boundaries and the final step below.
                        traces.write(json.dumps(dict(event='step_info', episode_seed=seed,
                                                     step=step+1, info=diagnostic_value(info)))+'\n')
                    if successes or terminated_success:
                        break
                    if (step+1)%5 == 0:
                        record_diagnostics(traces, seed, step+1, info)
                    if (step+1)%100 == 0:
                        print(json.dumps(dict(event='progress',episode=episode_index,step=step+1)),flush=True)
                record_diagnostics(traces, seed, step+1, info, terminal=True)
                if video_writer is not None:
                    video_writer.close()
                    video_writer = None
                row = dict(episode=episode_index,episode_seed=seed,policy_seed=args.policy_seed,
                           success=successes,terminated=terminated_success,steps=step+1,replans=replan,
                           seconds=time.perf_counter()-start,reset_seconds=reset_s,
                           inference_seconds=infer_s,env_step_seconds=step_s)
                episodes.append(row)
                journal.write(json.dumps(row)+'\n');journal.flush()
                print(json.dumps(row),flush=True)
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()
    successes = sum(x['success'] for x in episodes)
    result = dict(robot=args.robot,task=args.task,success_rate=successes/len(episodes),
                  successes=successes,num_per_task=len(episodes),few_shot=True,
                  policy_seed=args.policy_seed,seed_base=args.seed_base,request_seed_scheme='sha256-v1',
                  checkpoint_sha256=manifest['checkpoint_sha256'],
                  code_manifest_sha256=manifest['code_manifest_sha256'],
                  replan_steps=args.replan_steps,denoising_steps=4,step_budget=args.step_limit or STEP_LENGTHS[args.task],
                  renderer_icd=os.environ.get('VK_ICD_FILENAMES'),
                  success_termination_mismatches=sum(x['success']!=x['terminated'] for x in episodes),
                  total_seconds=time.perf_counter()-total_start,identity_effect=effect)
    result['privileged_diagnostics_recorded'] = args.diagnostics
    temp = target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(target)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
