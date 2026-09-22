"""MOTIF evaluation on ManiSkill: 3 embodiments x 6 tasks, 50 seeds per pair.

Written for this reproduction. Neither the paper, `buduz/MOTIF` nor
`Johnathan218/ManiSkill` provides a closed-loop evaluation harness, so the
rollout loop, the seeding, the per-task step budgets, the task instruction
strings and the two reported metrics are all our choices; see
notes/PROVENANCE.md for which of them are constrained by a source and which
are not.

Talking to a MOTIF server, as opposed to a single-embodiment policy server:

  * MOTIF is multi-embodiment, so the robot uid has to be announced with
    set_robot_uid before the first query. The *_rel uids are used because
    training runs on canonicalized data (PANDAWithRELDataConfig).
  * MOTIF returns an action *chunk* as a dict of modality arrays, not one
    action as a tensor, so actions are queued and the policy is re-queried
    every replan_steps steps.
  * MOTIF's transforms expect uint8 HWC frames under video.* keys. The state
    is sent split as state.position / state.rotation / state.gripper rather
    than under a single "state" key: ConcatTransform groups keys by splitting
    on a dot, so a dot-less key falls into `others` and silently skips
    min-max normalization. The repo's own experiment/eval_policy.py has this
    bug, and a policy driven that way receives unnormalized metres.

Cameras and state layout mirror the training data exactly: third_view_camera +
hand_camera at 224x224, state = [tcp_pose(xyz+wxyz), last gripper joint].

Metrics written to --save-path:
  <robot>_success_dict.json   per-task success rate + total for that robot
  aver.json                   Global: mean over all 18 pairs
  generalized_aver.json       Transfer: mean over the 6 few-shot pairs
"""
import sys
import os
import json
from dataclasses import dataclass
from typing import Annotated, Optional
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.benchmarking.profiling import Profiler
from mani_skill.utils.visualization.misc import images_to_video, tile_images
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
import mani_skill.examples.benchmarking.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from gymnasium.vector.async_vector_env import AsyncVectorEnv

# service.py lives with the rest of MOTIF's serving code.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "experiment", "eval"))
from service import ExternalRobotInferenceClient as _BaseInferenceClient


class ExternalRobotInferenceClient(_BaseInferenceClient):
    """Adds the set_robot_uid endpoint.

    The author-provided service.py predates MOTIF's multi-embodiment server: its
    client exposes only get_action. MOTIF's own experiment/eval/robot.py adds
    set_robot_uid, which the server requires before the first query so it can pick
    the embodiment projector and transforms. Importing MOTIF's copy here would drag
    the whole MOTIF package into the ManiSkill interpreter, so the one method is
    re-declared instead.
    """

    def set_robot_uid(self, tags: str):
        return self.call_endpoint("set_robot_uid", tags)

# gym robot_uids -> the uid MOTIF's EMBODIMENT_TAGS maps to a *_rel tag.
ROBOT_UID_TO_POLICY_UID = {
    "panda_wristcam": "panda_wristcam_rel",
    "xarm6_robotiq_wristcam": "xarm6_robotiq_wristcam_rel",
    "widowxai_wristcam": "widowxai_wristcam_rel",
}

TASKS = {
    "panda_wristcam": ["PushCube-v1", "PlaceSphere-v1", "PullCube-v1", "LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"],
    "xarm6_robotiq_wristcam": ["PushCube-v1", "PlaceSphere-v1", "PullCube-v1", "LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"],
    "widowxai_wristcam": ["PushCube-v1", "PlaceSphere-v1", "PullCube-v1", "LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"],
}

# The Few cells of the paper's interleaved allocation (Table 2a); their mean is
# the Transfer metric.
generalized_tasks = {
    "panda_wristcam": ["PushCube-v1", "PlaceSphere-v1"],
    "xarm6_robotiq_wristcam": ["PullCube-v1", "LiftPegUpright-v1"],
    "widowxai_wristcam": ["PickCube-v1", "StackCube-v1"],
}

# ADAPTED FOR THIS REPRODUCTION -- see eval_maniskill/dataset_spec.json.
# Each value below silently produces a plausible-but-wrong success rate if it
# disagrees with how demo_data_processed was built, so all of it is read from the
# spec file rather than hardcoded here.
_SPEC = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "dataset_spec.json")))
TASK_INSTRUCTIONS = _SPEC["task_instructions"]
GRIPPER_QPOS_COLS = _SPEC["gripper_qpos_cols"]
GRIPPER_RANGE = _SPEC["gripper_range"]
ACTION_GRIPPER_DIMS = _SPEC["action_gripper_dims"]


def _robot_key(robot_uids: str) -> str:
    for k in GRIPPER_QPOS_COLS:
        if robot_uids.startswith(k):
            return k
    raise KeyError(f"no dataset spec entry for {robot_uids}")


def _gripper_state(robot_uids: str, qpos) -> np.ndarray:
    """Reproduce maniskill_to_lerobot.py's gripper channel exactly: the driven
    joints from dataset_spec.json, min-max normalised to [0,1] per robot, matching
    the statistics the server's normalisation was fitted on."""
    rk = _robot_key(robot_uids)
    lo, hi = GRIPPER_RANGE[rk]
    raw = float(np.asarray(qpos)[GRIPPER_QPOS_COLS[rk]].sum())
    return np.array([np.clip((raw - lo) / (hi - lo), 0.0, 1.0)], dtype=np.float32)

STEP_LENGTHS = {
    "PickCube-v1": 500,
    "PullCube-v1": 500,
    "PushCube-v1": 500,
    "StackCube-v1": 800,
    "PullCubeTool-v1": 800,
    "PlaceSphere-v1": 500,
    "LiftPegUpright-v1": 700,
}


class ActionQueue:
    """Holds one predicted chunk, handing out one action per env step."""

    def __init__(self):
        self.queue = deque()

    def put(self, chunk):
        for i in range(chunk.shape[0]):
            self.queue.append(chunk[i])

    def get(self):
        return self.queue.popleft()

    def is_empty(self):
        return len(self.queue) == 0

    def clear(self):
        self.queue.clear()


@dataclass
class EvalConfig:
    host: str = "0.0.0.0"
    port: int = 5555
    url: Optional[str] = None
    resize_size: int = 256
    """Render at the resolution the TRAINING videos were written at (256). The
    transform stack still crops and resizes to 224 as in training; rendering straight
    to 224 skips VideoCrop and shifts the input distribution."""
    replan_steps: int = 5
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_ee_delta_pose"
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    cpu_sim: bool = True
    seed: int = 0
    control_freq: Optional[int] = 20   # ManiSkill default our demos were recorded at
    sim_freq: Optional[int] = 100      # ManiSkill default our demos were recorded at
    render_mode: str = "rgb_array"
    save_video: bool = False
    save_path: str = "./eval_result"
    shader: str = "default"
    num_per_task: int = 50
    step_budget: Optional[int] = None
    seed_base: int = 2025
    """Episode seeds are seed_base..seed_base+num_per_task-1; a disjoint base
    measures eval-seed variance on the same checkpoint."""
    """Override the per-task step budget for every task. OURS: no source gives an
    evaluation horizon. `Johnathan218/ManiSkill` registers these envs with
    max_episode_steps=50, but zero of our 830 motion-planned demonstrations finish
    within 50 steps (medians 76-206) and measured at 50 the policy scores 12% on
    Panda/PushCube against 82% at 500 -- a policy cannot exceed its expert, so 50
    cannot be the horizon behind Table 1. The 500/700/800 in STEP_LENGTHS is sized
    to comfortably exceed the longest demonstration of each task."""
    robot: Optional[str] = None
    """Evaluate only this gym robot_uids. Used to sharde the 18 pairs across processes."""
    task: Optional[str] = None
    """Evaluate only this env id. Used to shard the 18 pairs across processes."""


def main(args: EvalConfig):
    os.makedirs(args.save_path, exist_ok=True)
    profiler = Profiler(output_format="stdout")
    num_envs = args.num_envs
    sim_config = dict()
    if args.control_freq:
        sim_config["control_freq"] = args.control_freq
    if args.sim_freq:
        sim_config["sim_freq"] = args.sim_freq

    policy_client = ExternalRobotInferenceClient(host=args.host, port=args.port)

    # Rendered at the resolution the training videos were written at, so the
    # policy is not fed upscaled frames.
    sensor_configs = dict(
        shader_pack=args.shader, width=args.resize_size, height=args.resize_size
    )

    for robot_uids, tasks in TASKS.items():
        if args.robot is not None and robot_uids != args.robot:
            continue
        # set_robot_uid is global state on the server, so a shard must own a
        # server that no other embodiment is talking to concurrently.
        policy_client.set_robot_uid(ROBOT_UID_TO_POLICY_UID[robot_uids])
        total_successes = 0.0
        success_dict = {}
        for env_id in tasks:
            if args.task is not None and env_id != args.task:
                continue
            def make_env():
                def _init():
                    env = gym.make(
                        env_id,
                        obs_mode=args.obs_mode,
                        sim_config=sim_config,
                        robot_uids=robot_uids,
                        sensor_configs=sensor_configs,
                        human_render_camera_configs=dict(shader_pack=args.shader),
                        viewer_camera_configs=dict(shader_pack=args.shader),
                        render_mode=args.render_mode,
                        control_mode=args.control_mode,
                    )
                    return CPUGymWrapper(env)
                return _init

            if not args.cpu_sim:
                env = gym.make(
                    env_id,
                    num_envs=num_envs,
                    obs_mode=args.obs_mode,
                    robot_uids=robot_uids,
                    sensor_configs=sensor_configs,
                    human_render_camera_configs=dict(shader_pack=args.shader),
                    viewer_camera_configs=dict(shader_pack=args.shader),
                    render_mode=args.render_mode,
                    control_mode=args.control_mode,
                    sim_config=sim_config,
                )
                if isinstance(env.action_space, gym.spaces.Dict):
                    env = FlattenActionSpaceWrapper(env)
                base_env: BaseEnv = env.unwrapped
            else:
                env = (
                    AsyncVectorEnv([make_env() for _ in range(num_envs)])
                    if args.num_envs > 1
                    else make_env()()
                )
                base_env = make_env()().unwrapped

            base_env.print_sim_details()

            task_successes = 0.0
            q = ActionQueue()
            for seed in tqdm(range(args.num_per_task), desc=f"{robot_uids}/{env_id}"):
                images = []
                q.clear()
                terminated = False
                with torch.inference_mode():
                    env.reset(seed=seed + args.seed_base)
                    # The warmup step drew from an unseeded action-space RNG. The
                    # reset below should erase its effect, but seed it anyway so
                    # nothing in the episode depends on process-global state.
                    env.action_space.seed(seed + args.seed_base)
                    env.step(env.action_space.sample())  # warmup step
                    obs, info = env.reset(seed=seed + args.seed_base)
                    if args.save_video:
                        images.append(np.expand_dims(env.render(), axis=0))
                    task_description = TASK_INSTRUCTIONS[env_id]
                    step_length = args.step_budget or STEP_LENGTHS[env_id]

                    with profiler.profile("env.step", total_steps=step_length, num_envs=num_envs):
                        for _ in range(step_length):
                            if q.is_empty():
                                # State exactly as the training rows were built:
                                # tcp pose (xyz + wxyz) plus the last gripper joint.
                                state = np.concatenate(
                                    (obs["extra"]["tcp_pose"],
                                     _gripper_state(robot_uids, obs["agent"]["qpos"]))
                                ).astype(np.float32)
                                img = obs["sensor_data"]["third_view_camera"]["rgb"]
                                wrist_img = obs["sensor_data"]["hand_camera"]["rgb"]

                                # Send the state split by modality, never as a bare
                                # "state" key: ConcatTransform groups on the dot, and
                                # an unsplit key silently skips min-max normalization.
                                # Slices follow modality.json.
                                element = {
                                    "video.image": np.asarray(img, dtype=np.uint8)[None],
                                    "video.wrist_image": np.asarray(wrist_img, dtype=np.uint8)[None],
                                    "state.position": state[None, 0:3],
                                    "state.rotation": state[None, 3:7],
                                    "state.gripper": state[None, 7:8],
                                    "annotation.human.task_description": [task_description],
                                }
                                chunk = policy_client.get_action(element)
                                # widowxai predicts TWO gripper channels (independent
                                # carriages, modality.json gripper 6:8); panda/xarm6 one.
                                g = np.asarray(chunk["action.gripper"])
                                ngrip = ACTION_GRIPPER_DIMS[_robot_key(robot_uids)]
                                g = g.reshape(g.shape[0], -1)
                                assert g.shape[1] == ngrip, (
                                    f"{robot_uids}: server returned {g.shape[1]} gripper "
                                    f"channel(s), dataset spec says {ngrip}")
                                pred = np.concatenate(
                                    [
                                        np.asarray(chunk["action.position"]),
                                        np.asarray(chunk["action.rotation"]),
                                        g,
                                    ],
                                    axis=1,
                                )
                                q.put(pred[: args.replan_steps])

                            action = q.get()
                            obs, rew, terminated, truncated, info = env.step(action)
                            if args.save_video:
                                images.append(np.expand_dims(env.render(), axis=0))
                            if terminated:
                                task_successes += 1
                                total_successes += 1
                                break
                    profiler.log_stats("env.step")

                    if args.save_video:
                        images = [tile_images(rgbs, nrows=1) for rgbs in images]
                        images_to_video(
                            images,
                            output_dir=args.save_path,
                            video_name=f"{robot_uids}-{env_id}-{seed}--success={terminated}",
                            fps=30,
                        )
                        del images
            env.close()
            rate = task_successes / args.num_per_task
            print(f"Task Success Rate: {rate}")
            success_dict[env_id] = rate
            # One file per pair, so shards never write the same path and a
            # partially finished sweep can still be summarized.
            with open(f"{args.save_path}/{robot_uids}__{env_id}.json", "w") as f:
                json.dump(
                    {
                        "robot": robot_uids,
                        "task": env_id,
                        "success_rate": rate,
                        "successes": task_successes,
                        "num_per_task": args.num_per_task,
                        "few_shot": env_id in generalized_tasks[robot_uids],
                    },
                    f,
                    indent=4,
                )
        if success_dict:
            print(f"Robot mean: {sum(success_dict.values()) / len(success_dict)}")


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
