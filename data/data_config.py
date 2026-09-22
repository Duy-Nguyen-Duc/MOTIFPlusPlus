# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from data.dataset import ModalityConfig
from data.transform.base import ComposedModalityTransform, ModalityTransform
from data.transform.concat import ConcatTransform
from data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from model.transforms import GR00TTransform


@dataclass
class BaseDataConfig(ABC):
    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


#####################################################################################
# helper functions
#####################################################################################


def import_external_data_config(data_config_str: str) -> Optional[BaseDataConfig]:
    """
    Import and instantiate an external data configuration class.

    Format: "module_path:ClassName" (e.g., "my_configs:RobotConfig")
    Supports nested modules like "package.submodule:ClassName"
    """
    if ":" not in data_config_str:
        return None

    import importlib
    import os
    import sys
    from pathlib import Path

    # Add current working directory to Python path
    current_dir = str(Path(os.getcwd()).absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        module_path, class_name = data_config_str.split(":", 1)
        if not module_path or not class_name:
            raise ValueError(f"Invalid format: '{data_config_str}'. Use 'module:ClassName'")

        print(f"Loading external config: {module_path}.{class_name}")

        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            available = [
                n
                for n in dir(module)
                if not n.startswith("_") and isinstance(getattr(module, n), type)
            ]
            raise AttributeError(
                f"Class '{class_name}' not found in '{module_path}'. Available: {available}"
            )

        # assert if the class has 'transform' and 'modality_config' methods
        if not hasattr(getattr(module, class_name), "transform"):
            raise AttributeError(f"Class '{class_name}' does not have a 'transform' method")
        if not hasattr(getattr(module, class_name), "modality_config"):
            raise AttributeError(f"Class '{class_name}' does not have a 'modality_config' method")

        return getattr(module, class_name)()

    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        print(f"Config loading failed: {e}")
        print("Example: my_configs:MyConfig, package.submodule:ClassName")
        raise


def load_data_config(data_config_str: str) -> BaseDataConfig:
    """
    Get a data config class from a string.
    >>> load_data_config("so100")
    >>> get_data_config("dir.subdir.my_configs:RobotConfig")
    """
    if data_config_str in DATA_CONFIG_MAP:
        return DATA_CONFIG_MAP[data_config_str]
    data_config_cls = import_external_data_config(data_config_str)
    if data_config_cls is not None:
        return data_config_cls
    # Yellow warning color
    yellow = "\033[93m"
    reset = "\033[0m"
    raise ValueError(
        f"{yellow}Invalid data_config '{data_config_str}'. "
        f"Available options: {list(DATA_CONFIG_MAP.keys())}, "
        f"or use 'module:ClassName' for external configs{reset}"
    )


###########################################################################################
class PANDADataConfig(BaseDataConfig):
    video_keys = ["video.image","video.wrist_image"]
    # video_keys = ["video.image"]
    state_keys = ["state.position","state.rotation","state.gripper"]
    action_keys = ["action.position","action.rotation","action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class PANDAWithRELDataConfig(BaseDataConfig):
    video_keys = ["video.image","video.wrist_image"]
    # video_keys = ["video.image"]
    state_keys = ["state.position","state.rotation","state.gripper"]
    action_keys = ["action.position","action.rotation","action.gripper","action.rel_state_position","action.rel_state_rotation","action.rel_state_gripper","action.progress"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys if key != "action.progress"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=10,
                max_action_dim=10,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    
class PiperJointDataConfig(BaseDataConfig):
    video_keys = ["video.image","video.wrist_image_left"]
    state_keys = ["state.joint", "state.gripper"]
    action_keys = ["action.joint", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=10,
                max_action_dim=18,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms) 

class PiperEndposeDataConfig(BaseDataConfig):
    video_keys = ["video.image","video.wrist_image_left"]
    state_keys = ["state.position", "state.rotation", "state.gripper"]
    action_keys = ["action.position", "action.rotation", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.position": "min_max",
                    "state.gripper": "min_max",
                },
                target_rotations={
                    "state.rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.position": "min_max",
                    "action.gripper": "min_max",
                },
                target_rotations={"action.rotation": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
###########################################################################################

class PANDAWithRELObjDataConfig(PANDAWithRELDataConfig):
    """RELData plus the gripper->object offset (action.ee_to_obj).

    Opt-in: only embodiments registered as *_rel_obj load the extra dims, and
    they exist only in the object-augmented dataset built by
    data/process/append_object_action_dims.py. Every other arm is unaffected.
    """
    action_keys = PANDAWithRELDataConfig.action_keys + ["action.ee_to_obj"]


class PANDAWithRELPairDataConfig(PANDAWithRELDataConfig):
    """Two observation timesteps instead of one, for the DINO-target stage 1.

    That stage defines a motif as the change between O_t and O_{t+k}, so the
    batch has to carry both frames; every other config in this file returns
    `observation_indices = [0]` and images shaped [B, v, t=1, ...].

    Only the VIDEO modality takes the pair. State and language stay at [0]:
    stage 1 reads neither at t=1, `batch["language"]` is consumed as one string
    per sample by the InfoNCE term and the conditional discriminator, and a
    second index would silently turn both into length-2 sequences.

    PAIR_DELTA is in dataset steps. Demos were recorded at 20 Hz, so 20 steps is
    the ~1 second UniVLA calibrates its latent actions to. Near the end of an
    episode the loader clamps, which makes the target the final frame -- i.e. a
    smaller observed change, which is correct.
    """
    PAIR_DELTA = 20
    observation_indices = [0]      # state / language / GR00TTransform horizon

    def modality_config(self) -> dict:
        cfg = super().modality_config()
        cfg["video"] = ModalityConfig(
            delta_indices=[0, self.PAIR_DELTA],
            modality_keys=self.video_keys,
        )
        return cfg


class PANDAWithRELPairBaseDataConfig(PANDAWithRELPairDataConfig):
    """Frame pair, BASE CAMERA ONLY -- for the grounded stage 1.

    That stage's visual FDM reads `images[:, view_index, 0]` and `[:, view_index, 1]`
    with view_index=0, and nothing else in stage 1 touches images at all. With
    both views the wrist camera is read from the frame cache, cropped, resized,
    colour-jittered, collated and copied to the GPU on every sample, and then
    never indexed.

    Dropping it halves the image bytes per sample (2.4 -> 1.2 MB), which halves
    the /dev/shm footprint, the pfss read volume and the augmentation CPU. It
    changes nothing the model sees. Stage 2 keeps both views -- its perceiver
    consumes them -- via dataset_maniskill_contact_s2.yaml.
    """
    video_keys = ["video.image"]


DATA_CONFIG_MAP = {
    'xarm6': PANDADataConfig(),
    'panda': PANDADataConfig(),
    'xarm7': PANDADataConfig(),
    'widowxai': PANDADataConfig(),
    'xarm6_rel': PANDAWithRELDataConfig(),
    'panda_rel': PANDAWithRELDataConfig(),
    'xarm7_rel': PANDAWithRELDataConfig(),
    'widowxai_rel': PANDAWithRELDataConfig(),
    'panda_rel_obj': PANDAWithRELObjDataConfig(),
    'xarm6_rel_obj': PANDAWithRELObjDataConfig(),
    'widowxai_rel_obj': PANDAWithRELObjDataConfig(),
    'panda_rel_pair': PANDAWithRELPairDataConfig(),
    'xarm6_rel_pair': PANDAWithRELPairDataConfig(),
    'widowxai_rel_pair': PANDAWithRELPairDataConfig(),
    'panda_rel_pair_base': PANDAWithRELPairBaseDataConfig(),
    'xarm6_rel_pair_base': PANDAWithRELPairBaseDataConfig(),
    'widowxai_rel_pair_base': PANDAWithRELPairBaseDataConfig(),
    'piper_joint': PiperJointDataConfig(),
    'piper_endpose': PiperEndposeDataConfig(),

    "arx5_real": PANDADataConfig(),
    'piper_real': PANDADataConfig(),
    'xarm7_real': PANDADataConfig(),
    'arx5_real_rel': PANDAWithRELDataConfig(),
    'piper_real_rel': PANDAWithRELDataConfig(),
    'xarm7_real_rel': PANDAWithRELDataConfig(),
}
