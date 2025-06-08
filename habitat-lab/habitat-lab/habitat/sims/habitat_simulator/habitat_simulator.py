#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import magnum as mn
import numpy as np
from gym import spaces
from gym.spaces.box import Box
from omegaconf import DictConfig
from os import path as osp
import random
import multiprocessing

import habitat_sim
import habitat.articulated_agents.humanoids.kinematic_humanoid as kinematic_humanoid
from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)
from habitat.articulated_agent_controllers import Humanoid_OCRA

from habitat.config.default import get_agent_config
from habitat.core.batch_rendering.env_batch_renderer_constants import (
    KEYFRAME_OBSERVATION_KEY,
    KEYFRAME_SENSOR_PREFIX,
)
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from habitat.core.spaces import Space

from collections import deque
import cv2
import numpy as np
from habitat.utils.visualizations import maps

def get_obstacle(sim,height):
    map_array = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=0.01
    )

    boder = sim.pathfinder.get_bounds()
    boder = [boder[0][0],boder[0][2],boder[1][0],boder[1][2]]

    contours, hierarchy = cv2.findContours(map_array.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    outer_boundary = None
    inner_boundaries = []

    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            outer_boundary = contour.squeeze()
        else:  
            inner_boundaries.append(contour.squeeze())

    p1 = outer_boundary[outer_boundary[:, 0].argmin()]
    p2 = outer_boundary[outer_boundary[:, 0].argmax()]

    idx_p1 = np.where((outer_boundary[:, 0] == p1[0]) & (outer_boundary[:, 1] == p1[1]))[0][0]
    idx_p2 = np.where((outer_boundary[:, 0] == p2[0]) & (outer_boundary[:, 1] == p2[1]))[0][0]

    if idx_p1 < idx_p2:
        new_outer_boundary1 = np.concatenate((outer_boundary[idx_p1:idx_p2+1], 
            [[map_array.shape[1], map_array.shape[0]], [0, map_array.shape[0]], [0, 0], [p1[0],p1[1]]]))
        new_outer_boundary2 = np.concatenate((outer_boundary[idx_p2:],outer_boundary[:idx_p1+1], 
            [[0, 0], [map_array.shape[1], 0], [map_array.shape[1], map_array.shape[0]], [p2[0],p2[1]]]))
    else:
        new_outer_boundary1 = np.concatenate((outer_boundary[idx_p2:idx_p1+1], 
            [[map_array.shape[1], map_array.shape[0]], [0, map_array.shape[0]], [0, 0], [p2[0],p2[1]]]))
        new_outer_boundary2 = np.concatenate((outer_boundary[idx_p1:],outer_boundary[:idx_p2+1],  
            [[0, 0], [map_array.shape[1], 0], [map_array.shape[1], map_array.shape[0]], [p1[0],p1[1]]]))

    map_with_obstacles = np.ones(map_array.shape, dtype=np.uint8) * 255 

    inner_boundaries.append(new_outer_boundary1.reshape((-1,  2)).astype(np.int32))
    inner_boundaries.append(new_outer_boundary2.reshape((-1, 2)).astype(np.int32))
    obstacles = []
    try:
        for boundary in inner_boundaries:
            cv2.fillPoly(map_with_obstacles, [boundary], color=0)

        for boundary in inner_boundaries:
            real_boundary = []
            for i in range(boundary.shape[0]):
                x_real = boder[0] + (boundary[i][0] / map_array.shape[1]) * (boder[2] - boder[0])
                y_real = boder[1] + (boundary[i][1] / map_array.shape[0]) * (boder[3] - boder[1])
                real_boundary.append([x_real, y_real])
            obstacles.append(real_boundary)
    except Exception as e:
        print("error for obstacle:",e)
    # cv2.imwrite("Map.png", map_with_obstacles)
    return obstacles


class HumanPoseContrioller:
    def __init__(self,sim,human,motion_path,direction=mn.Vector3(0,0,1),motion_fps=10):
        self.walk_flag = False
        motion_list = [
            "data/action/13_23_stageii.pkl",  # 	sweep floor 
            "data/action/79_70_stageii.pkl",  # 	laughing
            "data/action/05_02_stageii.pkl"    # dance
        ]
        motion_path = random.choice(motion_list)
        self.human = human
        self.human.walk_flag = self.walk_flag
        self.sim = sim
        # We define here humanoid controller
        self.humanoid_controller = HumanoidSeqPoseController(motion_path,motion_fps=motion_fps)
        base_trans = self.human.base_transformation
        self.humanoid_controller.reset(base_trans)
        self.humanoid_controller.set_direction(direction) # 初始朝向，决定动作方向
    
    def move(self):
        self.humanoid_controller.calculate_pose()
        self.humanoid_controller.next_pose(True)

        # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
        new_pose = self.humanoid_controller.get_pose()

        new_joints = new_pose[:-16]
        new_pos_transform_base = new_pose[-16:]
        new_pos_transform_offset = new_pose[-32:-16]

        # When the array is all 0, this indicates we are not setting
        # the human joint
        if np.array(new_pos_transform_offset).sum() != 0:
            vecs_base = [
                mn.Vector4(new_pos_transform_base[i * 4 : (i + 1) * 4])
                for i in range(4)
            ]
            vecs_offset = [
                mn.Vector4(new_pos_transform_offset[i * 4 : (i + 1) * 4])
                for i in range(4)
            ]
            new_transform_offset = mn.Matrix4(*vecs_offset)
            new_transform_base = mn.Matrix4(*vecs_base)
            self.human.set_joint_transform(
                new_joints, new_transform_offset, new_transform_base
            )


# @title Continuous Path Follower Example { display-mode: "form" }
# @markdown A python Class to provide waypoints along a path given agent states
class ContinuousPathFollowerHuman:
    def __init__(self, sim, path, human_node, waypoint_threshold):
        self._sim = sim
        self._points = path.points[:]
        assert len(self._points) > 0
        self._length = path.geodesic_distance
        self._human_node = human_node
        self._threshold = waypoint_threshold
        self._step_size = 0.01
        self.progress = 0  # geodesic distance -> [0,1]
        self.waypoint = path.points[0]

        # setup progress waypoints
        _point_progress = [0]
        _segment_tangents = []
        _length = self._length
        for ix, point in enumerate(self._points):
            if ix > 0:
                segment = point - self._points[ix - 1]
                segment_length = np.linalg.norm(segment)
                segment_tangent = segment / segment_length
                _point_progress.append(
                    segment_length / _length + _point_progress[ix - 1]
                )
                # t-1 -> t
                _segment_tangents.append(segment_tangent)
        self._point_progress = _point_progress
        self._segment_tangents = _segment_tangents
        # final tangent is duplicated
        self._segment_tangents.append(self._segment_tangents[-1])

    def pos_at(self, progress):
        if progress <= 0:
            return self._points[0]
        elif progress >= 1.0:
            return self._points[-1]

        path_ix = 0
        for ix, prog in enumerate(self._point_progress):
            if prog > progress:
                path_ix = ix
                break

        segment_distance = self._length * (progress - self._point_progress[path_ix - 1])
        return (
            self._points[path_ix - 1]
            + self._segment_tangents[path_ix - 1] * segment_distance
        )

    def update_waypoint(self):
        if self.progress < 1.0:
            wp_disp = self.waypoint - self._human_node.base_pos
            wp_dist = np.linalg.norm(wp_disp)
            node_pos = self._human_node.base_pos
            step_size = self._step_size
            threshold = self._threshold
            while wp_dist < threshold:
                self.progress += step_size
                self.waypoint = self.pos_at(self.progress)
                if self.progress >= 1.0:
                    break
                wp_disp = self.waypoint - node_pos
                wp_dist = np.linalg.norm(wp_disp)


class HumanWalkController:
    def __init__(self,sim,human,human_config,motion_fps=10) -> None:
        self.walk_flag = True
        self.path = habitat_sim.ShortestPath()
        self.sim = sim
        self.human = human
        self.human.walk_flag = self.walk_flag
        self.humanoid_controller = HumanoidRearrangeController(human_config["motion_data_path"],motion_fps=motion_fps)
        self.humanoid_controller.reset(self.human.base_transformation)
        self.continuous_path_follower = None
        self.last_process = -1
        self.queue_len = 6
        
        self.queue = deque()    # 理论可以直接 self.queue = deque()，后面试试修改
        # self.get_random_path()
        self.set_gxy()
    
    # 函数用于添加新数据并判断队列中的元素是否相同
    def update_queue(self,data):
        # 将新数据添加到队列的右侧
        self.queue.append(data)
        
        # 如果队列长度超过self.queue_len，从左侧移除一个元素
        if len(self.queue) > self.queue_len :
            self.queue.popleft()
        
        # 检查队列中的所有元素是否相同
        all_equal = all(x == self.queue[0] for x in self.queue)
        return all_equal

    def human_move(self,target_pos):
        pose_diff = target_pos - self.human.base_pos
        self.humanoid_controller.calculate_walk_pose(pose_diff)
        new_pose = self.humanoid_controller.get_pose()

        new_joints = new_pose[:-16]
        new_pos_transform_base = new_pose[-16:]
        new_pos_transform_offset = new_pose[-32:-16]

        # When the array is all 0, this indicates we are not setting
        # the human joint
        if np.array(new_pos_transform_offset).sum() != 0:
            vecs_base = [
                mn.Vector4(new_pos_transform_base[i * 4 : (i + 1) * 4])
                for i in range(4)
            ]
            vecs_offset = [
                mn.Vector4(new_pos_transform_offset[i * 4 : (i + 1) * 4])
                for i in range(4)
            ]
            new_transform_offset = mn.Matrix4(*vecs_offset)
            new_transform_base = mn.Matrix4(*vecs_base)
            self.human.set_joint_transform(
                new_joints, new_transform_offset, new_transform_base
            )

    def human_move_vxy(self,vx,vy):
        self.humanoid_controller.calculate_walk_pose_vxy(vx=vx,vy=vy)
        new_pose = self.humanoid_controller.get_pose()
        self.human.set_vxy(self.humanoid_controller.vx,self.humanoid_controller.vy)

        new_joints = new_pose[:-16]
        new_pos_transform_base = new_pose[-16:]
        new_pos_transform_offset = new_pose[-32:-16]

        # When the array is all 0, this indicates we are not setting
        # the human joint
        if np.array(new_pos_transform_offset).sum() != 0:
            vecs_base = [
                mn.Vector4(new_pos_transform_base[i * 4 : (i + 1) * 4])
                for i in range(4)
            ]
            vecs_offset = [
                mn.Vector4(new_pos_transform_offset[i * 4 : (i + 1) * 4])
                for i in range(4)
            ]
            new_transform_offset = mn.Matrix4(*vecs_offset)
            new_transform_base = mn.Matrix4(*vecs_base)
            self.human.set_joint_transform(
                new_joints, new_transform_offset, new_transform_base
            )

    def get_random_path(self):
        found_path = False
        
        while not found_path:
            self.path.requested_start = self.human.base_pos
            requested_end = self.sim.pathfinder.get_random_navigable_point()
            requested_end = self.sim.pathfinder.snap_point(
                requested_end
            )
            requested_end[1] = self.human.base_pos[1]
            self.path.requested_end = requested_end

            found_path = self.sim.pathfinder.find_path(self.path)
        self.continuous_path_follower = ContinuousPathFollowerHuman(
            self.sim, self.path, self.human, waypoint_threshold=0.4
        )
    
    def set_gxy(self):
        try_times = 0
        found_path = False
        distance = 10
        
        while (not found_path):   # 找一个至少3m远可导航的点
            self.path.requested_start = self.human.base_pos
            requested_end = self.sim.pathfinder.get_random_navigable_point()
            requested_end = self.sim.pathfinder.snap_point(
                requested_end
            )
            # if self.human.px is not None:
            #     distance = (requested_end[0] - self.human.px)**2 + (requested_end[2] - self.human.gx)**2
            requested_end[1] = self.human.base_pos[1]
            self.path.requested_end = requested_end
            # self.path.requested_end = self.human.base_pos + mn.Vector3(0.0, 0.0, 10.51752)

            found_path = self.sim.pathfinder.find_path(self.path)  
            try_times += 1
            if try_times>100:
                break
        self.human.gx = requested_end[0]
        self.human.gy = requested_end[2]
        # if self.human.px:
        #     print("self.human.goal:",self.human.gx, self.human.gy, " now:", self.human.px, self.human.py)
        
        # return requested_end[0], requested_end[2]
    
    def run_follow_orca(self):
        distance = (self.human.gx - self.human.px)**2 + (self.human.gy - self.human.gx)**2
        distance = round(distance, 2) # keep 2 小数位
        collided = self.update_queue(distance)
        if collided or distance < 0.5:
            self.set_gxy()
            
    def run_follow_waypoint(self):
        if (self.continuous_path_follower.progress > 0.75):
            self.get_random_path()
        # if self.last_process == self.continuous_path_follower.progress:  # reset
            
        self.last_process = self.continuous_path_follower.progress
        self.continuous_path_follower.update_waypoint()
        end_pos = self.sim.step_filter(
            self.human.base_pos, mn.Vector3(self.continuous_path_follower.waypoint)
        )
        collided = self.update_queue(self.continuous_path_follower.progress)
        if collided:
            self.get_random_path()
        else:
            self.human_move(end_pos)

if TYPE_CHECKING:
    from torch import Tensor


def overwrite_config(
    config_from: DictConfig,
    config_to: Any,
    ignore_keys: Optional[Set[str]] = None,
    trans_dict: Optional[Dict[str, Callable]] = None,
) -> None:
    r"""Takes Habitat Lab config and Habitat-Sim config structures. Overwrites
    Habitat-Sim config with Habitat Lab values, where a field name is present
    in lowercase. Mostly used to avoid :ref:`sim_cfg.field = hapi_cfg.FIELD`
    code.
    Args:
        config_from: Habitat Lab config node.
        config_to: Habitat-Sim config structure.
        ignore_keys: Optional set of keys to ignore in config_to
        trans_dict: A Dict of str, callable which can be used on any value that has a matching key if not in ignore_keys.
    """

    def if_config_to_lower(config):
        if isinstance(config, DictConfig):
            return {
                key.lower(): val
                for key, val in config.items()
                if isinstance(key, str)
            }
        else:
            return config

    for attr, value in config_from.items():
        assert isinstance(attr, str)
        low_attr = attr.lower()
        if ignore_keys is None or low_attr not in ignore_keys:
            if hasattr(config_to, low_attr):
                if trans_dict is not None and low_attr in trans_dict:
                    setattr(config_to, low_attr, trans_dict[low_attr](value))
                else:
                    setattr(config_to, low_attr, if_config_to_lower(value))
            else:
                raise NameError(
                    f"""{low_attr} is not found on habitat_sim but is found on habitat_lab config.
                    It's also not in the list of keys to ignore: {ignore_keys}
                    Did you make a typo in the config?
                    If not the version of Habitat Sim may not be compatible with Habitat Lab version: {config_from}
                    """
                )


class HabitatSimSensor:
    sim_sensor_type: habitat_sim.SensorType
    _get_default_spec = Callable[..., habitat_sim.sensor.SensorSpec]
    _config_ignore_keys = {"height", "type", "width"}


@registry.register_sensor
class HabitatSimRGBSensor(RGBSensor, HabitatSimSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    sim_sensor_type = habitat_sim.SensorType.COLOR

    RGBSENSOR_DIMENSION = 3

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(
                self.config.height,
                self.config.width,
                self.RGBSENSOR_DIMENSION,
            ),
            dtype=np.uint8,
        )

    def get_observation(
        self, sim_obs: Dict[str, Union[np.ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, : self.RGBSENSOR_DIMENSION]  # type: ignore[index]
        return obs


@registry.register_sensor
class HabitatSimDepthSensor(DepthSensor, HabitatSimSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    _config_ignore_keys = {
        "max_depth",
        "min_depth",
        "normalize_depth",
    }.union(HabitatSimSensor._config_ignore_keys)
    sim_sensor_type = habitat_sim.SensorType.DEPTH

    min_depth_value: float
    max_depth_value: float

    def __init__(self, config: DictConfig) -> None:
        self.min_depth_value = config.min_depth
        self.max_depth_value = config.max_depth
        self.normalize_depth = config.normalize_depth
        if self.normalize_depth:
            self._obs_shape = spaces.Box(
                low=0,
                high=1,
                shape=(config.height, config.width, 1),
                dtype=np.float32,
            )
        else:
            self._obs_shape = spaces.Box(
                low=self.min_depth_value,
                high=self.max_depth_value,
                shape=(config.height, config.width, 1),
                dtype=np.float32,
            )

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        return self._obs_shape

    def get_observation(
        self, sim_obs: Dict[str, Union[np.ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, self.min_depth_value, self.max_depth_value)

            obs = np.expand_dims(
                obs, axis=2
            )  # make depth observation a 3D array
        else:
            obs = obs.clamp(self.min_depth_value, self.max_depth_value)  # type: ignore[attr-defined, unreachable]

            obs = obs.unsqueeze(-1)  # type: ignore[attr-defined]

        if self.normalize_depth:
            # normalize depth observation to [0, 1]
            obs = (obs - self.min_depth_value) / (
                self.max_depth_value - self.min_depth_value
            )

        return obs


@registry.register_sensor
class HabitatSimSemanticSensor(SemanticSensor, HabitatSimSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    sim_sensor_type = habitat_sim.SensorType.SEMANTIC

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(self.config.height, self.config.width, 1),
            dtype=np.int32,
        )

    def get_observation(
        self, sim_obs: Dict[str, Union[np.ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)
        # make semantic observation a 3D array
        if isinstance(obs, np.ndarray):
            obs = obs[..., None].astype(np.int32)
        else:
            obs = obs[..., None]
        return obs


# TODO Sensor Hierarchy needs to be redone here. These should not subclass camera sensors
@registry.register_sensor
class HabitatSimEquirectangularRGBSensor(HabitatSimRGBSensor):
    _get_default_spec = habitat_sim.EquirectangularSensorSpec


@registry.register_sensor
class HabitatSimEquirectangularDepthSensor(HabitatSimDepthSensor):
    _get_default_spec = habitat_sim.EquirectangularSensorSpec


@registry.register_sensor
class HabitatSimEquirectangularSemanticSensor(HabitatSimSemanticSensor):
    _get_default_spec = habitat_sim.EquirectangularSensorSpec


@registry.register_sensor
class HabitatSimFisheyeRGBSensor(HabitatSimRGBSensor):
    _get_default_spec = habitat_sim.FisheyeSensorDoubleSphereSpec


@registry.register_sensor
class HabitatSimFisheyeDepthSensor(HabitatSimDepthSensor):
    _get_default_spec = habitat_sim.FisheyeSensorDoubleSphereSpec


@registry.register_sensor
class HabitatSimFisheyeSemanticSensor(HabitatSimSemanticSensor):
    _get_default_spec = habitat_sim.FisheyeSensorDoubleSphereSpec


def check_sim_obs(
    obs: Union[np.ndarray, "Tensor", None], sensor: Sensor
) -> None:
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )

def print_scene_recur(scene, limit_output=10):
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None

import trimesh

def get_glb_model_size(glb_file_path):
    mesh = trimesh.load(glb_file_path)
    
    bounding_box = mesh.bounds
    
    size = bounding_box[1] - bounding_box[0]
    return size

@registry.register_simulator(name="Sim-v0")
class HabitatSim(habitat_sim.Simulator, Simulator):
    r"""Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """
    def make_human(self,humanoid_name,agent_pos,walk_flag=True,base_pos=None):
        if walk_flag:
            dis2obs = 0.2
        else:
            dis2obs = 0.5
        human_path = self.habitat_config.human_path
        humanoid_path = f"{human_path}/{humanoid_name}/{humanoid_name}.urdf"
        walk_pose_path = f"{human_path}/{humanoid_name}/{humanoid_name}_motion_data_smplx.pkl"
        humanoid_glb = f"{human_path}/{humanoid_name}/{humanoid_name}.glb"
        agent_config = DictConfig(
            {
                "articulated_agent_urdf": humanoid_path,
                "motion_data_path": walk_pose_path,
            }
        )
        if not osp.exists(humanoid_path):
            print(f"No humanoid file {humanoid_path}")
        kin_humanoid = kinematic_humanoid.KinematicHumanoid(agent_config, self)
        kin_humanoid.reconfigure()
        kin_humanoid.update()
        kin_humanoid.height = get_glb_model_size(humanoid_glb)[0]
        if base_pos is not None:
            kin_humanoid.base_pos = base_pos
        else:
            try_times = 0
            while 1:    # 防止放置过近
                base_pos = self.pathfinder.get_random_navigable_point()
                base_pos = self.pathfinder.snap_point(base_pos)
                base_pos[1] = agent_pos[1]
                if self.pathfinder.distance_to_closest_obstacle(base_pos) > dis2obs and\
                    (base_pos[0]-agent_pos[0])**2+(base_pos[2]-agent_pos[2])**2>1:
                    kin_humanoid.base_pos = base_pos
                    # break
                    # 防止人初始化得太集中,比如都在一个房间
                    flag_hu_ok = True
                    for other_human in self.human_list:
                        other_pos = other_human.base_pos
                        if (other_pos[0]-base_pos[0])**2+(other_pos[2]-base_pos[2])**2 < 16:
                            flag_hu_ok = False
                            continue
                    if flag_hu_ok:
                        break
                
                try_times +=1
                if try_times > 50:
                    # 删除人物模型
                    aom = self.get_articulated_object_manager()
                    aom.remove_object_by_handle(kin_humanoid.sim_obj.handle)
                    return None,None
        # 根据不同人物模型高度调整
        base_pos = kin_humanoid.base_pos
        base_pos[1] = base_pos[1]- 1.7 + kin_humanoid.height
        kin_humanoid.base_pos= base_pos
        return kin_humanoid,agent_config
    
    def remove_objects(self, aom):
        try:
            for handle, ao in aom.get_objects_by_handle_substring().items():
                obj_id = aom.get_object_id_by_handle(handle) 
                print("obj_id:",obj_id)
                if obj_id is not None:
                    aom.remove_object_by_id(obj_id)
        except Exception as e:
            print("remove human error!!! but we ignore this!!!")
            print(f"Exception: {e}")
    
    def init_human(self,agent_pos):
        obstacles = get_obstacle(self,agent_pos[1]) # 扫描障碍物
        # add the humanoid to the world via the wrapper
        aom = self.get_articulated_object_manager()

        try:
            for handle, ao in aom.get_objects_by_handle_substring().items():
                obj_id = aom.get_object_id_by_handle(handle) 
                if obj_id is not None:
                    aom.remove_object_by_id(obj_id)
        except:
            print("remove human error!!! but we ignore this!!!")
                
        self.human_list = []
        self.human_config_list = []
        type_list = self.habitat_config.human_list
        for _ in range(self.habitat_config.human_num):
            human_type = random.choice(type_list)
            num = random.random()
            if num < self.habitat_config.human_act_rate:   # 原地动作时
                kin_humanoid,human_config = self.make_human(human_type,agent_pos,walk_flag=False)
            else:
                kin_humanoid,human_config = self.make_human(human_type,agent_pos,walk_flag=True)
            if kin_humanoid is not None:
                self.human_list.append(kin_humanoid)
                self.human_config_list.append(human_config)
        
        # human controller
        self.Human_ocra = Humanoid_OCRA(obstacles)
        self.HumanControl_list = []
        for i in range(len(self.human_list)):
            if self.human_list[i].walk_flag:
                HumanControl = HumanWalkController(self,self.human_list[i],self.human_config_list[i],self.habitat_config.walk_fps)
            else:
                HumanControl = HumanPoseContrioller(self,self.human_list[i],self.human_config_list[i],self.habitat_config.pose_fps)
            self.HumanControl_list.append(HumanControl)

    def __init__(self, config: DictConfig) -> None:
        # 确保human生成的可复现性(加了居然会导致没办法找到路径，靠)
        # random.seed(config.seed)
        # np.random.seed(config.seed)
        self.collision_with_human = False
    
        self.habitat_config = config
        sim_sensors = []
        for agent_config in self.habitat_config.agents.values():
            for sensor_cfg in agent_config.sim_sensors.values():
                sensor_type = registry.get_sensor(sensor_cfg.type)

                assert (
                    sensor_type is not None
                ), "invalid sensor type {}".format(sensor_cfg.type)
                sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        super().__init__(self.sim_config)
        # load additional object paths specified by the dataset
        # TODO: Should this be moved elsewhere?
        obj_attr_mgr = self.get_object_template_manager()
        for path in self.habitat_config.additional_object_paths:
            obj_attr_mgr.load_configs(path)
        self._action_space = spaces.Discrete(
            len(
                self.sim_config.agents[
                    self.habitat_config.default_agent_id
                ].action_space
            )
        )
        self._prev_sim_obs: Optional[Observations] = None
        self.human_list = []
        # if self.habitat_config.enable_human == 2:
        #     pos_ini = self.get_agent_state().position
        #     pos_ini[1] = self.get_floor_height(pos_ini)
        #     self.init_human(pos_ini)

    def get_floor_height(self,pos_ini):
        h_list = []
        for a in [-4,-2,0,2,-4]:
            for b in [-4,-2,0,2,-4]:
                pos_nei = mn.Vector3(pos_ini[0]+a,pos_ini[1],pos_ini[2]+b)
                pos_nei = self.pathfinder.snap_point(pos_nei, island_index=-1)
                if not np.isnan(pos_nei[1]):
                    h_list.append(pos_nei[1])
        unique, counts = np.unique(np.array(h_list), return_counts=True)  
        max_index = np.argmax(counts)  
        return unique[max_index]  
    
    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.habitat_config.habitat_sim_v0,
            config_to=sim_config,
            # Ignore key as it gets propagated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        # sim_config.scene_dataset_config_file = (
        #     self.habitat_config.scene_dataset
        # )
        # sim_config.scene_dataset_config_file = (
        #     "/home/code/code/Hab3/V3_1/habitat-lab/PEANUT/habitat-challenge-data/data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
        # )
        sim_config.scene_dataset_config_file = (
            "/home/code/code/Hab3/V3_1/habitat-lab/PEANUT/habitat-challenge-data_v1/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        )
        sim_config.scene_id = self.habitat_config.scene
        lab_agent_config = get_agent_config(self.habitat_config)
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=lab_agent_config,
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "sim_sensors",
                "start_position",
                "start_rotation",
                "articulated_agent_urdf",
                "articulated_agent_type",
                "joint_start_noise",
                "joint_that_can_control",
                "motion_data_path",
                "ik_arm_urdf",
                "grasp_managers",
                "max_climb",
                "max_slope",
                "joint_start_override",
            },
        )

        # configure default navmesh parameters to match the configured agent
        if self.habitat_config.default_agent_navmesh:
            sim_config.navmesh_settings = habitat_sim.nav.NavMeshSettings()
            sim_config.navmesh_settings.set_defaults()
            sim_config.navmesh_settings.agent_radius = agent_config.radius
            sim_config.navmesh_settings.agent_height = agent_config.height
            sim_config.navmesh_settings.agent_max_climb = (
                lab_agent_config.max_climb
            )
            sim_config.navmesh_settings.agent_max_slope = (
                lab_agent_config.max_slope
            )
            sim_config.navmesh_settings.include_static_objects = (
                self.habitat_config.navmesh_include_static_objects
            )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[operator]
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                # TODO consider making trans_dict a sensor class var too.
                trans_dict={
                    "sensor_model_type": lambda v: getattr(
                        habitat_sim.FisheyeSensorModelType, v
                    ),
                    "sensor_subtype": lambda v: getattr(
                        habitat_sim.SensorSubType, v
                    ),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.habitat_config.habitat_sim_v0.gpu_gpu
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications

        agent_config.action_space = {
            0: habitat_sim.ActionSpec("stop"),
            1: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.habitat_config.forward_step_size
                ),
            ),
            2: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(
                    amount=self.habitat_config.turn_angle
                ),
            ),
            3: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(
                    amount=self.habitat_config.turn_angle
                ),
            ),
        }
        
        # 行动的人需要启动sim的物理属性
        if self.habitat_config.enable_human:
            sim_config.enable_physics = True
            sim_config.physics_config_file = "./data/default.physics_config.json"

        output = habitat_sim.Configuration(sim_config, [agent_config])
        output.enable_batch_renderer = (
            self.habitat_config.renderer.enable_batch_renderer
        )
        return output

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, agent_name in enumerate(
            self.habitat_config.agents_order
        ):
            agent_cfg = self.habitat_config.agents[agent_name]
            if agent_cfg.is_set_start_state:
                self.set_agent_state(
                    [float(k) for k in agent_cfg.start_position],
                    [float(k) for k in agent_cfg.start_rotation],
                    agent_id,
                )
                is_updated = True

        return is_updated
    
    def get_human(self):
        aom = self.get_articulated_object_manager()
        print("List of articulated objects:")
        for handle, ao in aom.get_objects_by_handle_substring().items():
            print(handle, "id", aom.get_object_id_by_handle(handle))
        
        
    
    def reset(self) -> Observations:
        self.collision_with_human = False
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        if self.habitat_config.enable_human == 1:
            pos_ini = self.get_agent_state().position
            # print("pos_ini:",pos_ini)
            pos_ini[1] = self.get_floor_height(pos_ini)
            # print("pos_ini get_floor_height:",pos_ini)
            self.init_human(pos_ini)
            self.get_human()
        if self.config.enable_batch_renderer:
            self.add_keyframe_to_observations(sim_obs)
            return sim_obs
        else:
            return self._sensor_suite.get_observations(sim_obs)

    def step(
        self, action: Optional[Union[str, np.ndarray, int]]
    ) -> Observations:
        
        # human step
        if self.habitat_config.enable_human:
            agent_state = self.get_agent_state()
            a_x,_,a_y = agent_state.position
            robot_state = [a_x,a_y,0,0,0.2]
            human_actions = self.Human_ocra.get_huamn_actions(self.human_list,robot_state=robot_state,robot_vis=self.habitat_config.robot_vis)
            # human_actions = self.Human_ocra.get_huamn_actions(self.human_list)
            for i,action_h in enumerate(human_actions):
                px,_,py = self.human_list[i].base_pos
                dis = (px-a_x)**2+(py-a_y)**2
                if dis < 0.25:
                    print("collision with human!**************")
                    self.collision_with_human = True
                    
                hc = self.HumanControl_list[i]
                if hc.walk_flag:
                    hc.human_move_vxy(vx=action_h.vx,vy=action_h.vy)
                    hc.run_follow_orca()
                    # hc.run_follow_waypoint() # 按照轨迹走，人与人之间可能会撞
                else:
                    hc.move()
                
            # self.step_physics(1.0 / 60.0)
            self.step_physics(1.0 / 200.0)

        if action is None:
            sim_obs = self.get_sensor_observations()
        else:
            sim_obs = super().step(action)
        
        self._prev_sim_obs = sim_obs
        if self.config.enable_batch_renderer:
            self.add_keyframe_to_observations(sim_obs)
            return sim_obs
        else:
            return self._sensor_suite.get_observations(sim_obs)

    def render(self, mode: str = "rgb") -> Any:
        r"""
        Args:
            mode: sensor whose observation is used for returning the frame,
                eg: "rgb", "depth", "semantic"

        Returns:
            rendered frame according to the mode
        """
        assert not self.config.enable_batch_renderer

        sim_obs = self.get_sensor_observations()
        observations = self._sensor_suite.get_observations(sim_obs)

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)
        if not isinstance(output, np.ndarray):
            # If it is not a numpy array, it is a torch tensor
            # The function expects the result to be a numpy array
            output = output.to("cpu").numpy()

        return output

    def reconfigure(
        self,
        habitat_config: DictConfig,
        ep_info: Optional[Episode] = None,
        should_close_on_new_scene: bool = True,
    ) -> None:
        # TODO(maksymets): Switch to Habitat-Sim more efficient caching
        is_same_scene = habitat_config.scene == self._current_scene
        self.habitat_config = habitat_config
        self.sim_config = self.create_sim_config(self._sensor_suite)
        if not is_same_scene:
            self._current_scene = habitat_config.scene
            if should_close_on_new_scene:
                self.close(destroy=False)
            super().reconfigure(self.sim_config)

        self._update_agents_state()

    def geodesic_distance(
        self,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[
            Sequence[float], Sequence[Sequence[float]], np.ndarray
        ],
        episode: Optional[Episode] = None,
    ) -> float:
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            if isinstance(position_b[0], (Sequence, np.ndarray)):
                path.requested_ends = np.array(position_b, dtype=np.float32)
            else:
                path.requested_ends = np.array(
                    [np.array(position_b, dtype=np.float32)]
                )
        else:
            path = episode._shortest_path_cache

        path.requested_start = np.array(position_a, dtype=np.float32)

        self.pathfinder.find_path(path)

        if episode is not None:
            episode._shortest_path_cache = path

        return path.geodesic_distance

    def action_space_shortest_path(
        self,
        source: AgentState,
        targets: Sequence[AgentState],
        agent_id: int = 0,
    ) -> List[ShortestPathPoint]:
        r"""
        Returns:
            List of agent states and actions along the shortest path from
            source to the nearest target (both included). If one of the
            target(s) is identical to the source, a list containing only
            one node with the identical agent state is returned. Returns
            an empty list in case none of the targets are reachable from
            the source. For the last item in the returned list the action
            will be None.
        """
        raise NotImplementedError(
            "This function is no longer implemented. Please use the greedy "
            "follower instead"
        )

    @property
    def up_vector(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    @property
    def forward_vector(self) -> np.ndarray:
        return -np.array([0.0, 0.0, 1.0])

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self.pathfinder.find_path(path)
        return path.points

    def sample_navigable_point(self) -> List[float]:
        return self.pathfinder.get_random_navigable_point().tolist()

    def is_navigable(self, point: List[float]) -> bool:
        return self.pathfinder.is_navigable(point)

    def semantic_annotations(self):
        r"""
        Returns:
            SemanticScene which is a three level hierarchy of semantic
            annotations for the current scene. Specifically this method
            returns a SemanticScene which contains a list of SemanticLevel's
            where each SemanticLevel contains a list of SemanticRegion's where
            each SemanticRegion contains a list of SemanticObject's.

            SemanticScene has attributes: aabb(axis-aligned bounding box) which
            has attributes aabb.center and aabb.sizes which are 3d vectors,
            categories, levels, objects, regions.

            SemanticLevel has attributes: id, aabb, objects and regions.

            SemanticRegion has attributes: id, level, aabb, category (to get
            name of category use category.name()) and objects.

            SemanticObject has attributes: id, region, aabb, obb (oriented
            bounding box) and category.

            SemanticScene contains List[SemanticLevels]
            SemanticLevel contains List[SemanticRegion]
            SemanticRegion contains List[SemanticObject]

            Example to loop through in a hierarchical fashion:
            for level in semantic_scene.levels:
                for region in level.regions:
                    for obj in region.objects:
        """
        return self.semantic_scene

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        return self.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        r"""Sets agent state similar to initialize_agent, but without agents
        creation. On failure to place the agent in the proper position, it is
        moved back to its previous pose.

        Args:
            position: list containing 3 entries for (x, y, z).
            rotation: list with 4 entries for (x, y, z, w) elements of unit
                quaternion (versor) representing agent 3D orientation,
                (https://en.wikipedia.org/wiki/Versor)
            agent_id: int identification of agent from multiagent setup.
            reset_sensors: bool for if sensor changes (e.g. tilt) should be
                reset).

        Returns:
            True if the set was successful else moves the agent back to its
            original pose and returns false.
        """
        agent = self.get_agent(agent_id)
        new_state = self.get_agent(agent_id).get_state()
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def distance_to_closest_obstacle(
        self, position: np.ndarray, max_search_radius: float = 2.0
    ) -> float:
        return self.pathfinder.distance_to_closest_obstacle(
            position, max_search_radius
        )

    def island_radius(self, position: Sequence[float]) -> float:
        return self.pathfinder.island_radius(position)

    @property
    def previous_step_collided(self):
        r"""Whether or not the previous step resulted in a collision

        Returns:
            bool: True if the previous step resulted in a collision, false otherwise

        Warning:
            This field is only updated when :meth:`step`, :meth:`reset`, or :meth:`get_observations_at` are
            called.  It does not update when the agent is moved to a new location.  Furthermore, it
            will _always_ be false after :meth:`reset` or :meth:`get_observations_at` as neither of those
            result in an action (step) being taken.
        """
        return self._prev_sim_obs.get("collided", False)

    def add_keyframe_to_observations(self, observations):
        r"""Adds an item to observations that contains the latest gfx-replay keyframe.
        This is used to communicate the state of concurrent simulators to the batch renderer between processes.

        :param observations: Original observations upon which the keyframe is added.
        """
        assert self.config.enable_batch_renderer

        assert KEYFRAME_OBSERVATION_KEY not in observations
        for _sensor_uuid, sensor in self._sensors.items():
            node = sensor._sensor_object.node
            transform = node.absolute_transformation()
            rotation = mn.Quaternion.from_matrix(transform.rotation())
            self.gfx_replay_manager.add_user_transform_to_keyframe(
                KEYFRAME_SENSOR_PREFIX + _sensor_uuid,
                transform.translation,
                rotation,
            )
        observations[
            KEYFRAME_OBSERVATION_KEY
        ] = self.gfx_replay_manager.extract_keyframe()
