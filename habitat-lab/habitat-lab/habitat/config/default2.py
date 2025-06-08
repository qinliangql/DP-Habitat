#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import yacs.config

# Default Habitat config node
class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


cn = Config

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_c = cn()
_c.seed = 100
# -----------------------------------------------------------------------------
# environment
# -----------------------------------------------------------------------------
_c.environment = cn()
_c.environment.max_episode_steps = 1000
_c.environment.max_episode_seconds = 10000000
_c.environment.iterator_options = cn()
_c.environment.iterator_options.cycle = True
_c.environment.iterator_options.shuffle = True
_c.environment.iterator_options.group_by_scene = True
_c.environment.iterator_options.num_episode_sample = -1
_c.environment.iterator_options.max_scene_repeat_episodes = -1
_c.environment.iterator_options.max_scene_repeat_steps = int(1e4)
_c.environment.iterator_options.step_repetition_range = 0.2
# -----------------------------------------------------------------------------
# task
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # navigation task
# -----------------------------------------------------------------------------
_c.task = cn()
_c.task.type = "Nav-v0"
_c.task.sensors = []
_c.task.measurements = []
_c.task.goal_sensor_uuid = "pointgoal"
_c.task.possible_actions = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
# -----------------------------------------------------------------------------
# # actions
# -----------------------------------------------------------------------------
actions = cn()
actions.stop = cn()
actions.stop.type = "StopAction"
# -----------------------------------------------------------------------------
# # navigation actions
# -----------------------------------------------------------------------------
actions.move_forward = cn()
actions.move_forward.type = "MoveForwardAction"
actions.turn_left = cn()
actions.turn_left.type = "TurnLeftAction"
actions.turn_right = cn()
actions.turn_right.type = "TurnRightAction"
actions.look_up = cn()
actions.look_up.type = "LookUpAction"
actions.look_down = cn()
actions.look_down.type = "LookDownAction"
actions.teleport = cn()
actions.teleport.type = "TeleportAction"
actions.velocity_control = cn()
actions.velocity_control.type = "VelocityAction"
actions.velocity_control.lin_vel_range = [0.0, 0.25]  # meters per sec
actions.velocity_control.ang_vel_range = [-10.0, 10.0]  # deg per sec
actions.velocity_control.min_abs_lin_speed = 0.025  # meters per sec
actions.velocity_control.min_abs_ang_speed = 1.0  # deg per sec
actions.velocity_control.time_step = 1.0  # seconds

_c.task.actions = actions
# -----------------------------------------------------------------------------
# # task sensors
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# pointgoal sensor
# -----------------------------------------------------------------------------
_c.task.pointgoal_sensor = cn()
_c.task.pointgoal_sensor.type = "PointGoalSensor"
_c.task.pointgoal_sensor.goal_format = "POLAR"
_c.task.pointgoal_sensor.dimensionality = 2
# -----------------------------------------------------------------------------
# pointgoal with gps+compass sensor
# -----------------------------------------------------------------------------
_c.task.pointgoal_with_gps_compass_sensor = _c.task.pointgoal_sensor.clone()
_c.task.pointgoal_with_gps_compass_sensor.type = (
    "PointGoalWithGPSCompassSensor"
)
# -----------------------------------------------------------------------------
# objectgoal sensor
# -----------------------------------------------------------------------------
_c.task.objectgoal_sensor = cn()
_c.task.objectgoal_sensor.type = "ObjectGoalSensor"
_c.task.objectgoal_sensor.goal_spec = "TASK_CATEGORY_ID"
_c.task.objectgoal_sensor.goal_spec_max_val = 50
# -----------------------------------------------------------------------------
# imagegoal sensor
# -----------------------------------------------------------------------------
_c.task.imagegoal_sensor = cn()
_c.task.imagegoal_sensor.type = "ImageGoalSensor"
# -----------------------------------------------------------------------------
# heading sensor
# -----------------------------------------------------------------------------
_c.task.heading_sensor = cn()
_c.task.heading_sensor.type = "HeadingSensor"
# -----------------------------------------------------------------------------
# compass sensor
# -----------------------------------------------------------------------------
_c.task.compass_sensor = cn()
_c.task.compass_sensor.type = "CompassSensor"
# -----------------------------------------------------------------------------
# gps sensor
# -----------------------------------------------------------------------------
_c.task.gps_sensor = cn()
_c.task.gps_sensor.type = "GPSSensor"
_c.task.gps_sensor.dimensionality = 2
# -----------------------------------------------------------------------------
# proximity sensor
# -----------------------------------------------------------------------------
_c.task.proximity_sensor = cn()
_c.task.proximity_sensor.type = "ProximitySensor"
_c.task.proximity_sensor.max_detection_radius = 2.0
# -----------------------------------------------------------------------------
# success measurement
# -----------------------------------------------------------------------------
_c.task.success = cn()
_c.task.success.type = "Success"
_c.task.success.success_distance = 0.2
# -----------------------------------------------------------------------------
# spl measurement
# -----------------------------------------------------------------------------
_c.task.spl = cn()
_c.task.spl.type = "SPL"
# -----------------------------------------------------------------------------
# soft-spl measurement
# -----------------------------------------------------------------------------
_c.task.soft_spl = cn()
_c.task.soft_spl.type = "SoftSPL"
# -----------------------------------------------------------------------------
# topdownmap measurement
# -----------------------------------------------------------------------------
_c.task.top_down_map = cn()
_c.task.top_down_map.type = "TopDownMap"
_c.task.top_down_map.max_episode_steps = _c.environment.max_episode_steps
_c.task.top_down_map.map_padding = 3
_c.task.top_down_map.map_resolution = 1024
_c.task.top_down_map.draw_source = True
_c.task.top_down_map.draw_border = True
_c.task.top_down_map.draw_shortest_path = True
_c.task.top_down_map.fog_of_war = cn()
_c.task.top_down_map.fog_of_war.draw = True
_c.task.top_down_map.fog_of_war.visibility_dist = 5.0
_c.task.top_down_map.fog_of_war.fov = 90
_c.task.top_down_map.draw_view_points = True
_c.task.top_down_map.draw_goal_positions = True
# axes aligned bounding boxes
_c.task.top_down_map.draw_goal_aabbs = True
# -----------------------------------------------------------------------------
# collisions measurement
# -----------------------------------------------------------------------------
_c.task.collisions = cn()
_c.task.collisions.type = "Collisions"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # eqa task
# -----------------------------------------------------------------------------
_c.task.actions.answer = cn()
_c.task.actions.answer.type = "AnswerAction"
# # eqa task question sensor
# -----------------------------------------------------------------------------
_c.task.question_sensor = cn()
_c.task.question_sensor.type = "QuestionSensor"
# -----------------------------------------------------------------------------
# # eqa task correct_answer measure for training
# -----------------------------------------------------------------------------
_c.task.correct_answer = cn()
_c.task.correct_answer.type = "CorrectAnswer"
# -----------------------------------------------------------------------------
# # eqa task answer sensor
# -----------------------------------------------------------------------------
_c.task.episode_info = cn()
_c.task.episode_info.type = "EpisodeInfo"
# -----------------------------------------------------------------------------
# # vln task instruction sensor
# -----------------------------------------------------------------------------
_c.task.instruction_sensor = cn()
_c.task.instruction_sensor.type = "InstructionSensor"
_c.task.instruction_sensor_uuid = "instruction"
# -----------------------------------------------------------------------------
# # distance_to_goal measurement
# -----------------------------------------------------------------------------
_c.task.distance_to_goal = cn()
_c.task.distance_to_goal.type = "DistanceToGoal"
_c.task.distance_to_goal.distance_to = "POINT"
# -----------------------------------------------------------------------------
# # answer_accuracy measurement
# -----------------------------------------------------------------------------
_c.task.answer_accuracy = cn()
_c.task.answer_accuracy.type = "AnswerAccuracy"
# -----------------------------------------------------------------------------
# simulator
# -----------------------------------------------------------------------------
_c.simulator = cn()
_c.simulator.type = "Sim-v0"
_c.simulator.action_space_config = "v0"
_c.simulator.forward_step_size = 0.25  # in metres
_c.simulator.scene = (
    "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
)
_c.simulator.scene_dataset = "default"  # the scene dataset to load in the MetaDataMediator. Should contain simulator.scene
_c.simulator.additional_object_paths = (
    []
)  # a list of directory or config paths to search in addition to the dataset for object configs
_c.simulator.seed = _c.seed
_c.simulator.turn_angle = 10  # angle to rotate left or right in degrees
_c.simulator.tilt_angle = 15  # angle to tilt the camera up or down in degrees
_c.simulator.default_agent_id = 0
# -----------------------------------------------------------------------------
# simulator sensors
# -----------------------------------------------------------------------------
simulator_sensor = cn()
simulator_sensor.height = 480
simulator_sensor.width = 640
simulator_sensor.position = [0, 1.25, 0]
simulator_sensor.orientation = [0.0, 0.0, 0.0]  # Euler's angles

# -----------------------------------------------------------------------------
# camera sensor
# -----------------------------------------------------------------------------
camera_sim_sensor = simulator_sensor.clone()
camera_sim_sensor.hfov = 90  # horizontal field of view in degrees
camera_sim_sensor.sensor_subtype = "PINHOLE"

simulator_depth_sensor = simulator_sensor.clone()
simulator_depth_sensor.min_depth = 0.0
simulator_depth_sensor.max_depth = 10.0
simulator_depth_sensor.normalize_depth = True

# -----------------------------------------------------------------------------
# rgb sensor
# -----------------------------------------------------------------------------
_c.simulator.rgb_sensor = camera_sim_sensor.clone()
_c.simulator.rgb_sensor.type = "HabitatSimRGBSensor"
# -----------------------------------------------------------------------------
# depth sensor
# -----------------------------------------------------------------------------
_c.simulator.depth_sensor = camera_sim_sensor.clone()
_c.simulator.depth_sensor.merge_from_other_cfg(simulator_depth_sensor)
_c.simulator.depth_sensor.type = "HabitatSimDepthSensor"
# -----------------------------------------------------------------------------
# semantic sensor
# -----------------------------------------------------------------------------
_c.simulator.semantic_sensor = camera_sim_sensor.clone()
_c.simulator.semantic_sensor.type = "HabitatSimSemanticSensor"
# -----------------------------------------------------------------------------
# equirect rgb sensor
# -----------------------------------------------------------------------------
_c.simulator.equirect_rgb_sensor = simulator_sensor.clone()
_c.simulator.equirect_rgb_sensor.type = "HabitatSimEquirectangularRGBSensor"
_c.simulator.equirect_rgb_sensor.height = 256
_c.simulator.equirect_rgb_sensor.width = 512
# -----------------------------------------------------------------------------
# equirect depth sensor
# -----------------------------------------------------------------------------
_c.simulator.equirect_depth_sensor = (
    _c.simulator.equirect_rgb_sensor.clone()
)
_c.simulator.equirect_depth_sensor.type = (
    "HabitatSimEquirectangularDepthSensor"
)
_c.simulator.equirect_depth_sensor.merge_from_other_cfg(simulator_depth_sensor)
# -----------------------------------------------------------------------------
# fisheye rgb sensor
# -----------------------------------------------------------------------------
_c.simulator.fisheye_rgb_sensor = simulator_sensor.clone()
_c.simulator.fisheye_rgb_sensor.type = "HabitatSimFisheyeRGBSensor"
_c.simulator.fisheye_rgb_sensor.height = 256
_c.simulator.fisheye_rgb_sensor.width = 256
_c.simulator.fisheye_rgb_sensor.fisheye_model = "double_sphere"
# -----------------------------------------------------------------------------
# fisheye depth sensor
# -----------------------------------------------------------------------------
_c.simulator.fisheye_depth_sensor = _c.simulator.fisheye_rgb_sensor.clone()
_c.simulator.fisheye_depth_sensor.type = "HabitatSimFisheyeDepthSensor"
_c.simulator.fisheye_depth_sensor.merge_from_other_cfg(simulator_depth_sensor)
# -----------------------------------------------------------------------------
# agent
# -----------------------------------------------------------------------------
_c.simulator.agent_0 = cn()
_c.simulator.agent_0.height = 1.5
_c.simulator.agent_0.radius = 0.1
_c.simulator.agent_0.sensors = ["rgb_sensor"]
_c.simulator.agent_0.is_set_start_state = False
_c.simulator.agent_0.start_position = [0, 0, 0]
_c.simulator.agent_0.start_rotation = [0, 0, 0, 1]
_c.simulator.agents = ["agent_0"]
# -----------------------------------------------------------------------------
# simulator habitat_sim_v0
# -----------------------------------------------------------------------------
_c.simulator.habitat_sim_v0 = cn()
_c.simulator.habitat_sim_v0.gpu_device_id = 0
# Use Habitat-Sim's GPU->GPU copy mode to return rendering results
# in PyTorch tensors.  Requires Habitat-Sim to be built
# with --with-cuda
# This will generally imply sharing CUDA tensors between processes.
# Read here: https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
# for the caveats that results in
_c.simulator.habitat_sim_v0.gpu_gpu = False
# Whether or not the agent slides on collisions
_c.simulator.habitat_sim_v0.allow_sliding = True
_c.simulator.habitat_sim_v0.enable_physics = False
_c.simulator.habitat_sim_v0.physics_config_file = (
    "./data/default.physics_config.json"
)
# Possibly unstable optimization for extra performance with concurrent rendering
_c.simulator.habitat_sim_v0.leave_context_with_background_renderer = False
# -----------------------------------------------------------------------------
# pyrobot
# -----------------------------------------------------------------------------
_c.pyrobot = cn()
_c.pyrobot.robots = ["locobot"]  # types of robots supported
_c.pyrobot.robot = "locobot"
_c.pyrobot.sensors = ["rgb_sensor", "depth_sensor", "bump_sensor"]
_c.pyrobot.base_controller = "proportional"
_c.pyrobot.base_planner = "none"
# -----------------------------------------------------------------------------
# sensors
# -----------------------------------------------------------------------------
pyrobot_visual_sensor = cn()
pyrobot_visual_sensor.height = 480
pyrobot_visual_sensor.width = 640
# -----------------------------------------------------------------------------
# rgb sensor
# -----------------------------------------------------------------------------
_c.pyrobot.rgb_sensor = pyrobot_visual_sensor.clone()
_c.pyrobot.rgb_sensor.type = "PyRobotRGBSensor"
_c.pyrobot.rgb_sensor.center_crop = False
# -----------------------------------------------------------------------------
# depth sensor
# -----------------------------------------------------------------------------
_c.pyrobot.depth_sensor = pyrobot_visual_sensor.clone()
_c.pyrobot.depth_sensor.type = "PyRobotDepthSensor"
_c.pyrobot.depth_sensor.min_depth = 0.0
_c.pyrobot.depth_sensor.max_depth = 5.0
_c.pyrobot.depth_sensor.normalize_depth = True
_c.pyrobot.depth_sensor.center_crop = False
# -----------------------------------------------------------------------------
# bump sensor
# -----------------------------------------------------------------------------
_c.pyrobot.bump_sensor = cn()
_c.pyrobot.bump_sensor.type = "PyRobotBumpSensor"
# -----------------------------------------------------------------------------
# actions locobot
# -----------------------------------------------------------------------------
_c.pyrobot.locobot = cn()
_c.pyrobot.locobot.actions = ["base_actions", "camera_actions"]
_c.pyrobot.locobot.base_actions = ["go_to_relative", "go_to_absolute"]
_c.pyrobot.locobot.camera_actions = ["set_pan", "set_tilt", "set_pan_tilt"]
# TODO: add support for arm actions
# -----------------------------------------------------------------------------
# dataset
# -----------------------------------------------------------------------------
_c.dataset = cn()
_c.dataset.type = "PointNav-v1"
_c.dataset.split = "train"
_c.dataset.scenes_dir = "data/scene_datasets"
_c.dataset.content_scenes = ["*"]
_c.dataset.data_path = (
    "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
)
# -----------------------------------------------------------------------------
# deprecated keys
# -----------------------------------------------------------------------------
_c.register_deprecated_key("task.SUCCESS_DISTANCE")
# -----------------------------------------------------------------------------

def get_config_habv2(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> cn:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _c.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config