#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import magnum as mn
import numpy as np
import copy

import quaternion
import math

from habitat.articulated_agent_controllers import (
    HumanoidBaseController,
    Motion,
)


class HumanoidSeqPoseController(HumanoidBaseController):
    """
    Humanoid Seq Pose Controller, replays a sequence of humanoid poses.

        :param motion_pose_path: file containing the motion poses we want to play.
        :param motion_fps: the FPS at which we should be advancing the pose.
        :param base_offset: what is the offset between the root of the character and their feet.
    """

    def __init__(
        self,
        motion_pose_path,
        motion_fps=30,
        base_offset=(0, 0.9, 0),
        start_end_index=None,   # (s,e)
    ):
        super().__init__(motion_fps, base_offset)

        if not os.path.isfile(motion_pose_path):
            raise RuntimeError(
                f"Path does {motion_pose_path} not exist. Reach out to the paper authors to obtain this data."
            )

        motion_info = np.load(motion_pose_path, allow_pickle=True)
        motion_info = motion_info["pose_motion"]
        if start_end_index is not None:
            if start_end_index[0] >= 0 and start_end_index[0] <= start_end_index[1] and start_end_index[1] <motion_info['joints_array'].shape[0]:
                self.humanoid_motion = Motion(
                    motion_info["joints_array"][start_end_index[0]:start_end_index[1]],
                    motion_info["transform_array"][start_end_index[0]:start_end_index[1]],
                    motion_info["displacement"],
                    motion_info["fps"],
                )
            else:
                raise RuntimeError(
                    f"{start_end_index} is not in range 0-{motion_info['joints_array'].shape[0]}"
                )
            
        else:
            self.humanoid_motion = Motion(
                motion_info["joints_array"],
                motion_info["transform_array"],
                motion_info["displacement"],
                motion_info["fps"],
            )
        self.motion_frame = 0
        self.ref_pose = mn.Matrix4()
        self.first_pose = mn.Matrix4(
            self.humanoid_motion.poses[0].root_transform
        )
        self.step_size = int(self.humanoid_motion.fps / self.motion_fps)
        self.base_transform_offset = mn.Matrix4()
        
        self.offset_rot = -np.pi / 2
        add_rot = mn.Matrix4.rotation(
            mn.Rad(self.offset_rot), mn.Vector3(0, 1.0, 0)
        )
        perm = mn.Matrix4.rotation(
            mn.Rad(self.offset_rot), mn.Vector3(0, 0, 1.0)
        )
        self.offset_transform_base = perm @ add_rot

    def reset(self, base_transformation: mn.Matrix4) -> None:
        """Reset the joints on the human. (Put in rest state)"""
        super().reset(base_transformation)
        self.motion_frame = 0

        # The first pose of the motion file will be set at base_transformation
        self.ref_pose = base_transformation
        self.base_transform_offset = mn.Matrix4()
        # self.obj_transform_base_first = mn.Matrix4()
        # self.calculate_pose()
    
    def set_direction(self, direction: mn.Vector3):
        look_at_path_T = mn.Matrix4.look_at(
            self.obj_transform_base.translation,
            self.obj_transform_base.translation + direction,
            mn.Vector3.y_axis(),
        )
        rot_offset = mn.Matrix4.rotation(
                    mn.Rad(-np.pi / 2), mn.Vector3(1, 0, 0)
                )
        self.obj_transform_base = look_at_path_T @ rot_offset
        self.obj_transform_base_first = look_at_path_T @ rot_offset
        

    def apply_base_transformation(
        self, base_transformation: mn.Matrix4
    ) -> None:
        """Sets the current pose to the base transformation, making the rest of poses are relative to this one"""
        self.base_transform_offset = mn.Matrix4()
        self.base_transform_offset.translation = (
            -base_transformation.translation + self.base_offset
        )

        self.calculate_pose()

    def next_pose(self, cycle=False) -> None:
        """
        Move to the next pose in the motion sequence

            :param cycle: boolean indicating whether we should stop or cycle when reaching the last pose
        """

        if cycle:
            self.motion_frame = (
                self.motion_frame + self.step_size
            ) % self.humanoid_motion.num_poses
        else:
            self.motion_frame = min(
                self.motion_frame + 1, self.humanoid_motion.num_poses - 1
            )

    def prev_pose(self, cycle=False) -> None:
        """
        Move to the previous pose in the motion sequence
            :param cycle: boolean indicating whether we should stop or cycle when reaching the first pose
        """

        if cycle:
            self.motion_frame = (
                self.motion_frame - self.step_size
            ) % self.humanoid_motion.num_poses
        else:
            self.motion_frame = max(0, self.motion_frame - 1)

    # def calculate_pose(self, advance_pose=False) -> None:
    #     """
    #     目前可以实现在固定区域指定朝向进行动作,不过还是有bug,主要是位置的方向在动作中是固定的方向;
    #     想要设定可以指定方向的理论上transalation要变,这个主要影响大移动的物体，其他就还好
    #     Set the joint transforms according to the current frame
    #         :param advance_pose: whether this function should move to the next pose
    #     """
    #     curr_transform = mn.Matrix4(
    #         self.humanoid_motion.poses[self.motion_frame].root_transform
    #     )
    #     # Remove the forward component, and orient according to forward_V
        
    #     # curr_transform.translation = (
    #     #     curr_transform.translation
    #     #     - self.first_pose.translation
    #     #     + self.ref_pose.translation
    #     # )
    #     # curr_transform.translation = curr_transform.translation - mn.Vector3(
    #     #     0, 0.9, 0
    #     # )
    #     curr_poses = self.humanoid_motion.poses[self.motion_frame].joints
    #     # self.obj_transform_offset = self.base_transform_offset @ curr_transform
    #     # rot_offset = mn.Matrix4.rotation(
    #     #     mn.Rad(-np.pi / 2), mn.Vector3(1, 0, 0)
    #     # )
    #     obj_transform_base = copy.deepcopy(self.obj_transform_base_first)
    #     tmp_diff = curr_transform.translation - self.first_pose.translation
    #     first_rotation_y = mn.Quaternion.from_matrix(self.first_pose.rotation()) 
    #     base_rotation_y = mn.Quaternion.from_matrix(obj_transform_base.rotation())
    #     # 将四元数投影到 XZ 平面上
    #     first_rotation_y_projected = mn.Vector2(first_rotation_y.vector.x, first_rotation_y.vector.z).normalized()
    #     base_rotation_y_projected = mn.Vector2(base_rotation_y.vector.x, base_rotation_y.vector.z).normalized()
    #     print("first_rotation_y_projected:",first_rotation_y_projected,"base_rotation_y_projected:",base_rotation_y_projected)

    #     # 计算两个投影后的向量之间的角度差异
    #     theta = mn.math.angle(first_rotation_y_projected, base_rotation_y_projected)
    #     print("theta:",theta)
    #     x_rotated = -tmp_diff.x * math.cos(theta) - tmp_diff.z * math.sin(theta)
    #     z_rotated = tmp_diff.x * math.sin(theta) - tmp_diff.z * math.cos(theta)
    #     tmp_diff2 = mn.Vector3(x_rotated,tmp_diff.y,z_rotated)



    #     print("tmp_diff:",tmp_diff,"tmp_diff2:",tmp_diff2)
    #     obj_transform_base.translation = self.obj_transform_base_first.translation + mn.Vector3(tmp_diff2)
    #     self.obj_transform_base = obj_transform_base 
    #     # self.obj_transform_base.translation = curr_transform.translation
        
    #     obj_transform = self.humanoid_motion.poses[self.motion_frame].root_transform
    #     obj_transform.translation = tmp_diff
    #     # add_rot = mn.Matrix4.rotation(mn.Rad(np.pi), mn.Vector3(0, 1, 0))
    #     # obj_transform = add_rot @ obj_transform
    #     obj_transform.translation *= mn.Vector3.x_axis() + mn.Vector3.y_axis()  # 将z轴变化置0
    #     self.obj_transform_offset = obj_transform

    #     self.joint_pose = curr_poses

    #     if advance_pose:
    #         self.next_pose()
    
    def calculate_pose(self, advance_pose=False) -> None:
        """
        目前可以实现在固定区域指定朝向进行动作,不过还是有bug,主要是位置的方向在动作中是固定的方向;
        想要设定可以指定方向的理论上transalation要变,这个主要影响大移动的物体，其他就还好 OK 版
        Set the joint transforms according to the current frame
            :param advance_pose: whether this function should move to the next pose
        """
        curr_transform = mn.Matrix4(
            self.humanoid_motion.poses[self.motion_frame].root_transform
        )
        # Remove the forward component, and orient according to forward_V
        
        curr_poses = self.humanoid_motion.poses[self.motion_frame].joints
        # self.obj_transform_offset = self.base_transform_offset @ curr_transform
        # rot_offset = mn.Matrix4.rotation(
        #     mn.Rad(-np.pi / 2), mn.Vector3(1, 0, 0)
        # )
        obj_transform_base = copy.deepcopy(self.obj_transform_base_first)
        tmp_diff = curr_transform.translation - self.first_pose.translation
        obj_transform_base.translation = self.obj_transform_base_first.translation + mn.Vector3(tmp_diff)
        self.obj_transform_base = obj_transform_base 
        # self.obj_transform_base.translation = curr_transform.translation
        
        obj_transform = self.humanoid_motion.poses[self.motion_frame].root_transform
        obj_transform.translation = tmp_diff
        # add_rot = mn.Matrix4.rotation(mn.Rad(np.pi), mn.Vector3(0, 1, 0))
        # obj_transform = add_rot @ obj_transform
        obj_transform.translation *= mn.Vector3.x_axis() + mn.Vector3.y_axis()  # 将z轴变化置0
        self.obj_transform_offset = obj_transform

        self.joint_pose = curr_poses

        if advance_pose:
            self.next_pose()

    # # 本以为改好的版本
    # def calculate_pose(self, advance_pose=False) -> None:
    #     """
    #     Set the joint transforms according to the current frame
    #         :param advance_pose: whether this function should move to the next pose
    #     """
    #     curr_transform = mn.Matrix4(
    #         self.humanoid_motion.poses[self.motion_frame].root_transform
    #     )
    #     curr_transform.translation = (
    #         curr_transform.translation
    #         - self.first_pose.translation
    #     )
    #     # curr_transform.translation = curr_transform.translation - mn.Vector3(
    #     #     0, 0.9, 0
    #     # )
    #     curr_poses = self.humanoid_motion.poses[self.motion_frame].joints
    #     self.obj_transform_offset = self.base_transform_offset @ curr_transform
        
    #     add_rot = self.offset_transform_base.inverted()
    #     final_transform = (self.obj_transform_base @ add_rot) @ self.obj_transform_offset
    #     self.obj_transform_base.translation = final_transform.translation
        
    #     # self.obj_transform_base.translation = self.obj_transform_offset.translation
        
    #     obj_transform = self.humanoid_motion.poses[self.motion_frame].root_transform
    #     obj_transform.translation = mn.Vector3(0, 0.0, 0)
    #     self.obj_transform_offset = obj_transform
        
        
        
    #     self.joint_pose = curr_poses

    #     if advance_pose:
    #         self.next_pose()
    
    # 原版
    # def calculate_pose(self, advance_pose=False) -> None:
    #     """
    #     Set the joint transforms according to the current frame
    #         :param advance_pose: whether this function should move to the next pose
    #     """
    #     curr_transform = mn.Matrix4(
    #         self.humanoid_motion.poses[self.motion_frame].root_transform
    #     )
    #     curr_transform.translation = (
    #         curr_transform.translation
    #         - self.first_pose.translation
    #         + self.ref_pose.translation
    #     )
    #     curr_transform.translation = curr_transform.translation - mn.Vector3(
    #         0, 0.9, 0
    #     )
    #     curr_poses = self.humanoid_motion.poses[self.motion_frame].joints
    #     self.obj_transform_offset = self.base_transform_offset @ curr_transform
        
        
    #     self.joint_pose = curr_poses

    #     if advance_pose:
    #         self.next_pose()