#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import numpy as np


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)
        self.dirt_delta = 0.0
        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        # Charger info
        # 充电桩信息
        self.charger_positions = []
        self.nearest_charger_dist = 999.0
        self.last_nearest_charger_dist = 999.0

        # Obstacle distance
        # 最近障碍物距离
        self.nearest_obstacle_dist = 999.0
        self.last_nearest_obstacle_dist = 999.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        # 获取和官方机器人最近的距离
        self.npc_positions = []
        self.nearest_npc_dist = 999.0
        self.last_nearest_npc_dist = 999.0

        self.charger_cells = set()
        self.collision_type = "none"   # none / npc / charger / obstacle
        self.terminated = False

    # 获取官方机器人的位置信息
    def _get_npc_positions(self, npcs):
        """Extract npc positions from frame_state['npcs']."""
        npc_positions = []
        for npc in npcs:
            pos = npc.get("pos", {})
            npc_positions.append((int(pos["x"]), int(pos["z"])))
        return npc_positions
    # 获取最近NPC
    def _calc_nearest_npc_dist(self):
        """Compute Euclidean distance to the nearest official robot."""
        if not self.npc_positions:
            return 999.0

        hx, hz = self.cur_pos
        min_dist = 999.0
        for nx, nz in self.npc_positions:
            dist = np.sqrt((nx - hx) ** 2 + (nz - hz) ** 2)
            min_dist = min(min_dist, dist)
        return float(min_dist)
    # 电量最接近充电桩的奖励
    def _calc_nearest_charger_dist(self):
        if not self.charger_positions:
            return 999.0

        hx, hz = self.cur_pos
        min_dist = 999.0
        for cx, cz in self.charger_positions:
            dist = ((cx - hx) ** 2 + (cz - hz) ** 2) ** 0.5
            min_dist = min(min_dist, dist)
        return float(min_dist)
    
    def _calc_nearest_obstacle_dist(self):
        """Find nearest obstacle distance from local view.

        从局部视野中找最近障碍物的欧氏距离。
        地图协议：0=障碍物/边界，1=已清扫，2=污渍
        """
        view = self._view_map
        if view is None:
            return 999.0

        obs_coords = np.argwhere(view == 0)
        if len(obs_coords) == 0:
            return 999.0

        center = self.VIEW_HALF
        dists = np.sqrt((obs_coords[:, 0] - center) ** 2 + (obs_coords[:, 1] - center) ** 2)

        # 去掉中心点自身的干扰（如果有的话）
        dists = dists[dists > 0]
        if len(dists) == 0:
            return 999.0

        return float(np.min(dists))

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]
        # 新增获取充电桩位置信息
        organs = frame_state.get("organs", [])
        self.charger_positions = self._get_charger_positions(organs)
        self.step_no = int(observation["step_no"])
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))
        # 更新最近充电桩距离
        self.last_nearest_charger_dist = self.nearest_charger_dist
        self.nearest_charger_dist = self._calc_nearest_charger_dist()
        # Update nearest obstacle distance / 更新最近障碍物距离
        self.last_nearest_obstacle_dist = self.nearest_obstacle_dist
        self.nearest_obstacle_dist = self._calc_nearest_obstacle_dist()
        # Battery / 电量
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Legal actions / 合法动作
        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        # npc相关的信息位置以及最近位置
        npcs = frame_state.get("npcs", [])
        self.npc_positions = self._get_npc_positions(npcs)

        self.last_nearest_npc_dist = self.nearest_npc_dist
        self.nearest_npc_dist = self._calc_nearest_npc_dist()

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)
        self.terminated = bool(env_obs.get("terminated", False))
        # 获取充电桩占用的格子
        organs = frame_state.get("organs", [])
        self.charger_cells = self._get_charger_cells(organs)
        self.collision_type = self._get_collision_type()
   
   
    # 获取充电桩位置信息
    def _get_charger_positions(self, organs):
        charger_positions = []
        for organ in organs:
            if int(organ.get("sub_type", -1)) == 1:
                pos = organ.get("pos", {})
                charger_positions.append((int(pos["x"]), int(pos["z"])))
        return charger_positions
    
    # 获取充电桩的位置
    def _get_charger_cells(self, organs):
        charger_cells = set()
        for organ in organs:
            if int(organ.get("sub_type", -1)) != 1:
                continue

            pos = organ.get("pos", {})
            cx = int(pos["x"])
            cz = int(pos["z"])
            w = int(organ.get("w", 1))
            h = int(organ.get("h", 1))

            half_w = w // 2
            half_h = h // 2

            for x in range(cx - half_w, cx + half_w + 1):
                for z in range(cz - half_h, cz + half_h + 1):
                    charger_cells.add((x, z))

        return charger_cells

    # 判断碰撞类型
    def _get_collision_type(self):
        """Classify collision type when episode terminates.

        返回:
        - "npc"      : 与官方机器人碰撞
        - "charger"  : 到达/接触充电桩
        - "obstacle" : 其他障碍物/边界
        - "none"     : 无碰撞
        """
        if not self.terminated:
            return "none"

        hx, hz = self.cur_pos

        # 1) 先判断是否碰到官方机器人
        if self.nearest_npc_dist <= 0.5:
            return "npc"

        # 2) 再判断是否在充电桩范围内
        if (hx, hz) in self.charger_cells:
            return "charger"

        # 3) 其余终止情形，视为普通障碍物/边界碰撞
        return "obstacle"
    def _update_passable(self, hx, hz):
        """Write local view into global passable map.

        将局部视野写入全局通行地图。
        """
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    # 0 = obstacle, 1/2 = passable
                    # 0 = 障碍, 1/2 = 可通行
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _get_local_view_feature(self):
        """Local view feature (49D): crop center 7×7 from 21×21.

        局部视野特征（49D）：从 21×21 视野中心裁剪 7×7。
        """
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        return (crop / 2.0).flatten()

    def _get_global_state_feature(self):
        """Global state feature (12D).

        全局状态特征（12D）。

        Dimensions / 维度说明：
        [0]  step_norm         step progress / 步数归一化 [0,1]
        [1]  battery_ratio     battery level / 电量比 [0,1]
        [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
        [3]  remaining_dirt    remaining dirt ratio / 剩余污渍比例 [0,1]
        [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
        [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
        [6]  ray_N_dirt        north ray distance / 向上（z-）方向最近污渍距离
        [7]  ray_E_dirt        east ray distance / 向右（x+）方向
        [8]  ray_S_dirt        south ray distance / 向下（z+）方向
        [9]  ray_W_dirt        west ray distance / 向左（x-）方向
        [10] nearest_dirt_norm nearest dirt Euclidean distance / 最近污渍欧氏距离归一化
        [11] dirt_delta        approaching dirt indicator / 是否在接近污渍（1=是, 0=否）
        """
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 4-directional ray to find nearest dirt
        # 四方向射线找最近污渍距离
        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W
        ray_dirt = []
        max_ray = 30   # 类似于雷达

        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break

                if self._view_map is not None:
                    local_r = x - hx + self.VIEW_HALF   # 计算这个点距离自己的相对位置
                    local_c = z - hz + self.VIEW_HALF   # 计算这个点的z坐标距离自己的相对位置

                    if 0 <= local_r < 21 and 0 <= local_c < 21:
                        cell = int(self._view_map[local_r, local_c])
                    else:
                        cell = 0

                    if cell == 2:
                        found = step
                        break

            ray_dirt.append(_norm(found, max_ray))

        # Nearest dirt Euclidean distance
        # 最近污渍欧氏距离
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)

        self.dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0
        
        return np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                ray_dirt[0],
                ray_dirt[1],
                ray_dirt[2],
                ray_dirt[3],
                nearest_dirt_norm,
                self.dirt_delta,
            ],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        return list(self._legal_act)
    
    # 将有用的全局特征送到模型里面。
    def feature_process(self, env_obs, last_action):
        """Generate 69D feature vector, legal action mask, and scalar reward.

        生成 69D 特征向量、合法动作掩码和标量奖励。
        """
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()  # 49D
        global_state = self._get_global_state_feature()  # 12D
        legal_action = self.get_legal_action()  # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)

        feature = np.concatenate([local_view, global_state, legal_arr])  # 69D

        reward,reward_info = self.reward_process()
        self.reward_info = reward_info
        return feature, legal_action, reward

    def reward_process(self):
        """Compute reward.

        奖励由以下几部分组成：
        1. 清扫奖励
        2. 接近污渍奖励
        3. 远离障碍物奖励（或过近惩罚）
        4. 低电量时接近充电桩奖励
        5. 时间惩罚
        """
        # 1) Cleaning reward / 清扫奖励
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        cleaning_reward = 0.1 * cleaned_this_step

        # 2) Approach dirt reward / 只要在接近污渍，就给奖励
        dirt_approach_reward = 0.0
        if self.step_no > 0 and self.dirt_delta > 0:
            dirt_approach_reward = 0.05

        # 3) NPC collision avoidance reward / 官方机器人避碰惩罚
        safe_npc_dist = 2.0
        collision_npc_dist = 0.5
        obstacle_reward = 0.0

        npc_avoid_reward = 0.0
        if self.nearest_npc_dist <= collision_npc_dist:
            npc_avoid_reward = -0.5 # 距离过近的时候给个很大的惩罚
        elif self.nearest_npc_dist < safe_npc_dist:
            npc_avoid_reward = -0.1 * (safe_npc_dist - self.nearest_npc_dist)

        # 4) Low-battery charger reward / 低电量时接近充电桩奖励（固定奖励）
        battery_ratio = self.battery / max(self.battery_max, 1)
        charger_reward = 0.0
        if (
            self.step_no > 0
            and battery_ratio < 0.3
            and self.nearest_charger_dist < self.last_nearest_charger_dist):
            charger_reward = 0.1
        # 5) Step penalty / 时间惩罚
        step_penalty = -0.001

        total_reward = (
            cleaning_reward
            + dirt_approach_reward
            + obstacle_reward
            + charger_reward
            + step_penalty
        )

        reward_info = {
            "reward": float(total_reward),
            "cleaning_reward": float(cleaning_reward),
            "dirt_approach_reward": float(dirt_approach_reward),
            "obstacle_reward": float(obstacle_reward),
            "charger_reward": float(charger_reward),
            "step_penalty": float(step_penalty),
        }

        return total_reward, reward_info