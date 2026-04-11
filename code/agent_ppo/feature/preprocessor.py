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
    if v is None:
        return 1.0
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10   # Full local view radius (21×21)
    LOCAL_HALF = 3   # Cropped view radius (7×7)

    ACTION_TO_DELTA = {
        0: (1, 0),    # 右
        1: (1, -1),   # 右上
        2: (0, -1),   # 上
        3: (-1, -1),  # 左上
        4: (-1, 0),   # 左
        5: (-1, 1),   # 左下
        6: (0, 1),    # 下
        7: (1, 1),    # 右下
    }

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 600
        self.last_battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)
        self.prev_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        # 全局通行图：0=障碍物，1=可通行
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # 最近污渍距离（8邻域步数）
        self.nearest_dirt_dist = None
        self.last_nearest_dirt_dist = None
        self.dirt_delta = 0.0

        # 充电桩信息
        self.charger_positions = []
        self.charger_cells = set()
        self.nearest_charger_dist = 999.0
        self.last_nearest_charger_dist = 999.0

        # 官方机器人信息
        self.npc_positions = []
        self.nearest_npc_dist = 999.0
        self.last_nearest_npc_dist = 999.0

        # 连续清扫 / 空闲统计
        self.clean_streak = 0   # 连续清扫步数
        self.idle_streak = 0    # 连续空闲步数

        # 终止 / 碰撞类型
        self.terminated = False
        self.collision_type = "none"  # none / npc / charger / obstacle

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        self.reward_info = {
            "reward": 0.0,
            "cleaning_reward": 0.0,
            "dirt_approach_reward": 0.0,
            "npc_avoid_reward": 0.0,
            "charger_reward": 0.0,
            "collision_reward": 0.0,
            "efficiency_reward": 0.0,
            "step_penalty": 0.0,
            # 兼容旧名字
            "obstacle_reward": 0.0,
        }

    # =========================
    # 解析环境信息
    # =========================

    def _get_npc_positions(self, npcs):
        """Extract npc positions from frame_state['npcs'].""" 
        npc_positions = []
        for npc in npcs:
            pos = npc.get("pos", {})
            npc_positions.append((int(pos["x"]), int(pos["z"])))
        return npc_positions

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

    def _get_charger_positions(self, organs):
        """Extract charger center positions."""
        charger_positions = []
        for organ in organs:
            if int(organ.get("sub_type", -1)) == 1:
                pos = organ.get("pos", {})
                charger_positions.append((int(pos["x"]), int(pos["z"])))
        return charger_positions

    def _get_charger_cells(self, organs):
        """Extract all occupied cells of chargers."""
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

    def _calc_nearest_charger_dist(self):
        """Compute Euclidean distance to the nearest charger center."""
        if not self.charger_positions:
            return 999.0

        hx, hz = self.cur_pos
        min_dist = 999.0
        for cx, cz in self.charger_positions:
            dist = np.sqrt((cx - hx) ** 2 + (cz - hz) ** 2)
            min_dist = min(min_dist, dist)
        return float(min_dist)

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
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _get_target_pos_from_last_action(self, last_action):
        """Infer intended next position from prev_pos and last_action."""
        if last_action not in self.ACTION_TO_DELTA:
            return self.cur_pos

        px, pz = self.prev_pos
        dx, dz = self.ACTION_TO_DELTA[last_action]
        return px + dx, pz + dz

    def _get_collision_type_from_next_pos(self, last_action):
        """Classify collision target from intended next grid.

        返回:
          - "npc"      : 下一位置撞到官方机器人
          - "charger"  : 下一位置进入充电桩区域
          - "obstacle" : 下一位置进入普通障碍物/墙壁
          - "none"     : 无碰撞
        """
        if last_action not in self.ACTION_TO_DELTA:
            return "none"

        nx, nz = self._get_target_pos_from_last_action(last_action)

        # 出界视为普通障碍物/墙壁
        if not (0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE):
            return "obstacle"

        # 官方机器人优先
        for px, pz in self.npc_positions:
            if (nx, nz) == (px, pz):
                return "npc"

        # 充电桩区域
        if (nx, nz) in self.charger_cells:
            return "charger"

        # 其他普通障碍物/墙壁
        if self.passable_map[nx, nz] == 0:
            return "obstacle"

        return "none"

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        # 保存上一时刻位置
        self.prev_pos = self.cur_pos

        self.step_no = int(observation["step_no"])
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        # 电量
        self.last_battery = self.battery
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # 合法动作
        self._legal_act = [
            int(x) for x in (
                observation.get("legal_act")
                or observation.get("legal_action")
                or [1] * 8
            )
        ]

        # 局部地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)

        # 官方机器人信息
        npcs = frame_state.get("npcs", [])
        self.npc_positions = self._get_npc_positions(npcs)
        self.last_nearest_npc_dist = self.nearest_npc_dist
        self.nearest_npc_dist = self._calc_nearest_npc_dist()

        # 充电桩信息
        organs = frame_state.get("organs", [])
        self.charger_positions = self._get_charger_positions(organs)
        self.charger_cells = self._get_charger_cells(organs)
        self.last_nearest_charger_dist = self.nearest_charger_dist
        self.nearest_charger_dist = self._calc_nearest_charger_dist()

        # 终止状态 + 碰撞类型
        self.terminated = bool(env_obs.get("terminated", False))
        self.collision_type = self._get_collision_type_from_next_pos(last_action)

    # =========================
    # 特征构造
    # =========================

    def _get_local_view_feature(self):
        """Local view feature (49D): crop center 7×7 from 21×21."""
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h:center + h + 1, center - h:center + h + 1]
        return (crop / 2.0).flatten()

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt distance in Chebyshev metric (8-neighbor steps)."""
        view = self._view_map
        if view is None:
            return None

        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return None

        center = self.VIEW_HALF
        dx = np.abs(dirt_coords[:, 0] - center)
        dz = np.abs(dirt_coords[:, 1] - center)
        dists = np.maximum(dx, dz)   # 8邻域下的最少步数
        return float(np.min(dists))

    def _get_global_state_feature(self):
        """Global state feature (12D)."""
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 四方向射线找最近污渍距离
        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W
        ray_dirt = []
        max_ray = 30

        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break

                if self._view_map is not None:
                    local_r = x - hx + self.VIEW_HALF
                    local_c = z - hz + self.VIEW_HALF

                    if 0 <= local_r < 21 and 0 <= local_c < 21:
                        cell = int(self._view_map[local_r, local_c])
                    else:
                        cell = 0

                    if cell == 2:
                        found = step
                        break

            ray_dirt.append(_norm(found, max_ray))

        # 最近污渍距离（8邻域最少步数）
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()

        if self.nearest_dirt_dist is None:
            nearest_dirt_norm = 1.0
        else:
            nearest_dirt_norm = _norm(self.nearest_dirt_dist, self.VIEW_HALF)

        if (
            self.last_nearest_dirt_dist is not None
            and self.nearest_dirt_dist is not None
            and self.nearest_dirt_dist < self.last_nearest_dirt_dist
        ):
            self.dirt_delta = 1.0
        else:
            self.dirt_delta = 0.0

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

    def get_legal_action(self):
        """Return legal action mask (8D list)."""
        return list(self._legal_act)

    def feature_process(self, env_obs, last_action):
        """Generate 69D feature vector, legal action mask, and scalar reward."""
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()      # 49D
        global_state = self._get_global_state_feature()  # 12D
        legal_action = self.get_legal_action()           # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)

        feature = np.concatenate([local_view, global_state, legal_arr])  # 69D

        reward, reward_info = self.reward_process()
        self.reward_info = reward_info

        return feature, legal_action, reward

    # =========================
    # 奖励函数接口
    # =========================

    def reward_cleaning(self, cleaned_this_step):
        """主任务奖励：清扫到新污渍。"""
        return 1.0 * cleaned_this_step

    def reward_approach_dirt(self):
        """接近污渍奖励：按8邻域最少步数差分给奖励。"""
        if self.step_no <= 0:
            return 0.0

        if self.last_nearest_dirt_dist is None or self.nearest_dirt_dist is None:
            return 0.0

        delta = self.last_nearest_dirt_dist - self.nearest_dirt_dist
        delta = np.clip(delta, -1.0, 1.0)
        return 0.15 * delta

    def reward_npc_avoid(self):
        """官方机器人避碰惩罚。"""
        safe_npc_dist = 2.0
        collision_npc_dist = 1.0

        if self.nearest_npc_dist <= collision_npc_dist:
            return -0.4
        if self.nearest_npc_dist < safe_npc_dist:
            return -0.10 * (safe_npc_dist - self.nearest_npc_dist)
        return 0.0

    def reward_low_battery_charger(self):
        """充电相关奖励。

        包含两部分：
        1. 低电量时接近充电桩奖励
        2. 当前步实际充电奖励（低电量更大，高电量很小）
        """
        if self.step_no <= 0 or not self.charger_positions:
            return 0.0

        reward = 0.0
        battery_ratio = self.battery / max(self.battery_max, 1)
        last_battery_ratio = self.last_battery / max(self.battery_max, 1)

        # --------------------------------
        # 1) 接近充电桩奖励
        # --------------------------------
        if self.last_nearest_charger_dist is not None and self.nearest_charger_dist is not None:
            delta_dist = self.last_nearest_charger_dist - self.nearest_charger_dist
            delta_dist = float(np.clip(delta_dist, -1.0, 1.0))

            # 只有低电量时，靠近充电桩才值得鼓励
            if battery_ratio < 0.15:
                reward += 0.30 * delta_dist
            elif battery_ratio < 0.30:
                reward += 0.18 * delta_dist
            else:
                # 高电量时如果还故意靠近充电桩，轻微抑制
                if delta_dist > 0:
                    reward += -0.02 * delta_dist

        # --------------------------------
        # 2) 当前步实际充电奖励
        # --------------------------------
        battery_gain = self.battery - self.last_battery
        if battery_gain > 0:
            gain = float(np.clip(battery_gain, 0.0, 20.0))

            # 上一步越缺电，这一步充上电的奖励越大
            if last_battery_ratio < 0.15:
                reward += 0.35 + 0.02 * gain
            elif last_battery_ratio < 0.30:
                reward += 0.20 + 0.01 * gain
            else:
                # 高电量时充电，只给一个很小奖励
                reward += 0.03 + 0.002 * gain

        return reward

    def reward_collision(self):
        """碰撞类型惩罚。

        - npc: 大惩罚
        - obstacle: 小惩罚
        - charger: 不惩罚
        """
        if self.collision_type == "npc":
            return -1.0
        if self.collision_type == "obstacle":
            return -0.3
        if self.collision_type == "charger":
            return 0.0
        return 0.0

    def reward_efficiency(self, cleaned_this_step):
        """效率奖励：奖励连续清扫，惩罚连续空闲。"""
        if cleaned_this_step > 0:
            self.clean_streak += 1
            self.idle_streak = 0
            return 0.05 * min(self.clean_streak, 2)

        self.idle_streak += 1
        self.clean_streak = 0
        if self.step_no > 0:
            return -0.01 * min(self.idle_streak, 3)
        return 0.0

    def reward_time(self):
        """时间惩罚。"""
        return -0.02

    def reward_process(self):
        """Compute total reward and reward breakdown."""
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)

        cleaning_reward = self.reward_cleaning(cleaned_this_step)
        dirt_approach_reward = self.reward_approach_dirt()
        npc_avoid_reward = self.reward_npc_avoid()
        charger_reward = self.reward_low_battery_charger()
        collision_reward = self.reward_collision()
        efficiency_reward = self.reward_efficiency(cleaned_this_step)
        step_penalty = self.reward_time()

        total_reward = (
            cleaning_reward
            + dirt_approach_reward
            #+ npc_avoid_reward
            + charger_reward
            #+ collision_reward
            + efficiency_reward
            + step_penalty
        )

        reward_info = {
            "reward": float(total_reward),
            "cleaning_reward": float(cleaning_reward),
            "dirt_approach_reward": float(dirt_approach_reward),
            "npc_avoid_reward": float(npc_avoid_reward),
            "charger_reward": float(charger_reward),
            "collision_reward": float(collision_reward),
            "efficiency_reward": float(efficiency_reward),
            "step_penalty": float(step_penalty),
            "charging_reward": 0.0,

            # 兼容旧监控命名
            "obstacle_reward": float(npc_avoid_reward + collision_reward),
        }

        return total_reward, reward_info