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
    GLOBAL_REDUCED_SIZE = 16

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
        self.last_pos = (0, 0)               # 上一步位置
        self.no_move_count = 0               # 连续未移动步数
        self.wall_bump_count = 0             # 连续撞墙/顶障碍步数
        self.last_action = -1                # 上一时刻动作

        self.dirt_delta = 0.0
        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        self.clean_streak = 0                # 连续清扫步数

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        # Charger info
        self.charger_positions = []
        self.charger_cells = set()
        self.nearest_charger_dist = 999.0
        self.last_nearest_charger_dist = 999.0

        # Obstacle distance
        self.nearest_obstacle_dist = 999.0
        self.last_nearest_obstacle_dist = 999.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        # NPC info
        self.npc_positions = []
        self.last_npc_positions = []
        self.nearest_npc_dist = 999.0
        self.last_nearest_npc_dist = 999.0

        # Trajectory / exploration / dirt memory
        self.visited_counts = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        self.explored_mask = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.global_dirt_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.unique_visited = 0

        self.collision_type = "none"   # none / npc / charger / obstacle
        self.terminated = False
        self.reward_info = {}

        # Low battery guide
        self.low_battery_threshold = 0.30
        self.very_low_battery_threshold = 0.15

        self.rule_action = -1          # 当前规则动作，-1 表示无
        self.use_rule_guide = 0.0      # 当前状态是否建议使用规则引导
        self.on_charger = 0.0          # 当前是否在充电桩区域

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

    def _calc_nearest_charger_dist(self):
        """计算当前位置到最近充电桩中心的欧氏距离。"""
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

        dists = dists[dists > 0]
        if len(dists) == 0:
            return 999.0

        return float(np.min(dists))

    def _get_action_dirs(self):
        """动作编号与环境协议严格一致。"""
        return [
            (1, 0),    # 0: 右
            (1, -1),   # 1: 右上
            (0, -1),   # 2: 上
            (-1, -1),  # 3: 左上
            (-1, 0),   # 4: 左
            (-1, 1),   # 5: 左下
            (0, 1),    # 6: 下
            (1, 1),    # 7: 右下
        ]

    def _is_charger_cell_global(self, x, z):
        """判断全局坐标 (x, z) 是否属于充电桩区域。"""
        return (int(x), int(z)) in self.charger_cells

    def _build_collision_free_legal_action(self):
        """根据局部视野图构造无碰撞 legal_action。

        规则：
        - map_info == 0 表示障碍/边界，不可通行
        - 但若目标格属于充电桩区域，则仍视为可通行
        """
        if self._view_map is None:
            return [1] * 8

        center = self.VIEW_HALF
        action_dirs = self._get_action_dirs()
        legal = [1] * 8
        hx, hz = self.cur_pos

        for act, (dx, dz) in enumerate(action_dirs):
            r = center + dx
            c = center + dz
            gx = hx + dx
            gz = hz + dz

            if not (0 <= r < self._view_map.shape[0] and 0 <= c < self._view_map.shape[1]):
                legal[act] = 0
                continue

            cell_value = int(self._view_map[r, c])
            if cell_value == 0 and not self._is_charger_cell_global(gx, gz):
                legal[act] = 0
                continue

        return [int(x) for x in legal]

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        # Reset rule-guide state every step / 每步先重置规则引导状态
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        self.step_no = int(observation["step_no"])
        self.last_action = int(last_action) if last_action is not None else -1

        # Reset rule-guide state every step / 每步先重置规则引导状态
        self.rule_action = -1
        self.use_rule_guide = 0.0

        # 先保存上一时刻位置，再更新当前位置
        self.last_pos = self.cur_pos
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        # Battery / 电量
        self.last_battery = self.battery
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Chargers
        organs = frame_state.get("organs", [])
        self.charger_positions = self._get_charger_positions(organs)
        self.charger_cells = self._get_charger_cells(organs)

        self.last_nearest_charger_dist = self.nearest_charger_dist
        self.nearest_charger_dist = self._calc_nearest_charger_dist()

        # NPCs
        npcs = frame_state.get("npcs", [])
        self.last_npc_positions = list(self.npc_positions)
        self.npc_positions = self._get_npc_positions(npcs)

        self.last_nearest_npc_dist = self.nearest_npc_dist
        self.nearest_npc_dist = self._calc_nearest_npc_dist()

        # Local view map (21×21)
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)
            self._update_exploration_and_dirt_map(hx, hz)

        # Update nearest obstacle distance after _view_map updated
        self.last_nearest_obstacle_dist = self.nearest_obstacle_dist
        self.nearest_obstacle_dist = self._calc_nearest_obstacle_dist()

        # Legal actions: recompute by map_info + charger cells
        self._legal_act = self._build_collision_free_legal_action()

        self._update_trajectory()
        self.terminated = bool(env_obs.get("terminated", False))
        self.collision_type = self._get_collision_type()

    def _get_charger_positions(self, organs):
        """提取所有充电桩中心位置。"""
        charger_positions = []
        for organ in organs:
            if int(organ.get("sub_type", -1)) == 1:
                pos = organ.get("pos", {})
                charger_positions.append((int(pos["x"]), int(pos["z"])))
        return charger_positions

    def _get_charger_cells(self, organs):
        """获取充电桩占据的所有格子坐标。

        默认按中心点 + w/h 展开。
        """
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

    def _get_collision_type(self):
        """Classify collision type when episode terminates."""
        if not self.terminated:
            return "none"

        hx, hz = self.cur_pos

        if self.nearest_npc_dist <= 0.5:
            return "npc"

        if (hx, hz) in self.charger_cells:
            return "charger"

        return "obstacle"

    def _update_passable(self, hx, hz):
        """Write local view into global passable map."""
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _update_exploration_and_dirt_map(self, hx, hz):
        """Update explored cells and global dirt memory from local view."""
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2
        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    cell = int(view[ri, ci])
                    self.explored_mask[gx, gz] = 1
                    self.global_dirt_map[gx, gz] = 1 if cell == 2 else 0

    def _update_trajectory(self):
        """Update visited count and unique visited counter for current position."""
        x, z = self.cur_pos
        if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
            if self.visited_counts[x, z] == 0:
                self.unique_visited += 1
            self.visited_counts[x, z] += 1

    def _get_battery_dead_penalty(self):
        """电量耗尽惩罚。"""
        if self.battery <= 0:
            return -2.0
        return 0.0

    def _get_local_view_feature(self):
        """Local view feature (441D): use full 21×21 field of view."""
        return (self._view_map / 2.0).flatten()

    def _downsample_grid(self, grid):
        """Average-pool a 128×128 map into a 16×16 coarse map."""
        block = self.GRID_SIZE // self.GLOBAL_REDUCED_SIZE
        coarse = grid.reshape(
            self.GLOBAL_REDUCED_SIZE,
            block,
            self.GLOBAL_REDUCED_SIZE,
            block,
        ).mean(axis=(1, 3))
        return coarse.astype(np.float32)

    def _get_global_map_feature(self):
        """Global map memory feature (1024D)."""
        explored_map = self.explored_mask.astype(np.float32)
        dirt_map = self.global_dirt_map.astype(np.float32)
        obstacle_map = explored_map * (1.0 - self.passable_map.astype(np.float32))
        visited_map = np.clip(self.visited_counts.astype(np.float32), 0.0, 5.0) / 5.0

        coarse_maps = [
            self._downsample_grid(explored_map),
            self._downsample_grid(dirt_map),
            self._downsample_grid(obstacle_map),
            self._downsample_grid(visited_map),
        ]
        return np.concatenate([m.flatten() for m in coarse_maps], axis=0)
    def _get_clean_streak_reward(self):
        """连续清扫奖励。

        规则：
        - 本步如果有新清扫，则 clean_streak += 1
        - 否则清零
        - streak 越长，奖励越高，但只保留三档
        - 低电量时，不再鼓励连续清扫，直接关闭
        """
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)

        if cleaned_this_step > 0:
            self.clean_streak += 1
        else:
            self.clean_streak = 0

        if self.clean_streak <= 1:
            return 0.0

        if self._is_low_battery_mode():
            return 0.0

        reward = 0.02 * min(self.clean_streak - 1, 3)
        return float(reward)
    def _is_low_battery_mode(self):
        """是否进入低电量模式。"""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
        return battery_ratio < self.low_battery_threshold

    def _get_no_move_penalty(self):
        """位置不变惩罚。

        规则：
        - 如果当前位置和上一时刻相同，则认为本步没有有效移动
        - 连续不动时惩罚逐步增大
        - 如果当前在充电桩上且电量正在恢复，则不惩罚，避免把“正常充电停留”也打掉
        """
        if self.step_no <= 0:
            return 0.0

        on_charger = (self.cur_pos in self.charger_cells) or (self.collision_type == "charger")
        battery_gain = self.battery - self.last_battery

        # 在充电桩上且电量正在增加，允许停留
        if on_charger and battery_gain > 0:
            self.no_move_count = 0
            return 0.0

        if self.cur_pos == self.last_pos:
            self.no_move_count += 1

            # 单步不动惩罚 + 连续不动增强惩罚
            penalty = -0.05 - 0.02 * min(self.no_move_count - 1, 3)
            return float(penalty)

        self.no_move_count = 0
        return 0.0

    def _is_last_action_blocked(self):
        """判断上一动作是否朝向不可通行格子（墙/障碍）。"""
        if self.last_action < 0 or self.last_action >= 8:
            return False

        if self._view_map is None:
            return False

        dx, dz = self._get_action_dirs()[self.last_action]
        center = self.VIEW_HALF
        r = center + dx
        c = center + dz
        gx = self.cur_pos[0] + dx
        gz = self.cur_pos[1] + dz

        if not (0 <= r < self._view_map.shape[0] and 0 <= c < self._view_map.shape[1]):
            return True

        cell_value = int(self._view_map[r, c])
        if cell_value == 0 and not self._is_charger_cell_global(gx, gz):
            return True
        return False

    def _get_wall_bump_penalty(self):
        """连续顶墙惩罚。

        当“本步位置未变 + 上一步动作确实是朝向障碍/边界”时，
        认定为撞墙/顶墙行为，快速加大惩罚，促使策略尽快换方向。
        """
        if self.step_no <= 0:
            return 0.0

        on_charger = (self.cur_pos in self.charger_cells) or (self.collision_type == "charger")
        battery_gain = self.battery - self.last_battery
        if on_charger and battery_gain > 0:
            self.wall_bump_count = 0
            return 0.0

        if self.cur_pos == self.last_pos and self._is_last_action_blocked():
            self.wall_bump_count += 1
            penalty = -0.12 - 0.06 * min(self.wall_bump_count - 1, 4)
            return float(penalty)

        self.wall_bump_count = 0
        return 0.0

    def _get_global_state_feature(self):
        """Global state feature (13D)."""
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        battery_current_norm = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

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

        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)
        self.dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0

        return np.array(
            [
                step_norm,
                battery_ratio,
                battery_current_norm,
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
        """Find nearest dirt Euclidean distance from local view."""
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
        """Return legal action mask (8D list)."""
        return list(self._legal_act)

    def _get_npc_velocity_feature(self):
        """npc速度特征: [vx_norm, vz_norm, speed_norm]."""
        if not self.npc_positions or not self.last_npc_positions:
            return np.zeros(3, dtype=np.float32)

        cx, cz = self.cur_pos
        cur_idx = int(np.argmin([((x - cx) ** 2 + (z - cz) ** 2) for x, z in self.npc_positions]))
        px, pz = self.npc_positions[cur_idx]

        last_idx = int(np.argmin([((x - px) ** 2 + (z - pz) ** 2) for x, z in self.last_npc_positions]))
        lx, lz = self.last_npc_positions[last_idx]

        vx = float(px - lx)
        vz = float(pz - lz)
        speed = float(np.sqrt(vx * vx + vz * vz))
        vmax = 5.0
        return np.array(
            [_norm(vx, vmax, -vmax), _norm(vz, vmax, -vmax), _norm(speed, vmax)],
            dtype=np.float32
        )

    def _get_npc_approach_direction_feature(self):
        """npc接近方向特征: [dir_x, dir_z, approaching]."""
        if not self.npc_positions:
            return np.zeros(3, dtype=np.float32)

        hx, hz = self.cur_pos
        nearest_idx = int(np.argmin([((x - hx) ** 2 + (z - hz) ** 2) for x, z in self.npc_positions]))
        nx, nz = self.npc_positions[nearest_idx]
        dx = float(nx - hx)
        dz = float(nz - hz)
        dist = float(np.sqrt(dx * dx + dz * dz))
        if dist > 1e-6:
            dir_x = dx / dist
            dir_z = dz / dist
        else:
            dir_x, dir_z = 0.0, 0.0

        approaching = 1.0 if self.nearest_npc_dist < self.last_nearest_npc_dist else 0.0
        return np.array([dir_x, dir_z, approaching], dtype=np.float32)

    def _get_trajectory_feature(self):
        """轨迹特征: [visit_norm, revisit_flag, unique_visit_ratio]."""
        x, z = self.cur_pos
        visit = 0
        if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
            visit = int(self.visited_counts[x, z])

        visit_norm = _norm(min(visit, 10), 10)
        revisit_flag = 1.0 if visit > 1 else 0.0
        total_cells = float(self.GRID_SIZE * self.GRID_SIZE)
        unique_visit_ratio = float(self.unique_visited) / total_cells
        return np.array([visit_norm, revisit_flag, unique_visit_ratio], dtype=np.float32)

    def _get_global_dirt_distribution_feature(self):
        """全局污渍统计特征: [known_dirt_ratio, nearby_dirt_density, nearest_dirt_norm]."""
        explored = float(np.sum(self.explored_mask))
        known_dirt = float(np.sum(self.global_dirt_map))
        known_dirt_ratio = known_dirt / max(explored, 1.0)

        hx, hz = self.cur_pos
        r = 5
        x0, x1 = max(hx - r, 0), min(hx + r + 1, self.GRID_SIZE)
        z0, z1 = max(hz - r, 0), min(hz + r + 1, self.GRID_SIZE)
        near_patch = self.global_dirt_map[x0:x1, z0:z1]
        nearby_dirt_density = float(np.mean(near_patch)) if near_patch.size > 0 else 0.0

        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 200.0)
        return np.array([known_dirt_ratio, nearby_dirt_density, nearest_dirt_norm], dtype=np.float32)

    def _get_exploration_uncleaned_feature(self):
        """探索/未清扫特征: [explore_ratio, unknown_ratio, uncleaned_ratio]."""
        total_cells = float(self.GRID_SIZE * self.GRID_SIZE)
        explored = float(np.sum(self.explored_mask))
        explore_ratio = explored / total_cells
        unknown_ratio = 1.0 - explore_ratio

        known_dirt = float(np.sum(self.global_dirt_map))
        uncleaned_ratio = known_dirt / max(explored, 1.0)
        return np.array([explore_ratio, unknown_ratio, uncleaned_ratio], dtype=np.float32)

    def _get_low_battery_feature(self):
        """低电量回充特征 (8D)."""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
        is_low_battery = 1.0 if battery_ratio < self.low_battery_threshold else 0.0
        is_very_low_battery = 1.0 if battery_ratio < self.very_low_battery_threshold else 0.0

        nearest_charger_dist_norm = _norm(self.nearest_charger_dist, 180.0)

        charger_dir_x = 0.0
        charger_dir_z = 0.0
        if self.charger_positions:
            hx, hz = self.cur_pos
            nearest_idx = int(np.argmin([
                (cx - hx) ** 2 + (cz - hz) ** 2 for cx, cz in self.charger_positions
            ]))
            cx, cz = self.charger_positions[nearest_idx]
            dx = float(cx - hx)
            dz = float(cz - hz)
            dist = float(np.sqrt(dx * dx + dz * dz))
            if dist > 1e-6:
                charger_dir_x = dx / dist
                charger_dir_z = dz / dist

        charger_approaching = 1.0 if self.nearest_charger_dist < self.last_nearest_charger_dist else 0.0
        on_charger = 1.0 if (self.cur_pos in self.charger_cells) or (self.collision_type == "charger") else 0.0

        self.on_charger = on_charger

        return np.array(
            [
                battery_ratio,
                is_low_battery,
                is_very_low_battery,
                nearest_charger_dist_norm,
                charger_dir_x,
                charger_dir_z,
                charger_approaching,
                on_charger,
            ],
            dtype=np.float32,
        )

    def feature_process(self, env_obs, last_action):
        """Generate feature vector, legal action mask, and scalar reward."""
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()          # 441D
        global_map = self._get_global_map_feature()          # 1024D
        global_state = self._get_global_state_feature()      # 13D
        legal_action = self.get_legal_action()               # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)
        npc_velocity = self._get_npc_velocity_feature()      # 3D
        npc_approach = self._get_npc_approach_direction_feature()  # 3D
        trajectory = self._get_trajectory_feature()          # 3D
        global_dirt = self._get_global_dirt_distribution_feature() # 3D
        exploration = self._get_exploration_uncleaned_feature()    # 3D
        low_battery_feat = self._get_low_battery_feature()   # 8D

        feature = np.concatenate(
            [
                local_view,
                global_map,
                global_state,
                legal_arr,
                npc_velocity,
                npc_approach,
                trajectory,
                global_dirt,
                exploration,
                low_battery_feat,
            ]
        )  # 1509D

        reward, reward_info = self.reward_process()
        self.reward_info = reward_info
        return feature, legal_action, reward

    def _get_cleaning_reward(self):
        """清扫奖励。

        高电量时鼓励清扫。
        低电量时只保留很弱的清扫奖励，避免完全失去清扫反馈，
        但不再让它与回充目标对抗。
        """
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        if cleaned_this_step <= 0:
            return 0.0

        base_reward = 0.1 * cleaned_this_step

        if self._is_low_battery_mode():
            base_reward *= 0.1

        return float(base_reward)

    def _get_dirt_approach_reward(self):
        """接近污渍奖励。

        低电量时关闭，避免与回充目标冲突。
        """
        if self._is_low_battery_mode():
            return 0.0

        if self.step_no > 0 and self.dirt_delta > 0:
            return 0.05
        return 0.0

    def _get_npc_avoid_reward(self):
        """官方机器人避碰惩罚。"""
        safe_npc_dist = 2.0
        collision_npc_dist = 0.5

        if self.nearest_npc_dist <= collision_npc_dist:
            return -0.5
        elif self.nearest_npc_dist < safe_npc_dist:
            return -0.1 * (safe_npc_dist - self.nearest_npc_dist)
        return 0.0

    def _get_low_battery_charger_approach_reward(self):
        """低电量时接近充电桩奖励。"""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
        if self.step_no <= 0 or battery_ratio >= self.low_battery_threshold:
            return 0.0

        delta = float(self.last_nearest_charger_dist - self.nearest_charger_dist)
        reward = 0.12 * np.clip(delta, -2.0, 2.0)
        return float(reward)

    def _get_low_battery_charging_reward(self):
        """低电量时到达充电桩和实际充电奖励。"""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
        on_charger = (self.cur_pos in self.charger_cells) or (self.collision_type == "charger")

        if battery_ratio >= self.low_battery_threshold or not on_charger:
            return 0.0

        reward = 1.0
        battery_gain = self.battery - self.last_battery
        if battery_gain > 0:
            reward += 0.3 + 0.02 * battery_gain

        return float(reward)

    def _get_low_battery_leave_charger_penalty(self):
        """低电量时远离充电桩惩罚。"""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
        if self.step_no <= 0 or battery_ratio >= self.low_battery_threshold:
            return 0.0

        delta = float(self.nearest_charger_dist - self.last_nearest_charger_dist)
        if delta > 0:
            return -0.05 * min(delta, 2.0)
        return 0.0

    def _get_step_penalty(self):
        """时间步惩罚。"""
        return -0.001

    def get_rule_action(self):
        """返回低电量时的规则动作。"""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)

        if battery_ratio >= self.low_battery_threshold:
            self.rule_action = -1
            self.use_rule_guide = 0.0
            return -1

        if not self.charger_positions:
            self.rule_action = -1
            self.use_rule_guide = 0.0
            return -1

        hx, hz = self.cur_pos

        nearest_idx = int(np.argmin([
            (cx - hx) ** 2 + (cz - hz) ** 2 for cx, cz in self.charger_positions
        ]))
        cx, cz = self.charger_positions[nearest_idx]

        dx = float(cx - hx)
        dz = float(cz - hz)

        if abs(dx) < 1e-6 and abs(dz) < 1e-6:
            self.rule_action = -1
            self.use_rule_guide = 0.0
            return -1

        dist = np.sqrt(dx * dx + dz * dz)
        target_dir = (dx / dist, dz / dist)

        action_dirs = self._get_action_dirs()

        best_act = -1
        best_score = -1e9

        for act, (ax, az) in enumerate(action_dirs):
            if self._legal_act[act] != 1:
                continue

            move_len = np.sqrt(ax * ax + az * az)
            move_dir = (ax / move_len, az / move_len)

            direction_score = move_dir[0] * target_dir[0] + move_dir[1] * target_dir[1]

            nx = hx + ax
            nz = hz + az
            new_dist = np.sqrt((cx - nx) ** 2 + (cz - nz) ** 2)
            approach_bonus = 1.0 if new_dist < dist else -1.0

            score = direction_score + 0.5 * approach_bonus

            if score > best_score:
                best_score = score
                best_act = act

        self.rule_action = best_act
        self.use_rule_guide = 1.0 if best_act >= 0 else 0.0
        return best_act

    def reward_process(self):
        """Compute reward with stage-aware design.

        分阶段奖励设计：
        - 高电量：以清扫为主
        - 低电量：以回充为主
        """
        low_battery_mode = self._is_low_battery_mode()

        # ===== 基础奖励 =====
        cleaning_reward = self._get_cleaning_reward()
        clean_streak_reward = self._get_clean_streak_reward()
        dirt_approach_reward = self._get_dirt_approach_reward()
        npc_avoid_reward = self._get_npc_avoid_reward()

        charger_approach_reward = self._get_low_battery_charger_approach_reward()
        leave_charger_penalty = self._get_low_battery_leave_charger_penalty()
        charging_reward = self._get_low_battery_charging_reward()
        battery_dead_penalty = self._get_battery_dead_penalty()

        no_move_penalty = self._get_no_move_penalty()
        wall_bump_penalty = self._get_wall_bump_penalty()
        step_penalty = self._get_step_penalty()

        obstacle_reward = npc_avoid_reward
        charger_reward = (
            charger_approach_reward
            + leave_charger_penalty
            + charging_reward
            + battery_dead_penalty
        )

        # ===== 分阶段组合 =====
        if not low_battery_mode:
            # 高电量：主任务是清扫
            total_reward = (
                cleaning_reward
                + clean_streak_reward
                + dirt_approach_reward
                + obstacle_reward
                + no_move_penalty
                + wall_bump_penalty
                + step_penalty
            )
        else:
            # 低电量：主任务切换为回充
            total_reward = (
                cleaning_reward          # 已在函数内部强降权
                + obstacle_reward
                + charger_reward
                + no_move_penalty
                + wall_bump_penalty
                + step_penalty
            )

        reward_info = {
            "reward": float(total_reward),
            "cleaning_reward": float(cleaning_reward),
            "clean_streak_reward": float(clean_streak_reward),
            "dirt_approach_reward": float(dirt_approach_reward),
            "obstacle_reward": float(obstacle_reward),
            "charger_reward": float(charger_reward),
            "npc_avoid_reward": float(npc_avoid_reward),
            "charger_approach_reward": float(charger_approach_reward),
            "leave_charger_penalty": float(leave_charger_penalty),
            "charging_reward": float(charging_reward),
            "battery_dead_penalty": float(battery_dead_penalty),
            "no_move_penalty": float(no_move_penalty),
            "wall_bump_penalty": float(wall_bump_penalty),
            "step_penalty": float(step_penalty),
            "clean_streak": int(self.clean_streak),
            "no_move_count": int(self.no_move_count),
            "wall_bump_count": int(self.wall_bump_count),
            "rule_action": int(self.rule_action),
            "use_rule_guide": float(self.use_rule_guide),
            "low_battery_mode": float(low_battery_mode),
            "battery_ratio": float(self.battery / max(self.battery_max, 1)),
            "nearest_charger_dist": float(self.nearest_charger_dist),
        }

        return total_reward, reward_info