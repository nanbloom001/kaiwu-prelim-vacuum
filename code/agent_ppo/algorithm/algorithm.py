#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm and rule-based coverage planner for Robot Vacuum.
"""

from __future__ import annotations

import heapq
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agent_ppo.conf.conf import Config


class Algorithm:

    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in optimizer.param_groups for p in pg["params"]]
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF
        self.var_beta = Config.BETA_START
        self.label_size = Config.ACTION_NUM

        self.train_step = 0
        self.last_report_time = 0

    def learn(self, list_sample_data: list) -> dict:
        obs = torch.stack([s.obs for s in list_sample_data]).to(self.device)
        legal_action = torch.stack([s.legal_action for s in list_sample_data]).to(self.device)
        act = torch.stack([s.act for s in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([s.prob for s in list_sample_data]).to(self.device)
        planner_prob = torch.stack([s.planner_prob for s in list_sample_data]).to(self.device)
        mix_alpha = torch.stack([s.mix_alpha for s in list_sample_data]).to(self.device)
        old_value = torch.stack([s.value for s in list_sample_data]).to(self.device)
        reward_sum = torch.stack([s.reward_sum for s in list_sample_data]).to(self.device)
        advantage = torch.stack([s.advantage for s in list_sample_data]).to(self.device)
        reward = torch.stack([s.reward for s in list_sample_data]).to(self.device)

        adv = advantage.squeeze(-1) if advantage.dim() > 1 else advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(obs)
        logits = rst_list[0]
        value_pred = rst_list[1]

        total_loss, info = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            planner_prob=planner_prob,
            mix_alpha=mix_alpha,
            old_value=old_value,
            reward_sum=reward_sum,
            advantage=adv,
        )

        total_loss.backward()
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        results = {"total_loss": float(total_loss.item())}
        now = time.time()
        if now - self.last_report_time >= 60:
            results.update(
                {
                    "value_loss": round(info["value_loss"], 4),
                    "policy_loss": round(info["policy_loss"], 4),
                    "entropy_loss": round(info["entropy_loss"], 4),
                    "bc_loss": round(info["bc_loss"], 4),
                    "approx_kl": round(info["approx_kl"], 4),
                    "mix_alpha": round(info["mix_alpha"], 4),
                    "reward": round(reward.mean().item(), 4),
                }
            )
            if self.logger:
                self.logger.info(
                    f"[step {self.train_step}] "
                    f"policy={results['policy_loss']} "
                    f"value={results['value_loss']} "
                    f"entropy={results['entropy_loss']} "
                    f"bc={results['bc_loss']} "
                    f"alpha={results['mix_alpha']} "
                    f"reward={results['reward']}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_time = now
        return results

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        planner_prob,
        mix_alpha,
        old_value,
        reward_sum,
        advantage,
    ):
        tdret = reward_sum.squeeze(-1) if reward_sum.dim() > 1 else reward_sum
        vp = value_pred.squeeze(-1) if value_pred.dim() > 1 else value_pred
        ov = old_value.squeeze(-1) if old_value.dim() > 1 else old_value

        vp_clipped = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = 0.5 * torch.max((tdret - vp) ** 2, (tdret - vp_clipped) ** 2).mean()

        policy_prob = self._masked_softmax(logits, legal_action)
        mixed_prob = self._mix_policy(policy_prob, planner_prob, mix_alpha, legal_action)
        entropy_loss = (-(mixed_prob * torch.log(mixed_prob.clamp(1e-9, 1.0))).sum(1)).mean()

        one_hot = F.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * mixed_prob).sum(1, keepdim=True)
        old_act_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_act_prob

        adv = advantage.unsqueeze(-1) if advantage.dim() == 1 else advantage
        policy_loss = torch.max(
            -ratio * adv,
            -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv,
        ).mean()

        bc_loss = -(planner_prob * torch.log(policy_prob.clamp(1e-9, 1.0))).sum(1).mean()

        alpha_mean = float(mix_alpha.mean().item())
        alpha_norm = np.clip(alpha_mean / max(Config.RESIDUAL_ALPHA_MAX, 1e-6), 0.0, 1.0)
        bc_coef = max(Config.BC_COEF_MIN, Config.BC_COEF_START * (1.0 - alpha_norm) ** 2)
        self.var_beta = Config.BETA_END + (Config.BETA_START - Config.BETA_END) * (1.0 - alpha_norm)

        approx_kl = (
            old_prob.clamp(1e-9, 1.0)
            * (torch.log(old_prob.clamp(1e-9, 1.0)) - torch.log(mixed_prob.clamp(1e-9, 1.0)))
        ).sum(1).mean()

        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss + bc_coef * bc_loss
        return total_loss, {
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy_loss": float(entropy_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "approx_kl": float(approx_kl.item()),
            "mix_alpha": alpha_mean,
        }

    def _masked_softmax(self, logits: torch.Tensor, legal_action: torch.Tensor) -> torch.Tensor:
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        logits = (logits - label_max) * legal_action
        logits = logits + 1e5 * (legal_action - 1)
        return F.softmax(logits, dim=1)

    def _mix_policy(
        self,
        policy_prob: torch.Tensor,
        planner_prob: torch.Tensor,
        mix_alpha: torch.Tensor,
        legal_action: torch.Tensor,
    ) -> torch.Tensor:
        alpha = mix_alpha
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        alpha = alpha.clamp(0.0, 1.0)
        mixed = (1.0 - alpha) * planner_prob + alpha * policy_prob
        mixed = mixed * legal_action
        return mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-9)


Position = Tuple[int, int]


@dataclass
class PolicyInfo:
    safe_action_mask: np.ndarray
    action_scores: np.ndarray
    chosen_action: int
    greedy_action: int
    target_mode: str
    target_pos: Optional[Position]
    target_distance: float
    battery: float
    battery_ratio: float
    charger_distance: float
    charger_slack: float
    nearest_npc_distance: float
    frontier_density: float
    local_dirty_ratio: float
    local_unknown_ratio: float
    new_known_cells: int
    on_charger: bool
    should_charge: bool


class CoveragePlanner:
    UNKNOWN = -1
    OBSTACLE = 0
    CLEAN = 1
    DIRT = 2

    MAP_SIZE = 128
    VIEW_RADIUS = 10
    ACTION_TO_DELTA: Sequence[Position] = (
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )
    DIAGONAL_ACTIONS = {1, 3, 5, 7}
    PATH_ACTION_ORDER = (1, 3, 5, 7, 0, 2, 4, 6)

    BASE_RETURN_MARGIN = 22.0
    NPC_RETURN_MARGIN = 28.0
    LOW_BATTERY_RATIO = 0.30
    EXIT_RETURN_RATIO = 0.95
    EXPANSION_KNOWN_RATIO = 0.78
    HARD_NPC_RADIUS = 2
    PATH_RISK_RADIUS = 4
    AGGRESSIVE_EDGE_STEPS = 500

    def __init__(self):
        self.reset()

    def set_episode_config(self, max_step=None, robot_count=None, charger_count=None, battery_max=None):
        if max_step is not None:
            self.episode_max_step = max(1, int(max_step))
        if robot_count is not None:
            self.episode_robot_count = max(1, int(robot_count))
        if charger_count is not None:
            self.episode_charger_count = max(1, int(charger_count))
        if battery_max is not None:
            self.episode_battery_max = max(1, int(battery_max))

    def reset(self):
        self.global_map = np.full((self.MAP_SIZE, self.MAP_SIZE), self.UNKNOWN, dtype=np.int8)
        self.visit_count = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.int16)
        self.charger_regions: List[Set[Position]] = []
        self.return_mode = False
        self.last_target_pos = None
        self.last_policy_info = None
        self.episode_max_step = 1000
        self.episode_robot_count = 4
        self.episode_charger_count = 4
        self.episode_battery_max = 200

    def update(self, env_obs: Any, last_action: int = -1) -> PolicyInfo:
        del last_action
        obs = self._unwrap_observation(env_obs)
        frame_state = self._get(obs, "frame_state", {})
        hero = self._parse_hero_state(frame_state)
        hero_pos = self._parse_position(self._get(hero, "pos", {}))
        battery = self._safe_float(self._get(hero, "battery", self.episode_battery_max), self.episode_battery_max)
        battery_max = max(
            1.0,
            self._safe_float(self._get(hero, "battery_max", self.episode_battery_max), self.episode_battery_max),
        )
        battery_ratio = float(np.clip(battery / battery_max, 0.0, 1.0))
        step_no = int(self._safe_float(self._get(obs, "step_no", 0), 0.0))
        map_info = self._parse_map_info(obs)
        npcs = self._parse_npc_positions(frame_state)
        self.charger_regions = self._parse_charger_regions(self._parse_organ_states(frame_state))

        new_known_cells = self._merge_local_view(hero_pos, map_info)
        self.visit_count[hero_pos[1], hero_pos[0]] += 1

        charger_cells = self._charger_cells()
        charger_path, charger_distance = self._path_to_any(hero_pos, charger_cells, allow_unknown=False)
        if charger_path is None:
            charger_path, charger_distance = self._path_to_any(hero_pos, charger_cells, allow_unknown=True)
        if not np.isfinite(charger_distance):
            charger_distance = 999.0

        on_charger = self._hero_on_charger(hero_pos)
        nearest_npc_distance = self._nearest_npc_dist(hero_pos, npcs)
        known_ratio = float(np.mean(self.global_map != self.UNKNOWN))
        local_dirty_ratio = float(np.mean(map_info == self.DIRT)) if map_info.size > 0 else 0.0
        local_unknown_ratio = self._local_unknown_ratio(hero_pos)
        frontier_density = self._frontier_density(hero_pos)

        should_charge = False
        if charger_cells:
            danger_margin = self.BASE_RETURN_MARGIN
            if nearest_npc_distance <= self.PATH_RISK_RADIUS:
                danger_margin += self.NPC_RETURN_MARGIN - self.BASE_RETURN_MARGIN
            if self.return_mode:
                should_charge = True
            elif battery_ratio <= self.LOW_BATTERY_RATIO:
                should_charge = True
            elif battery <= charger_distance + danger_margin:
                should_charge = True

        if on_charger and battery_ratio >= self.EXIT_RETURN_RATIO:
            self.return_mode = False
        elif should_charge:
            self.return_mode = True

        target_mode = "fallback"
        target_pos = None
        target_path = None
        target_distance = 999.0
        if self.return_mode and charger_cells:
            target_mode = "charge"
            target_path = charger_path
            target_distance = charger_distance
            if target_path:
                target_pos = target_path[-1]
        else:
            target_mode, target_pos, target_path, target_distance = self._select_coverage_plan(
                hero_pos=hero_pos,
                step_no=step_no,
                known_ratio=known_ratio,
                battery=battery,
                charger_distance=charger_distance,
            )

        action_scores, safe_action_mask = self._rank_legal_actions(
            hero_pos=hero_pos,
            target_mode=target_mode,
            target_pos=target_pos,
            target_path=target_path,
            npcs=npcs,
        )
        greedy_action = int(np.argmax(action_scores))
        chosen_action = greedy_action

        info = PolicyInfo(
            safe_action_mask=safe_action_mask.astype(np.float32),
            action_scores=action_scores.astype(np.float32),
            chosen_action=chosen_action,
            greedy_action=greedy_action,
            target_mode=target_mode,
            target_pos=target_pos,
            target_distance=float(target_distance),
            battery=float(battery),
            battery_ratio=float(battery_ratio),
            charger_distance=float(charger_distance),
            charger_slack=float(battery - charger_distance),
            nearest_npc_distance=float(nearest_npc_distance),
            frontier_density=float(frontier_density),
            local_dirty_ratio=float(local_dirty_ratio),
            local_unknown_ratio=float(local_unknown_ratio),
            new_known_cells=int(new_known_cells),
            on_charger=bool(on_charger),
            should_charge=bool(self.return_mode),
        )
        self.last_policy_info = info
        self.last_target_pos = target_pos
        return info

    def _select_coverage_plan(self, hero_pos, step_no, known_ratio, battery, charger_distance):
        charger_unknown = len(self.charger_regions) == 0
        if charger_unknown and step_no <= self.AGGRESSIVE_EDGE_STEPS:
            target = self._best_frontier_target(hero_pos, prefer_edge=True, battery=battery, charger_distance=charger_distance)
            if target[0] is not None:
                return "find_charger_edge", target[0], target[1], target[2]

        if known_ratio < self.EXPANSION_KNOWN_RATIO:
            target = self._best_frontier_target(hero_pos, prefer_edge=True, battery=battery, charger_distance=charger_distance)
            if target[0] is not None:
                return "edge_frontier", target[0], target[1], target[2]

        dirt_target = self._best_dirt_target(hero_pos, battery=battery, charger_distance=charger_distance)
        frontier_target = self._best_frontier_target(hero_pos, prefer_edge=False, battery=battery, charger_distance=charger_distance)
        if dirt_target[0] is not None and (frontier_target[0] is None or dirt_target[3] >= frontier_target[3] - 0.1):
            return "dirt", dirt_target[0], dirt_target[1], dirt_target[2]
        if frontier_target[0] is not None:
            return "frontier", frontier_target[0], frontier_target[1], frontier_target[2]
        return "fallback", None, None, 999.0

    def _best_dirt_target(self, hero_pos, battery, charger_distance):
        best = (None, None, 999.0, float("-inf"))
        dirt_cells = np.argwhere(self.global_map == self.DIRT)
        for row, col in dirt_cells[:512]:
            target = (int(col), int(row))
            path, dist = self._path_to_any(hero_pos, [target], allow_unknown=False)
            if path is None:
                continue
            if not self._target_is_safe(target, battery, charger_distance):
                continue
            dirt_gain = self._count_dirty_cells(target, 2)
            info_gain = self._count_unobserved_cells(target, 2)
            revisit_penalty = float(self.visit_count[target[1], target[0]]) * 0.15
            score = 3.0 * dirt_gain + 0.8 * info_gain - 0.08 * dist - revisit_penalty
            if score > best[3]:
                best = (target, path, dist, score)
        return best

    def _best_frontier_target(self, hero_pos, prefer_edge, battery, charger_distance):
        best = (None, None, 999.0, float("-inf"))
        for target in self._iter_frontiers():
            path, dist = self._path_to_any(hero_pos, [target], allow_unknown=False)
            if path is None:
                continue
            if not self._target_is_safe(target, battery, charger_distance):
                continue
            info_gain = self._count_unobserved_cells(target, 2)
            dirt_gain = self._count_dirty_cells(target, 2)
            edge_bonus = 0.0
            if prefer_edge:
                edge_bonus = self._edge_bonus(target)
            revisit_penalty = float(self.visit_count[target[1], target[0]]) * 0.12
            score = 2.2 * info_gain + 0.7 * dirt_gain + edge_bonus - 0.07 * dist - revisit_penalty
            if score > best[3]:
                best = (target, path, dist, score)
        return best

    def _rank_legal_actions(self, hero_pos, target_mode, target_pos, target_path, npcs):
        legal_scores = np.full((Config.ACTION_NUM,), -1e6, dtype=np.float32)
        safe_mask = np.zeros((Config.ACTION_NUM,), dtype=np.float32)
        next_action = self._next_path_action(target_path) if target_path else None

        for action, nxt in self._iter_neighbors(hero_pos):
            if not self._can_move_to(hero_pos, nxt, allow_unknown=(target_mode != "charge")):
                continue
            npc_dist = self._nearest_npc_dist(nxt, npcs)
            if npc_dist <= self.HARD_NPC_RADIUS and not self._hero_on_charger(nxt):
                continue
            safe_mask[action] = 1.0
            score = 0.0
            if target_pos is not None:
                score += 0.35 * (self._chebyshev_dist(hero_pos, target_pos) - self._chebyshev_dist(nxt, target_pos))
                score -= 0.05 * float(self.visit_count[nxt[1], nxt[0]])
            if next_action is not None and action == next_action:
                score += 3.0
            if target_mode == "charge":
                score += 0.12 * max(0.0, 6.0 - npc_dist)
            else:
                score += 0.08 * self._count_unobserved_cells(nxt, 1)
                score += 0.06 * self._count_dirty_cells(nxt, 1)
            legal_scores[action] = score

        if float(safe_mask.sum()) <= 0.5:
            for action, nxt in self._iter_neighbors(hero_pos):
                if self._can_move_to(hero_pos, nxt, allow_unknown=True):
                    safe_mask[action] = 1.0
                    legal_scores[action] = -0.01 * float(self.visit_count[nxt[1], nxt[0]])

        if float(safe_mask.sum()) <= 0.5:
            safe_mask[:] = 1.0
            legal_scores[:] = 0.0
        return legal_scores, safe_mask

    def _target_is_safe(self, target, battery, charger_distance):
        if not np.isfinite(charger_distance) or charger_distance >= 999.0 or not self.charger_regions:
            return True
        charger_from_target = self._heuristic_charger_distance(target)
        reserve = self.BASE_RETURN_MARGIN
        return float(battery) > float(charger_from_target) + reserve

    def _merge_local_view(self, hero_pos: Position, map_info: np.ndarray) -> int:
        half = map_info.shape[0] // 2
        new_known = 0
        for row in range(map_info.shape[0]):
            for col in range(map_info.shape[1]):
                gx = hero_pos[0] - half + col
                gz = hero_pos[1] - half + row
                if not self._in_bounds((gx, gz)):
                    continue
                if self.global_map[gz, gx] == self.UNKNOWN:
                    new_known += 1
                self.global_map[gz, gx] = int(map_info[row, col])
        return new_known

    def _iter_frontiers(self):
        known = np.argwhere((self.global_map == self.CLEAN) | (self.global_map == self.DIRT))
        for row, col in known:
            pos = (int(col), int(row))
            if self._count_unobserved_cells(pos, 1) > 0:
                yield pos

    def _frontier_density(self, hero_pos):
        total = 0
        frontier = 0
        for dz in range(-4, 5):
            for dx in range(-4, 5):
                pos = (hero_pos[0] + dx, hero_pos[1] + dz)
                if not self._in_bounds(pos):
                    continue
                total += 1
                if self._count_unobserved_cells(pos, 1) > 0:
                    frontier += 1
        return frontier / max(total, 1)

    def _local_unknown_ratio(self, hero_pos):
        unknown = 0
        total = 0
        for dz in range(-2, 3):
            for dx in range(-2, 3):
                pos = (hero_pos[0] + dx, hero_pos[1] + dz)
                if not self._in_bounds(pos):
                    continue
                total += 1
                if int(self.global_map[pos[1], pos[0]]) == self.UNKNOWN:
                    unknown += 1
        return unknown / max(total, 1)

    def _edge_bonus(self, pos):
        margin = min(pos[0], pos[1], self.MAP_SIZE - 1 - pos[0], self.MAP_SIZE - 1 - pos[1])
        return float(max(0.0, 12.0 - margin) * 0.4)

    def _path_to_any(self, start: Position, targets: Sequence[Position], allow_unknown: bool):
        targets = [target for target in targets if self._in_bounds(target)]
        if not targets:
            return None, 999.0
        target_set = set(targets)
        open_heap = [(0.0, start)]
        g_cost = {start: 0.0}
        parent: Dict[Position, Position] = {}
        best_goal = None

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur in target_set:
                best_goal = cur
                break
            for action in self.PATH_ACTION_ORDER:
                nxt = self._apply_move(cur, action)
                if not self._can_move_to(cur, nxt, allow_unknown=allow_unknown):
                    continue
                step_cost = 1.0
                new_cost = g_cost[cur] + step_cost
                if new_cost >= g_cost.get(nxt, float("inf")):
                    continue
                g_cost[nxt] = new_cost
                parent[nxt] = cur
                heuristic = min(float(self._chebyshev_dist(nxt, target)) for target in target_set)
                heapq.heappush(open_heap, (new_cost + heuristic, nxt))

        if best_goal is None:
            return None, 999.0
        return self._reconstruct_path(parent, start, best_goal), float(g_cost[best_goal])

    def _count_unobserved_cells(self, pos: Position, radius: int) -> int:
        return self._count_cells_of_type(pos, radius, self.UNKNOWN)

    def _count_dirty_cells(self, pos: Position, radius: int) -> int:
        return self._count_cells_of_type(pos, radius, self.DIRT)

    def _count_cells_of_type(self, pos: Position, radius: int, cell_type: int) -> int:
        count = 0
        for dz in range(-radius, radius + 1):
            gz = pos[1] + dz
            if not (0 <= gz < self.MAP_SIZE):
                continue
            for dx in range(-radius, radius + 1):
                gx = pos[0] + dx
                if not (0 <= gx < self.MAP_SIZE):
                    continue
                if int(self.global_map[gz, gx]) == cell_type:
                    count += 1
        return count

    def _parse_charger_regions(self, organs: Sequence[Any]) -> List[Set[Position]]:
        regions: List[Set[Position]] = []
        for organ in organs:
            if int(self._safe_float(self._get(organ, "sub_type", 0), 0.0)) != 1:
                continue
            pos = self._parse_position(self._get(organ, "pos", {}))
            w = max(1, int(self._safe_float(self._get(organ, "w", 3), 3.0)))
            h = max(1, int(self._safe_float(self._get(organ, "h", 3), 3.0)))
            half_w = w // 2
            half_h = h // 2
            region: Set[Position] = set()
            for dx in range(w):
                for dz in range(h):
                    region.add((pos[0] + dx, pos[1] + dz))
            for dx in range(-half_w, half_w + 1):
                for dz in range(-half_h, half_h + 1):
                    region.add((pos[0] + dx, pos[1] + dz))
            region = {cell for cell in region if self._in_bounds(cell)}
            if region:
                regions.append(region)
        return regions

    def _charger_cells(self) -> List[Position]:
        return [cell for region in self.charger_regions for cell in region]

    def _hero_on_charger(self, pos: Position) -> bool:
        return any(pos in region for region in self.charger_regions)

    def _heuristic_charger_distance(self, pos: Position) -> float:
        best = min(
            (float(self._chebyshev_dist(pos, cell)) for region in self.charger_regions for cell in region),
            default=float("inf"),
        )
        return best if np.isfinite(best) else 999.0

    def _next_path_action(self, path: Sequence[Position] | None) -> Optional[int]:
        if not path or len(path) < 2:
            return None
        cur, nxt = path[0], path[1]
        dx = int(np.clip(nxt[0] - cur[0], -1, 1))
        dz = int(np.clip(nxt[1] - cur[1], -1, 1))
        for action, delta in enumerate(self.ACTION_TO_DELTA):
            if delta == (dx, dz):
                return action
        return None

    def _can_move_to(self, cur: Position, nxt: Position, allow_unknown: bool) -> bool:
        if not self._in_bounds(cur) or not self._in_bounds(nxt):
            return False
        cell = int(self.global_map[nxt[1], nxt[0]])
        if cell == self.OBSTACLE:
            return False
        if cell == self.UNKNOWN and not allow_unknown:
            return False
        dx = int(np.clip(nxt[0] - cur[0], -1, 1))
        dz = int(np.clip(nxt[1] - cur[1], -1, 1))
        if dx != 0 and dz != 0:
            side_h = (cur[0] + dx, cur[1])
            side_v = (cur[0], cur[1] + dz)
            if not self._side_passable(side_h, allow_unknown) and not self._side_passable(side_v, allow_unknown):
                return False
        return True

    def _side_passable(self, pos: Position, allow_unknown: bool) -> bool:
        if not self._in_bounds(pos):
            return False
        cell = int(self.global_map[pos[1], pos[0]])
        return cell != self.OBSTACLE and not (cell == self.UNKNOWN and not allow_unknown)

    def _reconstruct_path(self, parent: Dict[Position, Position], start: Position, goal: Position) -> List[Position]:
        path = [goal]
        cur = goal
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    def _iter_neighbors(self, pos: Position) -> Iterable[Tuple[int, Position]]:
        for action in range(8):
            yield action, self._apply_move(pos, action)

    def _apply_move(self, pos: Position, action: int) -> Position:
        dx, dz = self.ACTION_TO_DELTA[action]
        return pos[0] + dx, pos[1] + dz

    def _nearest_npc_dist(self, pos: Position, npcs: Sequence[Position]) -> float:
        if not npcs:
            return 99.0
        return min(float(self._chebyshev_dist(pos, npc)) for npc in npcs)

    @staticmethod
    def _chebyshev_dist(a: Position, b: Position) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _in_bounds(self, pos: Position) -> bool:
        return 0 <= pos[0] < self.MAP_SIZE and 0 <= pos[1] < self.MAP_SIZE

    def _unwrap_observation(self, env_obs: Any) -> Any:
        if isinstance(env_obs, dict) and "observation" in env_obs:
            return env_obs["observation"]
        return env_obs

    def _parse_hero_state(self, frame_state: Any) -> Any:
        heroes = self._get(frame_state, "heroes", {})
        if isinstance(heroes, (list, tuple)):
            return heroes[0] if heroes else {}
        return heroes

    def _parse_npc_positions(self, frame_state: Any) -> List[Position]:
        npcs = self._get(frame_state, "npcs", [])
        if not isinstance(npcs, (list, tuple)):
            return []
        return [
            self._parse_position(self._get(npc, "pos", {}))
            for npc in npcs
            if self._get(npc, "pos", None) is not None
        ]

    def _parse_organ_states(self, frame_state: Any) -> List[Any]:
        organs = self._get(frame_state, "organs", [])
        return list(organs) if isinstance(organs, (list, tuple)) else []

    def _parse_map_info(self, obs: Any) -> np.ndarray:
        map_info = self._get(obs, "map_info", None)
        if map_info is None:
            return np.ones((21, 21), dtype=np.int8)
        arr = np.asarray(map_info, dtype=np.int8)
        return arr if arr.ndim == 2 else np.ones((21, 21), dtype=np.int8)

    def _parse_position(self, obj: Any) -> Position:
        x = int(self._safe_float(self._get(obj, "x", 0), 0.0))
        z = int(self._safe_float(self._get(obj, "z", 0), 0.0))
        return (
            int(np.clip(x, 0, self.MAP_SIZE - 1)),
            int(np.clip(z, 0, self.MAP_SIZE - 1)),
        )

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        if obj is None:
            return default
        return getattr(obj, key, default)

    @staticmethod
    def _safe_float(v: Any, default: float) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)
