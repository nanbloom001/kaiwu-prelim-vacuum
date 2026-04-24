#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm for Robot Vacuum.
清扫大作战 PPO 算法实现。

损失函数：
  total_loss = vf_coef × value_loss + policy_loss - beta × entropy_loss

  value_loss  : Clipped 价值损失，防止 Critic 更新幅度过大
  policy_loss : PPO Clip 策略损失（clip_param = 0.2）
  entropy_loss: 动作分布熵，鼓励探索（beta 初始 0.005）
"""

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
        self.model      = model
        self.optimizer  = optimizer
        self.parameters = [p for pg in optimizer.param_groups for p in pg["params"]]
        self.device     = device
        self.logger     = logger
        self.monitor    = monitor

        self.clip_param  = Config.CLIP_PARAM
        self.vf_coef     = Config.VF_COEF
        self.var_beta    = Config.BETA_START
        self.label_size  = Config.ACTION_NUM

        self.train_step       = 0
        self.last_report_time = 0

    # ── 训练入口 ──────────────────────────────────────────────────────────────

    def learn(self, list_sample_data: list) -> dict:
        """
        接收一批 SampleData，执行一步 PPO 梯度更新。

        数据来源：workflow 收集的轨迹样本（经 GAE 后处理）。

        Returns:
            dict: 包含 total_loss（以及定期上报的详细 loss）
        """
        # ── 组装 Batch Tensor ────────────────────────────────────────────────
        obs         = torch.stack([s.obs         for s in list_sample_data]).to(self.device)
        legal_action= torch.stack([s.legal_action for s in list_sample_data]).to(self.device)
        act         = torch.stack([s.act         for s in list_sample_data]).to(self.device).view(-1, 1)
        old_prob    = torch.stack([s.prob        for s in list_sample_data]).to(self.device)
        planner_prob= torch.stack([s.planner_prob for s in list_sample_data]).to(self.device)
        mix_alpha   = torch.stack([s.mix_alpha   for s in list_sample_data]).to(self.device)
        old_value   = torch.stack([s.value       for s in list_sample_data]).to(self.device)
        reward_sum  = torch.stack([s.reward_sum  for s in list_sample_data]).to(self.device)
        advantage   = torch.stack([s.advantage   for s in list_sample_data]).to(self.device)
        reward      = torch.stack([s.reward      for s in list_sample_data]).to(self.device)

        # ── 优势归一化（提升 PPO 训练稳定性）───────────────────────────────
        adv = advantage.squeeze(-1) if advantage.dim() > 1 else advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── 前向传播 ─────────────────────────────────────────────────────────
        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list    = self.model(obs)
        logits      = rst_list[0]   # (B, 8)
        value_pred  = rst_list[1]   # (B, 1)

        # ── 计算损失 ─────────────────────────────────────────────────────────
        total_loss, info = self._compute_loss(
            logits      = logits,
            value_pred  = value_pred,
            legal_action= legal_action,
            old_action  = act,
            old_prob    = old_prob,
            planner_prob= planner_prob,
            mix_alpha   = mix_alpha,
            old_value   = old_value,
            reward_sum  = reward_sum,
            advantage   = adv,
        )

        # ── 反向传播 & 梯度裁剪 ──────────────────────────────────────────────
        total_loss.backward()
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        # ── 定期上报监控指标 ─────────────────────────────────────────────────
        results = {"total_loss": total_loss.item()}
        now = time.time()
        if now - self.last_report_time >= 60:
            results.update({
                "value_loss"  : round(info["value_loss"],   4),
                "policy_loss" : round(info["policy_loss"],  4),
                "entropy_loss": round(info["entropy_loss"], 4),
                "bc_loss"     : round(info["bc_loss"],      4),
                "approx_kl"   : round(info["approx_kl"],    4),
                "mix_alpha"   : round(info["mix_alpha"],    4),
                "reward"      : round(reward.mean().item(), 4),
            })
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

    # ── 损失计算 ──────────────────────────────────────────────────────────────

    def _compute_loss(self, logits, value_pred, legal_action,
                      old_action, old_prob, planner_prob, mix_alpha,
                      old_value, reward_sum, advantage):
        """
        计算 PPO 三项损失。

        value_loss  : Clipped MSE，防止价值函数偏移过大
        policy_loss : PPO surrogate（重要性比率 clip）
        entropy_loss: 动作分布熵，系数为 beta（正则化探索）
        """
        # ── 价值损失（Clipped）¬────────────────────────────────────────────────
        tdret = reward_sum.squeeze(-1) if reward_sum.dim() > 1 else reward_sum
        vp    = value_pred.squeeze(-1) if value_pred.dim() > 1 else value_pred
        ov    = old_value.squeeze(-1)  if old_value.dim()  > 1 else old_value

        vp_clipped  = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss  = 0.5 * torch.max(
            (tdret - vp) ** 2,
            (tdret - vp_clipped) ** 2,
        ).mean()

        # ── 策略损失（PPO Clip）───────────────────────────────────────────────
        policy_prob = self._masked_softmax(logits, legal_action)
        mixed_prob  = self._mix_policy(policy_prob, planner_prob, mix_alpha, legal_action)
        entropy_loss = (-(mixed_prob * torch.log(mixed_prob.clamp(1e-9, 1.0))).sum(1)).mean()

        one_hot      = F.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob     = (one_hot * mixed_prob).sum(1, keepdim=True)            # (B, 1)
        old_act_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9) # (B, 1)

        ratio = new_prob / old_act_prob

        adv = advantage.unsqueeze(-1) if advantage.dim() == 1 else advantage  # (B, 1)

        policy_loss = torch.max(
            -ratio * adv,
            -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv,
        ).mean()

        # ── 总损失 ────────────────────────────────────────────────────────────
        bc_loss = -(planner_prob * torch.log(policy_prob.clamp(1e-9, 1.0))).sum(1).mean()

        alpha_mean = float(mix_alpha.mean().item())
        alpha_norm = np.clip(alpha_mean / max(Config.RESIDUAL_ALPHA_MAX, 1e-6), 0.0, 1.0)
        bc_coef = max(Config.BC_COEF_MIN, Config.BC_COEF_START * (1.0 - alpha_norm) ** 2)
        self.var_beta = Config.BETA_END + (Config.BETA_START - Config.BETA_END) * (1.0 - alpha_norm)

        approx_kl = (
            old_prob.clamp(1e-9, 1.0)
            * (
                torch.log(old_prob.clamp(1e-9, 1.0))
                - torch.log(mixed_prob.clamp(1e-9, 1.0))
            )
        ).sum(1).mean()

        total_loss = (
            self.vf_coef * value_loss
            + policy_loss
            - self.var_beta * entropy_loss
            + bc_coef * bc_loss
        )

        return total_loss, {
            "value_loss"  : value_loss.item(),
            "policy_loss" : policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "bc_loss"     : bc_loss.item(),
            "approx_kl"   : approx_kl.item(),
            "mix_alpha"   : alpha_mean,
        }

    def _masked_softmax(self, logits: torch.Tensor, legal_action: torch.Tensor) -> torch.Tensor:
        """
        对 logits 应用合法动作掩码后计算 softmax。
        非法动作位置加 -1e5 使其概率趋近 0。
        """
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        logits = (logits - label_max) * legal_action
        logits = logits + 1e5 * (legal_action - 1)   # 非法位置 → 极小值
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
        return mixed / mixed.sum(dim=1, keepdim=True).clamp(1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule-based coverage planner (inlined from planner.py)
# ═══════════════════════════════════════════════════════════════════════════════

# 位置 (Position)：地图上一个格子的 (x, z) 坐标。
# x 轴向右为正，z 轴向下为正，原点在地图左上角。
Position = Tuple[int, int]


@dataclass
class PolicyInfo:
    """
    每帧规划器输出的决策摘要，供外部（Agent.exploit）使用。

    字段说明（对应开发指南数据协议）：
        safe_action_mask     : 8 维浮点掩码，1.0 = 该动作安全可选，0.0 = NPC 危险区或不合法
        action_scores        : 8 维评分，越高越优先；safe_action_mask 为 0 的方向得分无效
        chosen_action        : argmax(action_scores)，规划器建议执行的动作索引 (0-7)
        greedy_action        : 与 chosen_action 相同（保留字段，供外部区分贪心/随机）
        target_mode          : 当前目标模式，可取值见下方常量注释
        target_pos           : 当前导航目标的全局 (x, z) 坐标，None 表示无目标
        target_distance      : 到目标的 A* 代价距离（步数），无路径时为 999.0
        battery              : 小悟机器人当前剩余电量（remaining_charge）
        battery_ratio        : battery / battery_max，范围 [0, 1]
        charger_distance     : 到最近充电桩 (charger) 的 A* 代价距离，无路径时为 999.0
        charger_slack        : battery - charger_distance，表示到充电桩的电量余量
        nearest_npc_distance : 与最近官方机器人 (NPC) 的 Chebyshev 距离（格数）
        frontier_density     : 以小悟为中心 9×9 区域内 frontier cell 的比例
        local_dirty_ratio    : 当前 21×21 视野 (map_info) 中污渍地面 (dirty tile) 的比例
        local_unknown_ratio  : 小悟周围 5×5 区域内未观测格子 (unobserved cell) 的比例（归一化到 25 格）
        new_known_cells      : 本帧新观测到的格子数量（首次从 UNKNOWN 变为已知）
        on_charger           : 小悟当前是否位于充电桩 (charger) 区域内
        should_charge        : 规划器是否判定当前必须返回充电
    """
    safe_action_mask:     np.ndarray
    action_scores:        np.ndarray
    chosen_action:        int
    greedy_action:        int
    target_mode:          str
    target_pos:           Optional[Position]
    target_distance:      float
    battery:              float
    battery_ratio:        float
    charger_distance:     float
    charger_slack:        float
    nearest_npc_distance: float
    frontier_density:     float
    local_dirty_ratio:    float
    local_unknown_ratio:  float
    new_known_cells:      int
    on_charger:           bool
    should_charge:        bool


class CoveragePlanner:
    """
    基于规则的区域覆盖规划器（Rule-based Coverage Planner）。

    核心职责（对应开发指南"行为策略"方向）：
      1. 全局地图记忆：将每帧 21×21 的局部视野 (map_info) 融合到 128×128 的全局地图
      2. 充电策略    ：根据 battery、charger_distance 判断何时必须返回充电桩 (charger)
      3. 覆盖目标选取：在全局地图上评分，选出信息增益最大的 frontier cell 或 dirty tile 作为导航目标
      4. 路径规划    ：A* 算法在全局地图上搜索到目标的最优路径
      5. NPC 躲避    ：屏蔽官方机器人 (NPC) 周围危险区域，保证小悟机器人不与 NPC 碰撞

    所有决策均基于手工规则，不含任何可学习参数。
    每帧调用一次 update()，返回 PolicyInfo 供 Agent.exploit() 使用。

    ── target_mode 取值说明 ──────────────────────────────────────────────────
        "charge"           : 返回充电桩模式，优先级最高
        "find_charger_edge": 充电桩未知，往地图边缘探索以寻找充电桩
        "edge_frontier"    : 扩张阶段，往已知区域边缘的 frontier cell 扩展
        "frontier"         : 开采阶段，选取普通 frontier cell 探索未知区域
        "dirt"             : 开采阶段，直接导航到已知的 dirty tile（污渍地面）
        "fallback"         : 无可达目标，停留原地等待
    """

    # ── Cell types（视野网格值协议，对应开发指南 map_info 字段）────────────────
    UNKNOWN  = -1   # 尚未被观测到的格子（全局地图专用，环境不会返回此值）
    OBSTACLE =  0   # 障碍物 / 地图边界（不可通行）
    CLEAN    =  1   # 已清扫的道路（可通行）
    DIRT     =  2   # 污渍地面（可通行，小悟经过后自动完成清扫并得分）

    # ── Map / view geometry（地图与视野几何参数）─────────────────────────────
    MAP_SIZE    = 128   # 全局地图尺寸（128×128 栅格化地图）
    VIEW_RADIUS = 10    # 视野域半径：以小悟为中心向四个方向各延伸 10 格，形成 21×21 观测范围

    # 8 个移动方向对应的坐标偏移量 (dx, dz)，索引与动作值一一对应。
    # 动作空间协议：0=右(→), 1=右上(↗), 2=上(↑), 3=左上(↖),
    #               4=左(←), 5=左下(↙), 6=下(↓), 7=右下(↘)
    ACTION_TO_DELTA: Sequence[Position] = (
        ( 1,  0),   # 0  右 (→)
        ( 1, -1),   # 1  右上 (↗) 斜向
        ( 0, -1),   # 2  上 (↑)
        (-1, -1),   # 3  左上 (↖) 斜向
        (-1,  0),   # 4  左 (←)
        (-1,  1),   # 5  左下 (↙) 斜向
        ( 0,  1),   # 6  下 (↓)
        ( 1,  1),   # 7  右下 (↘) 斜向
    )
    DIAGONAL_ACTIONS  = {1, 3, 5, 7}          # 斜向动作集合（防穿角规则适用）
    PATH_ACTION_ORDER = (1, 3, 5, 7, 0, 2, 4, 6)  # A*/BFS 扩展顺序：斜向优先，加快收敛

    # ── Decision thresholds（充电策略与 NPC 安全阈值）────────────────────────
    BASE_RETURN_MARGIN    = 22.0  # 触发返回充电的电量余量：battery <= charger_distance + 22 时开始返回
    NPC_RETURN_MARGIN     = 28.0  # 充电路径经过 NPC 风险区时额外增加的安全余量（避免路途中碰撞）
    LOW_BATTERY_RATIO     = 0.30  # 电量比低于此值时强制返回充电桩（应对充电桩距离估算误差）
    EXIT_RETURN_RATIO     = 0.95  # 抵达充电桩且电量比超过此值后退出返回模式，恢复正常清扫
    EXPANSION_KNOWN_RATIO = 0.78  # 全局地图已知比例超过此值时，从扩张阶段切换为开采阶段
    EXPANSION_STEP_LIMIT  = 450   # （保留参数）步数上限，历史版本扩张阶段的步数守卫
    HARD_NPC_RADIUS       = 2     # NPC 硬封锁半径：Chebyshev 距离 ≤ 2 的格子视为危险区，直接屏蔽
    PATH_RISK_RADIUS      = 4     # NPC 软风险半径：充电路径经过此范围内时增加 NPC_RETURN_MARGIN
    AGGRESSIVE_EDGE_STEPS = 500   # 前 500 步保持扩张阶段，强制优先探索地图边缘寻找充电桩

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle（生命周期）
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self):
        self.reset()

    def reset(self):
        """
        每局 (episode) 开始时重置所有内部状态。
        由 Agent.reset() 调用，确保跨局之间状态不污染。
        """
        # 128×128 全局地图，初始全部标记为 UNKNOWN（尚未观测）
        self.global_map   = np.full((self.MAP_SIZE, self.MAP_SIZE), self.UNKNOWN, dtype=np.int8)
        # 小悟访问计数：记录每格被经过的次数，用于抑制重复路径
        self.visit_count  = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.int16)
        # 小悟清扫标记：小悟自身清扫过的格子置 1（用于区分 NPC 清扫）
        self.hero_cleaned = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.uint8)
        # NPC 清扫标记：官方机器人 (NPC) 清扫过的格子置 1（该格子再清扫无效，需规避）
        self.npc_cleaned  = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.uint8)
        # 充电桩区域列表：每个元素是一个 Position 集合，代表一个充电桩 (charger) 的占地格子
        self.charger_regions: List[Set[Position]] = []
        self.step_no      = 0
        self.return_mode  = False               # 是否处于"返回充电"模式
        self.current_goal: Optional[Position] = None   # 当前导航目标格子
        self.current_mode = "explore"           # 当前 target_mode 字符串
        self.last_policy_info: Optional[PolicyInfo] = None
        self.current_step = 0                   # 当前步数 (step_no)，来自 observation
        self.episode_max_step = 1000
        self.episode_charger_count = 4
        self.episode_battery_max = 200

    def set_episode_config(self, max_step=None, robot_count=None, charger_count=None, battery_max=None):
        if max_step is not None:
            self.episode_max_step = max(1, int(max_step))
        if charger_count is not None:
            self.episode_charger_count = int(np.clip(charger_count, 1, 4))
        if battery_max is not None:
            self.episode_battery_max = int(np.clip(battery_max, 100, 999))

    def _dynamic_return_margin(self) -> float:
        charger_scarcity = (4.0 - float(self.episode_charger_count)) / 3.0
        low_capacity_factor = np.clip((260.0 - float(self.episode_battery_max)) / 160.0, 0.0, 1.0)
        long_horizon_bonus = 2.0 if self.episode_max_step >= 1500 else 0.0
        return self.BASE_RETURN_MARGIN + 6.0 * charger_scarcity + 4.0 * low_capacity_factor + long_horizon_bonus

    def _dynamic_low_battery_ratio(self) -> float:
        charger_scarcity = (4.0 - float(self.episode_charger_count)) / 3.0
        low_capacity_factor = np.clip((260.0 - float(self.episode_battery_max)) / 160.0, 0.0, 1.0)
        long_horizon_bonus = 0.02 if self.episode_max_step >= 1500 else 0.0
        return float(np.clip(
            self.LOW_BATTERY_RATIO + 0.05 * charger_scarcity + 0.03 * low_capacity_factor + long_horizon_bonus,
            0.28,
            0.46,
        ))

    def _dynamic_exit_return_ratio(self) -> float:
        charger_scarcity = (4.0 - float(self.episode_charger_count)) / 3.0
        low_capacity_factor = np.clip((260.0 - float(self.episode_battery_max)) / 160.0, 0.0, 1.0)
        return float(np.clip(self.EXIT_RETURN_RATIO + 0.02 * charger_scarcity + 0.01 * low_capacity_factor, 0.93, 0.99))

    # ─────────────────────────────────────────────────────────────────────────
    # Main update — called once per frame（每帧主调用）
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, env_obs: Any, last_action: int = -1) -> PolicyInfo:
        """
        每步由 Agent.exploit() 调用，完成一次完整的规划决策循环。

        流程：
          1. 解析 env_obs → 提取 hero 状态、NPC 位置、organ（充电桩）信息、map_info
          2. 更新全局地图记忆（将 21×21 局部视野融合进 128×128 全局地图）
          3. 构建 legal_action 掩码，再叠加 NPC 安全掩码得到 safe_action_mask
          4. 判断是否需要返回充电桩 (should_charge)
          5. 选择本帧导航目标 (coverage target 或 charger)
          6. 对 8 个方向评分，输出 PolicyInfo

        Args:
            env_obs     : 环境返回的原始观测，包含 observation / frame_state / env_info 等字段
            last_action : 上一步执行的动作索引 (0-7)，用于连续性加分

        Returns:
            PolicyInfo: 本帧完整决策摘要
        """
        # ── 解析 observation（观测信息）─────────────────────────────────────
        obs         = self._unwrap_observation(env_obs)
        env_info    = self._get(obs, "env_info", {})      # EnvInfo：全局环境信息
        frame_state = self._get(obs, "frame_state", {})   # FrameState：帧状态数据
        hero        = self._parse_hero_state(frame_state)  # HeroState：小悟机器人状态
        npcs        = self._parse_npc_positions(frame_state)  # list[Position]：NPC 全局坐标
        organs      = self._parse_organ_states(frame_state)   # list[OrganState]：充电桩物件

        # 小悟机器人当前位置 (x, z)；优先从 HeroState.pos 读取，回退到 EnvInfo.pos
        hero_pos      = self._parse_position(self._get(hero, "pos", self._get(env_info, "pos", {})))
        # 21×21 局部视野地图 (map_info)：0=障碍, 1=已清扫, 2=污渍
        map_grid      = self._parse_map_info(obs)
        # 当前剩余电量 (battery / remaining_charge)
        battery       = self._safe_float(self._get(hero, "battery",     self._get(env_info, "remaining_charge", 0)),    0.0)
        battery_max   = self._safe_float(self._get(hero, "battery_max", self._get(env_info, "battery_max",     200)), 200.0)
        battery_ratio = battery / max(battery_max, 1.0)   # 电量比 ∈ [0, 1]
        self.current_step = int(self._safe_float(self._get(obs, "step_no", 0), 0.0))

        # ── 更新全局地图记忆 ────────────────────────────────────────────────
        # 解析本帧充电桩 (charger) 位置，更新 charger_regions
        self.charger_regions = self._parse_charger_regions(organs)
        # step_cleaned_cells：本步清扫的坐标列表（来自 EnvInfo）
        cleaned_cells        = self._parse_position_list(self._get(env_info, "step_cleaned_cells", []))
        # 将局部视野 (map_info) 融合进全局地图，返回本帧新增已知格子数
        new_known_cells      = self._merge_local_map_observation(hero_pos, map_grid, cleaned_cells)
        # 将本步小悟清扫的格子标记为 CLEAN，并重置对应的 npc_cleaned 标记
        self._record_hero_cleaned_cells(cleaned_cells)
        # 将充电桩区域内的 UNKNOWN 格子标记为 CLEAN（充电桩视为可通行区域）
        self._mark_charger_cells_passable()
        # 更新访问计数（防止 int16 溢出）
        self.visit_count[hero_pos[1], hero_pos[0]] = min(
            np.iinfo(np.int16).max,
            self.visit_count[hero_pos[1], hero_pos[0]] + 1,
        )

        # ── 构建动作掩码 ────────────────────────────────────────────────────
        # legal_action mask：基于局部视野 (map_info) 判断哪些方向不会撞墙
        legal_mask = self._build_legal_action_mask(map_grid)
        # safe_action_mask：在 legal_mask 基础上，进一步屏蔽通向 NPC 硬封锁区的方向
        safe_mask  = self._apply_npc_avoidance_mask(hero_pos, legal_mask, npcs)

        # ── 充电桩 (charger) 返回逻辑 ───────────────────────────────────────
        on_charger = self._hero_on_charger(hero_pos)
        dynamic_return_margin = self._dynamic_return_margin()
        dynamic_low_battery_ratio = self._dynamic_low_battery_ratio()
        dynamic_exit_return_ratio = self._dynamic_exit_return_ratio()
        # 抵达充电桩且电量已满 → 退出返回模式，重置目标，恢复清扫
        if on_charger and battery_ratio >= dynamic_exit_return_ratio:
            self.return_mode  = False
            self.current_goal = None
            self.current_mode = "explore"

        # 规划到充电桩的路径（A*），获取路径、距离、目标格子
        charger_path, charger_distance, charger_target = self._plan_path_to_charger(hero_pos, npcs)
        charger_known       = bool(self.charger_regions)  # 是否已观测到充电桩
        # 若充电路径途经 NPC 风险区（PATH_RISK_RADIUS），则额外加大安全余量
        extra_return_margin = self.NPC_RETURN_MARGIN if self._path_enters_npc_risk_zone(charger_path, npcs) else 0.0

        # 充电决策：满足以下任一条件则触发返回充电
        #   1. 已处于返回模式（return_mode 一旦触发，持续到充满电）
        #   2. 剩余电量 <= 到充电桩的距离 + 安全余量（再不走就来不及了）
        #   3. 电量比过低（LOW_BATTERY_RATIO），强制充电
        should_charge = (
            self.return_mode
            or (charger_known and np.isfinite(charger_distance)
                and battery <= charger_distance + dynamic_return_margin + extra_return_margin)
            or (charger_known and battery_ratio <= dynamic_low_battery_ratio)
        )
        if should_charge:
            self.return_mode = True   # 一旦触发，锁定返回模式直至充满

        # ── 选择导航目标 ────────────────────────────────────────────────────
        if should_charge:
            # 充电模式：目标固定为充电桩 (charger)，复用已规划的路径
            target_mode, target_pos, target_distance, path = (
                "charge", charger_target, charger_distance, charger_path
            )
        else:
            # 覆盖模式：从全局地图中选出得分最高的 frontier cell 或 dirty tile
            target_mode, target_pos, target_distance, path = self._select_coverage_target(
                hero_pos, battery, charger_distance, npcs
            )

        # ── 对 8 个合法动作评分 ─────────────────────────────────────────────
        action_scores = self._rank_legal_actions(
            hero_pos=hero_pos,
            safe_mask=safe_mask,
            last_action=last_action,
            npcs=npcs,
            target_pos=target_pos,
            target_mode=target_mode,
            path=path,
            should_charge=should_charge,
        )

        # 若所有动作均被 NPC 封锁（safe_mask 全为 0），启用逃脱评分兜底
        if safe_mask.sum() <= 0:
            action_scores   = self._rank_fallback_actions(hero_pos, legal_mask, map_grid, npcs)
            fallback_action = int(np.argmax(action_scores))
            safe_mask       = np.zeros((8,), dtype=np.float32)
            safe_mask[fallback_action] = 1.0  # 兜底动作视为唯一安全选项

        # ── 组装 PolicyInfo 并返回 ──────────────────────────────────────────
        chosen_action = int(np.argmax(action_scores))
        # charger_slack = battery - charger_distance，正值表示还有余量，负值表示危险
        charger_slack = battery - charger_distance if np.isfinite(charger_distance) else battery

        policy_info = PolicyInfo(
            safe_action_mask     = safe_mask.astype(np.float32),
            action_scores        = action_scores.astype(np.float32),
            chosen_action        = chosen_action,
            greedy_action        = chosen_action,
            target_mode          = target_mode,
            target_pos           = target_pos,
            target_distance      = float(target_distance if np.isfinite(target_distance) else 999.0),
            battery              = float(battery),
            battery_ratio        = float(battery_ratio),
            charger_distance     = float(charger_distance if np.isfinite(charger_distance) else 999.0),
            charger_slack        = float(charger_slack),
            nearest_npc_distance = float(self._nearest_npc_dist(hero_pos, npcs)),
            frontier_density     = float(self._local_frontier_density(hero_pos)),
            local_dirty_ratio    = float(np.mean(map_grid == self.DIRT)),
            local_unknown_ratio  = float(self._count_unobserved_cells(hero_pos, 2)) / 25.0,
            new_known_cells      = int(new_known_cells),
            on_charger           = bool(on_charger),
            should_charge        = bool(should_charge),
        )

        self.last_policy_info = policy_info
        self.current_goal     = target_pos
        self.current_mode     = target_mode
        return policy_info

    # ─────────────────────────────────────────────────────────────────────────
    # Target selection（覆盖目标选取）
    # ─────────────────────────────────────────────────────────────────────────

    def _select_coverage_target(
        self,
        hero_pos:         Position,
        battery:          float,
        charger_distance: float,
        npcs:             Sequence[Position],
    ) -> Tuple[str, Optional[Position], float, List[Position]]:
        """
        从全局地图中选出最优的覆盖目标（frontier cell 或 dirty tile）。

        决策流程：
          1. 若当前目标 (current_goal) 仍有效（dirty 或 frontier），直接复用
          2. 否则遍历全局地图，对每个候选格子评分，选出得分最高者
          3. 为最优目标规划 A* 路径并返回

        评分公式（综合多项启发因子）：
          score = 3.6 * info_gain       # 格子周围 UNKNOWN 数（探索价值）
                + 2.6 * dirt_gain       # 格子周围 DIRT 数（清扫价值）
                + 2.4 * (是否 DIRT)     # 目标本身是污渍地面的额外奖励
                - 0.42 * dist           # BFS 步数距离（越近越好）
                - 0.30 * visit_penalty  # 已访问次数惩罚（避免原地打转）
                - 1.6  * npc_cleaned    # NPC 已清扫标记惩罚（该格子无效）
                + 0.22 * diagonal_bonus # 斜向对齐奖励（斜向移动效率更高）
                + 边缘 bonus            # 扩张阶段额外鼓励向地图边缘探索

        电量安全门控：battery <= dist + charger_need + reserve 的格子跳过，
        确保小悟在前往该格子后仍能返回充电桩。

        Args:
            hero_pos         : 小悟当前位置 (x, z)
            battery          : 当前剩余电量
            charger_distance : 到最近充电桩的 A* 代价距离
            npcs             : 所有官方机器人 (NPC) 的全局坐标列表

        Returns:
            (target_mode, target_pos, target_distance, path)
            target_mode    : "frontier" / "dirt" / "edge_frontier" / "find_charger_edge" / "fallback"
            target_pos     : 目标格子全局坐标，无目标时为 None
            target_distance: 到目标的 A* 代价，无目标时为 inf
            path           : 到目标的 A* 路径（Position 列表），无路径时为 []
        """
        # ── 步骤 1：复用仍有效的当前目标 ──────────────────────────────────
        if self._goal_is_still_valid(self.current_goal, hero_pos):
            path, distance = self._astar_path(hero_pos, [self.current_goal], False, npcs)
            if path and np.isfinite(distance):
                return self.current_mode, self.current_goal, distance, path

        # ── 步骤 2：全图遍历评分 ───────────────────────────────────────────
        # BFS 距离图：hero_pos 到每个可达格子的步数（不可达格子 = -1）
        distance_map    = self._bfs_distance_map(hero_pos)
        # 全局地图已知比例（已观测格子 / 总格子数）
        known_ratio     = (np.count_nonzero(self.global_map != self.UNKNOWN)
                           / float(self.MAP_SIZE * self.MAP_SIZE))
        charger_known   = bool(self.charger_regions)
        # 扩张阶段判定：前 500 步 / 已知比例 < 78% / 充电桩未知 → 仍处于扩张阶段
        expansion_phase = (
            self.current_step <= self.AGGRESSIVE_EDGE_STEPS
            or known_ratio < self.EXPANSION_KNOWN_RATIO
            or not charger_known
        )
        known_bbox = self._explored_bounding_box()  # 已探索区域的外接矩形
        reserve    = self._dynamic_return_margin() + 4.0  # 电量安全预留（基础余量 + 缓冲）

        best_score:  float             = -1e9
        best_target: Optional[Position] = None
        best_mode:   str               = "frontier"

        for gz in range(self.MAP_SIZE):
            for gx in range(self.MAP_SIZE):
                dist = distance_map[gz, gx]
                if dist < 0:
                    continue  # BFS 不可达，跳过

                pos         = (gx, gz)
                cell        = int(self.global_map[gz, gx])
                is_frontier = self._is_frontier_cell(pos)

                # 只考虑 frontier cell（已知区域边界）和 dirty tile（污渍地面）
                if not is_frontier and cell != self.DIRT:
                    continue

                # 电量安全门控：确保前往 pos 后还能回到充电桩
                charger_need = self._heuristic_charger_distance(pos) if charger_known else 0.0
                if charger_known and battery <= dist + charger_need + reserve:
                    continue

                # 候选格子特征计算
                info_gain         = self._count_unobserved_cells(pos, 2)   # 半径 2 内 UNKNOWN 格子数
                dirt_gain         = self._count_dirty_cells(pos, 2)        # 半径 2 内 DIRT 格子数
                visit_penalty     = float(min(8, int(self.visit_count[gz, gx])))  # 访问次数（上限 8）
                edge_bonus        = self._exploration_edge_bonus(pos, known_bbox) # 已知区域边缘奖励
                map_edge_bonus    = self._map_boundary_bonus(pos)                 # 地图物理边界奖励
                npc_clean_penalty = 1.6 * float(self.npc_cleaned[gz, gx])  # NPC 清扫惩罚
                diagonal_bonus    = 1.0 if abs(gx - hero_pos[0]) == abs(gz - hero_pos[1]) else 0.0

                # 基础评分
                score = (
                      3.6 * float(info_gain)
                    + 2.6 * float(dirt_gain)
                    + (2.4 if cell == self.DIRT else 0.0)
                    - 0.42 * float(dist)
                    - 0.30 * visit_penalty
                    - npc_clean_penalty
                    + 0.22 * diagonal_bonus
                )

                # 阶段相关边缘奖励
                if expansion_phase:
                    # 扩张阶段：强力鼓励向已知区域边缘推进（找充电桩 / 开拓新领域）
                    score += (4.2 if not charger_known else 2.8) * edge_bonus
                    score += (2.4 if not charger_known and is_frontier
                              else 1.5 if is_frontier
                              else 0.0)
                    # 充电桩未知时，排斥离边缘太远且非 dirty 的格子，避免浪费在内部区域
                    if not charger_known and edge_bonus < 0.25 and cell != self.DIRT:
                        score -= 2.0
                else:
                    # 开采阶段：边缘奖励降低，优先清扫已知的 dirty tile
                    score += 0.7 * edge_bonus

                if score > best_score:
                    best_score  = score
                    best_target = pos
                    best_mode   = self._label_target_mode(charger_known, expansion_phase, is_frontier)

        if best_target is None:
            return "fallback", None, float("inf"), []

        # ── 步骤 3：为最优目标规划 A* 路径 ────────────────────────────────
        path, distance = self._astar_path(hero_pos, [best_target], False, npcs)
        return best_mode, best_target, distance, path

    def _label_target_mode(
        self, charger_known: bool, expansion_phase: bool, is_frontier: bool
    ) -> str:
        """
        根据当前探索阶段和目标类型，确定 target_mode 标签字符串。

        映射规则（优先级从高到低）：
          1. 充电桩未知 + 扩张阶段  → "find_charger_edge"（首要任务：找到充电桩）
          2. 扩张阶段 + frontier    → "edge_frontier"（向已知边缘的 frontier 扩张）
          3. 普通 frontier          → "frontier"（常规探索）
          4. 其他（dirty tile）     → "dirt"（直接清扫污渍）

        Args:
            charger_known   : 是否已观测到充电桩 (charger)
            expansion_phase : 当前是否处于扩张阶段
            is_frontier     : 目标格子是否为 frontier cell

        Returns:
            str: target_mode 标签
        """
        if not charger_known and expansion_phase:
            return "find_charger_edge"
        if expansion_phase and is_frontier:
            return "edge_frontier"
        if is_frontier:
            return "frontier"
        return "dirt"

    # ─────────────────────────────────────────────────────────────────────────
    # Path planning（路径规划）
    # ─────────────────────────────────────────────────────────────────────────

    def _plan_path_to_charger(
        self, hero_pos: Position, npcs: Sequence[Position]
    ) -> Tuple[List[Position], float, Optional[Position]]:
        """
        规划从小悟当前位置到最近充电桩 (charger) 的路径。

        采用两阶段策略：
          1. 先尝试仅经过已知 (known) 格子的路径（更可靠）
          2. 若无已知路径，则允许穿越 UNKNOWN 格子（充电紧急时的备用方案）

        Args:
            hero_pos : 小悟当前位置 (x, z)
            npcs     : NPC 全局坐标列表（用于绕开 NPC 硬封锁区）

        Returns:
            (path, distance, charger_target)
            path           : A* 路径，无法规划时为 []
            distance       : A* 代价距离，无路径时为 inf
            charger_target : 路径终点（充电桩格子），无路径时为 None
        """
        charger_targets = self._charger_cells()  # 所有充电桩格子的平铺列表
        if not charger_targets:
            return [], float("inf"), None
        path, distance = self._astar_path(hero_pos, charger_targets, False, npcs)
        if path:
            return path, distance, path[-1]

        path, distance = self._astar_path(hero_pos, charger_targets, True, npcs)
        if path:
            return path, distance, path[-1]
        return [], float("inf"), None

    def _astar_path(
        self,
        start:         Position,
        targets:       Sequence[Position],
        allow_unknown: bool,
        npcs:          Sequence[Position],
    ) -> Tuple[List[Position], float]:
        """
        A* 最短路径搜索：从 start 到 targets 集合中任意一个目标。

        代价函数（step_cost）：
          基础代价 = 1.0（每步）
          + 2.0（若下一格为 UNKNOWN，探索未知代价高）
          + 0.15 × min(5, visit_count)（访问次数惩罚，避免重复路径）
          + 0.9  × npc_cleaned（NPC 清扫区惩罚，该区域无清扫收益）
          + 0.03（若为直线动作，斜向成本更低以鼓励对角线移动）
          + (5 - npc_dist) × 1.5（NPC 软风险距离惩罚，距离 NPC < 5 时生效）

        启发函数：Chebyshev 距离到最近目标（适用于 8 方向移动）

        特殊规则：
          - NPC 硬封锁区 (HARD_NPC_RADIUS) 内的格子直接跳过（除非该格子就是目标）
          - 斜向移动防穿角：两侧格子都是障碍时不允许对角线穿越（见 _can_move_to）

        Args:
            start         : 起点位置 (x, z)
            targets       : 目标位置集合（到达其中任意一个即终止）
            allow_unknown : 是否允许经过 UNKNOWN 格子（True 时代价更高但可穿越未知区域）
            npcs          : NPC 全局坐标列表

        Returns:
            (path, cost)
            path : 从 start 到目标的完整路径（含 start 和 goal）
            cost : 路径总代价，无路径时为 inf
        """
        target_set = {pos for pos in targets if self._in_bounds(pos)}
        if not target_set or not self._in_bounds(start):
            return [], float("inf")
        if start in target_set:
            return [start], 0.0

        # 优先队列：(f_score, g_cost, position)，f = g + heuristic
        pq:        List[Tuple[float, float, Position]] = [(0.0, 0.0, start)]
        best_cost: Dict[Position, float]               = {start: 0.0}
        parent:    Dict[Position, Position]            = {}

        while pq:
            _, cost, cur = heapq.heappop(pq)
            # 跳过已被更优路径超越的节点
            if cost > best_cost.get(cur, float("inf")) + 1e-6:
                continue
            if cur in target_set:
                return self._reconstruct_path(parent, start, cur), cost

            for action in self.PATH_ACTION_ORDER:
                nxt = self._apply_move(cur, action)
                if not self._can_move_to(cur, nxt, allow_unknown):
                    continue
                # 跳过 NPC 硬封锁区（目标格子除外，允许"冲进去充电"）
                if self._in_npc_hard_zone(nxt, npcs) and nxt not in target_set:
                    continue

                step_cost = 1.0
                if self.global_map[nxt[1], nxt[0]] == self.UNKNOWN:
                    step_cost += 2.0   # 未知格子探索代价
                step_cost += 0.15 * min(5, int(self.visit_count[nxt[1], nxt[0]]))
                step_cost += 0.9  * float(self.npc_cleaned[nxt[1], nxt[0]])
                if action not in self.DIAGONAL_ACTIONS:
                    step_cost += 0.03  # 直线动作轻微惩罚，鼓励斜向走
                npc_dist = self._nearest_npc_dist(nxt, npcs)
                if npc_dist < 5:
                    step_cost += (5.0 - npc_dist) * 1.5  # NPC 软风险区代价

                new_cost = cost + step_cost
                if new_cost + 1e-6 >= best_cost.get(nxt, float("inf")):
                    continue  # 已有更优路径，剪枝

                best_cost[nxt] = new_cost
                parent[nxt]    = cur
                # Chebyshev 启发值：到最近目标的下界
                heuristic      = min(self._chebyshev_dist(nxt, tgt) for tgt in target_set)
                heapq.heappush(pq, (new_cost + heuristic, new_cost, nxt))

        return [], float("inf")

    def _bfs_distance_map(self, start: Position) -> np.ndarray:
        """
        BFS 广度优先搜索：计算从 start 出发，到全图每个可达格子的步数。

        仅经过已知 (known) 可通行格子（CLEAN / DIRT），UNKNOWN 和 OBSTACLE 不可穿越。
        返回值用于 _select_coverage_target 中对所有候选目标的距离查询（O(1)）。

        Args:
            start : BFS 起点（小悟当前位置）

        Returns:
            dist : shape=(MAP_SIZE, MAP_SIZE) 的 int16 数组
                   dist[z, x] = 从 start 到 (x, z) 的最短步数；
                   -1 表示 BFS 不可达（障碍、UNKNOWN 或孤立区域）
        """
        dist = np.full((self.MAP_SIZE, self.MAP_SIZE), -1, dtype=np.int16)
        if not self._in_bounds(start):
            return dist

        queue: List[Position] = [start]
        dist[start[1], start[0]] = 0
        head = 0   # 用数组模拟队列，避免 deque 的开销

        while head < len(queue):
            cur      = queue[head]
            head    += 1
            cur_dist = int(dist[cur[1], cur[0]])
            for action in self.PATH_ACTION_ORDER:
                nxt = self._apply_move(cur, action)
                if not self._can_move_to(cur, nxt, allow_unknown=False):
                    continue
                if dist[nxt[1], nxt[0]] >= 0:
                    continue   # 已访问
                dist[nxt[1], nxt[0]] = cur_dist + 1
                queue.append(nxt)

        return dist

    # ─────────────────────────────────────────────────────────────────────────
    # Action scoring（动作评分）
    # ─────────────────────────────────────────────────────────────────────────

    def _rank_legal_actions(
        self,
        hero_pos:      Position,
        safe_mask:     np.ndarray,
        last_action:   int,
        npcs:          Sequence[Position],
        target_pos:    Optional[Position],
        target_mode:   str,
        path:          Sequence[Position],
        should_charge: bool,
    ) -> np.ndarray:
        """
        对 8 个动作方向进行综合评分，输出评分向量。

        评分因子（各项均作用于 safe_mask > 0.5 的合法方向）：
          +0.15          斜向动作奖励（斜向移动效率更高）
          +0.08          与上一步相同方向奖励（鼓励直线移动减少折返）
          +1.25          目标格子为 dirty tile 奖励（直接清扫）
          +0.07 * info   周围 UNKNOWN 格子数加分（探索价值）
          +0.04 * frontier 周围 frontier 格子数加分（边界扩展价值）
          -revisit       已访问次数惩罚（避免绕圈）
          -npc_penalty   NPC 软风险距离惩罚（越近扣分越多）
          -npc_clean     NPC 已清扫区惩罚
          +2.6 / +2.2    若该动作与 A* 路径首步一致，充电模式 +2.6，覆盖模式 +2.2
          +progress      向目标方向推进的 Chebyshev 进展量，充电时权重更高
          +0.9 * edge    edge_frontier / find_charger_edge 模式下的边缘奖励
          +4.0           charge 模式且目标格子就是充电桩区域
          +0.45          frontier / dirt 等模式下进入 frontier cell 的奖励

        Args:
            hero_pos      : 小悟当前位置
            safe_mask     : NPC 安全掩码（0.0 = 危险，1.0 = 安全）
            last_action   : 上一步动作索引
            npcs          : NPC 全局坐标列表
            target_pos    : 当前导航目标
            target_mode   : 当前目标模式字符串
            path          : 到目标的 A* 路径
            should_charge : 是否处于返回充电模式

        Returns:
            scores : shape=(8,) 的 float32 评分数组，被 safe_mask 屏蔽的方向为 -1e9
        """
        scores      = np.full((8,), -1e9, dtype=np.float32)
        # 从 A* 路径中提取下一步动作（即路径第 0 → 第 1 格的方向）
        path_action = self._next_path_action(path) if path else None
        known_bbox  = self._explored_bounding_box()
        edge_mode   = target_mode in ("edge_frontier", "find_charger_edge")
        current_npc_distance = self._nearest_npc_dist(hero_pos, npcs)

        for action in range(8):
            if safe_mask[action] <= 0.5:
                continue   # 被 NPC 安全掩码屏蔽，直接跳过
            nxt = self._apply_move(hero_pos, action)
            if not self._in_bounds(nxt):
                continue

            cell              = int(self.global_map[nxt[1], nxt[0]])
            info_gain         = self._count_unobserved_cells(nxt, 2)  # 半径 2 内 UNKNOWN 数
            frontier_gain     = self._count_unobserved_cells(nxt, 1)  # 半径 1 内 UNKNOWN 数（frontier 密度）
            revisit_penalty   = 0.18 * min(6, int(self.visit_count[nxt[1], nxt[0]]))
            npc_dist          = self._nearest_npc_dist(nxt, npcs)
            npc_penalty       = max(0.0, 7.5 - npc_dist) * 1.05   # 轻量增强近 NPC 风险惩罚
            npc_clean_penalty = 0.9 * float(self.npc_cleaned[nxt[1], nxt[0]])

            score = 0.0
            if action in self.DIAGONAL_ACTIONS: score += 0.15  # 斜向效率奖励
            if action == last_action:           score += 0.08  # 直线连续移动奖励
            if cell == self.DIRT:               score += 1.25  # 目标格子为污渍地面
            score += 0.07 * float(info_gain)
            score += 0.04 * float(frontier_gain)
            score -= revisit_penalty + npc_penalty + npc_clean_penalty

            # A* 路径首步对齐奖励：该动作与路径规划方向一致时大幅加分
            if path_action is not None and action == path_action:
                score += 2.6 if should_charge else 2.2

            # 向目标进展量：该动作让小悟更接近 target_pos 则加分
            if target_pos is not None:
                progress = float(self._chebyshev_dist(hero_pos, target_pos)
                                 - self._chebyshev_dist(nxt,      target_pos))
                score += (0.95 if should_charge else 0.45) * progress

            # 仅在非常接近 NPC 时，轻量奖励能拉开距离的动作，避免极短局。
            if current_npc_distance <= 3.0:
                npc_escape_progress = max(0.0, npc_dist - current_npc_distance)
                score += 1.2 * npc_escape_progress

            # 边缘模式（扩张阶段）额外奖励靠近已知区域边缘的动作
            if edge_mode:
                score += 0.9 * self._exploration_edge_bonus(nxt, known_bbox)

            # 充电模式：若该动作走到的格子就是充电桩，给予最高奖励
            if target_mode == "charge" and self._hero_on_charger(nxt):
                score += 4.0

            # 探索 / 清扫模式：进入 frontier cell 额外加分
            if target_mode in ("frontier", "dirt", "edge_frontier", "find_charger_edge"):
                if self._is_frontier_cell(nxt):
                    score += 0.45

            scores[action] = score

        return scores

    # ─────────────────────────────────────────────────────────────────────────
    # Fallback / escape scoring（兜底 / 逃脱评分）
    # ─────────────────────────────────────────────────────────────────────────

    def _rank_escape_actions(
        self, hero_pos: Position, legal_mask: np.ndarray, npcs: Sequence[Position]
    ) -> np.ndarray:
        """
        逃脱评分：仅凭借远离 NPC 的距离对合法动作排序。

        在所有安全动作被封锁（safe_mask 全为 0）时使用，以 legal_mask 为准，
        选出能最大化与最近 NPC 距离的方向。

        Args:
            hero_pos   : 小悟当前位置
            legal_mask : 基于局部视野 (map_info) 的合法动作掩码
            npcs       : NPC 全局坐标列表

        Returns:
            scores : shape=(8,) 的 float32 评分数组，被 legal_mask 屏蔽的方向为 -1e9
        """
        scores = np.full((8,), -1e9, dtype=np.float32)
        for action in range(8):
            if legal_mask[action] <= 0.5:
                continue
            nxt = self._apply_move(hero_pos, action)
            if not self._in_bounds(nxt):
                continue
            npc_dist       = self._nearest_npc_dist(nxt, npcs)
            diagonal_bonus = 0.1 if action in self.DIAGONAL_ACTIONS else 0.0
            scores[action] = float(npc_dist) + diagonal_bonus
        return scores

    def _rank_fallback_actions(
        self,
        hero_pos:   Position,
        legal_mask: np.ndarray,
        map_grid:   np.ndarray,
        npcs:       Sequence[Position],
    ) -> np.ndarray:
        """
        兜底评分：在所有方向均被 NPC 封锁时的最后备用策略。

        在逃脱评分基础上，对朝向障碍物方向（会原地踏步的动作）给予极高奖励，
        使小悟优先选择被障碍物阻挡的方向（停在原地而非走向 NPC）。

        「停在原地」的本质：向障碍物方向执行动作会让小悟停留原地（保留 1 步 + 1 电量代价），
        但至少不会撞上 NPC 导致任务失败。

        Args:
            hero_pos   : 小悟当前位置
            legal_mask : 合法动作掩码
            map_grid   : 当前 21×21 视野地图 (map_info)
            npcs       : NPC 全局坐标列表

        Returns:
            scores : shape=(8,) 的 float32 兜底评分数组
        """
        scores = self._rank_escape_actions(hero_pos, legal_mask, npcs)
        center = self.VIEW_RADIUS   # 局部视野中心格子坐标（= 10）

        for action, (dx, dz) in enumerate(self.ACTION_TO_DELTA):
            row, col = center + dz, center + dx
            if not (0 <= row < map_grid.shape[0] and 0 <= col < map_grid.shape[1]):
                continue

            blocked = int(map_grid[row, col]) == self.OBSTACLE
            if action in self.DIAGONAL_ACTIONS:
                # 斜向动作防穿角：两侧都是障碍时视为被阻挡
                side_h  = int(map_grid[center, center + dx]) != self.OBSTACLE
                side_v  = int(map_grid[center + dz, center]) != self.OBSTACLE
                blocked = blocked or not (side_h or side_v)

            if blocked:
                # 障碍物方向 → 该动作会让小悟停在原地，给予高分以逃离 NPC
                scores[action] = max(
                    scores[action],
                    100.0 + self._nearest_npc_dist(hero_pos, npcs),
                )

        return scores

    # ─────────────────────────────────────────────────────────────────────────
    # Global map maintenance（全局地图维护）
    # ─────────────────────────────────────────────────────────────────────────

    def _merge_local_map_observation(
        self,
        hero_pos:      Position,
        map_grid:      np.ndarray,
        cleaned_cells: Sequence[Position],
    ) -> int:
        """
        将当前帧的 21×21 局部视野 (map_info) 融合进 128×128 全局地图。

        融合规则：
          - 局部视野中每个格子的值直接覆盖全局地图对应位置
          - 若某格子在全局地图中曾是 DIRT，但本帧变为 CLEAN，且不在 step_cleaned_cells 中，
            则判定为 NPC 清扫，标记 npc_cleaned[z, x] = 1（该格子后续无清扫价值）

        Args:
            hero_pos      : 小悟当前位置，用于将局部坐标转换为全局坐标
            map_grid      : 21×21 局部视野数组（map_info）
            cleaned_cells : 本步 step_cleaned_cells，即小悟本步清扫的格子（全局坐标列表）

        Returns:
            new_known_cells : 本帧新观测到（从 UNKNOWN 变为已知）的格子数量
        """
        center           = self.VIEW_RADIUS   # 视野中心 = 小悟在局部网格中的坐标（10, 10）
        hero_cleaned_set = set(cleaned_cells) # 转为集合，O(1) 查询
        new_known        = 0

        for row in range(map_grid.shape[0]):
            gz = hero_pos[1] + (row - center)   # 局部行 → 全局 z 坐标
            if not (0 <= gz < self.MAP_SIZE):
                continue
            for col in range(map_grid.shape[1]):
                gx = hero_pos[0] + (col - center)   # 局部列 → 全局 x 坐标
                if not (0 <= gx < self.MAP_SIZE):
                    continue
                prev = int(self.global_map[gz, gx])
                val  = int(map_grid[row, col])
                if prev == self.UNKNOWN:
                    new_known += 1   # 首次观测，计入新增已知格子数
                # DIRT → CLEAN 且不在小悟清扫列表中 → NPC 清扫
                if prev == self.DIRT and val == self.CLEAN and (gx, gz) not in hero_cleaned_set:
                    self.npc_cleaned[gz, gx] = 1
                self.global_map[gz, gx] = val   # 覆盖更新全局地图

        return new_known

    def _record_hero_cleaned_cells(self, cleaned_cells: Sequence[Position]) -> None:
        """
        将本步 step_cleaned_cells 记录到全局地图：
          - 全局地图对应格子标记为 CLEAN
          - hero_cleaned[z, x] = 1（小悟清扫标记）
          - npc_cleaned[z, x] = 0（清除 NPC 清扫标记，以防误判）

        Args:
            cleaned_cells : step_cleaned_cells（小悟本步清扫的全局坐标列表）
        """
        for pos in cleaned_cells:
            if self._in_bounds(pos):
                self.global_map[pos[1], pos[0]]  = self.CLEAN
                self.hero_cleaned[pos[1], pos[0]] = 1
                self.npc_cleaned[pos[1], pos[0]]  = 0

    def _mark_charger_cells_passable(self) -> None:
        """
        将充电桩 (charger) 区域内所有 UNKNOWN 格子标记为 CLEAN。

        充电桩区域由 OrganState 解析得到（_parse_charger_regions），其内部格子
        一定是可通行的道路，但在全局地图上可能仍标记为 UNKNOWN。
        提前置为 CLEAN 可避免 A* 为这些格子付出 UNKNOWN 惩罚代价。
        """
        for region in self.charger_regions:
            for pos in region:
                if self._in_bounds(pos) and self.global_map[pos[1], pos[0]] == self.UNKNOWN:
                    self.global_map[pos[1], pos[0]] = self.CLEAN

    # ─────────────────────────────────────────────────────────────────────────
    # Action masks（动作掩码）
    # ─────────────────────────────────────────────────────────────────────────

    def _build_legal_action_mask(self, map_grid: np.ndarray) -> np.ndarray:
        """
        基于当前 21×21 局部视野 (map_info) 构建 legal_action 掩码。

        合法动作规则（对应开发指南"执行逻辑 / 移动规则"）：
          - 直线移动：目标格子不是 OBSTACLE 即合法
          - 斜向移动防穿角：目标格子可通行，且水平/垂直两侧格子至少有一个不是 OBSTACLE
          - 若所有方向均被障碍封锁（边角特殊情况），则强制开放全部 8 方向（避免卡死）

        Args:
            map_grid : 21×21 局部视野数组（0=障碍, 1=CLEAN, 2=DIRT）

        Returns:
            mask : shape=(8,) 的 float32 掩码，1.0 = 合法，0.0 = 不合法
        """
        mask = np.zeros((8,), dtype=np.float32)
        c    = self.VIEW_RADIUS   # 视野中心坐标（= 10）

        for action, (dx, dz) in enumerate(self.ACTION_TO_DELTA):
            row, col = c + dz, c + dx
            if not (0 <= row < map_grid.shape[0] and 0 <= col < map_grid.shape[1]):
                continue
            if int(map_grid[row, col]) == self.OBSTACLE:
                continue   # 目标格子是障碍物
            if action in self.DIAGONAL_ACTIONS:
                # 斜向防穿角：水平或垂直方向至少有一条可通行
                side_h = int(map_grid[c, c + dx]) != self.OBSTACLE
                side_v = int(map_grid[c + dz, c]) != self.OBSTACLE
                if not (side_h or side_v):
                    continue
            mask[action] = 1.0

        if mask.sum() <= 0:
            mask[:] = 1.0   # 全部被封锁时开放所有方向（容错处理）
        return mask

    def _apply_npc_avoidance_mask(
        self, hero_pos: Position, legal_mask: np.ndarray, npcs: Sequence[Position]
    ) -> np.ndarray:
        """
        在 legal_action 掩码基础上，屏蔽会进入 NPC 硬封锁区 (HARD_NPC_RADIUS) 的动作。

        NPC 硬封锁区：以官方机器人 (NPC) 为中心，Chebyshev 距离 ≤ 2 的 5×5 区域。
        小悟进入此区域即判定碰撞，任务立即失败（terminated = True）。

        Args:
            hero_pos   : 小悟当前位置
            legal_mask : 由 _build_legal_action_mask 生成的合法动作掩码
            npcs       : NPC 全局坐标列表

        Returns:
            safe_mask : shape=(8,) 的 float32 安全掩码（legal_mask 的子集）
        """
        safe_mask = legal_mask.copy()
        for action in range(8):
            if safe_mask[action] > 0.5 and self._in_npc_hard_zone(self._apply_move(hero_pos, action), npcs):
                safe_mask[action] = 0.0
        return safe_mask

    # ─────────────────────────────────────────────────────────────────────────
    # NPC safety queries（NPC 安全查询）
    # ─────────────────────────────────────────────────────────────────────────

    def _in_npc_hard_zone(self, pos: Position, npcs: Sequence[Position]) -> bool:
        """
        判断 pos 是否处于任意官方机器人 (NPC) 的硬封锁区。

        硬封锁区（HARD_NPC_RADIUS = 2）：以 NPC 为中心，Chebyshev 距离 ≤ 2 的区域，
        即以 NPC 为中心的 5×5 格范围。进入即碰撞，任务失败。

        Args:
            pos  : 待检查的格子坐标
            npcs : NPC 全局坐标列表

        Returns:
            bool: True 表示处于 NPC 硬封锁区内
        """
        return any(self._chebyshev_dist(pos, npc) <= self.HARD_NPC_RADIUS for npc in npcs)

    def _path_enters_npc_risk_zone(self, path: Sequence[Position], npcs: Sequence[Position]) -> bool:
        """
        检查充电路径中是否有格子经过 NPC 软风险区 (PATH_RISK_RADIUS = 4)。

        若充电路径途经 NPC 风险区，则返回充电的安全余量 (return_margin) 需要增加
        NPC_RETURN_MARGIN（20 格），以应对 NPC 可能移动导致路径被封锁的风险。

        Args:
            path : 充电路径（由 _plan_path_to_charger 规划得到）
            npcs : NPC 全局坐标列表

        Returns:
            bool: True 表示路径中至少有一格处于 NPC 软风险区内
        """
        return any(
            self._chebyshev_dist(pos, npc) <= self.PATH_RISK_RADIUS
            for pos in path
            for npc in npcs
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Frontier / goal analysis（frontier 分析与目标有效性）
    # ─────────────────────────────────────────────────────────────────────────

    def _local_frontier_density(self, hero_pos: Position) -> float:
        """
        计算小悟周围 9×9 区域内 frontier cell 的密度（比例）。

        frontier cell：已知（CLEAN / DIRT）且至少有一个相邻格子为 UNKNOWN 的格子。
        密度越高，说明周围可探索空间越多，该指标用于 PolicyInfo 中的 frontier_density 字段。

        Args:
            hero_pos : 小悟当前位置

        Returns:
            float: frontier cell 比例 ∈ [0, 1]
        """
        total = frontier = 0
        for dz in range(-4, 5):
            gz = hero_pos[1] + dz
            if not (0 <= gz < self.MAP_SIZE):
                continue
            for dx in range(-4, 5):
                gx = hero_pos[0] + dx
                if not (0 <= gx < self.MAP_SIZE):
                    continue
                total += 1
                if self._is_frontier_cell((gx, gz)):
                    frontier += 1
        return float(frontier) / float(max(total, 1))

    def _is_frontier_cell(self, pos: Position) -> bool:
        """
        判断 pos 是否为 frontier cell（已知区域与未知区域的边界格子）。

        定义：pos 本身为 CLEAN 或 DIRT，且至少有一个 8 方向相邻格子为 UNKNOWN。
        frontier cell 是探索扩张的优先目标，清扫它可以揭示新的地图区域。

        Args:
            pos : 待判断的格子全局坐标

        Returns:
            bool: True 表示 pos 是 frontier cell
        """
        if not self._in_bounds(pos):
            return False
        cell = int(self.global_map[pos[1], pos[0]])
        if cell not in (self.CLEAN, self.DIRT):
            return False   # UNKNOWN / OBSTACLE 本身不是 frontier
        return any(
            self._in_bounds(nxt) and int(self.global_map[nxt[1], nxt[0]]) == self.UNKNOWN
            for _, nxt in self._iter_neighbors(pos)
        )

    def _goal_is_still_valid(self, goal: Optional[Position], hero_pos: Position) -> bool:
        """
        判断当前导航目标 (current_goal) 是否仍然值得前往。

        以下情况认为目标仍有效：
          - charge / edge_frontier / find_charger_edge 模式：目标是充电桩或边缘，始终有效
          - 其他模式：目标格子仍是 DIRT（尚未被清扫）或仍是 frontier cell（未被揭示）

        以下情况认为目标已失效（需重新选取）：
          - goal 为 None（无目标）
          - goal 越界
          - goal == hero_pos（已到达）
          - 目标格子已被清扫（CLEAN）且不再是 frontier

        Args:
            goal     : 当前导航目标格子
            hero_pos : 小悟当前位置

        Returns:
            bool: True 表示目标仍有效，可继续沿用
        """
        if goal is None or not self._in_bounds(goal) or goal == hero_pos:
            return False
        if self.current_mode in ("charge", "edge_frontier", "find_charger_edge"):
            return True   # 这些模式下目标稳定，不需要每步重评
        cell = int(self.global_map[goal[1], goal[0]])
        return cell == self.DIRT or self._is_frontier_cell(goal)

    # ─────────────────────────────────────────────────────────────────────────
    # Spatial bonus helpers（空间奖励辅助函数）
    # ─────────────────────────────────────────────────────────────────────────

    def _explored_bounding_box(self) -> Optional[Tuple[int, int, int, int]]:
        """
        计算全局地图中所有已观测格子（非 UNKNOWN）的轴对齐外接矩形 (AABB)。

        用于 _exploration_edge_bonus：已知区域越小，边缘格子距中心越近，
        边缘奖励越高，从而引导小悟优先向未知区域推进。

        Returns:
            (min_x, max_x, min_z, max_z) 若存在已知格子；None 若全图均未知
        """
        known = np.argwhere(self.global_map != self.UNKNOWN)
        if known.size == 0:
            return None
        min_z, max_z = int(np.min(known[:, 0])), int(np.max(known[:, 0]))
        min_x, max_x = int(np.min(known[:, 1])), int(np.max(known[:, 1]))
        return min_x, max_x, min_z, max_z

    def _exploration_edge_bonus(self, pos: Position, bbox: Optional[Tuple[int, int, int, int]]) -> float:
        """
        计算 pos 相对于已知区域外接矩形 (bbox) 边缘的奖励值。

        越靠近已知区域边缘（即 bbox 的任意一边），奖励越高（最高 1.0，最低 0.0）。
        奖励衰减距离为 4 格：距边缘 0 格 → 1.0，距边缘 ≥ 4 格 → 0.0。

        Args:
            pos  : 候选格子全局坐标
            bbox : _explored_bounding_box() 返回的已知区域外接矩形，None 时返回 0.0

        Returns:
            float: 边缘奖励 ∈ [0.0, 1.0]
        """
        if bbox is None:
            return 0.0
        min_x, max_x, min_z, max_z = bbox
        edge_dist = min(
            abs(pos[0] - min_x), abs(pos[0] - max_x),
            abs(pos[1] - min_z), abs(pos[1] - max_z),
        )
        return max(0.0, 4.0 - float(edge_dist)) / 4.0

    def _map_boundary_bonus(self, pos: Position) -> float:
        """
        计算 pos 靠近 128×128 地图物理边界的奖励值。

        地图边缘往往有更多未探索区域（地图边界墙），奖励靠近边界的格子。
        奖励衰减距离为 12 格：距边界 0 格 → 1.0，距边界 ≥ 12 格 → 0.0。

        Args:
            pos : 候选格子全局坐标

        Returns:
            float: 地图边界奖励 ∈ [0.0, 1.0]
        """
        edge_dist = min(pos[0], pos[1], self.MAP_SIZE - 1 - pos[0], self.MAP_SIZE - 1 - pos[1])
        return max(0.0, 12.0 - float(edge_dist)) / 12.0

    # ─────────────────────────────────────────────────────────────────────────
    # Charger helpers（充电桩辅助函数）
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_charger_regions(self, organs: Sequence[Any]) -> List[Set[Position]]:
        """
        从 OrganState 列表中解析充电桩 (charger) 的占地格子集合。

        开发指南 OrganState 协议：
          sub_type = 1 表示充电桩；pos = {x, z}；w / h = 宽高（默认 3×3）。
        代码中取两种方式的并集以覆盖充电桩全部格子：
          方式 1：从 (pos.x, pos.z) 起，向右 w 格、向下 h 格
          方式 2：以 pos 为中心，向四周各延伸 half_w / half_h 格

        Args:
            organs : FrameState.organs 列表（OrganState 对象）

        Returns:
            list[set[Position]]: 每个元素是一个充电桩的全局格子集合
        """
        regions: List[Set[Position]] = []
        for organ in organs:
            # sub_type = 1 才是充电桩，过滤其他物件
            if int(self._safe_float(self._get(organ, "sub_type", 0), 0.0)) != 1:
                continue
            pos    = self._parse_position(self._get(organ, "pos", {}))
            w      = max(1, int(self._safe_float(self._get(organ, "w", 3), 3.0)))
            h      = max(1, int(self._safe_float(self._get(organ, "h", 3), 3.0)))
            half_w = w // 2
            half_h = h // 2

            region: Set[Position] = set()
            # 方式 1：右下角矩形
            for dx in range(w):
                for dz in range(h):
                    region.add((pos[0] + dx, pos[1] + dz))
            # 方式 2：中心对称矩形
            for dx in range(-half_w, half_w + 1):
                for dz in range(-half_h, half_h + 1):
                    region.add((pos[0] + dx, pos[1] + dz))

            region = {cell for cell in region if self._in_bounds(cell)}
            if region:
                regions.append(region)
        return regions

    def _charger_cells(self) -> List[Position]:
        """
        将所有充电桩区域的格子平铺为一个列表。

        用于 _astar_path 的 targets 参数：A* 以"到达其中任意一格"为终止条件。

        Returns:
            list[Position]: 所有充电桩格子的全局坐标列表
        """
        return [cell for region in self.charger_regions for cell in region]

    def _hero_on_charger(self, pos: Position) -> bool:
        """
        判断 pos 是否位于任意充电桩 (charger) 区域内。

        按开发指南规则：机器人抵达充电桩范围内，立即充满能量。
        用于检测小悟是否已到达充电桩（触发退出 return_mode）。

        Args:
            pos : 待检查的格子坐标（通常是 hero_pos 或候选动作的 nxt）

        Returns:
            bool: True 表示 pos 在某个充电桩区域内
        """
        return any(pos in region for region in self.charger_regions)

    def _heuristic_charger_distance(self, pos: Position) -> float:
        """
        计算从 pos 到最近充电桩格子的 Chebyshev 距离下界（启发式估计）。

        用于 _select_coverage_target 的电量安全门控：
          battery <= dist_to_target + heuristic_charger_from_target + reserve
        若满足则跳过该候选格子（无法保证安全返回充电桩）。

        Chebyshev 距离是实际 A* 代价的下界（8 方向移动时等价于曼哈顿距离的下界），
        用于估计不保守，但作为门控条件可接受（偶有误判，有 BASE_RETURN_MARGIN 兜底）。

        Args:
            pos : 候选格子的全局坐标

        Returns:
            float: 到最近充电桩格子的 Chebyshev 距离，充电桩未知时为 inf（实际由调用方处理）
        """
        best = min(
            (float(self._chebyshev_dist(pos, cell))
             for region in self.charger_regions for cell in region),
            default=float("inf"),
        )
        return best if np.isfinite(best) else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Neighbour count helpers（邻域格子计数）
    # ─────────────────────────────────────────────────────────────────────────

    def _count_unobserved_cells(self, pos: Position, radius: int) -> int:
        """
        统计以 pos 为中心、半径 radius 的正方形区域内 UNKNOWN（未观测）格子数量。

        用于：
          - _select_coverage_target 中的 info_gain（探索价值）
          - _rank_legal_actions 中的 info_gain / frontier_gain

        Args:
            pos    : 中心格子全局坐标
            radius : 邻域半径（正方形边长 = 2*radius + 1）

        Returns:
            int: 区域内 UNKNOWN 格子数
        """
        return self._count_cells_of_type(pos, radius, self.UNKNOWN)

    def _count_dirty_cells(self, pos: Position, radius: int) -> int:
        """
        统计以 pos 为中心、半径 radius 的正方形区域内 DIRT（污渍地面）格子数量。

        用于 _select_coverage_target 中的 dirt_gain（清扫价值）。

        Args:
            pos    : 中心格子全局坐标
            radius : 邻域半径

        Returns:
            int: 区域内 DIRT 格子数
        """
        return self._count_cells_of_type(pos, radius, self.DIRT)

    def _count_cells_of_type(self, pos: Position, radius: int, cell_type: int) -> int:
        """
        统计以 pos 为中心、半径 radius 的正方形区域内指定类型格子的数量。

        为 _count_unobserved_cells 和 _count_dirty_cells 提供共用实现。

        Args:
            pos       : 中心格子全局坐标
            radius    : 邻域半径（正方形半边长）
            cell_type : 目标格子类型（UNKNOWN / CLEAN / DIRT / OBSTACLE）

        Returns:
            int: 指定类型格子的数量
        """
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

    # ─────────────────────────────────────────────────────────────────────────
    # Path utilities（路径工具函数）
    # ─────────────────────────────────────────────────────────────────────────

    def _next_path_action(self, path: Sequence[Position]) -> Optional[int]:
        """
        从 A* 路径中提取下一步应执行的动作索引。

        通过比较路径第 0 格（当前位置）和第 1 格（下一格）的坐标差，
        匹配 ACTION_TO_DELTA 得到对应的动作值（0-7）。

        Args:
            path : A* 规划的路径（Position 列表，至少包含 start 和下一格）

        Returns:
            int  : 动作索引 (0-7)；路径不足 2 个格子时返回 None
        """
        if len(path) < 2:
            return None
        cur, nxt = path[0], path[1]
        dx = int(np.clip(nxt[0] - cur[0], -1, 1))
        dz = int(np.clip(nxt[1] - cur[1], -1, 1))
        for action, delta in enumerate(self.ACTION_TO_DELTA):
            if delta == (dx, dz):
                return action
        return None

    def _can_move_to(self, cur: Position, nxt: Position, allow_unknown: bool) -> bool:
        """
        检查从 cur 到 nxt 的移动是否可行（用于 A* 和 BFS）。

        规则（对应开发指南"斜向移动防穿角"规则）：
          1. cur / nxt 必须在地图范围内
          2. nxt 不能是 OBSTACLE
          3. 若 allow_unknown=False，nxt 不能是 UNKNOWN
          4. 若是斜向移动（dx ≠ 0 且 dz ≠ 0），水平侧和垂直侧格子不能同时是障碍
             （防止小悟穿越墙角）

        Args:
            cur           : 当前格子位置
            nxt           : 目标格子位置
            allow_unknown : 是否允许经过 UNKNOWN 格子

        Returns:
            bool: True 表示可移动
        """
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
            # 斜向移动：检查两侧通行性（防穿角）
            side_h = (cur[0] + dx, cur[1])   # 水平侧格子
            side_v = (cur[0],      cur[1] + dz)  # 垂直侧格子
            if not self._side_passable(side_h, allow_unknown) and not self._side_passable(side_v, allow_unknown):
                return False   # 两侧都不通行 → 禁止斜穿
        return True

    def _side_passable(self, pos: Position, allow_unknown: bool) -> bool:
        """
        判断斜向移动的侧边格子是否可通行（防穿角辅助）。

        Args:
            pos           : 侧边格子全局坐标
            allow_unknown : 是否允许 UNKNOWN 格子视为可通行

        Returns:
            bool: True 表示可通行（不是 OBSTACLE，且在 allow_unknown=False 时也不是 UNKNOWN）
        """
        if not self._in_bounds(pos):
            return False
        cell = int(self.global_map[pos[1], pos[0]])
        return cell != self.OBSTACLE and not (cell == self.UNKNOWN and not allow_unknown)

    def _reconstruct_path(
        self, parent: Dict[Position, Position], start: Position, goal: Position
    ) -> List[Position]:
        """
        通过父节点字典回溯 A* 路径，从 goal 反向追溯到 start。

        Args:
            parent : A* 搜索过程中记录的父节点映射 {当前格子: 来自格子}
            start  : 路径起点
            goal   : 路径终点（已到达目标）

        Returns:
            list[Position]: 从 start 到 goal 的顺序路径（含首尾）
        """
        path = [goal]
        cur  = goal
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # Primitive geometry（基础几何工具）
    # ─────────────────────────────────────────────────────────────────────────

    def _iter_neighbors(self, pos: Position) -> Iterable[Tuple[int, Position]]:
        """
        生成 pos 在 8 个方向的相邻格子。

        Args:
            pos : 中心格子坐标

        Yields:
            (action, neighbor_pos) : 动作索引 (0-7) 及对应的相邻格子坐标
        """
        for action in range(8):
            yield action, self._apply_move(pos, action)

    def _apply_move(self, pos: Position, action: int) -> Position:
        """
        根据动作索引计算移动后的新位置（不校验越界）。

        Args:
            pos    : 当前位置 (x, z)
            action : 动作索引 (0-7)，对应 ACTION_TO_DELTA 中的方向

        Returns:
            Position: 移动后的新坐标 (x + dx, z + dz)
        """
        dx, dz = self.ACTION_TO_DELTA[action]
        return pos[0] + dx, pos[1] + dz

    def _nearest_npc_dist(self, pos: Position, npcs: Sequence[Position]) -> float:
        """
        计算 pos 与所有官方机器人 (NPC) 中最近的 Chebyshev 距离。

        Args:
            pos  : 参考格子坐标
            npcs : NPC 全局坐标列表

        Returns:
            float: 最近 NPC 的 Chebyshev 距离；npcs 为空时返回 99.0（视为安全）
        """
        if not npcs:
            return 99.0
        return min(float(self._chebyshev_dist(pos, npc)) for npc in npcs)

    @staticmethod
    def _chebyshev_dist(a: Position, b: Position) -> int:
        """
        计算两个格子之间的 Chebyshev 距离（国际象棋棋盘距离）。

        在 8 方向移动中，Chebyshev 距离等于从 a 到 b 所需的最少步数（理想情况无障碍）。

        公式：max(|a.x - b.x|, |a.z - b.z|)

        Args:
            a, b : 两个格子的全局坐标

        Returns:
            int: Chebyshev 距离
        """
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _in_bounds(self, pos: Position) -> bool:
        """
        判断 pos 是否在 128×128 地图范围内。

        Args:
            pos : 格子全局坐标 (x, z)

        Returns:
            bool: True 表示 0 ≤ x < 128 且 0 ≤ z < 128
        """
        return 0 <= pos[0] < self.MAP_SIZE and 0 <= pos[1] < self.MAP_SIZE

    # ─────────────────────────────────────────────────────────────────────────
    # Observation parsing（观测数据解析）
    # ─────────────────────────────────────────────────────────────────────────

    def _unwrap_observation(self, env_obs: Any) -> Any:
        """
        从 env_obs 中提取 observation 字段（若存在）。

        env_obs 有两种形式：
          1. dict 形式：{"observation": Observation, ...}  → 返回 Observation
          2. 直接是 Observation 对象                       → 原样返回

        Args:
            env_obs : 环境返回的原始观测对象

        Returns:
            Any: Observation 对象或原始 env_obs
        """
        if isinstance(env_obs, dict) and "observation" in env_obs:
            return env_obs["observation"]
        return env_obs

    def _parse_hero_state(self, frame_state: Any) -> Any:
        """
        从 FrameState 中提取小悟机器人状态 (HeroState)。

        FrameState.heroes 可能是列表（多个 hero，取第一个）或单个对象。
        开发指南中小悟机器人数量固定为 1。

        Args:
            frame_state : FrameState 数据

        Returns:
            Any: HeroState 对象（含 pos / battery / battery_max / score 等字段）
        """
        heroes = self._get(frame_state, "heroes", {})
        if isinstance(heroes, (list, tuple)):
            return heroes[0] if heroes else {}
        return heroes

    def _parse_npc_positions(self, frame_state: Any) -> List[Position]:
        """
        从 FrameState 中提取所有官方机器人 (NPC) 的全局位置列表。

        NpcState 协议：每个 NPC 有 pos = {x, z}（全局绝对坐标）。
        仅保留有 pos 字段的 NPC（容错处理）。

        Args:
            frame_state : FrameState 数据

        Returns:
            list[Position]: NPC 全局坐标列表，无 NPC 时返回 []
        """
        npcs = self._get(frame_state, "npcs", [])
        if not isinstance(npcs, (list, tuple)):
            return []
        return [
            self._parse_position(self._get(npc, "pos", {}))
            for npc in npcs
            if self._get(npc, "pos", None) is not None
        ]

    def _parse_organ_states(self, frame_state: Any) -> List[Any]:
        """
        从 FrameState 中提取所有物件 (organ) 列表。

        目前开发指南中 organ 仅用于充电桩 (charger)，sub_type = 1。
        后续若有其他物件类型，由 _parse_charger_regions 中的 sub_type 过滤。

        Args:
            frame_state : FrameState 数据

        Returns:
            list[Any]: OrganState 对象列表
        """
        organs = self._get(frame_state, "organs", [])
        return list(organs) if isinstance(organs, (list, tuple)) else []

    def _parse_map_info(self, obs: Any) -> np.ndarray:
        """
        从 observation 中提取并转换 21×21 视野地图 (map_info)。

        map_info 协议（开发指南视野网格值协议）：
          0 = 障碍物/地图边界（OBSTACLE）
          1 = 已清扫的道路（CLEAN）
          2 = 污渍地面（DIRT）

        Args:
            obs : Observation 对象

        Returns:
            np.ndarray: shape=(21, 21) 的 int8 数组；
                        若 map_info 缺失或格式错误，返回全 1（全部视为 CLEAN）
        """
        map_info = self._get(obs, "map_info", None)
        if map_info is None:
            return np.ones((21, 21), dtype=np.int8)
        arr = np.asarray(map_info, dtype=np.int8)
        return arr if arr.ndim == 2 else np.ones((21, 21), dtype=np.int8)

    def _parse_position(self, obj: Any) -> Position:
        """
        从 Position 对象（或 dict）中提取 (x, z) 坐标并做范围裁剪。

        Position 协议（开发指南）：x = 横坐标（右为正），z = 纵坐标（下为正）。
        坐标裁剪到 [0, MAP_SIZE - 1] 防止越界。

        Args:
            obj : Position 对象或含 "x" / "z" 键的 dict

        Returns:
            Position: (x, z) 整数坐标，已裁剪到地图范围内
        """
        x = int(self._safe_float(self._get(obj, "x", 0), 0.0))
        z = int(self._safe_float(self._get(obj, "z", 0), 0.0))
        return (
            int(np.clip(x, 0, self.MAP_SIZE - 1)),
            int(np.clip(z, 0, self.MAP_SIZE - 1)),
        )

    def _parse_position_list(self, items: Any) -> List[Position]:
        """
        将 step_cleaned_cells 或其他 Position 列表批量解析为 Position 列表。

        Args:
            items : Position 对象或 dict 的列表

        Returns:
            list[Position]: 解析后的坐标列表，输入非列表时返回 []
        """
        if not isinstance(items, (list, tuple)):
            return []
        return [self._parse_position(item) for item in items]

    # ─────────────────────────────────────────────────────────────────────────
    # Generic helpers（通用辅助工具）
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        """
        通用属性 / 键值读取：支持 dict、None 和任意对象（通过 getattr）。

        使环境数据解析代码对 dict 和 protobuf 对象都兼容。

        Args:
            obj     : 源对象（dict / protobuf / None）
            key     : 要读取的字段名
            default : 字段不存在时的默认值

        Returns:
            Any: 字段值或 default
        """
        if isinstance(obj, dict):
            return obj.get(key, default)
        if obj is None:
            return default
        return getattr(obj, key, default)

    @staticmethod
    def _safe_float(v: Any, default: float) -> float:
        """
        安全地将任意值转换为 float，转换失败时返回 default。

        用于解析环境数据中可能为 None / 字符串 / protobuf 数值的字段（如 battery）。

        Args:
            v       : 待转换的值
            default : 转换失败时的默认浮点数

        Returns:
            float: 转换结果或 default
        """
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)
