"""LTSPPO expert utilities: NPC safety filter + charger planning signals + emergency fallback.

This module intentionally no longer provides regular charging policy guidance.

What remains:
  - NPC collision safety filter for legal-action masking
  - Weighted A* charger planning as a signal / fallback utility
  - Blocked-cell memory to stabilize replanning
  - Explicit extreme-emergency fallback path for agent runtime use
  - Helper methods for teacher / target reliability checks

What is removed from policy guidance:
  - No regular charging logit bias
  - No regular return-mode hysteresis controlling normal policy behavior

`return_mode` is kept only as a compatibility flag for "extreme emergency fallback
currently active", so existing runtime code can still suppress anti-stuck logic.
"""
from __future__ import annotations

import heapq
import numpy as np

from agent_ppo.conf.conf import Config


class ExpertPolicy:
    GRID = 128
    DELTAS = (
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    )

    # Compatibility / fallback control
    EXIT_EMERGENCY_RATIO = 0.12
    EMERGENCY_RATIO = float(getattr(Config, "EXPERT_EMERGENCY_BATTERY_RATIO", 0.05))
    EMERGENCY_SLACK_MARGIN = 0.0
    EMERGENCY_PATH_MARGIN = 2.0

    # Reliability thresholds for teacher signals
    RELIABLE_NPC_DIST = 4
    RELIABLE_SLACK_BUFFER = float(getattr(Config, "EXPERT_RELIABLE_SLACK_BUFFER", 12.0))
    RELIABLE_RETURN_RATIO = float(getattr(Config, "EXPERT_RELIABLE_RETURN_RATIO", 0.45))
    RELIABLE_PREPARE_RETURN_RATIO = float(getattr(Config, "EXPERT_RELIABLE_PREPARE_RETURN_RATIO", 0.65))

    # Blocked cell memory
    BLOCKED_TTL = 8

    # Cost-map parameters
    _INF_COST = 1e6
    _NPC_DANGER_MAX = 15.0
    _NPC_DANGER_DECAY = 2.0
    _NPC_DANGER_RADIUS = 8
    _UNEXPLORED_COST = 1.8
    _RECENT_TARGET_STEPS = 4
    _PLANNER_CHARGER_EVAL_LIMIT = int(getattr(Config, "PLANNER_CHARGER_EVAL_LIMIT", 3))

    def __init__(self):
        self._charger_list = []
        self.return_mode = False
        self._cached_path = []
        self._cached_distance = float("inf")
        self._cached_target = None
        self.blocked_cells = {}
        self._prev_pos = None
        self._last_emergency_reason = None

    def reset(self):
        """Reset per-episode state."""
        self._charger_list = []
        self.return_mode = False
        self._cached_path = []
        self._cached_distance = float("inf")
        self._cached_target = None
        self.blocked_cells = {}
        self._prev_pos = None
        self._last_emergency_reason = None

    # ------------------------------------------------------------------
    # Charger tracking
    # ------------------------------------------------------------------

    def update_chargers(self, prep):
        self._charger_list = []
        for organ in prep._organs:
            if int(organ.get("sub_type", 0)) != 1:
                continue
            pos = organ.get("pos") or {}
            cx, cz = int(pos.get("x", 0)), int(pos.get("z", 0))
            w = max(int(organ.get("w", 3)) // 2, 1)
            h = max(int(organ.get("h", 3)) // 2, 1)
            self._charger_list.append((cx, cz, w, h))

    def _is_on_charger(self, x, z):
        return any(abs(x - cx) <= w and abs(z - cz) <= h for cx, cz, w, h in self._charger_list)

    # ------------------------------------------------------------------
    # Layer 1: NPC safety filter
    # ------------------------------------------------------------------

    def filter_actions(self, prep, legal_action):
        """Block stepping onto NPC and moving toward nearby NPCs."""
        legal = list(legal_action)
        hx, hz = prep.cur_pos

        for npc in prep._npcs:
            pos = npc.get("pos") or {}
            nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
            ndx, ndz = nx - hx, nz - hz
            npc_dist = max(abs(ndx), abs(ndz))
            for idx, (dx, dz) in enumerate(self.DELTAS):
                if not legal[idx]:
                    continue
                nx2, nz2 = hx + dx, hz + dz
                if nx2 == nx and nz2 == nz:
                    legal[idx] = 0
                    continue
                if npc_dist <= 3:
                    new_dist = max(abs(nx2 - nx), abs(nz2 - nz))
                    if new_dist < npc_dist:
                        legal[idx] = 0
                elif npc_dist <= 5:
                    sx = (1 if ndx > 0 else -1) if ndx != 0 else 0
                    sz = (1 if ndz > 0 else -1) if ndz != 0 else 0
                    if dx == sx and dz == sz:
                        legal[idx] = 0

        if sum(legal) == 0:
            return list(legal_action)
        return legal

    # ------------------------------------------------------------------
    # Blocked cell memory
    # ------------------------------------------------------------------

    def update_blocked(self, prep, last_action):
        """Track cells that blocked movement (action executed but position unchanged)."""
        cur = prep.cur_pos

        expired = [k for k, v in self.blocked_cells.items() if v <= 1]
        for k in expired:
            del self.blocked_cells[k]
        for k in list(self.blocked_cells):
            self.blocked_cells[k] -= 1

        if self._prev_pos is not None and cur == self._prev_pos and last_action is not None and last_action >= 0:
            dx, dz = self.DELTAS[last_action]
            tx, tz = cur[0] + dx, cur[1] + dz
            if 0 <= tx < self.GRID and 0 <= tz < self.GRID:
                self.blocked_cells[(tx, tz)] = self.BLOCKED_TTL

        self._prev_pos = cur

    # ------------------------------------------------------------------
    # Public planning / reliability helpers
    # ------------------------------------------------------------------

    def get_charger_signal(self, prep, legal_action=None, last_action=-1, refresh_state=False):
        """Return structured charger-planning information without controlling policy."""
        self.update_chargers(prep)
        if refresh_state:
            self.update_blocked(prep, last_action)

        hx, hz = prep.cur_pos
        battery = float(getattr(prep, "battery", 0.0))
        battery_max = max(float(getattr(prep, "battery_max", 1.0)), 1.0)
        battery_ratio = battery / battery_max
        on_charger = self._is_on_charger(hx, hz)

        mode = int(getattr(prep, "current_mode", 0))
        contract_pressure = float(getattr(prep, "route_contract_pressure", 0.0))
        multi_target_eval = bool(
            int(getattr(prep, "total_charger", 1)) <= 2
            or battery_ratio <= float(getattr(Config, "UNKNOWN_PATH_RISK_BATTERY_RATIO", 0.45))
            or mode in (
                getattr(prep, "MODE_CONTRACT", 3),
                getattr(prep, "MODE_RETURN", 4),
            )
            or contract_pressure >= 0.25
            or getattr(prep, "route_anchor_center", None) is None
        )

        if multi_target_eval:
            charger_candidates = self._evaluate_charger_candidates(prep)
        else:
            charger_candidates = self._build_lightweight_charger_candidates(prep)
        best_candidate = charger_candidates[0] if charger_candidates else None

        charger_path = list(best_candidate.get("path", [])) if best_candidate else []
        charger_dist = float(best_candidate.get("astar_dist", float("inf"))) if best_candidate else float("inf")
        charger_target = tuple(best_candidate["center"]) if best_candidate and best_candidate.get("center") is not None else None
        path_source = str(best_candidate.get("path_source", "greedy_local")) if best_candidate else "greedy_local"

        if not charger_path:
            charger_dist = float(getattr(prep, "nearest_charger_dist", float("inf")))
            charger_target = charger_target or self._nearest_charger_center(hx, hz)

        margin = self._charge_margin(charger_path)
        slack = self._estimate_slack(prep, charger_dist)
        min_npc_dist = self._min_npc_dist(prep)
        unknown_path_ratio = self._unknown_path_ratio(prep, charger_path)
        target_gap = self._target_gap_from_candidates(charger_candidates)
        target_stable = self._target_stable(prep, charger_target)
        anchor_stable = self._anchor_stable(prep, charger_target)
        action_margin = self._best_action_margin(prep, legal_action, charger_target)
        suggested_action = None

        if charger_path and len(charger_path) >= 2:
            suggested_action = self._path_to_action(hx, hz, charger_path[1])

        if suggested_action is None:
            suggested_action = self._plan_to_charger(prep)

        if suggested_action is None and legal_action is not None:
            suggested_action = self._greedy_toward_charger(prep, legal_action)

        legal_and_safe = False
        if legal_action is not None and suggested_action is not None:
            legal_and_safe = bool(legal_action[suggested_action])

        reachable = bool(best_candidate.get("reachable", False)) if best_candidate else (bool(charger_path) or np.isfinite(charger_dist))
        target_reliable = self._is_target_signal_reliable(
            prep,
            legal_and_safe=legal_and_safe,
            reachable=reachable,
            charger_target=charger_target,
            charger_dist=charger_dist,
            slack=slack,
            min_npc_dist=min_npc_dist,
            unknown_path_ratio=unknown_path_ratio,
            target_gap=target_gap,
            target_stable=target_stable,
        )
        anchor_reliable = self._is_anchor_signal_reliable(
            reachable=reachable,
            charger_target=charger_target,
            charger_dist=charger_dist,
            target_gap=target_gap,
            target_stable=target_stable,
            anchor_stable=anchor_stable,
            unknown_path_ratio=unknown_path_ratio,
            min_npc_dist=min_npc_dist,
        )
        mode_reliable = self._is_mode_signal_reliable(
            battery_ratio=battery_ratio,
            slack=slack,
            min_npc_dist=min_npc_dist,
            target_reliable=target_reliable,
            on_charger=on_charger,
            reachable=reachable,
        )
        return_action_reliable = bool(
            target_reliable
            and suggested_action is not None
            and legal_and_safe
            and action_margin >= float(getattr(Config, "TEACHER_RETURN_ACTION_MARGIN_MIN", 1.0))
        )

        best_cheb_idx = 0
        best_astar_idx = 0
        selected_target_rank = 0
        all_known_path_count = 0
        total_known_route_count = 0
        topk_reachable_count = 0
        for idx, cand in enumerate(charger_candidates, start=1):
            if idx == 1:
                best_astar_idx = idx
            if idx == 1 and cand.get("is_nearest_cheb", False):
                best_cheb_idx = idx
            if cand.get("is_nearest_cheb", False) and best_cheb_idx == 0:
                best_cheb_idx = idx
            if cand.get("reachable", False) and np.isfinite(float(cand.get("astar_dist", float("inf")))):
                all_known_path_count += 1
                topk_reachable_count += 1
            total_known_route_count += int(max(float(cand.get("route_diversity", 0.0)), 0.0))
            if charger_target is not None and tuple(cand.get("center", ())) == tuple(charger_target):
                selected_target_rank = idx

        if best_cheb_idx == 0 and charger_candidates:
            best_cheb_idx = 1

        best_vs_second_gap = self._target_gap_from_candidates(charger_candidates)
        best_family_slacks = [
            float(cand.get("best_slack", float("-inf")))
            for cand in charger_candidates
            if bool(cand.get("reachable", False))
        ]
        multi_route_recoverability = float(np.clip(max(best_family_slacks), -1.0, 1.0)) if best_family_slacks else -1.0
        best_target_tangle_cost = float(best_candidate.get("best_tangle_cost", 0.0)) if best_candidate else 0.0
        best_target_edge_break_cost = float(best_candidate.get("best_edge_break_cost", 0.0)) if best_candidate else 0.0
        best_target_region_fragment_cost = float(best_candidate.get("best_region_fragment_cost", 0.0)) if best_candidate else 0.0
        best_target_route_diversity = float(best_candidate.get("route_diversity", 0.0)) if best_candidate else 0.0
        current_task_continuity_cost = float(
            np.clip(
                0.55 * best_target_edge_break_cost + 0.45 * best_target_region_fragment_cost,
                0.0,
                1.0,
            )
        )

        return {
            "battery_ratio": battery_ratio,
            "battery": battery,
            "on_charger": on_charger,
            "charger_dist": float(charger_dist),
            "charger_target": charger_target,
            "charger_path": charger_path,
            "margin": float(margin),
            "slack": float(slack),
            "min_npc_dist": float(min_npc_dist),
            "unknown_path_ratio": float(unknown_path_ratio),
            "target_gap": float(target_gap),
            "target_stable": bool(target_stable),
            "anchor_stable": bool(anchor_stable),
            "action_margin": float(action_margin),
            "suggested_action": suggested_action,
            "suggested_action_legal": legal_and_safe,
            "reachable": reachable,
            "path_source": path_source,
            "fallback_to_chebyshev": bool(path_source == "chebyshev"),
            "charger_candidates": charger_candidates,
            "all_charger_known_path_count": int(all_known_path_count),
            "planner_topk_reachable_count": int(topk_reachable_count),
            "planner_known_route_count_total": int(total_known_route_count),
            "best_astar_charger_idx": int(best_astar_idx),
            "best_cheb_charger_idx": int(best_cheb_idx),
            "selected_target_rank": int(selected_target_rank if selected_target_rank > 0 else 1 if charger_candidates else 0),
            "planner_best_target_best_cost": float(best_candidate.get("best_total_cost", float("inf"))) if best_candidate else float("inf"),
            "planner_best_target_safe_cost": float(best_candidate.get("best_safe_cost", float("inf"))) if best_candidate else float("inf"),
            "planner_best_target_unknown_ratio": float(best_candidate.get("unknown_path_ratio", 1.0)) if best_candidate else 1.0,
            "planner_best_target_route_diversity": best_target_route_diversity,
            "planner_best_vs_second_gap": float(best_vs_second_gap),
            "planner_multi_route_recoverability": multi_route_recoverability,
            "planner_best_target_tangle_cost": best_target_tangle_cost,
            "planner_best_target_edge_break_cost": best_target_edge_break_cost,
            "planner_best_target_region_fragment_cost": best_target_region_fragment_cost,
            "planner_current_task_continuity_cost": current_task_continuity_cost,
            "target_reliable": bool(target_reliable),
            "anchor_reliable": bool(anchor_reliable),
            "mode_reliable": bool(mode_reliable),
            "return_action_reliable": bool(return_action_reliable),
            "route_anchor": charger_target,
        }

    def is_target_teacher_reliable(self, prep, legal_action=None, last_action=-1):
        signal = self.get_charger_signal(prep, legal_action, last_action)
        return bool(signal["target_reliable"])

    def is_mode_teacher_reliable(self, prep, legal_action=None, last_action=-1):
        signal = self.get_charger_signal(prep, legal_action, last_action)
        return bool(signal["mode_reliable"])

    def get_teacher_guidance(self, prep, legal_action=None, last_action=-1, signal=None):
        """Return optional teacher guidance payload for future mode/target supervision."""
        signal = signal or self.get_charger_signal(prep, legal_action, last_action)
        if not (
            signal["target_reliable"]
            or signal["mode_reliable"]
            or signal["anchor_reliable"]
            or signal["return_action_reliable"]
        ):
            return None

        battery_ratio = signal["battery_ratio"]
        slack = signal["slack"]
        on_charger = signal["on_charger"]
        margin = float(signal.get("margin", 0.0))
        unknown_ratio = float(signal.get("unknown_path_ratio", 0.0))
        known_path_count = int(signal.get("all_charger_known_path_count", 0))

        local_dirt_density = float(getattr(prep, "local_dirt_density", 0.0))
        recoverability = float(getattr(prep, "future_recoverability_score", 0.0))
        contract_pressure = float(getattr(prep, "route_contract_pressure", 0.0))
        depart_steps = int(getattr(prep, "steps_since_charge", 999))

        if signal["min_npc_dist"] <= 2:
            mode = "evade"
        elif (
            battery_ratio <= Config.RETURN_BATTERY_RATIO
            or slack <= Config.RETURN_SLACK_THRESHOLD
            or recoverability <= Config.RETURN_RECOVERABILITY_THRESHOLD
            or margin <= Config.CHARGE_MARGIN_LOW
        ):
            mode = "return"
        elif (
            battery_ratio <= Config.CONTRACT_BATTERY_RATIO
            or slack <= Config.PREPARE_RETURN_SLACK_THRESHOLD
            or recoverability <= Config.CONTRACT_RECOVERABILITY_THRESHOLD
            or contract_pressure >= 0.5
            or margin <= Config.CHARGE_MARGIN_WARN
            or (
                known_path_count < min(int(getattr(prep, "total_charger", 1)), 2)
                and battery_ratio <= Config.UNKNOWN_PATH_RISK_BATTERY_RATIO
                and unknown_ratio >= Config.UNKNOWN_PATH_RISK_THRESHOLD
            )
        ):
            mode = "contract"
        elif depart_steps <= Config.DEPART_STEPS:
            mode = "depart"
        elif local_dirt_density >= Config.HARVEST_DIRT_DENSITY:
            mode = "harvest"
        else:
            mode = "expand"

        return {
            "mode": mode,
            "route_mode": mode,
            "route_anchor": signal["charger_target"],
            "target": signal["charger_target"],
            "action": signal["suggested_action"],
            "return_action": signal["suggested_action"],
            "mode_teacher_mask": float(signal["mode_reliable"]),
            "target_teacher_mask": float(signal["target_reliable"]),
            "route_anchor_teacher_mask": float(signal["anchor_reliable"]),
            "return_action_teacher_mask": float(signal["return_action_reliable"]),
            "signal": signal,
        }

    # ------------------------------------------------------------------
    # Extreme emergency fallback
    # ------------------------------------------------------------------

    def get_emergency_fallback(self, prep, legal_action, last_action=-1):
        """Return explicit charger fallback only in extreme emergency.

        This is runtime-oriented and intentionally much narrower than the old
        return-mode controller. It only activates when the robot is close to
        battery death and charger reachability is already critical.
        """
        legal_action = list(legal_action)
        signal = self.get_charger_signal(prep, legal_action, last_action, refresh_state=True)

        battery_ratio = signal["battery_ratio"]
        on_charger = signal["on_charger"]
        slack = signal["slack"]
        charger_dist = signal["charger_dist"]
        margin = signal["margin"]
        suggested_action = signal["suggested_action"]

        if self.return_mode:
            should_keep = (
                not on_charger
                and battery_ratio < self.EXIT_EMERGENCY_RATIO
                and (
                    slack <= self.RELIABLE_SLACK_BUFFER
                    or battery_ratio <= self.EMERGENCY_RATIO * 1.5
                )
            )
            if not should_keep:
                self.return_mode = False
                self._last_emergency_reason = None

        should_trigger = (
            not on_charger
            and suggested_action is not None
            and legal_action[suggested_action]
            and (
                (battery_ratio <= self.EMERGENCY_RATIO and slack <= self.EMERGENCY_SLACK_MARGIN)
                or (battery_ratio <= self.EMERGENCY_RATIO * 0.8)
                or (np.isfinite(charger_dist) and prep.battery <= charger_dist + self.EMERGENCY_PATH_MARGIN)
            )
        )

        if should_trigger:
            self.return_mode = True
            if battery_ratio <= self.EMERGENCY_RATIO and slack <= self.EMERGENCY_SLACK_MARGIN:
                reason = "battery_and_slack_critical"
            elif battery_ratio <= self.EMERGENCY_RATIO * 0.8:
                reason = "battery_ratio_critical"
            else:
                reason = "path_margin_critical"
            self._last_emergency_reason = reason
            return {
                "active": True,
                "action": suggested_action,
                "reason": reason,
                "signal": signal,
            }

        return {
            "active": False,
            "action": None,
            "reason": self._last_emergency_reason,
            "signal": signal,
        }

    # ------------------------------------------------------------------
    # Legacy-compatible interfaces used by runtime
    # ------------------------------------------------------------------

    def get_override(self, prep, legal_action, last_action=-1):
        """Legacy compatibility: only returns extreme-emergency fallback now."""
        fallback = self.get_emergency_fallback(prep, legal_action, last_action)
        if fallback["active"] and fallback["action"] is not None:
            return True, fallback["action"]
        return False, None

    def get_logit_bias(self, prep, legal_action, last_action=-1):
        """Return only NPC avoidance bias; no regular charging / return bias."""
        # Keep compatibility flag aligned with emergency state only.
        _ = self.get_emergency_fallback(prep, legal_action, last_action)

        bias = np.zeros(8, dtype=np.float32)
        hx, hz = prep.cur_pos

        for npc in prep._npcs:
            pos = npc.get("pos") or {}
            nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
            ndx, ndz = nx - hx, nz - hz
            npc_dist = max(abs(ndx), abs(ndz))
            if npc_dist > 6 or npc_dist < 1:
                continue
            for idx, (dx, dz) in enumerate(self.DELTAS):
                if not legal_action[idx]:
                    continue
                nlen = max(max(abs(ndx), abs(ndz)), 1.0)
                dot = (dx * ndx + dz * ndz) / nlen
                if dot > 0:
                    close = (6.0 - npc_dist) / 6.0
                    bias[idx] -= 2.0 * dot * close

        return bias

    # ------------------------------------------------------------------
    # Planner helpers
    # ------------------------------------------------------------------

    def _greedy_toward_charger(self, prep, legal_action):
        hx, hz = prep.cur_pos
        best_act = None
        best_dist = float("inf")
        for idx, (dx, dz) in enumerate(self.DELTAS):
            if not legal_action[idx]:
                continue
            nx, nz = hx + dx, hz + dz
            dist = self._charger_heuristic(nx, nz)
            if dist < best_dist:
                best_dist = dist
                best_act = idx
        return best_act

    def _charge_margin(self, path):
        """Dynamic safety margin based on path complexity."""
        if len(path) < 2:
            return 20.0
        turns = 0
        blocked_count = 0
        prev_delta = None
        for i in range(1, len(path)):
            cur = path[i - 1]
            nxt = path[i]
            delta = (
                int(np.clip(nxt[0] - cur[0], -1, 1)),
                int(np.clip(nxt[1] - cur[1], -1, 1)),
            )
            if prev_delta is not None and delta != prev_delta:
                turns += 1
            prev_delta = delta
            if nxt in self.blocked_cells:
                blocked_count += 1
        margin = 18.0 + 0.35 * float(turns) + 1.2 * float(blocked_count)
        return min(margin, 40.0)

    # ------------------------------------------------------------------
    # Path caching
    # ------------------------------------------------------------------

    def _plan_to_charger_cached(self, prep):
        """Use cached charger path if still valid, otherwise replan."""
        hx, hz = prep.cur_pos

        if self._cached_path:
            try:
                idx = self._cached_path.index((hx, hz))
                remaining = self._cached_path[idx:]
                if len(remaining) >= 2:
                    nxt = remaining[1]
                    if nxt not in self.blocked_cells:
                        return remaining, float(len(remaining) - 1), self._cached_target
            except ValueError:
                pass

        chargers = self._charger_list
        if not chargers:
            return [], float("inf"), None

        def is_goal(x, z):
            return any(abs(x - cx) <= w and abs(z - cz) <= h for cx, cz, w, h in chargers)

        h_func = self._charger_heuristic

        cost_map = self._build_cost_map(prep, npc_weight=1.0)
        act, path, dist = self._weighted_astar_full(prep, cost_map, is_goal, h_func)
        if act is not None and path:
            self._cached_path = path
            self._cached_distance = dist
            self._cached_target = self._match_target_to_charger(path[-1])
            return path, dist, self._cached_target

        cost_map = self._build_cost_map(prep, npc_weight=0.3)
        act, path, dist = self._weighted_astar_full(prep, cost_map, is_goal, h_func)
        if act is not None and path:
            self._cached_path = path
            self._cached_distance = dist
            self._cached_target = self._match_target_to_charger(path[-1])
            return path, dist, self._cached_target

        return [], float("inf"), None

    @staticmethod
    def _path_to_action(hx, hz, next_pos):
        dx = int(np.clip(next_pos[0] - hx, -1, 1))
        dz = int(np.clip(next_pos[1] - hz, -1, 1))
        for idx, (adx, adz) in enumerate(ExpertPolicy.DELTAS):
            if adx == dx and adz == dz:
                return idx
        return None

    # ------------------------------------------------------------------
    # Cost-map construction
    # ------------------------------------------------------------------

    def _build_cost_map(self, prep, npc_weight=1.0, unexplored_cost=None, blocked_penalty=4.0):
        G = self.GRID
        base_unexplored_cost = self._UNEXPLORED_COST if unexplored_cost is None else float(unexplored_cost)
        cost = np.full((G, G), base_unexplored_cost, dtype=np.float32)

        explored = prep.explored_map >= 0.5
        base_passable = explored & (prep.passable_map >= 0.5)
        cost[base_passable] = 1.0

        visit_penalty = np.clip(prep.visit_count * 0.15, 0, 0.75)
        cost[base_passable] += visit_penalty[base_passable]

        cost[explored & (prep.passable_map < 0.5)] = self._INF_COST

        for (bx, bz), ttl in self.blocked_cells.items():
            if 0 <= bx < G and 0 <= bz < G and cost[bx, bz] < self._INF_COST:
                cost[bx, bz] += float(blocked_penalty)

        if npc_weight > 0:
            for npc in prep._npcs:
                pos = npc.get("pos") or {}
                nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
                radius = self._NPC_DANGER_RADIUS
                x0, x1 = max(nx - radius, 0), min(nx + radius + 1, G)
                z0, z1 = max(nz - radius, 0), min(nz + radius + 1, G)

                lx = np.arange(x0, x1, dtype=np.float32)
                lz = np.arange(z0, z1, dtype=np.float32)
                xx, zz = np.meshgrid(lx, lz, indexing="ij")
                dist = np.maximum(np.abs(xx - nx), np.abs(zz - nz))
                danger = npc_weight * self._NPC_DANGER_MAX * np.exp(-dist / self._NPC_DANGER_DECAY)
                danger[dist > radius] = 0.0

                region = cost[x0:x1, z0:z1]
                safe = region < self._INF_COST
                region[safe] += danger[safe]

        return cost

    # ------------------------------------------------------------------
    # Weighted A* search
    # ------------------------------------------------------------------

    def _weighted_astar_full(self, prep, cost_map, is_goal, h_func):
        sx, sz = prep.cur_pos
        if is_goal(sx, sz):
            return 0, [(sx, sz)], 0.0

        G = self.GRID
        inf = 1e9
        dist_arr = np.full((G, G), inf, dtype=np.float32)
        dist_arr[sx, sz] = 0.0
        first_act = np.full((G, G), -1, dtype=np.int8)
        parent = {}
        closed = np.zeros((G, G), dtype=np.bool_)

        counter = 0
        heap = [(h_func(sx, sz), counter, sx, sz)]

        while heap:
            _, _, x, z = heapq.heappop(heap)
            if closed[x, z]:
                continue
            closed[x, z] = True
            if is_goal(x, z):
                path = self._reconstruct_path(parent, (sx, sz), (x, z))
                fa = int(first_act[x, z])
                return fa if fa >= 0 else 0, path, float(dist_arr[x, z])

            cur_d = dist_arr[x, z]
            for idx, (dx, dz) in enumerate(self.DELTAS):
                nx, nz = x + dx, z + dz
                if not (0 <= nx < G and 0 <= nz < G) or closed[nx, nz]:
                    continue
                step_cost = cost_map[nx, nz]
                if step_cost >= self._INF_COST:
                    continue
                if dx != 0 and dz != 0:
                    c1 = (0 <= x + dx < G) and cost_map[x + dx, z] < self._INF_COST
                    c2 = (0 <= z + dz < G) and cost_map[x, z + dz] < self._INF_COST
                    if not c1 and not c2:
                        continue
                move = float(Config.DIAGONAL_MOVE_COST if (dx != 0 and dz != 0) else Config.ORTHOGONAL_MOVE_COST)
                new_d = cur_d + step_cost * move
                if new_d < dist_arr[nx, nz]:
                    dist_arr[nx, nz] = new_d
                    first_act[nx, nz] = first_act[x, z] if first_act[x, z] >= 0 else idx
                    parent[(nx, nz)] = (x, z)
                    counter += 1
                    heapq.heappush(heap, (new_d + h_func(nx, nz), counter, nx, nz))

        return None, [], float("inf")

    @staticmethod
    def _reconstruct_path(parent, start, goal):
        path = [goal]
        cur = goal
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    def _weighted_astar(self, prep, cost_map, is_goal, h_func):
        sx, sz = prep.cur_pos
        if is_goal(sx, sz):
            return None

        G = self.GRID
        inf = 1e9
        dist_arr = np.full((G, G), inf, dtype=np.float32)
        dist_arr[sx, sz] = 0.0
        first_act = np.full((G, G), -1, dtype=np.int8)
        closed = np.zeros((G, G), dtype=np.bool_)

        counter = 0
        heap = [(h_func(sx, sz), counter, sx, sz)]

        while heap:
            _, _, x, z = heapq.heappop(heap)
            if closed[x, z]:
                continue
            closed[x, z] = True
            if is_goal(x, z):
                fa = int(first_act[x, z])
                return fa if fa >= 0 else 0

            cur_d = dist_arr[x, z]
            for idx, (dx, dz) in enumerate(self.DELTAS):
                nx, nz = x + dx, z + dz
                if not (0 <= nx < G and 0 <= nz < G) or closed[nx, nz]:
                    continue
                step_cost = cost_map[nx, nz]
                if step_cost >= self._INF_COST:
                    continue
                if dx != 0 and dz != 0:
                    c1 = (0 <= x + dx < G) and cost_map[x + dx, z] < self._INF_COST
                    c2 = (0 <= z + dz < G) and cost_map[x, z + dz] < self._INF_COST
                    if not c1 and not c2:
                        continue
                move = float(Config.DIAGONAL_MOVE_COST if (dx != 0 and dz != 0) else Config.ORTHOGONAL_MOVE_COST)
                new_d = cur_d + step_cost * move
                if new_d < dist_arr[nx, nz]:
                    dist_arr[nx, nz] = new_d
                    first_act[nx, nz] = first_act[x, z] if first_act[x, z] >= 0 else idx
                    counter += 1
                    heapq.heappush(heap, (new_d + h_func(nx, nz), counter, nx, nz))

        return None

    # ------------------------------------------------------------------
    # Charger heuristic / planning
    # ------------------------------------------------------------------

    def _charger_heuristic(self, x, z):
        if not self._charger_list:
            return 0.0
        return min(max(abs(x - cx) - w, abs(z - cz) - h, 0) for cx, cz, w, h in self._charger_list)

    def _plan_to_charger(self, prep):
        chargers = self._charger_list
        if not chargers:
            return None

        def is_goal(x, z):
            return any(abs(x - cx) <= w and abs(z - cz) <= h for cx, cz, w, h in chargers)

        h_func = self._charger_heuristic

        cost_map = self._build_cost_map(prep, npc_weight=1.0)
        act = self._weighted_astar(prep, cost_map, is_goal, h_func)
        if act is not None:
            return act

        cost_map = self._build_cost_map(prep, npc_weight=0.3)
        act = self._weighted_astar(prep, cost_map, is_goal, h_func)
        if act is not None:
            return act

        return None

    # ------------------------------------------------------------------
    # Internal reliability helpers
    # ------------------------------------------------------------------

    def _estimate_slack(self, prep, charger_dist):
        if hasattr(prep, "charger_slack"):
            try:
                return float(prep.charger_slack)
            except (TypeError, ValueError):
                pass
        if not np.isfinite(charger_dist):
            return float("-inf")
        return float(getattr(prep, "battery", 0.0) - charger_dist)

    def _min_npc_dist(self, prep):
        hx, hz = prep.cur_pos
        min_npc_dist = float("inf")
        for npc in prep._npcs:
            pos = npc.get("pos") or {}
            nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
            min_npc_dist = min(min_npc_dist, max(abs(nx - hx), abs(nz - hz)))
        return min_npc_dist

    def _nearest_charger_center(self, hx, hz):
        if not self._charger_list:
            return None
        return min(
            ((cx, cz) for cx, cz, _, _ in self._charger_list),
            key=lambda item: max(abs(item[0] - hx), abs(item[1] - hz)),
        )

    def _match_target_to_charger(self, pos):
        if pos is None:
            return None
        x, z = pos
        for cx, cz, w, h in self._charger_list:
            if abs(x - cx) <= w and abs(z - cz) <= h:
                return (cx, cz)
        return self._nearest_charger_center(x, z)

    def _is_target_signal_reliable(
        self,
        prep,
        legal_and_safe,
        reachable,
        charger_target,
        charger_dist,
        slack,
        min_npc_dist,
        unknown_path_ratio,
        target_gap,
        target_stable,
    ):
        if not reachable or charger_target is None or not legal_and_safe:
            return False
        if not np.isfinite(charger_dist):
            return False
        if min_npc_dist <= self.RELIABLE_NPC_DIST:
            return False
        if slack <= -self.RELIABLE_SLACK_BUFFER:
            return False
        if unknown_path_ratio > float(getattr(Config, "TEACHER_UNKNOWN_PATH_RATIO_MAX", 0.20)):
            return False
        if not target_stable:
            return False
        if target_gap < float(getattr(Config, "TEACHER_TARGET_MARGIN_MIN", 3.0)):
            return False
        return True

    def _is_mode_signal_reliable(self, battery_ratio, slack, min_npc_dist, target_reliable, on_charger, reachable):
        if min_npc_dist <= 2:
            return True
        if on_charger and battery_ratio < self.RELIABLE_RETURN_RATIO:
            return True
        if battery_ratio <= self.RELIABLE_RETURN_RATIO or slack <= 0.0:
            return bool(reachable)
        if battery_ratio <= self.RELIABLE_PREPARE_RETURN_RATIO or slack <= self.RELIABLE_SLACK_BUFFER:
            return bool(reachable or target_reliable)
        return False

    def _is_anchor_signal_reliable(
        self,
        reachable,
        charger_target,
        charger_dist,
        target_gap,
        target_stable,
        anchor_stable,
        unknown_path_ratio,
        min_npc_dist,
    ):
        if not reachable or charger_target is None or not np.isfinite(charger_dist):
            return False
        if min_npc_dist <= self.RELIABLE_NPC_DIST:
            return False
        if unknown_path_ratio > float(getattr(Config, "TEACHER_UNKNOWN_PATH_RATIO_MAX", 0.20)):
            return False
        if target_gap < float(getattr(Config, "TEACHER_ANCHOR_MARGIN_MIN", 4.0)):
            return False
        if not target_stable or not anchor_stable:
            return False
        return True

    def _unknown_path_ratio(self, prep, path):
        if not path or len(path) <= 1:
            return 0.0
        unknown = 0
        total = 0
        for x, z in path[1:]:
            total += 1
            if not (0 <= x < self.GRID and 0 <= z < self.GRID):
                unknown += 1
                continue
            if float(prep.explored_map[x, z]) < 0.5:
                unknown += 1
        if total <= 0:
            return 0.0
        return float(unknown) / float(total)

    def _target_gap_from_candidates(self, candidates):
        scored = [float(cand.get("score", float("inf"))) for cand in candidates if np.isfinite(float(cand.get("score", float("inf"))))]
        scored.sort()
        if len(scored) < 2:
            return float("inf")
        return float(scored[1] - scored[0])

    def _target_gap(self, prep, charger_target, charger_dist):
        if charger_target is None:
            return 0.0
        candidates = []
        for _, dx, dz, cx, cz, _, _ in getattr(prep, "all_charger_info", []):
            cheb = float(max(abs(dx), abs(dz)))
            if (cx, cz) == tuple(charger_target) and np.isfinite(charger_dist):
                candidates.append(float(charger_dist))
            else:
                candidates.append(cheb)
        candidates = sorted(candidates)
        if len(candidates) < 2:
            return float("inf")
        return float(candidates[1] - candidates[0])

    def _target_stable(self, prep, charger_target):
        if charger_target is None:
            return False
        checker = getattr(prep, "is_teacher_target_stable", None)
        if checker is None:
            return True
        return bool(checker(tuple(charger_target)))

    def _anchor_stable(self, prep, charger_target):
        if charger_target is None:
            return False
        checker = getattr(prep, "is_route_anchor_stable", None)
        if checker is None:
            return True
        return bool(checker(tuple(charger_target)))

    def _best_action_margin(self, prep, legal_action, charger_target):
        if legal_action is None or charger_target is None:
            return 0.0
        hx, hz = prep.cur_pos
        tx, tz = charger_target
        scores = []
        for idx, (dx, dz) in enumerate(self.DELTAS):
            if not legal_action[idx]:
                continue
            nx, nz = hx + dx, hz + dz
            progress = max(abs(hx - tx), abs(hz - tz)) - max(abs(nx - tx), abs(nz - tz))
            scores.append(float(progress))
        if len(scores) < 2:
            return float("inf") if scores else 0.0
        scores.sort(reverse=True)
        return float(scores[0] - scores[1])

    def _charger_goal_heuristic(self, charger):
        cx, cz, w, h = charger

        def h_func(x, z):
            return max(abs(x - cx) - w, abs(z - cz) - h, 0)

        return h_func

    def _plan_to_specific_charger(self, prep, charger, primary_cost_map=None, relaxed_cost_map=None):
        primary_cost_map = primary_cost_map if primary_cost_map is not None else self._build_cost_map(prep, npc_weight=1.0)
        relaxed_cost_map = relaxed_cost_map if relaxed_cost_map is not None else self._build_cost_map(prep, npc_weight=0.3)
        cx, cz, w, h = charger

        def is_goal(x, z):
            return abs(x - cx) <= w and abs(z - cz) <= h

        h_func = self._charger_goal_heuristic(charger)
        act, path, dist = self._weighted_astar_full(prep, primary_cost_map, is_goal, h_func)
        if act is not None and path:
            return path, dist, "astar"
        act, path, dist = self._weighted_astar_full(prep, relaxed_cost_map, is_goal, h_func)
        if act is not None and path:
            return path, dist, "astar_relaxed"
        return [], float("inf"), "chebyshev"

    def _plan_route_family(self, prep, charger, family_kind, primary_cost_map=None, relaxed_cost_map=None):
        if family_kind == "best_cost_route":
            return self._plan_to_specific_charger(
                prep,
                charger,
                primary_cost_map=primary_cost_map,
                relaxed_cost_map=relaxed_cost_map,
            )

        if family_kind == "low_unknown_route":
            low_unknown_primary = self._build_cost_map(prep, npc_weight=0.8, unexplored_cost=3.2, blocked_penalty=4.5)
            low_unknown_relaxed = self._build_cost_map(prep, npc_weight=0.2, unexplored_cost=2.6, blocked_penalty=4.5)
            return self._plan_to_specific_charger(
                prep,
                charger,
                primary_cost_map=low_unknown_primary,
                relaxed_cost_map=low_unknown_relaxed,
            )

        safe_primary = self._build_cost_map(prep, npc_weight=1.6, unexplored_cost=2.0, blocked_penalty=6.0)
        safe_relaxed = self._build_cost_map(prep, npc_weight=0.8, unexplored_cost=1.8, blocked_penalty=6.0)
        return self._plan_to_specific_charger(
            prep,
            charger,
            primary_cost_map=safe_primary,
            relaxed_cost_map=safe_relaxed,
        )

    def _heading_conflict(self, prep, path):
        if not path or len(path) < 2:
            return 0.0
        if len(getattr(prep, "_trajectory", [])) < 3:
            return 0.0

        prev = prep._trajectory[-2]
        cur = prep._trajectory[-1]
        recent_delta = (
            int(np.clip(cur[0] - prev[0], -1, 1)),
            int(np.clip(cur[1] - prev[1], -1, 1)),
        )
        first_step = path[1]
        hx, hz = prep.cur_pos
        return_delta = (
            int(np.clip(first_step[0] - hx, -1, 1)),
            int(np.clip(first_step[1] - hz, -1, 1)),
        )
        if recent_delta == (0, 0) or return_delta == (0, 0):
            return 0.0
        if recent_delta == return_delta:
            return 0.0
        if recent_delta == (-return_delta[0], -return_delta[1]):
            return 1.0
        return 0.5

    def _route_separation(self, prep, path, horizon=6):
        if not path or len(path) < 2:
            return 0.0
        hx, hz = prep.cur_pos
        local_task_value = (
            0.45 * float(np.clip(getattr(prep, "local_dirt_density", 0.0), 0.0, 1.0))
            + 0.25 * float(np.clip(float(getattr(prep, "dirty_adjacent", 0.0)) / 4.0, 0.0, 1.0))
            + 0.20 * float(np.clip(getattr(prep, "local_frontier_density", 0.0), 0.0, 1.0))
            + 0.10 * float(np.clip(float(getattr(prep, "new_explored_cells", 0.0)) / 6.0, 0.0, 1.0))
        )
        if local_task_value <= 0.0:
            return 0.0

        leave_step = horizon
        for idx, (px, pz) in enumerate(path[1 : horizon + 1], start=1):
            if max(abs(px - hx), abs(pz - hz)) > 2:
                leave_step = idx
                break
        route_separation = 1.0 - float(leave_step - 1) / max(float(horizon), 1.0)
        return float(np.clip(route_separation, 0.0, 1.0))

    def _summarize_route_family(self, prep, path, astar_dist, path_source, family_kind):
        reachable = bool(path) and np.isfinite(astar_dist)
        unknown_ratio = self._unknown_path_ratio(prep, path) if reachable else 1.0
        slack = self._estimate_slack(prep, astar_dist)
        cost_length = float(astar_dist) / max(float(self.GRID), 1.0) if reachable else 1.0
        cost_unknown = float(unknown_ratio)

        path_cells = path[1:] if len(path) > 1 else []
        if reachable and path_cells:
            revisit_mean = float(
                np.mean(
                    [
                        float(np.clip(prep.visit_count[x, z] / 6.0, 0.0, 1.0))
                        for x, z in path_cells
                        if 0 <= x < self.GRID and 0 <= z < self.GRID
                    ]
                )
            )
            safety_mean = float(
                np.mean(
                    [
                        float(np.clip(prep.npc_risk_map[x, z], 0.0, 1.0))
                        for x, z in path_cells
                        if 0 <= x < self.GRID and 0 <= z < self.GRID
                    ]
                )
            )
        else:
            revisit_mean = 1.0 if not reachable else 0.0
            safety_mean = 1.0 if not reachable else 0.0

        task_interrupt = (
            0.45 * float(np.clip(getattr(prep, "local_dirt_density", 0.0), 0.0, 1.0))
            + 0.25 * float(np.clip(float(getattr(prep, "dirty_adjacent", 0.0)) / 4.0, 0.0, 1.0))
            + 0.20 * float(np.clip(getattr(prep, "local_frontier_density", 0.0), 0.0, 1.0))
            + 0.10 * float(np.clip(float(getattr(prep, "new_explored_cells", 0.0)) / 6.0, 0.0, 1.0))
        )
        if not reachable:
            task_interrupt = 1.0

        loop_proxy = float(
            np.clip(
                0.5 * float(np.clip(float(getattr(prep, "same_region_streak", 0.0)) / 8.0, 0.0, 1.0))
                + 0.5 * (
                    1.0 - float(np.clip(float(getattr(prep, "recent_unique_cells_20", 0.0)) / 20.0, 0.0, 1.0))
                ),
                0.0,
                1.0,
            )
        )
        cost_tangle = float(
            np.clip(
                0.50 * float(np.clip(float(getattr(prep, "path_cross_count_50", 0.0)) / 10.0, 0.0, 1.0))
                + 0.30 * (1.0 - float(np.clip(float(getattr(prep, "coverage_efficiency_20", 1.0)), 0.0, 1.0)))
                + 0.20 * loop_proxy,
                0.0,
                1.0,
            )
        )
        cost_edge_break = float(
            np.clip(float(np.clip(getattr(prep, "local_frontier_density", 0.0), 0.0, 1.0)) * self._heading_conflict(prep, path), 0.0, 1.0)
        )
        cost_region_fragment = float(np.clip(task_interrupt * self._route_separation(prep, path), 0.0, 1.0))
        slack_risk = float(np.clip(-float(slack) / 12.0, 0.0, 1.0)) if reachable else 1.0

        state_gate = 1.0
        battery_ratio = float(getattr(prep, "battery", 0.0)) / max(float(getattr(prep, "battery_max", 1.0)), 1.0)
        if battery_ratio <= float(getattr(Config, "RETURN_BATTERY_RATIO", 0.18)):
            state_gate = 0.0
        elif battery_ratio <= float(getattr(Config, "CONTRACT_BATTERY_RATIO", 0.28)):
            state_gate = 0.4

        weighted_tangle = cost_tangle * state_gate
        weighted_edge_break = cost_edge_break * state_gate
        weighted_region_fragment = cost_region_fragment * state_gate

        cost_total = (
            1.00 * cost_length
            + 0.60 * slack_risk
            + 0.35 * cost_unknown
            + 0.25 * safety_mean
            + 0.18 * revisit_mean
            + 0.12 * task_interrupt
            + 0.08 * weighted_tangle
            + 0.05 * weighted_edge_break
            + 0.04 * weighted_region_fragment
        )
        if not reachable:
            cost_total += float(getattr(Config, "TARGET_SELECT_UNREACHABLE_PENALTY", 18.0))

        return {
            "family_kind": family_kind,
            "path": list(path),
            "path_source": str(path_source),
            "reachable": bool(reachable),
            "astar_dist": float(astar_dist if np.isfinite(astar_dist) else float("inf")),
            "unknown_ratio": float(cost_unknown),
            "slack": float(slack),
            "cost_total": float(cost_total),
            "cost_length": float(cost_length),
            "cost_unknown": float(cost_unknown),
            "cost_safety": float(safety_mean),
            "cost_revisit": float(revisit_mean),
            "cost_task_interrupt": float(task_interrupt),
            "cost_tangle": float(weighted_tangle),
            "cost_edge_break": float(weighted_edge_break),
            "cost_region_fragment": float(weighted_region_fragment),
        }

    def _build_route_family_set(self, prep, charger, primary_cost_map=None, relaxed_cost_map=None):
        families = []
        for family_kind in ("best_cost_route", "low_unknown_route", "safe_route"):
            path, astar_dist, path_source = self._plan_route_family(
                prep,
                charger,
                family_kind,
                primary_cost_map=primary_cost_map,
                relaxed_cost_map=relaxed_cost_map,
            )
            families.append(self._summarize_route_family(prep, path, astar_dist, path_source, family_kind))
        return families

    def _evaluate_charger_candidates(self, prep):
        hx, hz = prep.cur_pos
        if not self._charger_list:
            return []

        primary_cost_map = self._build_cost_map(prep, npc_weight=1.0)
        relaxed_cost_map = self._build_cost_map(prep, npc_weight=0.3)

        ordered = sorted(
            self._charger_list,
            key=lambda item: max(abs(hx - item[0]) - item[2], abs(hz - item[1]) - item[3], 0),
        )
        eval_targets = []
        known_centers = set()

        def _add_target(center):
            if center is None:
                return
            center = tuple(center)
            if center in known_centers:
                return
            for charger in ordered:
                if (charger[0], charger[1]) == center:
                    eval_targets.append(charger)
                    known_centers.add(center)
                    return

        _add_target(self._cached_target)
        _add_target(getattr(prep, "route_anchor_center", None))
        for charger in ordered:
            if len(eval_targets) >= self._PLANNER_CHARGER_EVAL_LIMIT:
                break
            _add_target((charger[0], charger[1]))

        candidates = []
        nearest_cheb_center = (ordered[0][0], ordered[0][1]) if ordered else None
        for charger in ordered:
            center = (charger[0], charger[1])
            cheb_dist = float(max(abs(hx - charger[0]) - charger[2], abs(hz - charger[1]) - charger[3], 0))
            if center in known_centers:
                route_families = self._build_route_family_set(
                    prep,
                    charger,
                    primary_cost_map=primary_cost_map,
                    relaxed_cost_map=relaxed_cost_map,
                )
                reachable_families = [family for family in route_families if family["reachable"]]
                best_family = min(reachable_families, key=lambda item: item["cost_total"]) if reachable_families else route_families[0]
                safe_family = min(reachable_families, key=lambda item: item["cost_safety"]) if reachable_families else route_families[-1]
                low_unknown_family = min(reachable_families, key=lambda item: item["unknown_ratio"]) if reachable_families else route_families[1]
                path = list(best_family.get("path", []))
                astar_dist = float(best_family.get("astar_dist", float("inf")))
                path_source = str(best_family.get("path_source", "chebyshev"))
                reachable = bool(best_family.get("reachable", False))
                unknown_ratio = float(best_family.get("unknown_ratio", 1.0))
                route_diversity = float(len(reachable_families))
                best_total_cost = float(best_family.get("cost_total", float("inf")))
                best_safe_cost = float(safe_family.get("cost_total", float("inf")))
                best_tangle_cost = float(best_family.get("cost_tangle", 0.0))
                best_edge_break_cost = float(best_family.get("cost_edge_break", 0.0))
                best_region_fragment_cost = float(best_family.get("cost_region_fragment", 0.0))
                best_slack = float(max((family.get("slack", float("-inf")) for family in reachable_families), default=float("-inf")))
            else:
                route_families = []
                path = []
                astar_dist = cheb_dist
                path_source = "chebyshev"
                reachable = False
                unknown_ratio = 1.0
                route_diversity = 0.0
                best_total_cost = float(
                    cheb_dist
                    + float(getattr(Config, "TARGET_SELECT_UNKNOWN_COST", 10.0)) * unknown_ratio
                    + float(getattr(Config, "TARGET_SELECT_UNREACHABLE_PENALTY", 18.0))
                )
                best_safe_cost = best_total_cost
                best_tangle_cost = 1.0
                best_edge_break_cost = 1.0
                best_region_fragment_cost = 1.0
                best_slack = float("-inf")

            selection_score = float(
                1.00 * best_total_cost
                + 0.25 * best_safe_cost
                + 0.20 * unknown_ratio
                + 0.08 * best_tangle_cost
                + 0.05 * best_edge_break_cost
                + 0.04 * best_region_fragment_cost
                - 0.12 * max(route_diversity - 1.0, 0.0)
            )
            candidates.append(
                {
                    "center": center,
                    "path": path,
                    "astar_dist": float(astar_dist),
                    "dist": cheb_dist,
                    "reachable": bool(reachable),
                    "unknown_path_ratio": float(unknown_ratio),
                    "score": selection_score,
                    "path_source": path_source,
                    "is_nearest_cheb": bool(center == nearest_cheb_center),
                    "route_families": route_families,
                    "route_diversity": route_diversity,
                    "best_total_cost": best_total_cost,
                    "best_safe_cost": best_safe_cost,
                    "best_tangle_cost": best_tangle_cost,
                    "best_edge_break_cost": best_edge_break_cost,
                    "best_region_fragment_cost": best_region_fragment_cost,
                    "best_slack": best_slack,
                }
            )

        candidates.sort(key=lambda item: (not item["reachable"], item["score"], item["dist"]))
        return candidates

    def _build_lightweight_charger_candidates(self, prep):
        hx, hz = prep.cur_pos
        path, dist, target = self._plan_to_charger_cached(prep)
        candidates = []
        for idx, (_, dx, dz, cx, cz, _, _) in enumerate(getattr(prep, "all_charger_info", [])):
            cheb_dist = float(max(abs(dx), abs(dz)))
            is_target = target is not None and (cx, cz) == tuple(target)
            reachable = bool(path) and is_target
            unknown_ratio = self._unknown_path_ratio(prep, path) if reachable else 1.0
            astar_dist = float(dist if reachable and np.isfinite(dist) else cheb_dist)
            base_cost = float(
                astar_dist
                + float(getattr(Config, "TARGET_SELECT_UNKNOWN_COST", 10.0)) * unknown_ratio
                + (0.0 if reachable else float(getattr(Config, "TARGET_SELECT_UNREACHABLE_PENALTY", 18.0)))
            )
            candidates.append(
                {
                    "center": (cx, cz),
                    "path": list(path) if reachable else [],
                    "astar_dist": astar_dist,
                    "dist": cheb_dist,
                    "reachable": bool(reachable),
                    "unknown_path_ratio": float(unknown_ratio),
                    "score": base_cost,
                    "path_source": "astar" if reachable else "chebyshev",
                    "is_nearest_cheb": bool(idx == 0),
                    "route_families": [],
                    "route_diversity": float(1.0 if reachable else 0.0),
                    "best_total_cost": base_cost,
                    "best_safe_cost": base_cost,
                    "best_tangle_cost": 0.0 if reachable else 1.0,
                    "best_edge_break_cost": 0.0 if reachable else 1.0,
                    "best_region_fragment_cost": 0.0 if reachable else 1.0,
                    "best_slack": float(self._estimate_slack(prep, astar_dist) if reachable else float("-inf")),
                }
            )
        candidates.sort(key=lambda item: (not item["reachable"], item["score"], item["dist"]))
        return candidates
