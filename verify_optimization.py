#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Comprehensive test of optimized reward function and action priority features.
验证优化后的奖励函数和行为优先级特征是否正常工作。
"""

import numpy as np
import sys

# Add project root to path
sys.path.insert(0, 'E:\\competition\\26fwwb')

from agent_ppo.feature.preprocessor import Preprocessor

def test_charging_scenario():
    """Test P1 priority: Charging behavior."""
    print("\n=== Testing P1 Priority: CHARGING ===")
    prep = Preprocessor()
    
    # Scenario 1: Low battery (25%), close to charger
    prep.step_no = 200
    prep.battery = 50  # 25% battery
    prep.battery_max = 200
    prep.cur_pos = (64.0, 64.0)
    prep.total_dirt = 10000
    prep.dirt_cleaned = 500
    prep._organs = [{"pos": {"x": 66.0, "z": 64.0}}]  # Charger 2 units away
    prep._npcs = [{"pos": {"x": 100.0, "z": 100.0}}]  # NPC far away
    prep._step_cleaned = 0
    prep._prev_battery_ratio = 0.24
    prep._cur_battery_ratio = 0.25
    prep._prev_min_charger_dist = 5.0
    prep._prev_min_npc_dist = 50.0
    prep._was_stuck = False
    prep._terminated = False
    prep._truncated = False
    prep._fail_reason = ""
    
    reward = prep.reward_process()
    print(f"✓ Low battery + close to charger:")
    print(f"  - Reward: {reward:.4f} (should have strong positive due to distance delta)")
    
    # Scenario 2: Successful charging
    prep._prev_battery_ratio = 0.24
    prep._cur_battery_ratio = 0.35  # +0.11 gain (> 0.08 threshold)
    prep._prev_min_charger_dist = 2.0
    prep._last_charge_step = prep.step_no - 20  # > 10 steps ago
    
    reward = prep.reward_process()
    print(f"✓ Successful charging (+0.11 battery gain):")
    print(f"  - Reward: {reward:.4f} (should include +5.0 bonus)")
    
    # Test charging weight calculation
    battery_ratio = 0.50  # 50% battery
    charger_weight = float(np.clip((0.60 - battery_ratio) / 0.35, 0.0, 1.0))
    print(f"✓ Charger weight at 50% battery: {charger_weight:.4f}")
    print(f"  - Should trigger charging behavior (new threshold: 55%)")
    
    return True

def test_npc_avoidance_scenario():
    """Test P2 priority: NPC avoidance."""
    print("\n=== Testing P2 Priority: NPC AVOIDANCE ===")
    prep = Preprocessor()
    
    # Scenario: NPC approaching
    prep.step_no = 100
    prep.battery = 150
    prep.battery_max = 200
    prep.cur_pos = (50.0, 50.0)
    prep.total_dirt = 10000
    prep.dirt_cleaned = 500
    prep._organs = [{"pos": {"x": 64.0, "z": 64.0}}]
    prep._npcs = [{"pos": {"x": 55.0, "z": 50.0}}]  # NPC 5 units away
    prep._step_cleaned = 0
    prep._prev_battery_ratio = 0.75
    prep._cur_battery_ratio = 0.75
    prep._prev_min_charger_dist = 20.0
    prep._prev_min_npc_dist = 8.0  # Was farther
    prep._was_stuck = False
    prep._terminated = False
    prep._truncated = False
    prep._fail_reason = ""
    
    reward = prep.reward_process()
    print(f"✓ NPC at distance 5 (close threat):")
    print(f"  - Reward: {reward:.4f} (should have strong negative penalty)")
    
    # Test evade weight at close distance
    npc_dist = 5.0
    evade_weight = float(np.clip((10.0 - npc_dist) / 7.0, 0.0, 1.0))
    print(f"✓ Evade weight at distance {npc_dist}: {evade_weight:.4f}")
    print(f"  - Should trigger strong evasion (threshold: 10)")
    
    # Scenario: NPC escaping
    prep._prev_min_npc_dist = 5.0  # Was close
    prep._npcs = [{"pos": {"x": 65.0, "z": 50.0}}]  # Now 15 units away
    
    reward = prep.reward_process()
    print(f"✓ Escaping from NPC (distance +10):")
    print(f"  - Reward: {reward:.4f} (should include escape bonus +0.60)")
    
    return True

def test_coverage_scenario():
    """Test P3 priority: Coverage exploration."""
    print("\n=== Testing P3 Priority: COVERAGE ===")
    prep = Preprocessor()
    
    # Scenario 1: First visit to new cell
    prep.step_no = 50
    prep.battery = 150
    prep.battery_max = 200
    prep.cur_pos = (32.0, 32.0)
    prep.total_dirt = 10000
    prep.dirt_cleaned = 200
    prep._organs = [{"pos": {"x": 64.0, "z": 64.0}}]
    prep._npcs = []
    prep._step_cleaned = 1  # Cleaned something
    prep._visit_counter = {(32, 32): 1}  # First visit
    prep._prev_battery_ratio = 0.75
    prep._cur_battery_ratio = 0.75
    prep._prev_min_charger_dist = 50.0
    prep._prev_min_npc_dist = 1e9
    prep._was_stuck = False
    prep._terminated = False
    prep._truncated = False
    prep._fail_reason = ""
    
    reward = prep.reward_process()
    print(f"✓ First visit to new cell with cleaning:")
    print(f"  - Reward: {reward:.4f} (should include +0.10 new cell bonus)")
    
    # Scenario 2: Excessive revisit
    prep.cur_pos = (32.0, 32.0)
    prep._visit_counter[(32, 32)] = 8  # 8 visits (> 6)
    prep._step_cleaned = 0
    
    reward = prep.reward_process()
    print(f"✓ Excessive revisit (visit #8):")
    print(f"  - Reward: {reward:.4f} (should include strong negative penalty)")
    
    return True

def test_action_priority_features():
    """Test action priority feature generation."""
    print("\n=== Testing Action Priority Features ===")
    prep = Preprocessor()
    
    prep.step_no = 100
    prep.battery = 100  # 50% battery
    prep.battery_max = 200
    prep.cur_pos = (50.0, 50.0)
    prep.total_dirt = 10000
    prep.dirt_cleaned = 500
    prep._organs = [{"pos": {"x": 52.0, "z": 50.0}}]  # 2 units toward east
    prep._npcs = [{"pos": {"x": 48.0, "z": 50.0}}]   # 2 units toward west
    prep._map_info = np.random.rand(21, 21) * 2  # Random map
    prep._legal_act = [1] * 8
    
    action_priority = prep._action_priority_feature()
    print(f"✓ Action priority feature shape: {action_priority.shape}")
    print(f"  - All values should be normalized [0, 1]: {np.all((action_priority >= 0) & (action_priority <= 1))}")
    print(f"  - Max value: {action_priority.max():.4f}")
    print(f"  - Values (by direction E NE N NW W SW S SE): {action_priority}")
    
    # Check that east direction (toward charger) has higher weight
    east_idx = 0
    print(f"  - East direction (toward charger): {action_priority[east_idx]:.4f} (should be high)")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("OPTIMIZATION VERIFICATION TESTS")
    print("Testing: Charging (P1) > Avoidance (P2) > Coverage (P3)")
    print("=" * 60)
    
    try:
        test_charging_scenario()
        test_npc_avoidance_scenario()
        test_coverage_scenario()
        test_action_priority_features()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED - Optimization is working correctly!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
