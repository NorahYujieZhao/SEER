import json


def generate_clock_cycles(j_val, k_val, cycles):
    return [{"j": j_val, "k": k_val, "areset": "0"} for _ in range(cycles)]


def stimulus_gen():
    scenarios = []

    # Scenario 1: Initial Reset Verification
    scenario1 = {
        "scenario": "Initial Reset Verification",
        "input variable": [
            {"j": "1", "k": "1", "areset": "1"},
            {"j": "0", "k": "0", "areset": "1"},
            {"j": "1", "k": "0", "areset": "1"},
            {"j": "0", "k": "1", "areset": "1"},
        ],
    }
    scenarios.append(scenario1)

    # Scenario 2: OFF to ON Transition
    scenario2 = {
        "scenario": "OFF to ON Transition",
        "input variable": [
            {"j": "0", "k": "0", "areset": "1"},  # Reset to OFF
            {"j": "0", "k": "0", "areset": "0"},  # Stay in OFF
            {"j": "1", "k": "0", "areset": "0"},  # Transition to ON
            {"j": "1", "k": "1", "areset": "0"},  # Should be in ON
            {"j": "1", "k": "0", "areset": "0"},  # Stay in ON
        ],
    }
    scenarios.append(scenario2)

    # Scenario 3: ON to OFF Transition
    scenario3 = {
        "scenario": "ON to OFF Transition",
        "input variable": [
            {"j": "1", "k": "0", "areset": "0"},  # Get to ON
            {"j": "0", "k": "1", "areset": "0"},  # Transition to OFF
            {"j": "1", "k": "1", "areset": "0"},  # Stay in OFF
            {"j": "0", "k": "1", "areset": "0"},  # Stay in OFF
        ],
    }
    scenarios.append(scenario3)

    # Scenario 4: State Retention OFF
    scenario4 = {
        "scenario": "State Retention OFF",
        "input variable": [
            {"j": "0", "k": "0", "areset": "1"},  # Reset to OFF
            {"j": "0", "k": "0", "areset": "0"},  # Stay in OFF
            {"j": "0", "k": "1", "areset": "0"},  # Stay in OFF
            {"j": "0", "k": "0", "areset": "0"},  # Stay in OFF
            {"j": "0", "k": "1", "areset": "0"},  # Stay in OFF
        ],
    }
    scenarios.append(scenario4)

    # Scenario 5: State Retention ON
    scenario5 = {
        "scenario": "State Retention ON",
        "input variable": [
            {"j": "1", "k": "0", "areset": "0"},  # Get to ON
            {"j": "0", "k": "0", "areset": "0"},  # Stay in ON
            {"j": "1", "k": "0", "areset": "0"},  # Stay in ON
            {"j": "0", "k": "0", "areset": "0"},  # Stay in ON
            {"j": "1", "k": "0", "areset": "0"},  # Stay in ON
        ],
    }
    scenarios.append(scenario5)

    # Scenario 6: Reset During Operation
    scenario6 = {
        "scenario": "Reset During Operation",
        "input variable": [
            {"j": "1", "k": "0", "areset": "0"},  # Get to ON
            {"j": "0", "k": "0", "areset": "0"},  # Stay in ON
            {"j": "1", "k": "0", "areset": "1"},  # Reset while in ON
            {"j": "1", "k": "0", "areset": "1"},  # Stay reset
            {"j": "0", "k": "0", "areset": "0"},  # Normal operation
        ],
    }
    scenarios.append(scenario6)

    # Scenario 7: Input Toggle Stress
    scenario7 = {
        "scenario": "Input Toggle Stress",
        "input variable": [
            {"j": "0", "k": "0", "areset": "0"},
            {"j": "1", "k": "0", "areset": "0"},
            {"j": "0", "k": "1", "areset": "0"},
            {"j": "1", "k": "1", "areset": "0"},
            {"j": "1", "k": "0", "areset": "0"},
            {"j": "0", "k": "1", "areset": "0"},
            {"j": "1", "k": "1", "areset": "0"},
            {"j": "0", "k": "0", "areset": "0"},
        ],
    }
    scenarios.append(scenario7)

    # Scenario 8: Invalid Input Handling
    scenario8 = {
        "scenario": "Invalid Input Handling",
        "input variable": [
            {"j": "x", "k": "0", "areset": "0"},
            {"j": "0", "k": "x", "areset": "0"},
            {"j": "z", "k": "0", "areset": "0"},
            {"j": "0", "k": "z", "areset": "0"},
            {"j": "x", "k": "x", "areset": "0"},
        ],
    }
    scenarios.append(scenario8)

    return scenarios


if __name__ == "__main__":
    result = stimulus_gen()
    # 将结果转换为 JSON 字符串
    if isinstance(result, list):
        result = json.dumps(result, indent=4)
    elif not isinstance(result, str):
        result = json.dumps(result, indent=4)

    with open("stimulus.json", "w") as f:
        f.write(result)
