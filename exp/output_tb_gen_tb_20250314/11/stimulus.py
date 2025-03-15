import json


def stimulus_gen():
    scenarios = []

    # Helper function to create a stimulus dictionary
    def create_stimulus(scenario_name, input_list):
        return {"scenario": scenario_name, "input variable": input_list}

    # Scenario 1: All Input Combinations
    all_combinations = [
        {"x": "0", "y": "0"},
        {"x": "0", "y": "1"},
        {"x": "1", "y": "0"},
        {"x": "1", "y": "1"},
    ]
    scenarios.append(create_stimulus("All Input Combinations", all_combinations))

    # Scenario 2: Input Transitions
    transitions = [
        {"x": "0", "y": "0"},
        {"x": "1", "y": "0"},
        {"x": "1", "y": "1"},
        {"x": "0", "y": "1"},
        {"x": "0", "y": "0"},
    ]
    scenarios.append(create_stimulus("Input Transitions", transitions))

    # Scenario 3: Invalid Input States
    invalid_states = [
        {"x": "x", "y": "0"},
        {"x": "0", "y": "x"},
        {"x": "z", "y": "1"},
        {"x": "1", "y": "z"},
        {"x": "x", "y": "x"},
    ]
    scenarios.append(create_stimulus("Invalid Input States", invalid_states))

    # Scenario 4: Setup Time Verification
    setup_time = [
        {"x": "0", "y": "0"},
        {"x": "1", "y": "0"},
        {"x": "0", "y": "1"},
        {"x": "1", "y": "1"},
    ]
    scenarios.append(create_stimulus("Setup Time Verification", setup_time))

    # Scenario 5: Hold Time Verification
    hold_time = [
        {"x": "1", "y": "1"},
        {"x": "1", "y": "1"},
        {"x": "0", "y": "0"},
        {"x": "0", "y": "0"},
    ]
    scenarios.append(create_stimulus("Hold Time Verification", hold_time))

    # Scenario 6: Simultaneous Input Changes
    simultaneous = [
        {"x": "0", "y": "0"},
        {"x": "1", "y": "1"},
        {"x": "0", "y": "0"},
        {"x": "1", "y": "1"},
    ]
    scenarios.append(create_stimulus("Simultaneous Input Changes", simultaneous))

    # Scenario 7: Output Stability
    stability = [
        {"x": "0", "y": "0"},
        {"x": "0", "y": "0"},
        {"x": "1", "y": "1"},
        {"x": "1", "y": "1"},
    ]
    scenarios.append(create_stimulus("Output Stability", stability))

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
