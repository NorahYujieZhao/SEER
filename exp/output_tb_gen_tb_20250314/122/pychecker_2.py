import json
from typing import dict


class GoldenDUT:
    def __init__(self):
        # Initialize state to OFF (False)
        self.current_state = False

    def load(self, stimulus_dict: dict[str, any]):
        stimulus_outputs = []

        for stimulus in stimulus_dict["input variable"]:
            # Extract input signals
            areset = int(stimulus["areset"])
            j = int(stimulus["j"])
            k = int(stimulus["k"])

            # Handle asynchronous reset
            if areset:
                self.current_state = False
            else:
                # State transition logic
                if not self.current_state:  # OFF state
                    if j:
                        self.current_state = True
                else:  # ON state
                    if k:
                        self.current_state = False

            # Moore output logic - depends only on current state
            out = 1 if self.current_state else 0
            stimulus_outputs.append({"out": str(out)})

        return {
            "scenario": stimulus_dict["scenario"],
            "output variable": stimulus_outputs,
        }


def check_output(stimulus_list):

    dut = GoldenDUT()
    tb_outputs = []

    for stimulus in stimulus_list:
        tb_outputs.append(dut.load(stimulus))

    return tb_outputs


if __name__ == "__main__":

    with open("stimulus.json", "r") as f:
        stimulus_data = json.load(f)

    if isinstance(stimulus_data, dict):
        stimulus_list = stimulus_data.get("input variable", [])
    else:
        stimulus_list = stimulus_data

    outputs = check_output(stimulus_list)

    print(json.dumps(outputs, indent=2))
