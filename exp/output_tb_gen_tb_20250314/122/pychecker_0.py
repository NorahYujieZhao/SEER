import json
from typing import Any, Dict


class GoldenDUT:
    # State definitions
    OFF = 0
    ON = 1

    def __init__(self):
        # Initialize state to OFF (reset state)
        self.current_state = self.OFF
        self.out = 0

    def load(self, stimulus_dict: Dict[str, Any]) -> Dict[str, Any]:
        stimulus_outputs = []

        for stimulus in stimulus_dict["input variable"]:
            # Extract inputs
            areset = int(stimulus.get("areset", "0"))
            j = int(stimulus.get("j", "0"))
            k = int(stimulus.get("k", "0"))

            # Handle asynchronous reset
            if areset:
                self.current_state = self.OFF
            else:
                # State transitions
                if self.current_state == self.OFF:
                    if j:
                        self.current_state = self.ON
                else:  # ON state
                    if k:
                        self.current_state = self.OFF

            # Output logic (Moore machine)
            self.out = 1 if self.current_state == self.ON else 0

            # Append output to results
            stimulus_outputs.append({"out": str(self.out)})

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
