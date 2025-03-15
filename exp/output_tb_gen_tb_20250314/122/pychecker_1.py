import json
from typing import Any, Dict


class GoldenDUT:
    def __init__(self):
        # Define states as constants
        self.STATE_OFF = 0
        self.STATE_ON = 1
        # Initialize state to OFF as per reset behavior
        self.current_state = self.STATE_OFF

    def load(self, stimulus_dict: Dict[str, Any]) -> Dict[str, Any]:
        stimulus_outputs = []

        for stimulus in stimulus_dict["input variable"]:
            # Extract inputs
            j = int(stimulus.get("j", "0"))
            k = int(stimulus.get("k", "0"))
            areset = int(stimulus.get("areset", "0"))

            # Handle asynchronous reset
            if areset:
                self.current_state = self.STATE_OFF
            else:
                # State transition logic
                if self.current_state == self.STATE_OFF:
                    if j:
                        self.current_state = self.STATE_ON
                else:  # STATE_ON
                    if k:
                        self.current_state = self.STATE_OFF

            # Output logic (Moore machine)
            out = 1 if self.current_state == self.STATE_ON else 0
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
