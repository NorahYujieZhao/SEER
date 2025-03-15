import json


class GoldenDUT:
    def __init__(self):
        # No state registers needed for combinational logic
        pass

    def _to_bool(self, val):
        # Handle undefined/high-impedance cases
        if val in ["X", "x", "Z", "z"]:
            return None
        return bool(int(val))

    def load(self, stimulus_dict: dict[str, any]):
        output_values = []

        for stimulus in stimulus_dict["input variable"]:
            # Extract and convert inputs
            x = self._to_bool(stimulus["x"])
            y = self._to_bool(stimulus["y"])

            # Handle undefined cases
            if x is None or y is None:
                z = "X"
            else:
                # Implement z = (x^y) & x
                z = str(int((x ^ y) & x))

            output_values.append({"z": z})

        return {"scenario": stimulus_dict["scenario"], "output variable": output_values}


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
