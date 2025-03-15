class GoldenDUT:
    def __init__(self):
        # Initialize output register
        self.z_reg = "x"

    def load(self, signal_vector):
        # Extract input signals
        x = signal_vector["x"]
        y = signal_vector["y"]

        # Compute z = (x^y) & x
        self.z_reg = (x ^ y) & x

        return self.z_reg


def collect_expected_output(vectors_in):
    golden_dut = GoldenDUT()
    expected_outputs = []
    for vector in vectors_in:
        q = golden_dut.load(vector)
        if vector["check_en"]:
            expected_outputs.append(q)

    return expected_outputs


def SignalTxt_to_dictlist(txt: str):
    lines = txt.strip().split("\n")
    signals = []
    for line in lines:
        signal = {}
        line = line.strip().split(", ")
        for item in line:
            if "scenario" in item:
                item = item.split(": ")
                signal["scenario"] = item[1]
            else:
                item = item.split(" = ")
                key = item[0]
                value = item[1]
                if "x" not in value and "z" not in value:
                    signal[key] = int(value)
                else:
                    signal[key] = value
        signals.append(signal)
    return signals


with open("TBout.txt", "r") as f:
    txt = f.read()
vectors_in = SignalTxt_to_dictlist(txt)
tb_outputs = collect_expected_output(vectors_in)
print(tb_outputs)
