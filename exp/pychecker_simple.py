import json
from typing import List, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from mage.log_utils import get_logger
from mage.token_counter import TokenCounter
from mage.utils import extract_code

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert in RTL design and Python programming. You can always write correct Python code to verify RTL functionality."""

GENERATION_PROMPT = """Your current task is: write a python class "GoldenDUT". This python class can represent the golden DUT (the ideal one). In your "GoldenDUT", you should do the following things:

- a. write a method "def __init__(self)". Set the inner states/values of the golden DUT. These values have suffix "_reg". The initial value of these inner values is "x", but later will be digits. The "__init__" method has no input parameters except "self".
- b. write a method "def load(self, signal_vector)". This method is to load the important input signals and the inner values of "GoldenDUT" shall change according to the input signals. There is no clock signal in the input signal vector, every time the "load" method is called, it means a new clock cycle. The initial values "x" should be changed according to the input signals. This method will return the expected output values, e.g., the changed inner values.
- c. write other methods you need, they can be called by "load".
- e. Write a method "def out_expected(self, signal_vector)". This method should return a dictionary of the output signals and their expected values.

You can use binary (like 0b1101), hexadecimal (like 0x1a) or normal number format in python. But the signal vector input to GoldenDUT is always in decimal format.

<problem_description>
{problem_description}
</problem_description>

<checker_spec>
{checker_spec}
</checker_spec>

Please only generate the Python code for the GoldenDUT class, no other words.
"""

CHECKER_TAIL = """
def collect_expected_output(vectors_in):
    golden_dut = GoldenDUT()
    expected_outputs = []
    for vector in vectors_in:
        q=golden_dut.load(vector)
        if vector["check_en"]:
            expected_outputs.append(q)

    return expected_outputs

def SignalTxt_to_dictlist(txt:str):
    lines = txt.strip().split("\\n")
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
"""

EXAMPLE_OUTPUT = r"""
Here are some examples of the GoldenDUT python code generation:
Example 1:
<example>
"input_spec": "Build a 64-bit arithmetic shift register, with synchronous load. The shifter can shift both left and right, and by 1 or 8 bit positions, selected by "amount." Assume the right shit is an arithmetic right shift. "
"response":  “
```python
class GoldenDUT:
    def __init__(self):
        # Initialize internal state register
        self.q_reg = 0

    def load(self, signal_vector):
        # Load the input signals and update the internal state
        load = signal_vector['load']
        ena = signal_vector['ena']
        data = signal_vector['data']
        amount = signal_vector['amount']
        current_q = self.q_reg

        if load:
            # Load data into the shift register
            self.q_reg = data & 0xFFFFFFFFFFFFFFFF  # Ensure 64-bit width
        elif ena:
            if amount == 0b00:
                # Shift left by 1
                self.q_reg = (current_q << 1) & 0xFFFFFFFFFFFFFFFF
            elif amount == 0b01:
                # Shift left by 8
                self.q_reg = (current_q << 8) & 0xFFFFFFFFFFFFFFFF
            elif amount == 0b10:
                # Arithmetic shift right by 1
                self.q_reg = (current_q >> 1) | ((current_q & 0x8000000000000000) >> 1)
            elif amount == 0b11:
                # Arithmetic shift right by 8
                self.q_reg = (current_q >> 8) | ((current_q & 0x8000000000000000) >> 8)

        return self.q_reg


``` “

<\example>
"""


class PyChecker:
    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter
        self.history: List[ChatMessage] = []
        self.max_trials = 3

    def reset(self):
        self.history = []

    def get_order_prompt_messages(self) -> List[ChatMessage]:
        return [
            ChatMessage(output_format="".join(json.dumps(EXAMPLE_OUTPUT, indent=4))),
        ]

    def chat(
        self, problem_description: str, checker_spec: str, python_rules: str = ""
    ) -> Tuple[bool, str]:
        """Generate Python checker code for the given problem

        Args:
            problem_description: Problem description text
            checker_spec: Checker specification text
            python_rules: Optional Python rules/guidelines

        Returns:
            Tuple[bool, str]: (success, generated code)
        """
        prompt = GENERATION_PROMPT.format(
            problem_description=problem_description,
            checker_spec=checker_spec,
            python_rules=python_rules,
        )

        messages = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(content=prompt, role=MessageRole.USER),
        ]

        response, token_cnt = self.token_counter.count_chat(messages)
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"Response: {response.message.content}")

        try:
            # Extract Python code from response
            code = extract_code(response.message.content, "python")[-1]
            # Add checker tail code
            full_code = code + CHECKER_TAIL
            return True, full_code
        except Exception as e:
            logger.error(f"Failed to generate checker code: {e}")
            return False, str(e)
