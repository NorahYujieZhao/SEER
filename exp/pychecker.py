import json
from typing import Dict

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from mage.gen_config import get_llm
from mage.log_utils import get_logger
from mage.prompts import ORDER_PROMPT
from mage.token_counter import TokenCounter, TokenCounterCached
from pydantic import BaseModel

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert in RTL design and Python programming. You can always write correct Python code to verify RTL functionality."""

GENERATION_PROMPT = """Your job is to implement a Python class named "GoldenDUT" that represents the ideal (golden) model of the RTL design.
This class must realize the logic and functionality specified in the provided RTL code spec and module header.
You are expected to translate RTL constructs into Python methods and attributes that simulate both combinational and sequential logic.
Ensure that your implementation is structured to facilitate testing and verification against expected simulation outputs.

In your GoldenDUT class, please do the following:

1. Initialization

Implement the method def __init__(self).
In this method, define and initialize all internal state variables (registers). Each internal state should have a suffix _reg.
Set the initial value of these internal states to "x" (note that later these will be updated to actual digit values).

2. Signal Loading and State Update

Implement the method def load(self, signal_vector).
This method takes one parameter, signal_vector, which is always provided in decimal format.
Each time load is called, it represents a new clock cycle. Use the values from the signal_vector to update the internal state registers accordingly.
The method should return the expected output values (i.e., the updated internal states).

3. Additional Helper Methods

If necessary, write any additional methods that may be called within the load method to organize the code and handle specific sub-tasks.


You have the following extra details to help guide your implementation:

<description>
{description}
</description>

<module_header>
{module_header}
</module_header>

Notes:

You may use binary (e.g., 0b1101), hexadecimal (e.g., 0x1a), or normal number formats in your Python code, but remember that the input signal vector to GoldenDUT will always be in decimal format.
Please incorporate the RTL code specification and the module header information as necessary to ensure the logic in GoldenDUT correctly represents the intended behavior.

Here are some examples of python code of GoldenDUT:
{examples_prompt}
"""

EXAMPLE_OUTPUT_FORMAT = {
    "reasoning": "All reasoning steps and advices to generate the python code of the GoldenDUT class",
    "python_code": "The python code of the GoldenDUT class",
}
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

        signals.append(signal)
    return signals

with open("TBout.txt", "r") as f:
    txt = f.read()
vectors_in = SignalTxt_to_dictlist(txt)
tb_outputs = collect_expected_output(vectors_in)
print(tb_outputs)
"""

ONE_SHOT_EXAMPLES = r"""
Here are some examples of the GoldenDUT python code generation:
Example 1:

<example>
    <input_spec>
        Build a 64-bit arithmetic shift register, with synchronous load. The shifter can shift both left and right, and by 1 or 8 bit positions, selected by "amount." Assume the right shit is an arithmetic right shift.


    </input_spec>
    <python_code>
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
    </python_code>
</example>

"""


class PyOutputFormat(BaseModel):
    reasoning: str
    python_code: str


class PyChecker:
    def __init__(self, model: str, max_token: int, provider: str, cfg_path: str):
        self.model = model
        self.llm = get_llm(
            model=model, max_token=max_token, provider=provider, cfg_path=cfg_path
        )
        self.token_counter = (
            TokenCounterCached(self.llm)
            if TokenCounterCached.is_cache_enabled(self.llm)
            else TokenCounter(self.llm)
        )

    def reset(self):
        self.history = []

    def parse_output(self, response: ChatResponse) -> PyOutputFormat:
        try:
            output_json_obj: Dict = json.loads(response.message.content, strict=False)
            ret = PyOutputFormat(
                reasoning=output_json_obj["reasoning"],
                python_code=output_json_obj["python_code"],
            )
        except json.decoder.JSONDecodeError as e:
            ret = PyOutputFormat(
                reasoning=f"Json Decode Error: {str(e)}", python_code=""
            )
        return ret

    def run(
        self,
        problem_description: str,
        header: str,
        python_path: str,
        circuit_type: str = "SEQ",
    ) -> str:
        """Generate Python checker code for the given problem

        Args:
            problem_description: Problem description text
            checker_spec: Checker specification text
            python_rules: Optional Python rules/guidelines

        Returns:
            Tuple[bool, str]: (success, generated code)
        """
        prompt = GENERATION_PROMPT.format(
            description=problem_description,
            module_header=header,
            examples_prompt=ONE_SHOT_EXAMPLES,
        )

        messages = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(content=prompt, role=MessageRole.USER),
            ChatMessage(
                content=ORDER_PROMPT.format(
                    output_format="".join(json.dumps(EXAMPLE_OUTPUT_FORMAT, indent=4))
                ),
                role=MessageRole.USER,
            ),
        ]

        response, token_cnt = self.token_counter.count_chat(messages)
        py_output = self.parse_output(response).python_code + CHECKER_TAIL

        logger.info(f"Token count: {token_cnt}")
        logger.info(f"Response: {response.message.content}")

        with open(python_path, "w") as f:
            f.write(py_output)

        return True, py_output
