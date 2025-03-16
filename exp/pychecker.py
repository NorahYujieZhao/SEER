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
GENERATION_PROMPT = """
Your task is to implement a Python class named "GoldenDUT" representing the ideal (golden) model of the RTL design.

You will receive input stimuli formatted explicitly as JSON:

{{
  "scenario": "scenario_name",
  "input variable": [
    {{"variable_name": (string)variable_value}},
    {{"variable_name": (string)variable_value}},
    {{"variable_name": (string)variable_value}}
  ]
}}

And the python code should return the outputs aligned with the RTL module outputs as JSON:

{{
  "scenario": "scenario_name",
  "output variable": [
    {{"variable_name": (string)variable_value}},
    {{"variable_name": (string)variable_value}},
    {{"variable_name": (string)variable_value}}
  ]
}}
Each scenario contains multiple input variables. Your primary goal is to implement a Python class whose outputs precisely match the functionality and logic described by the provided RTL specification (`spec`) and module header.

## Implementation Requirements:

### 1. Initialization (__init__ method)

Implement the following method exactly:

def __init__(self):
    '''
    Initialize all internal state registers.
    Each internal register/state variable must align with the module header.
    Explicitly initialize these states according to the RTL specification.
    '''
    pass  # Initialize your state variables here

### 2. Signal Loading and State Updates (load method)

Implement the method exactly as shown:

def load(self, stimulus_dict: Dict[str, any]):
    '''
    stimulus_dict: a dictionary formatted as shown above.
    Parse each input variable and use it to perform RTL state updates.
    Please note input variable is in string format and you need to convert it to the corresponding type.
    Returns a dictionary of the outputs aligned with the RTL module outputs and updated states for verification.
    '''
    pass  # Implement your signal update logic here

### 3. Helper methods (optional)

You may implement additional helper methods if needed to organize your code clearly.

## Important RTL-to-Python Simulation Considerations:

To accurately replicate RTL behavior in Python, explicitly handle the following:

<instructions>
{instructions}
</instructions>
---

Additional information for your implementation:

<description>
{description}
</description>

<module_header>
{module_header}
</module_header>

---

Python implementation examples (GoldenDUT):

{examples_prompt}
"""


instructions = """
1. **Undefined (X) and High-Impedance (Z) Values:**

    - Python lacks native support for RTL-specific undefined ('X') and high-impedance ('Z') states.
    - **Recommended Solution:**
        Use BinaryValue for all signal conversions:
            from cocotb.binary import BinaryValue
            signal = BinaryValue(value_str)
        Preserve X/Z states through binstr property.
        Use BinaryValue.value or BinaryValue.integer or BinaryValue.signed_integer to get the value of the signal.

    Usage Example:
        def load(self, stimulus_dict: Dict[str, any]):
            try:
                x = BinaryValue(stimulus_dict['x'])
                y = BinaryValue(stimulus_dict['y'])
                z = BinaryValue(x.integer ^ y.integer & x.integer)
                return {'z': str(z.integer)}


These solutions provide clear guidance on accurately modeling RTL behavior in Python simulations, enhancing clarity and maintainability.
"""


EXAMPLE_OUTPUT_FORMAT = {
    "reasoning": "All reasoning steps and advices to generate the python code of the GoldenDUT class",
    "python_code": "The python code of the GoldenDUT class",
}

PythonHeader = """
import json
from typing import Dict, List, Union

    def load(self, stimulus: Dict[str, str]):
        try:

            if stimulus['x'] in ['x', 'X', 'z', 'Z']:
                return {'z': stimulus['x']}


            x = int(stimulus['x'])
            y = int(stimulus['y'])


            z = (x ^ y) & x

            return {'z': str(z)}
        except ValueError:

            return {'z': 'x'}

"""
CHECKER_TAIL = """
def check_output(stimulus_list):

    dut = GoldenDUT()
    tb_outputs = []


    for stimulus in stimulus_list:
        if 'check_en' in stimulus:
            tb_outputs.append(dut.load(stimulus))
        else:
            dut.load(stimulus)

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
"""

ONE_SHOT_EXAMPLES = r"""
Here are some examples of the GoldenDUT python code generation:
Example 1:

<example>
    <input_spec>
        Build a 64-bit arithmetic shift register, with synchronous load. The shifter can shift both left and right, and by 1 or 8 bit positions, selected by "amount." Assume the right shit is an arithmetic right shift.


    </input_spec>
    <module_header>
    module top_module(\n\tinput clk,\n\tinput load,\n\tinput ena,\n\tinput [1:0] amount,\n\tinput [63:0] data,\n\toutput reg [63:0] q);
    </module_header>
    <python_code>
    class GoldenDUT:
        def __init__(self):
            # Initialize internal state register
            self.q_reg = 0

        def load(self, stimulus_dict: Dict[str, Any]):
            stimulus_outputs = []
            for stimulus in stimulus_dict['input variable']:
                # Load the input signals and update the internal state
                load = stimulus['load']
                ena = stimulus['ena']
                data = stimulus['data']
                amount = stimulus['amount']
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
                stimulus_outputs.append(self.q_reg)
            output_dict = {
                "scenario": stimulus_dict['scenario'],
                "output variable": stimulus_outputs,

            }

            return output_dict
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
            instructions=instructions,
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
        py_output = (
            PythonHeader + "\n" + self.parse_output(response).python_code + CHECKER_TAIL
        )

        logger.info(f"Token count: {token_cnt}")
        logger.info(f"Response: {response.message.content}")

        with open(python_path, "w") as f:
            f.write(py_output)

        return True, py_output
