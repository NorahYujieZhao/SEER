import json
from typing import Dict, List

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from mage.gen_config import get_llm
from mage.log_utils import get_logger
from mage.prompts import ORDER_PROMPT
from mage.token_counter import TokenCounter, TokenCounterCached
from pydantic import BaseModel

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are an expert in RTL design. You can always write SystemVerilog code with no syntax errors and always reach correct functionality. You can always generate correct testbenches for your RTL designs.
"""

GENERATION_PROMPT = """
Your task is to generate Python method named "stimulus_gen" to produce a list of Dictionary-formatted stimulus sequences for testing a given DUT (Device Under Test). If necessary, write any additional methods that may be called within the load method to organize the code and handle specific sub-tasks. The information you have is:

1. The problem description that guides student to write the RTL code (DUT)
2. The header of the "DUT"
3. The instruction for writing the testbench
4. The list of testbench scenarios description

The object of the stimulus list format should strictly follow the dictionary structure below:
{{
  "scenario": "scenario_name1",
  "input variable": [
    {{"variable_name": variable_value}},
    {{"variable_name": variable_value}},
    {{"variable_name": variable_value}}
  ]
}}

Each input variable sequence should be customized based on the given specific scenario description, typically including:

a. Typical operations
b. Edge cases and corner cases
c. Boundary conditions
d. Error handling
e. Valid and invalid inputs
f. Timing verification requirements

Please follow these steps:

1. First, analyze the given test scenarios description.

2. Generate Python method named "stimulus_gen" follow the instruction:
<instruction>
{instruction}
</instruction>



Here is the information you have:
1. <description>
{description}
</description>

2. <module_header>
{module_header}
</module_header>



3. <testbench_scenarios>
{testbench_scenarios}
</testbench_scenarios>

Please generate the testbench following the format in the example below:
<example>
{example}
</example>
"""

Instructions_for_Python_Code = """
Instructions for the Python Code:
[Most importantly]
1. Carefully read and interpret each description in the list of testbench scenarios.
2. Write a Python method named stimulus_gen that returns a list of dictionary-formatted stimulus sequences.
3. Ensure the length of the generated list matches exactly the number of provided testbench scenarios.
4. The stimulus_gen method can call and rely on any additional helper methods or sub-methods as needed to generate the stimulus sequences clearly and efficiently.
5. Clearly define and document any helper methods that you use.
6. The output should be a list of dictionaries, each dictionary is a stimulus sequence following the format:{{
  "scenario": "scenario_name1",
  "input variable": [
    {{"variable_name": (str)variable_value}},
    {{"variable_name": (str)variable_value}},
    {{"variable_name": (str)variable_value}}
  ]
}}.
7. The variable names in the "input variable" should be the input variables in the DUT module header.
[Some hints]
1. Input Variable Conformance: Ensure all input variables in the stimulus sequence strictly conform to the DUT module header definition (variable names, bit widths, data types). Clearly indicate variable types (binary, integer, etc.) and bit widths according to the DUT module header.

2. Special Verilog Values Handling:

Include scenarios explicitly testing special Verilog values such as 'x' (unknown) and 'z' (high impedance).

Ensure your Python code can represent these special states accurately in the JSON output.

3. Code Clarity and Maintainability:

Clearly document each step and scenario in comments.

Structure the code logically (use functions for clarity).

Consider edge cases involving timing and synchronization relevant to the RTL module's operation.

4. Specific Recommendations for stimulus_gen Module:

Leverage Python loops (for, while) to efficiently generate repetitive or sequential test inputs.

Use parameterized functions or loops to cover various input ranges and boundary conditions systematically.

Ensure scalability by avoiding hard-coded scenarios; instead, use loop-driven generation for comprehensive coverage.

[Return Value Format]
The stimulus_gen function should either:
1. Return a JSON-formatted string directly, or
2. Return a list/dictionary that can be JSON serialized
The function's output will be automatically converted to a JSON string before writing to file.
"""

EXTRA_PROMPT_SEQ = """

"""
python_code_header = """
import json
"""
EXAMPLE_OUTPUT = {
    "reasoning": "Analyze the scenario description and think how to generate the stimulus sequence",
    "stimulus_gen_code": "python code to generate stimulus sequence",
}
ONE_SHOT_EXAMPLE = """
Here are some examples of SystemVerilog testbench code:
Example 1:
<example>
    <input_spec>
        Rule 110 is a one-dimensional cellular automaton with interesting properties (such as being Turing-complete). There is a one-dimensional array of cells (on or off). At each time step, the state of each cell changes. In Rule 110, the next state of each cell depends only on itself and its two neighbours, according to the following table:\n// Left | Center | Right | Center's next state\n// 1 | 1 | 1 | 0\n// 1 | 1 | 0 | 1\n// 1 | 0 | 1 | 1\n// 1 | 0 | 0 | 0\n// 0 | 1 | 1 | 1\n// 0 | 1 | 0 | 1\n// 0 | 0 | 1 | 1\n// 0 | 0 | 0 | 0 \n// In this circuit, create a 512-cell system (q[511:0]), and advance by one time step each clock cycle. The synchronous active high load input indicates the state of the system should be loaded with data[511:0]. Assume the boundaries (q[-1] and q[512]) are both zero (off).
    </input_spec>

    <stimulus_gen_code>
    import json
    import random

def stimulus_gen():
    scenario = "Example Stimulus"
    stimulus = []

    def repeat_steps(load_val, data_val, count):
        for _ in range(count):
            stimulus.append({"load": load_val, "data": data_val})

    # Step 1: data = 0 with bit 0 set -> data = 1, load = 1 for 3 clock cycles
    data_val = 1
    load_val = 1
    repeat_steps(load_val, data_val, 3)

    # Then load = 0 for 10 cycles
    load_val = 0
    repeat_steps(load_val, data_val, 10)

    # Step 2: data = 0 with bit 256 set -> data = (1 << 256), load = 1 for 3 cycles
    data_val = 1 << 256
    load_val = 1
    repeat_steps(load_val, data_val, 3)

    # Then load = 0 for 1000 cycles
    load_val = 0
    repeat_steps(load_val, data_val, 1000)

    # Step 3: data = 0x4df, load = 1 for 1 cycle
    data_val = 0x4df
    load_val = 1
    repeat_steps(load_val, data_val, 1)

    # Then load = 0 for 1000 cycles
    load_val = 0
    repeat_steps(load_val, data_val, 1000)

    # Step 4: data = random (simulating $random), load = 1 for 1 cycle
    data_val = random.getrandbits(32)  # or getrandbits(512) if needed
    load_val = 1
    repeat_steps(load_val, data_val, 1)

    # Then load = 0 for 1000 cycles
    load_val = 0
    repeat_steps(load_val, data_val, 1000)

    # Step 5: data = 0, load = 1 for 20 cycles
    data_val = 0
    load_val = 1
    repeat_steps(load_val, data_val, 20)

    # Next cycle -> data = 2
    data_val = 2
    repeat_steps(load_val, data_val, 1)

    # Next cycle -> data = 4
    data_val = 4
    repeat_steps(load_val, data_val, 1)

    # Next cycle -> data = 9, load = 0
    data_val = 9
    load_val = 0
    repeat_steps(load_val, data_val, 1)

    # Next cycle -> data = 12
    data_val = 12
    repeat_steps(load_val, data_val, 1)

    # Finally repeat 100 cycles
    repeat_steps(load_val, data_val, 100)

    # Build the final JSON structure
    return json.dumps({
        "scenario": scenario,
        "input variable": stimulus
    }, indent=4)


</stimulus_gen_code>
</example>
"""
tail = """
if __name__ == "__main__":
    result = stimulus_gen()
    # 将结果转换为 JSON 字符串
    if isinstance(result, list):
        result = json.dumps(result, indent=4)
    elif not isinstance(result, str):
        result = json.dumps(result, indent=4)

    with open("stimulus.json", "w") as f:
        f.write(result)
"""


class TBOutputFormat(BaseModel):
    reasoning: str
    stimulus_gen_code: str


class TB_Generator:
    def __init__(
        self,
        model: str,
        max_token: int,
        provider: str,
        cfg_path: str,
        stimulus_python_path: str,
    ):
        self.model = model
        self.llm = get_llm(
            model=model, max_token=max_token, provider=provider, cfg_path=cfg_path
        )
        self.token_counter = (
            TokenCounterCached(self.llm)
            if TokenCounterCached.is_cache_enabled(self.llm)
            else TokenCounter(self.llm)
        )

        self.stimulus_python_path = stimulus_python_path

    def parse_output(self, response: ChatResponse) -> TBOutputFormat:
        try:
            output_json_obj: Dict = json.loads(response.message.content, strict=False)
            ret = TBOutputFormat(
                reasoning=output_json_obj["reasoning"],
                stimulus_gen_code=output_json_obj["stimulus_gen_code"],
            )
            return ret
        except json.decoder.JSONDecodeError:
            return TBOutputFormat(reasoning="", stimulus_gen_code="")

    def generate(self, messages: List[ChatMessage]) -> ChatResponse:
        logger.info(f" input message: {messages}")
        resp, token_cnt = self.token_counter.count_chat(messages)
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"{resp.message.content}")
        return resp

    def run(
        self,
        input_spec: str,
        header: str,
        tb_scenario_description: str,
        circuit_type: str = "SEQ",
    ) -> str:
        msg = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=GENERATION_PROMPT.format(
                    description=input_spec,
                    module_header=header,
                    example=ONE_SHOT_EXAMPLE,
                    instruction=Instructions_for_Python_Code,
                    testbench_scenarios=tb_scenario_description,
                ),
                role=MessageRole.USER,
            ),
        ]
        if circuit_type == "SEQ":
            msg.append(ChatMessage(content=EXTRA_PROMPT_SEQ, role=MessageRole.USER))
        msg.append(
            ChatMessage(
                content=ORDER_PROMPT.format(
                    output_format="".join(json.dumps(EXAMPLE_OUTPUT, indent=4))
                ),
                role=MessageRole.USER,
            )
        )

        response = self.generate(msg)
        # 确保在生成代码前添加必要的导入
        stimulus_py_code = (
            "import json\n" + self.parse_output(response).stimulus_gen_code + tail
        )

        with open(self.stimulus_python_path, "w") as f:
            f.write(stimulus_py_code)

        logger.info(f"Get response from {self.model}: {response}")
        return stimulus_py_code
