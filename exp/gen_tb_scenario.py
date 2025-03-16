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
You are an expert in RTL design. You can always write SystemVerilog code with no syntax errors and always reach correct functionality. You can always generate correct testbenches for your RTL designs. Based on this analysis, you must generate detailed testbench scenarios in structured JSON format. Clearly state your reasoning for each scenario.
"""

GENERATION_PROMPT = """
Your task is to write verilog testbench scenarios discription for an verilog RTL module code (we call it as "DUT", device under test) according to the problem description.

The information you have is:

1. The problem description that guides student to write the RTL code (DUT)
2. The header of the "DUT"
3. The instruction for writing the testbench

Analyze its behavior thoroughly and create representative testbench scenarios. Clearly state your reasoning for each scenario. Structure the scenarios as JSON descriptions, ensuring each scenario covers critical aspects, including:
a. Typical operations
b. Edge cases and corner cases
c. Boundary conditions
d. Error handling
e. Valid and invalid inputs
f. Timing verification requirements




Here is the information you have:
1. <description>
{description}
</description>

2. <module_header>
{module_header}
</module_header>

3. <instruction>
{instruction}
</instruction>

Please generate the testbench following the format in the example below:
<example>
{example}
</example>
"""

EXTRA_PROMPT = """

The variables are already declared. The clock signal is already prepared. This output will be used to check the correctness of the DUT's output later.
"""

EXTRA_PROMPT_SEQ = """

"""

EXAMPLE_OUTPUT = {
    "reasoning": "Analyze the technical specification and infer the test scenarios",
    "testbench_scenarios": [
        {
            "scenario": "The testbench scenario name",
            "description": "The description of the testbench scenario",
        }
    ],
}
FEW_SHOT_EXAMPLE = """
Here are some examples of SystemVerilog testbench scenarios descriptions:
Example 1:
<example>
    <input_spec>
        Implement an 8-bit synchronous up-counter with:
        - Clock (clk) rising-edge triggered
        - Active-high asynchronous reset (RST)
        - Active-high enable (EN)
        - Output value wraps from 255 to 0 on overflow
    </input_spec>

    <testbench_scenarios>
   {
"scenario": "Basic Counting",
"description": "Enable counting by asserting EN high for several clock cycles and verify counter increments each cycle."
},
{
"scenario": "Asynchronous Reset",
"description": "Assert RST asynchronously between clock edges and verify immediate counter reset to zero."
},
{
"scenario": "Simultaneous Enable and Reset",
"description": "Assert RST and EN simultaneously, ensuring reset behavior overrides counting operation immediately."
},
{
"scenario": "Counter Rollover",
"description": "Set counter value to 255, enable counting, and verify it rolls over to 0 on next increment."
},
{
"scenario": "Power-on Initialization",
"description": "Verify counter initializes to 0x00 after power-on without explicit reset."
}
{
"scenario": "Timing Requirements Verification",
"description": "Check setup and hold times of the EN signal relative to the clock and ensure counter increments correctly."
},
{
"scenario": "Boundary Conditions",
"description": "Test minimum (0) and maximum (255) counter values explicitly, verifying accurate transition and handling of edge boundary conditions."
}


    </testbench_scenarios>
</example>

Example 2:  
<example>  
    <input_spec>  
        Implement a synchronous FIFO buffer with:  
        - 8-bit data width, 16 entries depth  
        - wr_en/rd_en controls with full/empty flags  
        - Synchronous reset (active-high)  
        All signals sampled on rising clock edge.  
    </input_spec>  

    <testbench_scenarios>  
    {
"scenario": "Consecutive Write-Read",
"description": "Write 16 values (0x00-0x0F) then read them back, verifying flags and data integrity."
}
{  
"scenario": "Alternating Operations",  
"description": "Interleave write and read operations every clock cycle, checking pointer updates and flag stability."  
},  
{  
"scenario": "Full Condition Handling",  
"description": "Attempt to write to full FIFO (0xAA pattern) and verify data preservation and full flag persistence."  
},  
{
"scenario": "Reset During Operation",
"description": "Trigger reset when wr_ptr=4'hC, confirm empty state and pointer reset to 4'h0."
}
{  
"scenario": "Concurrent Read/Write",  
"description": "Simultaneously assert wr_en and rd_en for 10 cycles with different data patterns, checking coherency and flag behavior."  
},  
{  
"scenario": "Pointer Wrap-around",  
"description": "Fill/empty FIFO 3 consecutive times to verify circular buffer addressing and pointer rollover."  
},
{
    "scenario": "Partial Fill/Drain",
    "description": "Write 8 entries, then read 8 entries, checking almost_full/almost_empty flags at 75% capacity."
}
    </testbench_scenarios>  
</example>

Example 3:  
<example>  
    <input_spec>  
        Design a PWM controller with:  
        - 10-bit duty cycle resolution  
        - 100MHz clock input  
        - Synchronous duty cycle update register  
        - Asynchronous emergency stop (active-low)  
        Output should update immediately at period boundaries.  
    </input_spec>  

    <testbench_scenarios>  
   {
"scenario": "Duty Cycle Ramp",
"description": "Increment duty cycle from 0x000 to 0x3FF in 0x100 steps, verifying pulse width with 1% tolerance."
}
{
"scenario": "Asynchronous Abort",
"description": "Assert emergency_stop_n during active pulse, verify output shutdown within 10ns (independent of clock)."
}
{  
"scenario": "Boundary Values",  
"description": "Test 0x000 (0%), 0x3FF (100%), and 0x200 (50%) duty cycles for exact period matching and edge alignment."  
},  
{  
"scenario": "Concurrent Update",  
"description": "Change duty cycle register mid-period while maintaining previous output, verify seamless transition at next period boundary."  
},  
{
"scenario": "Clock Frequency Scaling",
"description": "Modify input clock (50-200MHz) and validate PWM period inversely scales with frequency (±1 cycle error)."
}
{  
"scenario": "Invalid Input Handling",  
"description": "Apply out-of-range duty values (>0x3FF) and verify register clamping behavior and output stability."  
}  
    </testbench_scenarios>  
</example> 
"""


class ScenarioItem(BaseModel):
    scenario: str
    description: str


class TBOutputFormat(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    reasoning: str
    testbench_scenarios: List[ScenarioItem]


class TB_Generator_Scenario:
    def __init__(
        self,
        model: str,
        max_token: int,
        provider: str,
        cfg_path: str,
        tb_scenarios_path: str,
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
        self.tb_scenarios_path = tb_scenarios_path

    def parse_output(self, response: ChatResponse) -> TBOutputFormat:
        try:
            output_json_obj: Dict = json.loads(response.message.content, strict=False)
            scenarios = output_json_obj["testbench_scenarios"]
            if isinstance(scenarios, dict):
                scenarios = [scenarios]
            elif not isinstance(scenarios, list):
                scenarios = []

            ret = TBOutputFormat(
                reasoning=output_json_obj["reasoning"], testbench_scenarios=scenarios
            )
        except json.decoder.JSONDecodeError:
            ret = TBOutputFormat(reasoning="", testbench_scenarios=[])
        return ret

    def generate(self, messages: List[ChatMessage]) -> ChatResponse:
        logger.info(f" input message: {messages}")
        resp, token_cnt = self.token_counter.count_chat(messages)
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"{resp.message.content}")
        return resp

    def run(self, input_spec: str, header: str, circuit_type: str = "SEQ") -> str:
        msg = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=GENERATION_PROMPT.format(
                    description=input_spec,
                    module_header=header,
                    example=FEW_SHOT_EXAMPLE,
                    instruction=EXTRA_PROMPT,
                ),
                role=MessageRole.USER,
            ),
            ChatMessage(
                content=EXTRA_PROMPT,
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
        tb_scenarios = self.parse_output(response).testbench_scenarios

        # 将列表转换为格式化的字符串
        scenarios_str = ""
        for scenario in tb_scenarios:
            scenarios_str += f"scenario: {scenario.scenario}\n"
            scenarios_str += f"description: {scenario.description}\n\n"

        # 写入文件
        with open(self.tb_scenarios_path, "w") as f:
            f.write(scenarios_str)

        logger.info(f"Get response from {self.model}: {response}")

        return scenarios_str  # 返回字符串而不是列表
