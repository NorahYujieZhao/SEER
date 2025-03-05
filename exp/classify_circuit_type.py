import json
from typing import Dict

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from mage.gen_config import get_llm
from mage.log_utils import get_logger
from mage.prompts import ORDER_PROMPT
from mage.token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
You are an expert in RTL design. You can always write SystemVerilog code with no syntax errors and always reach correct functionality.
"""

GENERATION_PROMPT = r"""
 I will provide you with SystemVerilog specification. Your job is to determine whether this code implements combinational logic (CMB) or sequential logic (SEQ). Then, please explain your reasoning in detail, pointing out the specific signals or language constructs that lead you to your conclusion.


Instructions:

Carefully read and analyze the provided code.
Determine whether it describes a purely combinational module (CMB) or a sequential module (SEQ).

{example_prompt}
<input_spec>
{input_spec}
</input_spec>
"""

EXAMPLE_OUTPUT_FORMAT = {
    "reasoning": "All reasoning to analyze the circuit type",
    "classification": "CMB or SEQ (do not use any other words)",
}

CLASSIFICATION_1_SHOT_EXAMPLES = r"""
Here are some examples:
Example 1:
<example> "input_spec": " // Module: simple_counter // Interface: // input logic clk, rst_n // output logic [3:0] count // // Specification: // 1. On every rising edge of clk, if rst_n is low, count resets to 0. // 2. Otherwise, count increments by 1. ", "reasoning": r" The design explicitly uses a clock (clk) and a reset signal (rst_n) to control state transitions. Since the counter updates its value on a clock edge, it clearly implements sequential logic. ", "classification": "SEQ" </example>

Example 2:
<example> "input_spec": " // Module: adder // Interface: // input logic [7:0] a, b // output logic [7:0] sum // // Specification: // 1. The module computes the sum of inputs a and b combinationally. // 2. There is no clock or state element involved. ", "reasoning": r" The absence of any clock or state-related signals and the direct assignment of the output based on inputs indicate that the module is purely combinational. ", "classification": "CMB" </example>
"""

EXTRA_ORDER_PROMPT = r"""
VERY IMPORTANT: Please only include "classification" in your response.
Do not include any other information in your response, like 'json', 'example', 'Let me analyze','input_spec' or '<output_format>'.
Key instruction: Direct output, no extra comments.
As a reminder, please directly provide the content without adding any extra comments or explanations.
"""


class circuit_type_classifier:
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
        # self.token_counter.token_cnts['circuit_type_classifier'] = []
        # self.history = []
        # self.max_trials = 15

    def run(self, input_spec: str) -> Dict:
        # self.token_counter.reset()
        if isinstance(self.token_counter, TokenCounterCached):
            self.token_counter.set_enable_cache(True)
        print(f"Setting token counter tag to {self.__class__.__name__}")
        self.token_counter.set_cur_tag(self.__class__.__name__)
        msg = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=GENERATION_PROMPT.format(
                    input_spec=input_spec, example_prompt=CLASSIFICATION_1_SHOT_EXAMPLES
                ),
                role=MessageRole.USER,
            ),
            ChatMessage(
                content=ORDER_PROMPT.format(
                    output_format="".join(json.dumps(EXAMPLE_OUTPUT_FORMAT, indent=4))
                ),
                role=MessageRole.USER,
            ),
        ]
        print(f"Generating response ")
        response, token_cnt = self.token_counter.count_chat(msg)
        print(f"Response: {response.message.content}")

        logger.info(f"Token count: {token_cnt}")
        logger.info(f"{response.message.content}")
        self.token_counter.log_token_stats()

        # response = self.generate(msg)
        logger.info(f"Get response from {self.model}: {response.message.content}")
        try:
            # output_json_obj: Dict = json.loads(response.message.content, strict=False)

            # use this for Deepseek r1 and claude-3-5-sonnet
            # if self.model == "claude-3-5-sonnet-20241022":
            #     output_json_obj: Dict = json.loads("".join(response.choices[0].message.content.split("\n")[1:]), strict=False)
            # else:
            #     output_json_obj: Dict = json.loads(response.choices[0].message.content, strict=False)
            output_json_obj: Dict = json.loads(response.message.content, strict=False)

            classification = output_json_obj["classification"]
            logger.info(f"Succeed to parse response, Classification: {classification}")
        except json.decoder.JSONDecodeError as e:
            print(f"Json parse error: {e}")
            logger.info(f"Json parse error: {e}")
            print(response)
            return None

        return output_json_obj
