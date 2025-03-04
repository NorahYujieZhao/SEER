import json
from typing import Dict, List

from llama_index.core.llms import LLM
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from mage.log_utils import get_logger, set_log_dir, switch_log_to_file, switch_log_to_stdout
from mage.gen_config import get_llm
from mage.token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
You are an expert in RTL design. You can always write SystemVerilog code with no syntax errors and always reach correct functionality.
You can always provide a precise and unambiguous RTL design specification.
"""

GENERATION_PROMPT = r"""
Analyze the provided SystemVerilog specification which is ambiguous. 
Based on the reasons for these ambiguities, modify the specification to eliminate any unclear aspects. 
Ensure that all the ambiguities are resolved.
Ensure that the revised specification is precise and unambiguous.
Note: You should provide more than one candidate (possible revisions for the specification).
<input_spec>
{input_spec}
</input_spec>

Reasons for ambiguity:
<reasons>
{reasons}
</reasons>
"""

ORDER_PROMPT = r"""
Your response will be processed by a program, not human.
So, please provide the modified specification only.
Your response should include multiple candidates, each marked with a number.
Each candidate should be a complete specification which has no ambiguity.
DO NOT include any other information in your response, like 'json', 'reasoning' or '<output_format>'.
Your response should be in the following format:

<example_output_format>
{example_output_format}
</example_output_format>
"""

EXAMPLE_OUTPUT_FORMAT = r"""
{
    "candidates": [
        {
            "number": 1,
            "spec": <modified_spec_1>
        },
        {
            "number": 2,
            "spec": <modified_spec_2>
        }
    ]
}
"""

SELECT_PROMPT = r"""
Please select the best specification from the candidates provided.
The selected specification should be precise and unambiguous.
<output_candidates>
{output_candidates}
</output_candidates>

Your output should only include the selected specification.
DO NOT include any other information in your response, like 'json', 'reasoning' or '<output_format>'.
"""

SELECT_REF_PROMPT = r"""
Please select the best specification from the candidates provided.
The selected specification should be precise and unambiguous and match the reference provided.
<output_candidates>
{output_candidates}
</output_candidates>

<RTL_reference>
{RTL_reference}
</RTL_reference>

Your output should only include the selected specification.
DO NOT include any other information in your response, like 'json', 'reasoning' or '<output_format>'.
"""

class ambiguous_fixer:
    def __init__(self, model: str, max_token: int, provider: str, cfg_path: str, use_golden_ref: bool = False):
        self.model = model
        self.llm = get_llm(model=model, max_token=max_token, provider=provider, cfg_path=cfg_path)
        self.token_counter = (
            TokenCounterCached(self.llm)
            if TokenCounterCached.is_cache_enabled(self.llm)
            else TokenCounter(self.llm)
        )
        self.use_golden_ref = use_golden_ref
    
    def generate(self, messages: List[ChatMessage]) -> ChatResponse:
        logger.info(f"Fixer input message: {messages}")
        resp, token_cnt = self.token_counter.count_chat(messages)
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"{resp.message.content}")
        return resp
        
    def run(self, input_spec: str, reasons: str, golden_ref: str = None) -> str:
        msg = [
            ChatMessage(
                content=SYSTEM_PROMPT,
                role=MessageRole.SYSTEM
            ),
            ChatMessage(
                content=GENERATION_PROMPT.format(
                    input_spec=input_spec,
                    reasons=reasons
                ),
                role=MessageRole.USER
            ),
            ChatMessage(
                content=ORDER_PROMPT.format(
                    example_output_format=EXAMPLE_OUTPUT_FORMAT
                ),
                role=MessageRole.SYSTEM
            )
        ]
        response = self.generate(msg)
        self.token_counter.log_token_stats()
        
        logger.info(f"Get response from {self.model}: {response}")

        if self.use_golden_ref:
            msg = [
                ChatMessage(
                    content=SYSTEM_PROMPT,
                    role=MessageRole.SYSTEM
                ),
                ChatMessage(
                    content=SELECT_REF_PROMPT.format(
                        output_candidates=response.message.content,
                        RTL_reference=golden_ref
                    ),
                    role=MessageRole.USER
                )
            ]
        else:
            msg = [
                ChatMessage(
                    content=SYSTEM_PROMPT,
                    role=MessageRole.SYSTEM
                ),
                ChatMessage(
                    content=SELECT_PROMPT.format(
                        output_candidates=response.message.content
                    ),
                    role=MessageRole.USER
                )
            ]
        self.token_counter.reset()
        response = self.generate(msg)
        logger.info(f"Get response from {self.model}: {response}")

        return response.message.content
