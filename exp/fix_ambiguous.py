import json
from typing import Dict

from llama_index.core.llms import LLM

from llama_index.llms.anthropic import Anthropic
# from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.openai import OpenAI

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from mage_rtl.log_utils import get_logger, set_log_dir, switch_log_to_file, switch_log_to_stdout

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
You are an expert in RTL design. You can always write SystemVerilog code with no syntax errors and always reach correct functionality.
"""

GENERATION_PROMPT = r"""
Analyze the provided SystemVerilog specification which is ambiguous. 
Based on the reasons for these ambiguities provided below, modify the specification to eliminate any unclear aspects. 
Ensure that the revised specification is precise and unambiguous.
<input_spec>
{input_spec}
</input_spec>

Reasons for ambiguity:
<reasons>
{reasons}
</reasons>

Your response will be processed by a program, not human.
So, please provide the modified specification only.
DO NOT include any other information in your response, like 'json', 'reasoning' or '<output_format>'.
"""

class ambiguous_fixer:
    def __init__(self, model: str, api_key: str, max_tokens: int):
        self.model = model
        # self.llm =OpenAI(model=args.model, api_key=api_key, api_base="https://api.bianxie.ai/v1")
        # self.llm = Anthropic(model=args.model, api_key=api_key, base_url="https://api.bianxie.ai/v1")
        self.llm = OpenAI(model=model, api_key=api_key)
    
    def run(self, input_spec: str, reasons: str) -> str:
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
            )
        ]

        response = self.llm.chat(
            messages=msg
        )
        
        logger.info(f"Get response from {self.model}: {response}")
        return response.message.content