import json
from typing import Dict

from llama_index.core.llms import LLM

from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from mage.log_utils import get_logger, set_log_dir, switch_log_to_file, switch_log_to_stdout
from mage.gen_config import get_llm

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
You are an expert in RTL design. You can always write SystemVerilog code with no syntax errors and always reach correct functionality.
You can always provide a precise and unambiguous RTL design specification.
"""

GENERATION_PROMPT = r"""
Analyze the provided SystemVerilog specification which is ambiguous. 
Based on the reasons for these ambiguities and candidates for eliminating the ambiguities provided below, modify the specification to eliminate any unclear aspects. 
FOR EACH AMBIGUITY, CHOSE ONE OF THE CANDIDATES AND MODIFY THE SPECIFICATION ACCORDINGLY.
YOU ARE NOT ALLOWED TO CHANGE THE MEANING OF THE SPECIFICATION.
Ensure that all the ambiguities are resolved.
Ensure that the revised specification is precise and unambiguous.
<input_spec>
{input_spec}
</input_spec>

Reasons for ambiguity:
<reasons>
{reasons}
</reasons>

Candidates for eliminating the ambiguities:
<candidates>
{candidates}
</candidates>

Your response will be processed by a program, not human.
So, please provide the modified specification only.
DO NOT include any other information in your response, like 'json', 'reasoning' or '<output_format>'.
"""

class ambiguous_fixer:
    def __init__(self, model: str, max_token: int, provider: str, cfg_path: str):
        self.model = model
        self.llm = get_llm(model=model, max_token=max_token, provider=provider, cfg_path=cfg_path)
    
    def run(self, input_spec: str, reasons: str, candidates: str) -> str:
        msg = [
            ChatMessage(
                content=SYSTEM_PROMPT,
                role=MessageRole.SYSTEM
            ),
            ChatMessage(
                content=GENERATION_PROMPT.format(
                    input_spec=input_spec,
                    reasons=reasons,
                    candidates=candidates
                ),
                role=MessageRole.USER
            )
        ]

        response = self.llm.chat(
            messages=msg
        )
        
        logger.info(f"Get response from {self.model}: {response}")
        return response.message.content
