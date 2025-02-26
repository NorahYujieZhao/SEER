import json
from typing import Dict

from llama_index.core.llms import LLM

from llama_index.llms.anthropic import Anthropic
# from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.deepseek import DeepSeek

# use this for Deepseek r1 and claude-3-5-sonnet
from openai import OpenAI

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
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        # self.llm =OpenAI(model=args.model, api_key=api_key, api_base="https://api.bianxie.ai/v1")
        # self.llm = Anthropic(model=args.model, api_key=api_key, base_url="https://api.bianxie.ai/v1")
        self.llm = OpenAI(api_key=api_key, base_url=base_url)
    
    def run(self, input_spec: str, reasons: str) -> str:
        
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": GENERATION_PROMPT.format(input_spec=input_spec, reasons=reasons)},
            ],
            stream=False
        )
        logger.info(f"Get response from {self.model}: {response}")

        return response.choices[0].message.content