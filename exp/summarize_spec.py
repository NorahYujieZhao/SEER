import json
from typing import Dict
from typing import List

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from mage.gen_config import get_llm
from mage.log_utils import get_logger, set_log_dir, switch_log_to_file
from mage.token_counter import TokenCounter, TokenCounterCached

from datetime import datetime
import argparse
import json
import os
import re
from mage.gen_config import Config

from llama_index.llms.openai import OpenAI

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
You are an expert in RTL design. You can always write SystemVerilog code with no syntax errors and always reach correct functionality.
"""

GENERATION_PROMPT = r"""
I have a code specification for RTL design. Your task is to create a brief summary of the specification that provides an overview without including important details. 
The summary should be abstract enough that a code agent would not be able to write a correct code implementation based on this information but would understand the general problem.

First, think about the key aspects of the specification without revealing crucial details. 
Then, create a summary that captures the essence of the specification without providing enough information for resolution. 
Note: You are not allowed to change the header of the specification or the module interface. Your ouput should also include the interface of the module and basic requirements to ask agent to implement the code.

Here is the specification: 
<specification>
{spec}
</specification>
"""

EXTRA_PROMPT = r"""
Your response should be like this:
<response>
I would like you to implement a module named TopModule with the following
interface. All input and output ports are one bit unless otherwise
specified.

<interface>

<summarized_spec>
</response>

Your response will be processed by a program, not human.
So, please provide the modified specification only.
VERT IMPORTANT:
DO NOT include any other information in your response, like 'json', 'reasoning', '<response>' or '<specification>'.
"""

class spec_summarizer:
    def __init__(
        self,
        model: str,
        max_token: int,
        provider: str,
        cfg_path: str,
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

    def generate(self, messages: List[ChatMessage]) -> ChatResponse:
        logger.info(f"Fixer input message: {messages}")
        resp, token_cnt = self.token_counter.count_chat(messages)
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"{resp.message.content}")
        return resp
    
    def run(self, input_spec: str) -> str:
        msg = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=GENERATION_PROMPT.format(
                    spec=input_spec,
                ),
                role=MessageRole.USER,
            ),
            ChatMessage(
                content=EXTRA_PROMPT,
            ),
        ]
        response = self.generate(msg)
        self.token_counter.log_token_stats()

        logger.info(f"Get response from {self.model}: {response}")
        return response.message.content

if __name__ == "__main__":
    args_dict = {
        "temperature": 0,
        "top_p": 1,
        "model": "claude-3-7-sonnet@20250219",
        "provider": "vertexanthropic",
        "filter_instance": "Prob131|Prob134|Prob135|Prob15",
        "folder_path": "../verilog-eval/dataset_spec-to-rtl",
        "run_identifier": "sum_spec",
        "key_cfg_path": "../key.cfg",
    }

    args = argparse.Namespace(**args_dict)
    Config(args.key_cfg_path)
    switch_log_to_file()

    summarizer = spec_summarizer(
        model=args.model,
        max_token=4096,
        provider=args.provider,
        cfg_path=args.key_cfg_path,
    )

    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = f"output_ambiguous_{args.run_identifier}_{timestamp}"
    log_dir = f"log_ambiguous_{args.run_identifier}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    count = 0

    for root, _dirs, files in os.walk(args.folder_path):
        for file in files:
            if re.match(f"({args.filter_instance}).*_prompt\\.txt$", file):
                count += 1
                print(f"Processing problem {count}: {file}")
                task_id = file.replace("_prompt.txt", "")
                output_dir_per_task = f"{output_dir}/{task_id}"
                log_dir_per_task = f"{log_dir}/{task_id}"
                os.makedirs(output_dir_per_task, exist_ok=True)
                os.makedirs(log_dir_per_task, exist_ok=True)
                set_log_dir(log_dir_per_task)

                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    input_spec = f.read()

                summarized_spec = summarizer.run(input_spec)
                output_file_path = os.path.join(output_dir_per_task, f"{task_id}_prompt_summarized.txt")
                with open(output_file_path, "w") as f:
                    f.write(summarized_spec)