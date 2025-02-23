import argparse
import os
import json
import re
from typing import Dict

from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.deepseek import DeepSeek

from mage_rtl.gen_config import Config
from mage_rtl.prompts import ORDER_PROMPT
from datetime import datetime

args_dict = {
    # "model": "deepseek-chat",
    "model": "gpt-4o-2024-08-06",
    "filter_instance": "Prob011_norgate|Prob012",
    # "filter_instance": "Prob01",
    # "filter_instance": "Prob011_norgate",
    # "filter_instance": "",
    # "folder_path": "../verilog-eval/dataset_code-complete-iccad2023",
    "folder_path": "../verilog-eval/dataset_spec-to-rtl",
    "run_identifier": "run_test_deepseek"
}

SYSTEM_PROMPT = r"""
You are an expert in RTL design. You can always write SystemVerilog code with no syntax errors and alwaysreach correct functionality.
"""

# GENERATION_PROMPT = r"""
# Analyze the following SystemVerilog code specification for potential ambiguities. 
# Identify if the specification allows for the creation of at least two distinct RTL code implementations that both fully comply with the given specification. 
# Highlight the specific parts of the specification that lead to these ambiguities and suggest possible clarifications to resolve them.

# <input_spec>
# {input_spec}
# </input_spec>
# """

GENERATION_PROMPT = r"""
Analyze the provided SystemVerilog specification for functional ambiguities that could lead to diverging implementations. 
Focus on identifying cases where the spec allows at least two logically valid but functionally distinct RTL interpretations 
(e.g., differing timing behaviors, state transitions, or output conditions). 

Execute the following steps strictly:

1. Ambiguity Detection
   Highlight exact specification clauses (quote text) with multiple interpretations.  
   Classify ambiguity type:  
    Timing Unspecified (e.g., missing clock-edge relationships)  
    State Machine Overlap (e.g., undefined priority between concurrent transitions)  
    Boundary Condition Gaps (e.g., undefined reset values or unhandled edge cases)  
    Interface Protocol Violation (e.g., ambiguous handshake signal timing)  
   Note: You should only pick out ambiguities that lead to diverging RTL implementations and substantive impact.

2. Implementation Contrast
   For each ambiguity, generate two minimal code snippets showing conflicting implementations.  
   Annotate how each version would produce different simulation waveforms (e.g., signal timing diagrams).  

3. Resolution Proposal
   Suggest SPEC modification using IEEE SystemVerilog standard terminology.  
   Provide assertion examples (SVA) to enforce intended behavior.

<input_spec>
{input_spec}
</input_spec>
"""
EXAMPLE_OUTPUT = {
    "reasoning": r"""
[Ambiguity 1]  
    Source Clause: "The module shall assert 'valid' when data is ready."  
    Type: Timing Unspecified  
    Conflict Implementations:  
        // Version A: Combinational valid (no latency)
            assign valid = (data_ready);
  
        // Version B: Pipelined valid (1-cycle latency)
            always_ff @(posedge clk) 
            valid <= data_ready;

    Waveform Impact:
        Version A: valid follows data_ready immediately
        Version B: valid lags data_ready by 1 cycle

Clarification: "valid must be asserted combinatorially when data_ready is high."

SVA Assertion:
    assert property (@(data_ready) data_ready |-> \#\#0 valid);
""",
    "classification": "ambigious or unambigious (do not use any other words)",
}


def main():
    args = argparse.Namespace(**args_dict)
    cfg = Config("./key.cfg")
    # llm = DeepSeek(model=args.model, api_key=cfg["BIANXIE_API_KEY"], api_base="https://api.bianxie.ai/v1")
    llm =OpenAI(model=args.model, api_key=cfg["BIANXIE_API_KEY"], api_base="https://api.bianxie.ai/v1")
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = f"output_{args.run_identifier}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    total_spec = 0
    ambiguous_spec = 0

    for root, dirs, files in os.walk(args.folder_path):
        for file in files:
            if re.match(f"({args.filter_instance}).*_prompt\\.txt$", file):
                total_spec += 1
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                msg = [
                    ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
                    ChatMessage(content=GENERATION_PROMPT.format(input_spec=content), role=MessageRole.USER),
                    ChatMessage(
                        content=ORDER_PROMPT.format(
                            output_format="".join(json.dumps(EXAMPLE_OUTPUT, indent=4))
                        ),
                        role=MessageRole.USER,
                    ),
                ]
                response = llm.chat(msg)
                try:
                    output_json_obj: Dict = json.loads(response.message.content, strict=False)
                    classification = output_json_obj["classification"]
                except json.decoder.JSONDecodeError as e:
                    print(f"Error: {e}")
                    print(response)
                    break

                output_file_path = os.path.join(output_dir, f"output_{file}")
                with open(output_file_path, 'w') as output_file:
                    json.dump(output_json_obj, output_file, indent=4)
                if classification == "ambigious":
                    ambiguous_spec += 1
                print(f"File: {file}, Classification: {classification}")
    print(f"Total Spec: {total_spec}, Ambiguous Spec: {ambiguous_spec}")

if __name__ == "__main__":
    main()