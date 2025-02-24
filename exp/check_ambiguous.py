import argparse
import os
import json
import re
from typing import Dict

from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from llama_index.llms.anthropic import Anthropic
# from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.deepseek import DeepSeek

# use this for Deepseek r1 and claude-3-5-sonnet
from openai import OpenAI

from mage_rtl.gen_config import Config
from mage_rtl.prompts import ORDER_PROMPT
from datetime import datetime

from mage_rtl.log_utils import get_logger, set_log_dir, switch_log_to_file, switch_log_to_stdout

logger = get_logger(__name__)

args_dict = {
    # "model": "deepseek-reasoner",
    # "model": "gpt-4o-2024-08-06",
    "model": "claude-3-5-sonnet-20241022",
    "filter_instance": "Prob011|Prob012|Prob013|Prob014|Prob015|Prob152|Prob153|Prob154|Prob155|Prob156",
    # "filter_instance": "Prob01",
    # "filter_instance": "Prob155",
    # "filter_instance": "",
    # "folder_path": "../verilog-eval/dataset_code-complete-iccad2023",
    "folder_path": "../verilog-eval/dataset_spec-to-rtl",
    # "run_identifier": "check_gpt4o",
    # "run_identifier": "check_deepseekr1",
    "run_identifier": "check_claude3-5",
}

SYSTEM_PROMPT = r"""
You are an expert in RTL design. You can always write SystemVerilog code with no syntax errors and alwaysreach correct functionality.
"""

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

{example_prompt}
<input_spec>
{input_spec}
</input_spec>
"""
# EXAMPLE_OUTPUT_FORMAT = {
#     "reasoning": r"""
# [Ambiguity 1]  
#     Source Clause: "The module shall assert 'valid' when data is ready."  
#     Type: Timing Unspecified  
#     Conflict Implementations:  
#         // Version A: Combinational valid (no latency)
#             assign valid = (data_ready);
  
#         // Version B: Pipelined valid (1-cycle latency)
#             always_ff @(posedge clk) 
#             valid <= data_ready;

#     Waveform Impact:
#         Version A: valid follows data_ready immediately
#         Version B: valid lags data_ready by 1 cycle

# Clarification: "valid must be asserted combinatorially when data_ready is high."

# SVA Assertion:
#     assert property (@(data_ready) data_ready |-> \#\#0 valid);
# """,
#     "classification": "ambiguous or unambiguous (do not use any other words)",
# }

EXAMPLE_OUTPUT_FORMAT = {
    "reasoning": "All reasoning steps and advices to avoid ambiguous",
    "classification": "ambiguous or unambiguous (do not use any other words)",
}

CLASSIFICATION_4_SHOT_EXAMPLES=r"""
Here are some examples of RTL ambiguity detection:
Example 1:
<example>
    "input_spec": "
// Module: data_processor
// Interface:   
//   input logic clk, rst_n
//   input logic [31:0] data_in
//   output logic data_valid
//
// Specification:
// 1. The module shall process incoming data when enable=1
// 2. data_valid must be asserted when processing completes
// 3. Processing takes 1-3 cycles depending on data value
    ",
    "reasoning": r"
[Ambiguity 1]  
    Source Clause: "Processing takes 1-3 cycles depending on data value"
    Type: Timing Unspecified
    Conflict Implementations:
        // Version A: Fixed 3-cycle pipeline
        always_ff @(posedge clk) begin
            if (data_in[31]) pipe_stage <= 3;
            else            pipe_stage <= 1;
        end
        
        // Version B: Variable latency
        always_comb begin
            case(data_in[31:30])
                2'b11: latency = 3;
                2'b01: latency = 2;
                default: latency = 1;
            endcase
        end

    Waveform Impact:
        Version A: Ignores 2-bit encoding, uses single-bit decision
        Version B: Fully utilizes 2-bit control field
        
    Clarification: "Processing latency shall be determined by bits[31:30] as:
                   2'b11:3 cycles, 2'b01:2 cycles, others:1 cycle"
                   
    SVA Assertion:
        assert property (@(posedge clk) disable iff (!rst_n)
            (enable && data_in[31:30]==2'b11) |-> \#\#3 data_valid);
        assert property (@(posedge clk) disable iff (!rst_n)
            (enable && data_in[31:30]==2'b01) |-> \#\#2 data_valid);
        assert property (@(posedge clk) disable iff (!rst_n)
            (enable && data_in[31:30]!=2'b11 && data_in[31:30]!=2'b01) |-> \#\#1 data_valid);
    
",
    "classification": "ambiguous"
</example>

Example 2:
<example>
    "input_spec": "
// Module: arbiter
// Requirements:
// - Supports 4 requestors with priority 0(highest)-3(lowest)
// - grant signal must be asserted within 2 cycles after req
// - Preemption allowed for higher priority requests
    ",
    "reasoning": r"
[Ambiguity 1]  
    Source Clause: "Preemption allowed for higher priority requests"
    Type: State Machine Overlap
    Conflict Implementations:
        // Version A: Immediate preemption
        always_comb begin
            grant = 4'b0;
            priority casez(req)
                4'b???1: grant = 4'b0001;
                4'b??10: grant = 4'b0010;
                4'b?100: grant = 4'b0100;
                4'b1000: grant = 4'b1000;
            endcase
        end
        
        // Version B: Cycle-boundary preemption
        always_ff @(posedge clk) begin
            if(current_grant && higher_priority_req)
                grant <= 1 << get_highest_priority(req);
        end

    Waveform Impact:
        Version A: Mid-cycle grant changes
        Version B: Grants update only at clock edges
        
    Clarification: "Preemption shall only occur at clock boundaries"
    
    SVA Assertion:
        assert property (@(posedge clk) 
            $changed(grant) |-> !$isunknown(clk));
",
    "classification": "ambiguous"
</example>

Example 3:
<example>
    "input_spec": "
// Module: serial_parser
// Functionality:
// - Start parsing when start_pulse=1
// - Detect sync pattern 0xA5 in first 2 bytes
// - Assert error_flag if invalid header within 16 cycles
// - All operations synchronous to clk (100MHz)
    ",
[Ambiguity 1]  
    Source Clause: "Detect sync pattern 0xA5 in first 2 bytes"
    Type: Boundary Condition Gap
    Conflict Implementations:
        // Version A: Check first 16 bits
        assign sync_ok = (data_stream[15:0] == 16'hA5);
        
        // Version B: Check any consecutive 8 bits
        always_comb begin
            sync_ok = 0;
            for(int i=0; i<16; i++)
                if(data_stream[i+:8] == 8'hA5)
                    sync_ok = 1;
        end

    Waveform Impact:
        Version A: Detects sync only at bits[15:0]
        Version B: May detect sync at bits[7:0], [8:1], ..., [23:16]
        
    Clarification: "Sync pattern must match the first two bytes exactly"
    
    SVA Assertion:
        assert property (@(posedge clk) 
            start_pulse |-> \#\#0 data_stream[15:0]==16'hA5);
",
    "classification": "ambiguous"
</example>

Example 4:
<example>
    "input_spec": "
// Module: crc_generator
// Specification:
// - Compute CRC32 on 32-bit input data
// - Initial value: 0xFFFFFFFF
// - Polynomial: 0x04C11DB7 (LSB-first)
// - Output registered at end of calculation
// - Calculation completes exactly 32 cycles after en=1
    ",
    "reasoning": r"
[Analysis]  
    Key Clarifications:
        1. Explicit timing constraint (32 cycles)
        2. Defined initialization value
        3. Specified polynomial format (LSB-first)
        4. Registration requirement
        
    Valid Implementation:
        always_ff @(posedge clk) begin
            if (en) begin
                crc <= 32'hFFFF_FFFF;
                counter <= 0;
            end
            else if (counter < 32) begin
                crc <= next_crc(crc, data);
                counter <= counter + 1;
            end
        end
        assign crc_out = (counter==32) ? crc : '0;
        
    Waveform Consistency:
        All implementations will show:
        - Fixed 32-cycle latency
        - Identical polynomial processing
        - Synchronous output registration
",
    "classification": "unambiguous"
</example>
"""


def main():
    args = argparse.Namespace(**args_dict)
    cfg = Config("./key.cfg")

    output_path = f"./output_{args.run_identifier}"
    log_path = f"./log_{args.run_identifier}"
    switch_log_to_file()

    # llm = DeepSeek(model=args.model, api_key=cfg["BIANXIE_API_KEY"], api_base="https://api.bianxie.ai/v1")
    # llm =OpenAI(model=args.model, api_key=cfg["BIANXIE_API_KEY"], api_base="https://api.bianxie.ai/v1")
    # llm = Anthropic(model=args.model, api_key=cfg["BIANXIE_API_KEY"], base_url="https://api.bianxie.ai/v1")
    # use this for Deepseek r1 and claude-3-5-sonnet
    client = OpenAI(api_key=cfg["BIANXIE_API_KEY"], base_url="https://api.bianxie.ai/v1")

    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = f"output_{args.run_identifier}_{timestamp}"
    log_dir = f"log_{args.run_identifier}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    total_spec = 0
    ambiguous_spec = 0
    summary_file_path = os.path.join(output_dir, "summary.txt")

    for root, dirs, files in os.walk(args.folder_path):
        for file in files:
            if re.match(f"({args.filter_instance}).*_prompt\\.txt$", file):
                task_id = file.replace("_prompt.txt", "")
                output_dir_per_task = f"{output_dir}/{task_id}"
                log_dir_per_task = f"{log_dir}/{task_id}"
                os.makedirs(output_dir_per_task, exist_ok=True)
                os.makedirs(log_dir_per_task, exist_ok=True)
                set_log_dir(log_dir_per_task)

                total_spec += 1
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()

                # msg = [
                #     ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
                #     ChatMessage(content=GENERATION_PROMPT.format(input_spec=content, example_prompt = CLASSIFICATION_4_SHOT_EXAMPLES), role=MessageRole.USER),
                #     ChatMessage(
                #         content=ORDER_PROMPT.format(
                #             output_format="".join(json.dumps(EXAMPLE_OUTPUT_FORMAT, indent=4))
                #         ),
                #         role=MessageRole.USER,
                #     ),
                # ]
                # response = llm.chat(msg)

                # use this for Deepseek r1 and claude-3-5-sonnet
                msg = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": GENERATION_PROMPT.format(input_spec=content, example_prompt = CLASSIFICATION_4_SHOT_EXAMPLES)},
                    {"role": "user", "content": ORDER_PROMPT.format(output_format="".join(json.dumps(EXAMPLE_OUTPUT_FORMAT, indent=4)))},
                ]
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=msg,
                    stream=False
                )

                logger.info(f"Get response from {args.model}: {response}")
                try:
                    # output_json_obj: Dict = json.loads(response.message.content, strict=False)

                    # use this for Deepseek r1 and claude-3-5-sonnet
                    output_json_obj: Dict = json.loads(response.choices[0].message.content, strict=False)
                    
                    classification = output_json_obj["classification"]
                    logger.info(f"Succeed to parse response, Classification: {classification}")
                except json.decoder.JSONDecodeError as e:
                    print(f"Error: {e}")
                    logger.info(f"Error: {e}")
                    print(response)
                    break

                output_file_path = os.path.join(output_dir_per_task, f"check_ambiguous.json")
                with open(output_file_path, 'w') as output_file:
                    json.dump(output_json_obj, output_file, indent=4)
                if classification == "ambiguous":
                    ambiguous_spec += 1
                with open(summary_file_path, 'a') as summary_file:
                    summary_file.write(f"File: {file}, Classification: {classification}\n")
                print(f"File: {file}, Classification: {classification}")
    with open(summary_file_path, 'a') as summary_file:
        summary_file.write(f"Total Spec: {total_spec}, Ambiguous Spec: {ambiguous_spec}\n")
    print(f"Total Spec: {total_spec}, Ambiguous Spec: {ambiguous_spec}")

if __name__ == "__main__":
    main()