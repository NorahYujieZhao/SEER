import json
from typing import Dict, List

from llama_index.core.llms import LLM

from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

# use this for Deepseek r1 and claude-3-5-sonnet
# from openai import OpenAI

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from mage.gen_config import get_llm

from mage.log_utils import get_logger, set_log_dir, switch_log_to_file, switch_log_to_stdout
from mage.prompts import ORDER_PROMPT

from mage.token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
You are an expert in RTL design. You can always write SystemVerilog code with no syntax errors and always reach correct functionality.
"""

GENERATION_PROMPT = r"""
Analyze the provided SystemVerilog specification for functional ambiguities that could lead to diverging implementations. 
Focus on identifying cases where the spec allows at least two logically valid but functionally distinct RTL interpretations 
(e.g., differing timing behaviors, state transitions, or output conditions). 

Execute the following steps strictly:

1. Ambiguity Detection
   Highlight exact specification clauses (quote text) with multiple interpretations.  
   If there are multiple ambiguities, list them separately.
   Classify ambiguity type:  
    Timing Unspecified (e.g., missing clock-edge relationships)  
    State Machine Overlap (e.g., undefined priority between concurrent transitions)  
    Boundary Condition Gaps (e.g., undefined reset values or unhandled edge cases)  
    Interface Protocol Violation (e.g., ambiguous handshake signal timing)
   There may be other types of ambiguities not listed here.
   There may be multiple types for a single ambiguity or multiple ambiguities in a single spec.
   Note: Ambiguous specifications may not always lead to diverging or wrong implementations.
   You should only pick out ambiguities that lead to diverging RTL implementations and substantive impact.

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

EXAMPLE_OUTPUT_FORMAT = {
    "reasoning": "All reasoning steps and advices to avoid ambiguous",
    "classification": "ambiguous or unambiguous (do not use any other words)",
}

CLASSIFICATION_5_SHOT_EXAMPLES=r"""
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

Example 5:
<example>
    "input_spec": "
// Module: memory_controller
// Interface:
//   input logic clk, rst_n
//   input logic [7:0] addr_offset
//   input logic [31:0] base_addr
//   output logic [31:0] phys_addr
//   output logic refresh_ack
//
// Specification:
// 1. Physical address = base_addr + addr_offset when request valid
// 2. Auto-refresh must occur every 100 cycles if no active requests
// 3. refresh_ack should assert within 2 cycles of refresh start
// 4. Address calculation uses unsigned arithmetic
    ",
    "reasoning": r"
[Ambiguity 1]  
    Source Clause: "Physical address = base_addr + addr_offset when request valid"
    Type: Unspecified Calculation
    Conflict Implementations:
        // Version A: Simple addition
        assign phys_addr = base_addr + addr_offset;
        
        // Version B: Byte-offset scaling
        assign phys_addr = base_addr + (addr_offset << 2);  // Assume 4-byte granularity

    Waveform Impact:
        Version A: Address increments by 1 per offset
        Version B: Address increments by 4 per offset
        
    Clarification: "addr_offset represents 4-byte words, calculation should be base_addr + (offset << 2)"
    
    SVA Assertion:
        assert property (@(posedge clk) 
            request_valid |-> phys_addr == (base_addr + (addr_offset << 2)));

[Ambiguity 2]  
    Source Clause: "Auto-refresh must occur every 100 cycles if no active requests"
    Type: Timing Window Overlap
    Conflict Implementations:
        // Version X: Strict cycle counting
        always_ff @(posedge clk) begin
            if (counter == 99 && !active_request) begin
                refresh <= 1;
                counter <= 0;
            end
        end
        
        // Version Y: Overlapping refresh windows
        always_ff @(posedge clk) begin
            if (counter >= 100 && !active_request) begin
                refresh <= 1;
                counter <= 0;
            end
        end

    Waveform Impact:
        Version X: Exact 100-cycle intervals
        Version Y: Allows refresh at cycle 100, 101, etc.
        
    Clarification: "Refresh must occur precisely every 100 cycles, resetting counter after refresh start"
    
    SVA Assertion:
        assert property (@(posedge clk) 
            $rose(refresh) |-> $past(counter,1) == 99);
",
    "classification": "ambiguous"
</example>
"""

EXTRA_ORDER_PROMPT = r"""
VERY IMPORTANT: Please only include "reasoning" and "classification" in your response.
Do not include any other information in your response, like 'json', 'example', 'Let me analyze','input_spec' or '<output_format>'.
Key instruction: Direct output, no extra comments.
As a reminder, please directly provide the content without adding any extra comments or explanations.
"""

class ambiguous_classifier:
    def __init__(self, model: str, max_token: int, provider: str, cfg_path: str):
        self.model = model
        self.llm = get_llm(model=model, max_token=max_token, provider=provider, cfg_path=cfg_path)
        self.token_counter = (
            TokenCounterCached(self.llm)
            if TokenCounterCached.is_cache_enabled(self.llm)
            else TokenCounter(self.llm)
        )
    def run(self, input_spec: str) -> Dict:

        msg = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=GENERATION_PROMPT.format(
                    input_spec=input_spec,
                    example_prompt=CLASSIFICATION_5_SHOT_EXAMPLES
                ),
                role=MessageRole.USER
            ),
            ChatMessage(
                content=ORDER_PROMPT.format(
                    output_format="".join(json.dumps(EXAMPLE_OUTPUT_FORMAT, indent=4))
                ),
                role=MessageRole.USER
            ),
        ]
        response, token_cnt = self.token_counter.count_chat(msg)
        
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"{response.message.content}")
        self.token_counter.log_token_stats()

        #response = self.generate(msg)
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