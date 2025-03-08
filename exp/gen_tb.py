from typing import List

from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse
from mage.gen_config import get_llm
from mage.log_utils import get_logger, set_log_dir, switch_log_to_file
from mage.token_counter import TokenCounter, TokenCounterCached

from datetime import datetime
import argparse
import json
import os
import re
from mage.gen_config import Config

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are an expert in RTL design. You can always write SystemVerilog code with no syntax errors and always reach correct functionality. You can always generate correct testbenches for your RTL designs.
"""

GENERATION_PROMPT = """
Your task is to write a verilog testbench for an verilog RTL module code (we call it as "DUT", device under test). 
The information we have is the problem description that guides student to write the RTL code (DUT) and the header of the "DUT". 
Our target is to generate the verilog testbench for the DUT. This testbench can check if the DUT in verilog satisfies all technical requirements of the problem description.
The infomation we have is 
- 1. the problem description that guides student to write the RTL code (DUT) and the header of the "DUT". 
- 2. the module header.
- 3. the testbench structure
- 4. the instruction of writing our testbench

The testbench does not need to check the DUT's output but only export the signals of DUT. Please export the signals of DUT to a file named "TBout.txt" at the end of each scenario. You need to use $fopen, $fdisplay, $fclose to export the signals. The exported data will be send to a python script to check the correctness of DUT.

The variables are already declared. The clock signal is already prepared. This output will be used to check the correctness of the DUT's output later.

the signals you save is the input and output of DUT, you should determine the signals according to DUT's header. Please determine the input signal's exact values. 
Note: please complete the last initial code part (marked in the given testbench template). You should give me the completed full code. The testbench template above is to help you generate the code. You must use %%d when exporting values.
please generate the full testbench code. please only reply verilog codes, no other words.

When running testbench, for one time point, you should export 1 line.
There could be multiple $fdisplay statements under one scenario, which means multiple test stimuli in one scenario.

Very important:
you should consider the rules of an ideal DUT to generate expected values in each test scenario.

Attention: before $fdisplay, you should always have a delay statement to make sure the signals are stable.

here is the header and description of the DUT:
<description>
{description}
</description>

You should write the testbench following the format of example below.
<example>
{example}
</example>
"""

EXTRA_PROMPT = """
Your response will be processed by a program, not human.
So, please provide the full testbench code only.
DO NOT include any other information in your response, like 'json', 'reasoning' or '<output_format>'.
"""

EXTRA_PROMPT_SEQ = """
Note: This circuit is a sequential circuit.
please only use "#10" as the delay when you need. If you need longer delay, you can use multiple "#10", such as "#10; #10; #10;". Avoid meaningless long delay in your code.
If you need a loop in a scenario to check multiple time points, use "repeat" loop. for exmaple:
```
// scenario x
scenario = x;
signal_1 = 1;
repeat(5) begin
    $fdisplay(file, "scenario: %d, clk = %d, signal_1 = %d", scenario, clk, signal_1);
    #10;
end

Note: You need to add a [check] tag in the $fdisplay statement in the last time point of each scenario. This tag is used to check the correctness of the DUT's output.
For example:
$fdisplay(file, "[check]scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q);
"""

EXAMPLE_OUTPUT = """
`timescale 1ns / 1ps
module testbench;
reg  clk;
reg  load;
reg  ena;
reg [1:0] amount;
reg [63:0] data;
wire [63:0] q;

integer file, scenario;
// DUT instantiation
top_module DUT (
    .clk(clk),
    .load(load),
    .ena(ena),
    .amount(amount),
    .data(data),
    .q(q)
);
// Clock generation
initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

initial begin
    file = $fopen("TBout.txt", "w");
end
// Scenario Based Test
initial begin
    // scenario 1
    scenario = 1;
    load = 1;
    ena = 0;
    amount = 2'b00;
    data = 64'hAAAAAAAAAAAAAAAA;
    $fdisplay(file, "scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q);
    #10; // Cycle 1
    $fdisplay(file, "[check]scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q); #10; 

    // scenario 2
    scenario = 2;
    load = 0;
    ena = 1;
    $fdisplay(file, "scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q); #10; 
    amount = 2'b00;
    $fdisplay(file, "scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q);
    #10; // Cycle 2
    $fdisplay(file, "[check]scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q); #10; 

    // scenario 3
    scenario = 3;
    amount = 2'b01;
    $fdisplay(file, "scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q);
    #10; // Cycle 3
    $fdisplay(file, "[check]scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q); #10; 

    // scenario 4
    scenario = 4;
    amount = 2'b10;
    $fdisplay(file, "scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q);
    #10; // Cycle 4
    $fdisplay(file, "[check]scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q); #10; 

    // scenario 5
    scenario = 5;
    amount = 2'b11;
    $fdisplay(file, "scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q);
    #10; // Cycle 5
    $fdisplay(file, "[check]scenario: %d, clk = %d, load = %d, ena = %d, amount = %d, data = %d, q = %d", scenario, clk, load, ena, amount, data, q); #10; 

    $fclose(file);
    $finish;
end

endmodule
"""

class tb_genarator:
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
        logger.info(f" input message: {messages}")
        resp, token_cnt = self.token_counter.count_chat(messages)
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"{resp.message.content}")
        return resp
    
    def run(self, input_spec: str, circuit_type: str = "SEQ") -> str:
        msg = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=GENERATION_PROMPT.format(
                    description=input_spec, example=EXAMPLE_OUTPUT
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
        "provider_fixer": "vertexanthropic",
        "filter_instance": "Prob131|Prob134|Prob135",
        "folder_path": "../verilog-eval/dataset_spec-to-rtl",
        "run_identifier": "gentb_test",
        "key_cfg_path": "../key.cfg",
    }

    args = argparse.Namespace(**args_dict)
    Config(args.key_cfg_path)
    switch_log_to_file()

    tb_genarator = tb_genarator(
        model=args.model,
        max_token=8192,
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

                output_tb = tb_genarator.run(input_spec)
                output_file_path = os.path.join(output_dir_per_task, f"{task_id}_tb.v")
                with open(output_file_path, "w") as f:
                    f.write(output_tb)