import json
from inspect import signature
from typing import Any, Dict, List, Tuple

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from .log_utils import get_logger
from .prompts import ORDER_PROMPT
from .sim_reviewer import SimReviewer, check_syntax
from .token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
You are an expert in RTL design and code optimization.
Your job is to use actions to edit and optimize the provided SystemVerilog RTL code.
Based on the supplied Verilog testbench and coverage report, you must identify and remove redundant parts of the RTL code that are causing the output line coverage to be less than 100%.
Ensure that the modified RTL code retains the intended functionality as verified by the testbench.
Moreover, you need to ensure 100% functionality correct rate when running the same testbench using the modified RTL code, since the original RTL code already reached the 100% functionality correct rate.
The actions below are available:
<actions>
{actions}
</actions>
"""

ACTION_PROMPT = r"""
<action>
<command>{command}</command>
<signature>{signature}</signature>
<description>{description}</description>
</action>
"""

INIT_EDITION_PROMPT = r"""
The following information is provided to assist your work:
1. A Verilog testbench which has been verified correct.
2. A coverage report indicating that the RTL module's output line coverage is below 100% due to redundant code.
3. The original RTL code that requires modification.
<testbench>
{testbench}
</testbench>
<coverage_report>
{coverage_report}
</coverage_report>
<rtl_code>
{rtl_code}
</rtl_code>

[Hints]:
- Analyze the testbench and coverage report to pinpoint which parts of the RTL code are not being exercised.
- Identify redundant logic that can be safely removed or refactored without affecting the intended behavior.
- Ensure that the modifications lead to 100% line coverage while maintaining the functional integrity of the module.
- Remember, the modified RTL code must achieve 100% functionality correctness with the given testbench, as the original RTL code already did so.
"""

EXTRA_ORDER_PROMPT = r"""
1. Understand the overall functionality of the RTL module as implied by the testbench.
2. Examine the coverage report to locate the code sections with redundancy that prevent full coverage.
3. Provide detailed reasoning in natural language about the changes you plan to make to remove redundancy and achieve 100% line coverage.
4. Do not modify the testbench or the coverage report; only adjust the RTL code.
5. Ensure that all outputs and internal signals maintain their correct behavior after modifications.
6. Use clear, well-commented RTL code, and ensure proper declarations (e.g., using logic, wire, or reg as appropriate).
7. Your final response should include both the revised RTL code and an explanation of the modifications made.

The file content which is going to be edited is given below:
<rtl_code>
{rtl_code}
</rtl_code>
"""

# The prompt above comes from:
# @misc{ho2024verilogcoderautonomousverilogcoding,
#       title={VerilogCoder: Autonomous Verilog Coding Agents with Graph-based Planning and Abstract Syntax Tree (AST)-based Waveform Tracing Tool},
#       author={Chia-Tung Ho and Haoxing Ren and Brucek Khailany},
#       year={2024},
#       eprint={2408.08927},
#       archivePrefix={arXiv},
#       primaryClass={cs.AI},
#       url={https://arxiv.org/abs/2408.08927},
# }

ACTION_OUTPUT_PROMPT = r"""
Output after running given action:
<action_output>
{action_output}
</action_output>
"""

EXAMPLE_OUTPUT = {
    "reasoning": "All reasoning steps",
    "action_input": {
        "command": "remove_content_redundancy",
        "args": {
            "old_content": "if (a && b && c) begin\n    out <= 1;\nend else if (a && b) begin\n    out <= 1;\nend",
            "new_content": "if (a && b) begin\n    out <= 1;\nend",
        },
    },
}


class ActionInput(BaseModel):
    command: str
    args: Dict[str, Any]


class RTLEditorStepOutput(BaseModel):
    reasoning: str
    action_input: ActionInput


class RTLCoverageEditor:
    def __init__(
        self,
        token_counter: TokenCounter,
        sim_reviewer: SimReviewer,
    ):
        self.token_counter = token_counter
        self.history: List[ChatMessage] = []
        self.max_trials = 3
        self.succeed_history_max_length = 2
        self.fail_history_max_length = 2
        self.is_done = False
        # self.last_mismatch_cnt: int | None = None
        self.last_coverage: float | None = None
        self.sim_reviewer = sim_reviewer

    def reset(self):
        self.is_done = False
        self.history = []
        # self.last_mismatch_cnt: int | None = None
        self.last_coverage: float | None = None

    def write_rtl(self, content: str) -> None:
        with open(self.rtl_path, "w") as f:
            f.write(content)

    def read_rtl(self) -> str:
        with open(self.rtl_path, "r") as f:
            return f.read()

    def replace_sanity_check(self) -> Dict[str, Any]:
        # Run syntax check and simulation check sequentially
        is_syntax_pass, syntax_output = check_syntax(self.rtl_path)
        if is_syntax_pass:
            syntax_output = "Syntax check passed."
        if not is_syntax_pass:
            return {
                "is_syntax_pass": False,
                "is_sim_pass": False,
                "error_msg": syntax_output,
                "line_coverage": 0.0,
                "branch_coverage": 0.0,
            }

        # Get both line and branch coverage
        is_sim_pass, line_coverage, branch_coverage, sim_output = (
            self.sim_reviewer.coverage_review_tb()
        )
        assert isinstance(line_coverage, float)
        assert isinstance(branch_coverage, float)

        return {
            "is_syntax_pass": True,
            "is_sim_pass": is_sim_pass,
            "error_msg": "" if is_sim_pass else sim_output,
            "line_coverage": line_coverage,
            "branch_coverage": branch_coverage,
        }

    def judge_replace_action_execution(
        self,
        old_content: str,
        new_content: str,
        action_name: str,
        old_file_content: str,
    ) -> Dict[str, Any]:
        sanity_check = self.replace_sanity_check()
        ret = {
            "is_action_executed": False,
            **sanity_check,
        }
        if not ret["is_syntax_pass"]:
            assert isinstance(ret["error_msg"], str)
            ret["error_msg"] += (
                f"Syntax check failed. {action_name} not executed."
                f"old_content: {old_content},"
                f"new_content: {new_content}"
            )
            self.write_rtl(old_file_content)
            return ret

        coverage = ret["line_coverage"]
        if not ret["is_sim_pass"]:
            # Must maintain 100% functionality
            logger.info("Simulation failed. Action not executed.")
            self.write_rtl(old_file_content)
            return ret

        if self.last_coverage is not None and coverage <= self.last_coverage:
            # Only accept changes that improve coverage
            logger.info(f"Coverage {coverage} not improved from {self.last_coverage}")
            self.write_rtl(old_file_content)
            ret["error_msg"] += "Coverage not improved. Action not executed."

        else:
            # Accept change
            logger.info(f"Coverage improved from {self.last_coverage} to {coverage}")
            self.last_coverage = coverage
            ret["is_action_executed"] = True
            if abs(coverage - 100.0) < 0.001:  # Check if we reached 100%
                self.is_done = True

        return ret

    def remove_content_redundancy(
        self, old_content: str, new_content: str
    ) -> Dict[str, Any]:
        """
        Replace redundant content in the RTL file with more concise content while maintaining functionality.
        Syntax check and coverage check are performed after the replacement.
        The replacement must:
        1. Maintain 100% functional correctness (all testbench cases must still pass)
        2. Improve line coverage by removing redundant code
        3. Only replace content that appears exactly once in the file

        Input:
            old_content: The old content of the file.
            new_content: The new content to replace the matching line.
        Output:
            A dictionary containing :
                1. Whether the action is executed.
                2. The error message if the action is not executed.
                3. Other information like Current coverage percentage (coverage), syntax check result and simulation check result.

        Example:
            Before:
            <example_rtl>
                always @(posedge clk) begin
                    if (a && b && c) begin
                        out <= 1;  // This line is covered
                    end else if (a && b) begin
                        out <= 1;  // This line is never reached (redundant)
                    end
                end
            </example_rtl>
            Action:
            <action_input>
                "command": "remove_content_redundancy",
                "args": {
                    "old_content": "if (a && b && c) begin\n    out <= 1;\nend else if (a && b) begin\n    out <= 1;\nend",
                    "new_content": "if (a && b) begin\n    out <= 1;\nend"
                },
            </action_input>
            Now:
            <example_rtl>
                always @(posedge clk) begin
                    if (a && b) begin
                        out <= 1;  // This line is now always reachable
                    end
                end
            </example_rtl>
        """
        old_file_content = self.read_rtl().expandtabs(4)
        old_content = old_content.expandtabs(4)
        new_content = new_content.expandtabs(4)

        # Check if old_str is unique in the file
        logger.info(f"Old File Content:\n{old_file_content}")
        logger.info(f"Target old Content:\n{old_content}")
        logger.info(f"Target new Content:\n{new_content}")

        occurrences = old_file_content.count(old_content)
        if occurrences == 0:
            return {
                "is_action_executed": False,
                "new_content": "",
                "coverage": 0.0,
                "error_msg": f"Cannot find old_content in current RTL. remove_content_redundancy not executed.",
            }
        # elif occurrences > 1:
        #     return {
        #         "is_action_executed": False,
        #         "new_content": "",
        #         "coverage": 0.0,
        #         "error_msg": f"Find multiple old_content in current RTL. remove_content_redundancy not executed.",
        #     }
        # whether need to check if new_content is unique in the file

        # Replace old_str with new_str
        new_file_content = old_file_content.replace(old_content, new_content)
        self.write_rtl(new_file_content)
        ret = self.judge_replace_action_execution(
            old_content, new_content, "remove_content_redundancy", old_file_content
        )
        # ret["new_file_content"] = new_file_content
        return ret

    def generate(self, messages: List[ChatMessage]) -> ChatResponse:
        logger.info(f"RTL editor input message: {messages}")
        resp, token_cnt = self.token_counter.count_chat(messages)
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"{resp.message.content}")
        return resp

    def gen_action_prompt(self, function) -> str:
        return ACTION_PROMPT.format(
            command=function.__name__,
            signature=signature(function),
            description=function.__doc__,
        )

    def get_init_prompt_messages(self) -> List[ChatMessage]:
        actions = [self.remove_content_redundancy]
        actions_prompt = SYSTEM_PROMPT.format(
            actions="".join([self.gen_action_prompt(action) for action in actions])
        )
        system_prompt = ChatMessage(content=actions_prompt, role=MessageRole.SYSTEM)

        with open(self.tb_path, "r") as f:
            testbench = f.read()

        with open(self.rtl_path, "r") as f:
            rtl_code = f.read()

        edit_init_prompt = ChatMessage(
            content=INIT_EDITION_PROMPT.format(
                testbench=testbench,
                coverage_report=self.coverage_report,
                rtl_code=rtl_code,
            ),
            role=MessageRole.USER,
        )
        ret = [system_prompt, edit_init_prompt]
        if (
            isinstance(self.token_counter, TokenCounterCached)
            and self.token_counter.enable_cache
        ):
            self.token_counter.add_cache_tag(ret[-1])
        return ret

    def get_order_prompt_messages(self) -> List[ChatMessage]:
        with open(self.rtl_path, "r") as f:
            rtl_code = f.read()
        return [
            ChatMessage(
                content=ORDER_PROMPT.format(
                    output_format="".join(json.dumps(EXAMPLE_OUTPUT, indent=4))
                )
                + EXTRA_ORDER_PROMPT.format(rtl_code=rtl_code),
                role=MessageRole.USER,
            ),
        ]

    def parse_output(self, response: ChatResponse) -> RTLEditorStepOutput:
        output_json_obj: Dict = json.loads(response.message.content, strict=False)
        action_input = output_json_obj["action_input"]
        command = action_input["command"]

        args = action_input["args"]
        return RTLEditorStepOutput(
            reasoning=output_json_obj["reasoning"],
            action_input=ActionInput(command=command, args=args),
        )

    def run_action(self, action_input: ActionInput) -> Dict[str, Any]:
        logger.info(f"Action input: {action_input}")
        action = getattr(self, action_input.command)
        action_output = action(**action_input.args)
        logger.info(f"Action output: {action_output}")
        return action_output

    def get_action_output_message(self, output: Dict[str, Any]) -> List[ChatMessage]:
        return [
            ChatMessage(
                content=ACTION_OUTPUT_PROMPT.format(
                    action_output=json.dumps(output, indent=4)
                ),
                role=MessageRole.USER,
            ),
        ]

    def chat(
        self,
        spec: str,
        output_dir_per_run: str,
        rtl_code: str,
        testbench: str,
        coverage_report: str,
    ) -> Tuple[bool, str]:
        # 1. Initialize the history
        # 2. Generate the initial prompt messages (with functool information)
        # 3. Loop for the max trials:
        #     - Generate the order prompt messages
        #     - Generate & parse the response
        #     - Generate & parse the tool call
        #     - If called
        if isinstance(self.token_counter, TokenCounterCached):
            self.token_counter.set_enable_cache(True)
        self.history = []
        self.token_counter.set_cur_tag(self.__class__.__name__)
        self.spec = spec
        self.output_dir_per_run = output_dir_per_run
        self.tb_path = f"{output_dir_per_run}/tb.sv"
        self.rtl_path = f"{output_dir_per_run}/rtl.sv"
        self.coverage_report = coverage_report

        # Write initial files
        with open(self.rtl_path, "w") as f:
            f.write(rtl_code)
        with open(self.tb_path, "w") as f:
            f.write(testbench)

        # Get initial coverage
        initial_check = self.replace_sanity_check()
        self.last_coverage = initial_check["coverage"]
        logger.info(f"Initial line coverage: {self.last_coverage}%")

        if self.last_coverage == 100.0:
            logger.info("Initial coverage is 100%. No need to edit.")
            return (True, rtl_code)
        if not initial_check["is_syntax_pass"]:
            logger.info("Syntax check failed. No need to edit.")
            return (False, rtl_code)
        if not initial_check["is_sim_pass"]:
            logger.info("Simulation check failed. No need to edit.")
            return (False, rtl_code)

        # Start the editing process
        self.history.extend(self.get_init_prompt_messages())
        is_pass = False
        succeed_history: List[ChatMessage] = []
        fail_history: List[ChatMessage] = []
        for i in range(self.max_trials):
            logger.info(f"RTL Editing: round {i + 1} / {self.max_trials}")
            response = self.generate(
                self.history
                + succeed_history
                + fail_history
                + self.get_order_prompt_messages()
            )
            new_contents = [response.message]
            action_input = self.parse_output(response).action_input
            action_output = self.run_action(action_input)
            if self.is_done:  # Reached 100% coverage while maintaining functionality
                is_pass = True
                break
            new_contents.extend(self.get_action_output_message(action_output))
            assert len(new_contents) == 2, f"new_contents: {new_contents}"

            if action_output["is_action_executed"]:
                logger.info(
                    f"Coverage improved to: {action_output.get('coverage', 0.0)}%"
                )
                fail_history = []
                succeed_history.extend(new_contents)
                if len(succeed_history) > self.succeed_history_max_length:
                    succeed_history = succeed_history[
                        -self.succeed_history_max_length :
                    ]
            else:
                fail_history.extend(new_contents)
                if len(fail_history) > self.fail_history_max_length:
                    fail_history = fail_history[-self.fail_history_max_length :]

        with open(self.rtl_path, "r") as f:
            final_rtl_code = f.read()
        return (is_pass, final_rtl_code)
