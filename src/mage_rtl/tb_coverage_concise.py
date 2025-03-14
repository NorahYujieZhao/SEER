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
You are an expert in RTL design and testbench development.
Your job is to use actions to edit and enhance the provided SystemVerilog testbench.
Based on the supplied RTL code and coverage report, you must identify untested parts of the RTL and add test cases to the testbench to achieve 100% line coverage and maximize branch coverage.
Ensure that the modified testbench maintains all existing test cases while adding new ones to exercise previously uncovered lines and branches.
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
1. The RTL module that needs to be fully tested.
2. A coverage report indicating that the RTL module's line and/or branch coverage is below 100%.
3. The current testbench that needs to be enhanced.
<rtl_code>
{rtl_code}
</rtl_code>
<coverage_report>
{coverage_report}
</coverage_report>
<testbench>
{testbench}
</testbench>

[Hints]:
- Analyze the coverage report to identify which lines and branches of the RTL code are not being exercised.
- Study the RTL code to understand the conditions needed to reach those uncovered lines and branches.
- Examine the existing testbench to understand its structure and how to add new test cases.
- Add test vectors or scenarios that will trigger the execution of the uncovered lines and branches.
- Pay special attention to conditional statements (if/else, case) to ensure all branches are tested.
- Ensure that your additions to the testbench are syntactically correct and follow the existing style.
- Focus on achieving 100% line coverage and maximizing branch coverage while maintaining all existing test functionality.
"""

EXTRA_ORDER_PROMPT = r"""
1. Understand the overall functionality of the RTL module and the existing testbench.
2. Examine the coverage report to locate the code sections that are not being tested (both lines and branches).
3. Provide detailed reasoning in natural language about the changes you plan to make to the testbench.
4. Do not modify the RTL code; only adjust the testbench.
5. Ensure that all existing test cases remain intact and functional.
6. Use clear, well-commented testbench code, and ensure proper declarations.
7. Your final response should include both the revised testbench code and an explanation of the modifications made.
8. Prioritize achieving 100% line coverage first, then focus on improving branch coverage.

The file content which is going to be edited is given below:
<testbench>
{testbench}
</testbench>
"""

ACTION_OUTPUT_PROMPT = r"""
Output after running given action:
<action_output>
{action_output}
</action_output>
"""

EXAMPLE_OUTPUT = {
    "reasoning": "All reasoning steps",
    "action_input": {
        "command": "enhance_testbench",
        "args": {
            "old_content": "initial begin\n  a = 0; b = 0;\n  #10 a = 1;\n  #10 $finish;\nend",
            "new_content": "initial begin\n  a = 0; b = 0;\n  #10 a = 1;\n  #10 b = 1;\n  #10 a = 0; b = 0;\n  #10 $finish;\nend",
        },
    },
}


class ActionInput(BaseModel):
    command: str
    args: Dict[str, Any]


class TBEditorStepOutput(BaseModel):
    reasoning: str
    action_input: ActionInput


class TBCoverageEditor:
    def __init__(
        self,
        token_counter: TokenCounter,
        sim_reviewer: SimReviewer,
    ):
        self.token_counter = token_counter
        self.history: List[ChatMessage] = []
        self.max_trials = 5
        self.succeed_history_max_length = 3
        self.fail_history_max_length = 3
        self.is_done = False
        self.last_line_coverage: float | None = None
        self.last_branch_coverage: float | None = None
        self.sim_reviewer = sim_reviewer

    def reset(self):
        self.is_done = False
        self.history = []
        self.last_line_coverage = None
        self.last_branch_coverage = None

    def write_tb(self, content: str) -> None:
        with open(self.tb_path, "w") as f:
            f.write(content)

    def read_tb(self) -> str:
        with open(self.tb_path, "r") as f:
            return f.read()

    def replace_sanity_check(self) -> Dict[str, Any]:
        # Run syntax check and simulation check sequentially
        is_syntax_pass, syntax_output = check_syntax(self.tb_path)
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
            self.write_tb(old_file_content)
            return ret

        line_coverage = ret["line_coverage"]
        branch_coverage = ret["branch_coverage"]

        if not ret["is_sim_pass"]:
            # Must maintain simulation correctness
            logger.info("Simulation failed. Action not executed.")
            self.write_tb(old_file_content)
            return ret

        # Check if either line or branch coverage improved
        line_improved = (
            self.last_line_coverage is None or line_coverage >= self.last_line_coverage
        )
        branch_improved = (
            self.last_branch_coverage is None
            or branch_coverage >= self.last_branch_coverage
        )

        if line_improved and branch_improved:
            # Accept change if both line and branch coverage improved/stayed the same
            logger.info(
                f"Coverage improved: Line {line_coverage}% (was {self.last_line_coverage}%), "
                f"Branch {branch_coverage}% (was {self.last_branch_coverage}%)"
            )
            self.last_line_coverage = line_coverage
            self.last_branch_coverage = branch_coverage
            ret["is_action_executed"] = True

            # Check if we've reached our coverage goals
            line_goal_met = abs(line_coverage - 100.0) < 0.001
            branch_goal_met = abs(branch_coverage - 100.0) < 0.001

            if line_goal_met and branch_goal_met:
                self.is_done = True
                logger.info("Both line and branch coverage goals achieved!")
            elif line_goal_met:
                logger.info(
                    "Line coverage goal achieved! Still working on branch coverage."
                )
            elif branch_goal_met:
                logger.info(
                    "Branch coverage goal achieved! Still working on line coverage."
                )
        else:
            # rej changes
            logger.info(
                f"Coverage not improved: Line {line_coverage}% (was {self.last_line_coverage}%), "
                f"Branch {branch_coverage}% (was {self.last_branch_coverage}%)"
            )
            self.write_tb(old_file_content)
            ret["error_msg"] = (
                "not both line and branch coverage improved. Action not executed."
            )

        return ret

    def enhance_testbench(self, old_content: str, new_content: str) -> Dict[str, Any]:
        """
        Enhance the testbench by replacing part of it with new content that improves line and branch coverage.
        Syntax check and coverage check are performed after the replacement.
        The enhancement must:
        1. Maintain simulation correctness (no errors or failures)
        2. Improve line coverage by adding test cases for uncovered lines
        3. Improve branch coverage by testing all possible conditions in the RTL
        4. Only replace content that appears exactly once in the file

        Input:
            old_content: The existing testbench content to be replaced
            new_content: The enhanced testbench content with additional test cases

        Output:
            A dictionary containing:
                1. Whether the action is executed (is_action_executed)
                2. Error message if action failed (error_msg)
                3. Syntax check result (is_syntax_pass)
                4. Simulation result (is_sim_pass)
                5. Current line coverage percentage (line_coverage)
                6. Current branch coverage percentage (branch_coverage)

        Example:
            Before (line coverage 80%, branch coverage 50%):
            <example_tb>
                initial begin
                  a = 0; b = 0;
                  #10 a = 1;
                  #10 $finish;
                end
            </example_tb>

            Action:
            <action_input>
                "command": "enhance_testbench",
                "args": {
                    "old_content": "initial begin\n  a = 0; b = 0;\n  #10 a = 1;\n  #10 $finish;\nend",
                    "new_content": "initial begin\n  a = 0; b = 0;\n  #10 a = 1;\n  #10 b = 1;\n  #10 a = 0; b = 0;\n  #10 a = 1; b = 1;\n  #10 $finish;\nend"
                },
            </action_input>

            After (line coverage 100%, branch coverage 100%):
            <example_tb>
                initial begin
                  a = 0; b = 0;
                  #10 a = 1;
                  #10 b = 1;
                  #10 a = 0; b = 0;
                  #10 a = 1; b = 1;
                  #10 $finish;
                end
            </example_tb>
        """
        old_file_content = self.read_tb().expandtabs(4)
        old_content = old_content.expandtabs(4)
        new_content = new_content.expandtabs(4)

        # Log the changes for debugging
        logger.info(f"Old Testbench Content:\n{old_file_content}")
        logger.info(f"Target old Content:\n{old_content}")
        logger.info(f"Target new Content:\n{new_content}")

        # Verify unique occurrence
        occurrences = old_file_content.count(old_content)
        if occurrences == 0:
            return {
                "is_action_executed": False,
                "error_msg": f"Cannot find old_content in current testbench. enhance_testbench not executed.",
                "line_coverage": self.last_line_coverage or 0.0,
                "branch_coverage": self.last_branch_coverage or 0.0,
            }
        elif occurrences > 1:
            return {
                "is_action_executed": False,
                "error_msg": f"Find multiple old_content in current testbench. enhance_testbench not executed.",
                "line_coverage": self.last_line_coverage or 0.0,
                "branch_coverage": self.last_branch_coverage or 0.0,
            }

        # Make the replacement
        new_file_content = old_file_content.replace(old_content, new_content)
        self.write_tb(new_file_content)

        # Verify the change maintains functionality and improves coverage
        ret = self.judge_replace_action_execution(
            old_content, new_content, "enhance_testbench", old_file_content
        )
        return ret

    def generate(self, messages: List[ChatMessage]) -> ChatResponse:
        logger.info(f"TB editor input message: {messages}")
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
        actions = [self.enhance_testbench]
        actions_prompt = SYSTEM_PROMPT.format(
            actions="".join([self.gen_action_prompt(action) for action in actions])
        )
        system_prompt = ChatMessage(content=actions_prompt, role=MessageRole.SYSTEM)

        with open(self.rtl_path, "r") as f:
            rtl_code = f.read()

        with open(self.tb_path, "r") as f:
            testbench = f.read()

        edit_init_prompt = ChatMessage(
            content=INIT_EDITION_PROMPT.format(
                rtl_code=rtl_code,
                coverage_report=self.coverage_report,
                testbench=testbench,
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
        with open(self.tb_path, "r") as f:
            testbench = f.read()
        return [
            ChatMessage(
                content=ORDER_PROMPT.format(
                    output_format="".join(json.dumps(EXAMPLE_OUTPUT, indent=4))
                )
                + EXTRA_ORDER_PROMPT.format(testbench=testbench),
                role=MessageRole.USER,
            ),
        ]

    def parse_output(self, response: ChatResponse) -> TBEditorStepOutput:
        output_json_obj: Dict = json.loads(response.message.content, strict=False)
        action_input = output_json_obj["action_input"]
        command = action_input["command"]

        args = action_input["args"]
        return TBEditorStepOutput(
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
        output_dir_per_run: str,
        rtl_code: str,
        testbench: str,
        coverage_report: str,
    ) -> Tuple[bool, str]:
        # Initialize token counter and history
        if isinstance(self.token_counter, TokenCounterCached):
            self.token_counter.set_enable_cache(True)
        self.history = []
        self.token_counter.set_cur_tag(self.__class__.__name__)

        # Store inputs
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
        self.last_line_coverage = initial_check["line_coverage"]
        self.last_branch_coverage = initial_check["branch_coverage"]
        logger.info(
            f"Initial coverage: Line {self.last_line_coverage}%, Branch {self.last_branch_coverage}%"
        )

        if (
            abs(self.last_line_coverage - 100.0) < 0.001
            and abs(self.last_branch_coverage - 100.0) < 0.001
        ):
            logger.info(
                "Initial coverage is already 100% for both line and branch. No need to enhance testbench."
            )
            return (True, testbench)

        if not initial_check["is_syntax_pass"]:
            logger.info("Syntax check failed. No need to enhance testbench.")
            return (False, testbench)
        if not initial_check["is_sim_pass"]:
            logger.info("Simulation check failed. No need to enhance testbench.")
            return (False, testbench)

        # Start the editing process
        self.history.extend(self.get_init_prompt_messages())
        is_pass = False
        succeed_history: List[ChatMessage] = []
        fail_history: List[ChatMessage] = []

        for i in range(self.max_trials):
            logger.info(
                f"Testbench Enhancement for Line/Branch Coverage: round {i + 1} / {self.max_trials}"
            )
            response = self.generate(
                self.history
                + succeed_history
                + fail_history
                + self.get_order_prompt_messages()
            )
            new_contents = [response.message]
            action_input = self.parse_output(response).action_input
            action_output = self.run_action(action_input)

            if self.is_done:  # Reached 100% coverage for both line and branch
                is_pass = True
                break

            new_contents.extend(self.get_action_output_message(action_output))
            assert len(new_contents) == 2, f"new_contents: {new_contents}"

            if action_output["is_action_executed"]:
                logger.info(
                    f"Coverage improved to: Line {action_output.get('line_coverage', 0.0)}%, "
                    f"Branch {action_output.get('branch_coverage', 0.0)}%"
                )
                fail_history = []
                succeed_history.extend(new_contents)
                if len(succeed_history) > self.succeed_history_max_length:
                    succeed_history = succeed_history[
                        -self.succeed_history_max_length :
                    ]
            else:
                logger.info(f"Edit failed: {action_output.get('error_msg', '')}")
                fail_history.extend(new_contents)
                if len(fail_history) > self.fail_history_max_length:
                    fail_history = fail_history[-self.fail_history_max_length :]

        with open(self.tb_path, "r") as f:
            final_testbench = f.read()

        # Check final coverage
        final_check = self.replace_sanity_check()
        logger.info(
            f"Final coverage: Line {final_check['line_coverage']}%, Branch {final_check['branch_coverage']}%"
        )

        # Consider it a success if line coverage is 100%, even if branch coverage isn't perfect
        is_pass = abs(final_check["line_coverage"] - 100.0) < 0.001

        return (is_pass, final_testbench)
