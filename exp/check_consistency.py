import json
from pathlib import Path
from typing import Dict, List, Tuple

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from mage_rtl.log_utils import get_logger
from mage_rtl.token_counter import TokenCounter

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
You are an expert in RTL design and verification.
Your task is to review a natural-language RTL specification and a time-series of input/output signal data (in JSON format).
You must determine whether the observed input/output behavior matches the expected logic described in the RTL specification.
If there are mismatches, identify them and propose or describe the needed actions to resolve them (or highlight the issues).
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
1. RTL specification: A natural-language RTL specification describing the expected hardware behavior.
2. scenario_discription: Additional context or scenario details that help understand expected functionalities or edge conditions.
3. testbench: A Verilog testbench provided as a time-series of input/output signal data (in JSON format). Each entry represents simulation time steps showing input signals and the observed outputs.

<RTL specification>
{spec}
</RTL specification>

<scenario_discription>
{scenario_discription}
</scenario_discription>

<testbench>
{testbench}
</testbench>

[Task]:
1. **Interpret the RTL specification** and understand the intended logic.
2. **Analyze the scenario_discription** for additional context or specific operational corners.
3. **Review the testbench** (time-series JSON) and compare the observed input/output sequences against the expected behavior from the RTL specification.
4. Determine whether the observed behavior **matches** or **does not match** what the specification dictates.
   - If it does not match, **identify** and **describe** the mismatch or possible cause of the discrepancy.
5. Compile the results into the final structure, producing a scenario-by-scenario breakdown:
   - For each scenario (e.g., "Scenario1", "Scenario2", etc.):
     - Provide a short textual explanation of the reasoning (why you believe it matches or not).
     - Indicate "yes" or "no" for `if matches`.
     - If "no", fill in `unmatched action` with a brief explanation of the mismatch or an action you would take to resolve it.

When you finish your analysis, structure the output as shown below, enclosed by the <action_output> tags.

NOTE:
- Do **not** alter the content or format of the testbench; only analyze it.
- Do **not** alter the RTL specification; only interpret it.
- The final answer should strictly follow the JSON format specified in the EXAMPLE_OUTPUT.

"""

EXTRA_ORDER_PROMPT = r"""
1. Review all provided information in detail.
2. Ensure your analysis aligns with the specified RTL logic and scenario context.
3. For each scenario in the input data, provide a concise explanation of whether the observed signals match the specification.
4. If mismatches are found, specify them clearly, including what is wrong or how to correct it.
5. Format the final result according to the EXAMPLE_OUTPUT JSON structure, placing it inside the <action_output> tags as instructed.
"""

ACTION_OUTPUT_PROMPT = r"""
Output after running given action:
<action_output>
{action_output}
</action_output>
"""

EXAMPLE_OUTPUT = {
    "Scenario1": {
        "reasoning": "Here's why it matches or does not match.",
        "if matches": "yes",
        "unmatched action": "",
    },
    "Scenario2": {
        "reasoning": "Here's why it matches or does not match.",
        "if matches": "no",
        "unmatched action": "Potential mismatch explanation or recommended action.",
    },
}


class ScenarioResult(BaseModel):
    reasoning: str
    if_matches: str  # "yes" or "no"
    unmatched_action: str


class ConsistencyChecker:
    def __init__(
        self,
        token_counter: TokenCounter,
        exp_dir: str,
    ):
        self.token_counter = token_counter
        self.exp_dir = Path(exp_dir)

    def get_init_prompt_messages(self) -> List[ChatMessage]:
        """Generate initial prompt messages."""
        system_prompt = ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM)

        spec, scenario, testbench = self.load_input_files()

        init_prompt = ChatMessage(
            content=INIT_EDITION_PROMPT.format(
                spec=spec, scenario_discription=scenario, testbench=testbench
            ),
            role=MessageRole.USER,
        )

        return [system_prompt, init_prompt]

    def get_order_prompt_messages(self) -> List[ChatMessage]:
        """Generate order prompt messages."""
        return [
            ChatMessage(
                content=EXTRA_ORDER_PROMPT
                + "\n"
                + f"EXAMPLE_OUTPUT = {json.dumps(EXAMPLE_OUTPUT, indent=4)}",
                role=MessageRole.USER,
            ),
        ]

    def chat(self) -> Tuple[bool, str]:
        """Single chat interaction to check consistency."""
        if isinstance(self.token_counter, TokenCounterCached):
            self.token_counter.set_enable_cache(True)
        self.token_counter.set_cur_tag(self.__class__.__name__)

        # Generate response
        messages = self.get_init_prompt_messages() + self.get_order_prompt_messages()
        response = self.generate(messages)

        # Parse response
        scenario_results = self.parse_response(response.message.content)

        # Check results
        all_match = all(
            result.if_matches == "yes" for result in scenario_results.values()
        )

        if all_match:
            return True, "All scenarios match the specification."

        # If we get here, there are mismatches
        error_msg = (
            "The following scenarios don't match the specification:\n"
            + "\n".join(
                f"Scenario {name}: {result.unmatched_action}"
                for name, result in scenario_results.items()
                if result.if_matches == "no"
            )
        )
        return False, error_msg

    def load_input_files(self) -> Tuple[str, str, str]:
        """Load the spec, scenario description and testbench files."""
        with open(self.exp_dir / "spec.txt", "r") as f:
            spec = f.read()

        with open(self.exp_dir / "TB_scenario.txt", "r") as f:
            scenario = f.read()

        with open(self.exp_dir / "testbench.json", "r") as f:
            testbench = f.read()

        return spec, scenario, testbench

    def generate(self, messages: List[ChatMessage]) -> ChatResponse:
        """Generate response using the token counter."""
        logger.info(f"Consistency checker input message: {messages}")
        resp, token_cnt = self.token_counter.count_chat(messages)
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"Response: {resp.message.content}")
        return resp

    def parse_response(self, content: str) -> Dict[str, ScenarioResult]:
        """Parse the LLM response into structured scenario results."""
        scenarios = {}
        current_scenario = None
        current_data = {}

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("Scenario") and line.endswith(":"):
                if current_scenario and current_data:
                    scenarios[current_scenario] = ScenarioResult(**current_data)
                current_scenario = line[:-1]
                current_data = {}
                continue

            if '"reasoning"' in line:
                current_data["reasoning"] = (
                    line.split(":", 1)[1].strip().strip('"').strip(",")
                )
            elif '"if matches"' in line:
                current_data["if_matches"] = (
                    line.split(":", 1)[1].strip().strip('"').strip(",").lower()
                )
            elif '"unmatched action"' in line:
                current_data["unmatched_action"] = (
                    line.split(":", 1)[1].strip().strip('"').strip(",")
                )

        if current_scenario and current_data:
            scenarios[current_scenario] = ScenarioResult(**current_data)

        return scenarios


def check_and_fix_implementation(exp_dir: str, token_counter: TokenCounter) -> bool:
    """
    Main function to check consistency and fix implementation if needed.
    Returns True if all scenarios match after potential fixes.
    """
    checker = ConsistencyChecker(token_counter, exp_dir)
    success, message = checker.chat()

    if not success:
        logger.error(f"Consistency check failed: {message}")

        # Load necessary files for fix generation
        exp_dir_path = Path(exp_dir)
        with open(exp_dir_path / "spec.txt", "r") as f:
            spec = f.read()
        with open(exp_dir_path / "TB_scenario.txt", "r") as f:
            scenario = f.read()
        with open(exp_dir_path / "testbench.json", "r") as f:
            testbench = f.read()
        with open(exp_dir_path / "pychecker_0.py", "r") as f:
            generated_python_code = f.read()

        # Prepare messages for fix generation
        messages = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=INIT_EDITION_PROMPT.format(
                    spec=spec,
                    scenario_discription=scenario,
                    testbench=testbench,
                    coverage_report=message,  # Use the error message as coverage report
                    generated_python_code=generated_python_code,
                ),
                role=MessageRole.USER,
            ),
            ChatMessage(
                content=EXTRA_ORDER_PROMPT.format(
                    generated_python_code=generated_python_code
                ),
                role=MessageRole.USER,
            ),
        ]

        # Generate fix
        resp, _ = token_counter.count_chat(messages)

        # Extract the modified code from the response
        response_content = resp.message.content
        if "<modified_python_code>" in response_content:
            modified_code = (
                response_content.split("<modified_python_code>")[1]
                .split("</modified_python_code>")[0]
                .strip()
            )
        else:
            # Fallback: try to extract the first code block
            code_blocks = response_content.split("```python")
            if len(code_blocks) > 1:
                modified_code = code_blocks[1].split("```")[0].strip()
            else:
                logger.error("Could not extract modified code from response")
                return False

        # Save the modified code
        output_path = exp_dir_path / "pychecker_0_new.py"
        with open(output_path, "w") as f:
            f.write(modified_code)

        logger.info(f"Modified code saved to: {output_path}")

        # Run consistency check again with new implementation
        # Note: You might want to implement a mechanism to use the new file
        # return check_and_fix_implementation(exp_dir, token_counter)

    return success


if __name__ == "__main__":
    # Example usage
    from mage_rtl.token_counter import TokenCounterCached

    exp_dir = "exp/check_consistency_20250410"
    token_counter = TokenCounterCached()

    success = check_and_fix_implementation(exp_dir, token_counter)
    print(f"Consistency check {'passed' if success else 'failed'}")
