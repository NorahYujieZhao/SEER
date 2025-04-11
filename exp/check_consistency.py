from pathlib import Path
from typing import Dict, List, Tuple

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from mage_rtl.log_utils import get_logger
from mage_rtl.token_counter import TokenCounter

logger = get_logger(__name__)


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

        # Load prompt template
        with open(self.exp_dir / "check_consistency_prompt.txt", "r") as f:
            prompt_content = f.read()
            self.system_prompt = prompt_content.split("ANALYSIS_PROMPT")[0].strip()
            self.analysis_prompt = prompt_content.split('ANALYSIS_PROMPT = r"""')[
                1
            ].strip()[:-4]

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

    def check_consistency(self) -> Tuple[bool, str, Dict[str, ScenarioResult]]:
        """
        Check consistency between spec and testbench.
        Returns:
            - bool: True if all scenarios match
            - str: Error message if any scenarios don't match
            - Dict: Parsed scenario results
        """
        spec, scenario, testbench = self.load_input_files()

        # Prepare messages
        messages = [
            ChatMessage(content=self.system_prompt, role=MessageRole.SYSTEM),
            ChatMessage(
                content=self.analysis_prompt.format(
                    spec=spec, scenario_discription=scenario, testbench=testbench
                ),
                role=MessageRole.USER,
            ),
        ]

        # Generate and parse response
        response = self.generate(messages)
        scenario_results = self.parse_response(response.message.content)

        # Check if all scenarios match
        all_match = all(
            result.if_matches == "yes" for result in scenario_results.values()
        )

        error_msg = ""
        if not all_match:
            mismatched = [
                f"Scenario {scenario}: {result.unmatched_action}"
                for scenario, result in scenario_results.items()
                if result.if_matches == "no"
            ]
            error_msg = (
                "The following scenarios don't match the specification:\n"
                + "\n".join(mismatched)
            )

        return all_match, error_msg, scenario_results


def check_and_fix_implementation(exp_dir: str, token_counter: TokenCounter) -> bool:
    """
    Main function to check consistency and fix implementation if needed.
    Returns True if all scenarios match after potential fixes.
    """
    checker = ConsistencyChecker(token_counter, exp_dir)
    all_match, error_msg, results = checker.check_consistency()

    if not all_match:
        logger.error(f"Consistency check failed: {error_msg}")

        # Here you would implement logic to fix pychecker_0.py
        # This could involve generating a new prompt to fix the implementation
        # based on the error_msg

        # For example:
        fix_prompt = f"""
        The current implementation in pychecker_0.py has consistency issues:
        {error_msg}

        Please modify the implementation to fix these issues while maintaining
        the correct behavior for other scenarios.
        """
        print(fix_prompt)

        # TODO: Implement the fix logic using the fix_prompt
        # This would involve:
        # 1. Generating a response with the fix_prompt
        # 2. Parsing the response to get code changes
        # 3. Applying changes to pychecker_0.py
        # 4. Running consistency check again

    return all_match


if __name__ == "__main__":
    # Example usage
    from mage_rtl.token_counter import TokenCounterCached

    exp_dir = "exp/check_consistency_20250410"
    token_counter = TokenCounterCached()

    success = check_and_fix_implementation(exp_dir, token_counter)
    print(f"Consistency check {'passed' if success else 'failed'}")
