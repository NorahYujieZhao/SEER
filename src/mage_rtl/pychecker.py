from typing import List, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from .log_utils import get_logger
from .token_counter import TokenCounter
from .utils import extract_code

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert in RTL design and Python programming. You can always write correct Python code to verify RTL functionality."""

GENERATION_PROMPT = """Your current task is: write a python class "GoldenDUT". This python class can represent the golden DUT (the ideal one). In your "GoldenDUT", you should do the following things:

- a. Write a method "def __init__(self)". Set the inner states/values of the golden DUT. The "__init__" method has no input parameters except "self".
- b. Write a method "def load(self, signal_vector)". This method is to load the important input signals and get the expected output signals. it should return the expected output values. It can call other methods to help computing the expected output. It will be called by other inner methods later.
- c. write other methods you need, they can be called by "__init__", "load" .
- d. the input and output of "load" are both the signal vector. The signal vector is a dictionary, the key is the signal name, the value is the signal value.

You can use binary (like 0b1101), hexadecimal (like 0x1a) or normal number format in python. But the signal vector input to GoldenDUT is always in decimal format.

<problem_description>
{problem_description}
</problem_description>

<checker_spec>
{checker_spec}
</checker_spec>

Please only generate the Python code for the GoldenDUT class, no other words.
"""

CHECKER_TAIL = """

def SignalTxt_to_dictlist(txt:str):
    lines = txt.strip().split("\\n")
    signals = []
    for line in lines:
        signal = {}
        line = line.strip().split(", ")
        for item in line:
            if "scenario" in item:
                item = item.split(": ")
                signal["scenario"] = item[1]
            else:
                item = item.split(" = ")
                key = item[0]
                value = item[1]
                if "x" not in value and "z" not in value:
                    signal[key] = int(value)
                else:
                    signal[key] = value
        signals.append(signal)
    return signals

with open("TBout.txt", "r") as f:
    txt = f.read()
vectors_in = SignalTxt_to_dictlist(txt)
golden_dut = GoldenDUT()
tb_outputs = golden_dut.load(vectors_in)
print(tb_outputs)
"""


class PyChecker:
    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter
        self.history: List[ChatMessage] = []
        self.max_trials = 3

    def reset(self):
        self.history = []

    def chat(
        self, problem_description: str, checker_spec: str, python_rules: str = ""
    ) -> Tuple[bool, str]:
        """Generate Python checker code for the given problem

        Args:
            problem_description: Problem description text
            checker_spec: Checker specification text
            python_rules: Optional Python rules/guidelines

        Returns:
            Tuple[bool, str]: (success, generated code)
        """
        prompt = GENERATION_PROMPT.format(
            problem_description=problem_description,
            checker_spec=checker_spec,
            python_rules=python_rules,
        )

        messages = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(content=prompt, role=MessageRole.USER),
        ]

        response, token_cnt = self.token_counter.count_chat(messages)
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"Response: {response.message.content}")

        try:
            # Extract Python code from response
            code = extract_code(response.message.content, "python")[-1]
            # Add checker tail code
            full_code = code + CHECKER_TAIL
            return True, full_code
        except Exception as e:
            logger.error(f"Failed to generate checker code: {e}")
            return False, str(e)
