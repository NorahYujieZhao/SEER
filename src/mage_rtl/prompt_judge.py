import json
from typing import Dict, List

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from .log_utils import get_logger
from .prompts import ORDER_PROMPT
from .token_counter import TokenCounter, TokenCounterCached
from .utils import add_lineno

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
"""

GENERATION_PROMPT = r"""You are going to post the following SystemVerilog code specification to students. 
Identify if there's any ambiguity in this problem that allows for two distinct RTL code implementations that fully comply with the given specification.
If you think the spec is ambiguous, set answer = True, otherwise set answer = False.

<input_spec>
{input_spec}
"""

class TBOutputFormat(BaseModel):
    reasoning: str
    answer: bool
    

EXAMPLE_OUTPUT = {
    "reasoning": "because there are two ways to interpret this problems",
    "answer": True,
}

EXTRA_ORDER_PROMPT = r"""
"""


class PromptJudge:
    def __init__(
        self,
        token_counter: TokenCounter,
    ):
        self.token_counter = token_counter
        self.history: List[ChatMessage] = []

    def reset(self):
        self.history = []

    def generate(self, messages: List[ChatMessage]) -> ChatResponse:
        logger.info(f"Prompt judge input message: {messages}")
        resp, token_cnt = self.token_counter.count_chat(messages)
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"{resp.message.content}")
        return resp

    def get_init_prompt_messages(
        self,
        input_spec: str,
    ) -> List[ChatMessage]:
        ret = [
            # ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=GENERATION_PROMPT.format(
                    input_spec=input_spec,
                ),
                role=MessageRole.USER,
            ),
        ]
        return ret

    def get_order_prompt_messages(self) -> List[ChatMessage]:
        return [
            ChatMessage(
                content=ORDER_PROMPT.format(
                    output_format="".join(json.dumps(EXAMPLE_OUTPUT, indent=4))
                ),
                role=MessageRole.USER,
            ),
        ]

    def parse_output(self, response: ChatResponse) -> TBOutputFormat:
        output_json_obj: Dict = json.loads(response.message.content, strict=False)
        return TBOutputFormat(
            reasoning=output_json_obj["reasoning"],
            answer=output_json_obj["answer"],
        )

    def chat(
        self,
        input_spec: str,
    ) -> bool:
        if isinstance(self.token_counter, TokenCounterCached):
            self.token_counter.set_enable_cache(False)
        self.history = []
        self.token_counter.set_cur_tag(self.__class__.__name__)
        self.history.extend(
            self.get_init_prompt_messages(
                input_spec
            )
        )
        self.history.extend(self.get_order_prompt_messages())
        response = self.generate(self.history)
        resp_obj = self.parse_output(response)
        return resp_obj.answer, resp_obj.reasoning
