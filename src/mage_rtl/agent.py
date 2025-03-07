import json
import os
import re
import sys
import traceback
from typing import Tuple

from llama_index.core.llms import LLM

from .log_utils import get_logger, set_log_dir, switch_log_to_file, switch_log_to_stdout
from .rtl_editor import RTLEditor
from .rtl_generator import RTLGenerator
from .sim_judge import SimJudge
from .sim_reviewer import SimReviewer
from .tb_generator import TBGenerator
from .token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)


class TopAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.token_counter = (
            TokenCounterCached(llm)
            if TokenCounterCached.is_cache_enabled(llm)
            else TokenCounter(llm)
        )
        self.sim_max_retry = 1
        self.rtl_max_candidates = 1
        self.rtl_selected_candidates = 1
        self.is_ablation = False
        self.redirect_log = False
        self.output_path = "./output"
        self.log_path = "./log"
        self.golden_tb_path: str | None = None
        self.golden_rtl_blackbox_path: str | None = None
        self.tb_gen: TBGenerator | None = None
        self.rtl_gen: RTLGenerator | None = None
        self.sim_reviewer: SimReviewer | None = None
        self.sim_judge: SimJudge | None = None
        self.rtl_edit: RTLEditor | None = None

    def set_output_path(self, output_path: str) -> None:
        self.output_path = output_path

    def set_log_path(self, log_path: str) -> None:
        self.log_path = log_path

    def set_ablation(self, is_ablation: bool) -> None:
        self.is_ablation = is_ablation

    def set_redirect_log(self, new_value: bool) -> None:
        self.redirect_log = new_value
        if self.redirect_log:
            switch_log_to_file()
        else:
            switch_log_to_stdout()

    def write_output(self, content: str, file_name: str) -> None:
        assert self.output_dir_per_run
        with open(f"{self.output_dir_per_run}/{file_name}", "w") as f:
            f.write(content)

    def run_instance(
        self, spec: str, testbench_num: int = 1, rtl_num: int = 1
    ) -> Tuple[bool, str]:
        """
        Run a single instance of the benchmark
        Return value:
        - is_pass: bool, whether the instance passes the golden testbench
        - rtl_code: str, the generated RTL code
        """
        assert self.tb_gen
        assert self.rtl_gen
        assert self.sim_reviewer
        # assert self.sim_judge
        # assert self.rtl_edit

        # 生成多个testbench
        testbenches = []
        self.tb_gen.reset()
        self.tb_gen.set_golden_tb_path(self.golden_tb_path)
        if not self.golden_tb_path:
            print("No golden testbench provided")
            logger.info("No golden testbench provided")

        for i in range(testbench_num):
            print(f"Generating testbench {i}")
            logger.info(f"Generating testbench {i}")
            testbench, interface = self.tb_gen.chat(spec)
            logger.info(f"Initial tb {i}:")
            logger.info(testbench)
            logger.info("Initial if:")
            logger.info(interface)
            self.write_output(testbench, f"tb_{i}.sv")
            testbenches.append(testbench)
        self.write_output(interface, "if.sv")

        # 生成多个RTL代码
        self.rtl_gen.reset()
        logger.info(spec)
        rtl_codes = []
        for i in range(rtl_num):
            logger.info(f"Generating RTL {i}")
            is_syntax_pass, rtl_code = self.rtl_gen.chat(
                input_spec=spec,
                interface=interface,
                rtl_path=os.path.join(self.output_dir_per_run, f"rtl_{i}.sv"),
                enable_cache=True,
            )
            if is_syntax_pass:
                self.write_output(rtl_code, f"rtl_{i}.sv")
                logger.info(f"Initial rtl {i}:")
                logger.info(rtl_code)
                rtl_codes.append(rtl_code)
            else:
                logger.info(f"RTL {i} syntax check failed")

        # 创建结果矩阵
        sim_results = []  # List[List[Tuple[bool, int, str]]]
        best_rtl_idx = 0
        best_tb_idx = 0
        min_mismatch = float("inf")

        # 对每个RTL和每个testbench进行仿真
        for i, rtl_code in enumerate(rtl_codes):
            sim_row = []
            for j, testbench in enumerate(testbenches):
                # 写入当前要测试的RTL和testbench
                self.write_output(rtl_code, "rtl.sv")
                self.write_output(testbench, "tb.sv")

                # 运行仿真
                is_sim_pass, sim_mismatch_cnt, sim_log = self.sim_reviewer.review()
                sim_row.append(sim_mismatch_cnt)

                # 更新最佳结果
                if sim_mismatch_cnt < min_mismatch:
                    min_mismatch = sim_mismatch_cnt
                    best_rtl_idx = i
                    best_tb_idx = j

                logger.info(
                    f"RTL {i} with TB {j}: pass={is_sim_pass}, mismatch={sim_mismatch_cnt}"
                )
            sim_results.append(sim_row)

        # 保存仿真结果矩阵
        sim_matrix = {
            "sim_results": sim_results,
            "best_rtl_idx": best_rtl_idx,
            "best_tb_idx": best_tb_idx,
            "min_mismatch": min_mismatch,
        }
        self.write_output(json.dumps(sim_matrix, indent=4), "sim_matrix.json")

        # 使用最佳的RTL代码
        best_rtl = rtl_codes[best_rtl_idx]
        self.write_output(best_rtl, "rtl_best.sv")

        return min_mismatch == 0, best_rtl

    def run_instance_ablation(self, spec: str) -> Tuple[bool, str]:
        """
        Run a single instance of the benchmark in ablation mode
        Return value:
        - is_pass: bool, whether the instance passes the golden testbench
        - rtl_code: str, the generated RTL code
        """
        assert self.rtl_gen

        self.rtl_gen.reset()
        logger.info(spec)
        # Current ablation: only run RTL generation with syntax check
        is_syntax_pass, rtl_code = self.rtl_gen.ablation_chat(
            input_spec=spec, rtl_path=os.path.join(self.output_dir_per_run, "rtl.sv")
        )
        self.write_output(rtl_code, "rtl.sv")
        return is_syntax_pass, rtl_code

    def _run(self, spec: str, testbench_num: int, rtl_num: int) -> Tuple[bool, str]:
        try:
            if os.path.exists(f"{self.output_dir_per_run}/properly_finished.tag"):
                os.remove(f"{self.output_dir_per_run}/properly_finished.tag")
            self.token_counter.reset()
            self.sim_reviewer = SimReviewer(
                self.output_dir_per_run,
                self.golden_rtl_blackbox_path,
            )
            self.rtl_gen = RTLGenerator(self.token_counter)
            self.tb_gen = TBGenerator(self.token_counter)
            # self.sim_judge = SimJudge(self.token_counter)
            # self.rtl_edit = RTLEditor(
            #    self.token_counter, sim_reviewer=self.sim_reviewer
            # )
            print(
                f"Running instance with testbench_num: {testbench_num} and rtl_num: {rtl_num}"
            )
            ret = (
                self.run_instance(spec, testbench_num, rtl_num)
                if not self.is_ablation
                else self.run_instance_ablation(spec)
            )
            self.token_counter.log_token_stats()
            with open(f"{self.output_dir_per_run}/properly_finished.tag", "w") as f:
                f.write("1")
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            ret = False, f"Exception: {exc_info[1]}"
        return ret

    def run(
        self,
        benchmark_type_name: str,
        task_id: str,
        spec: str,
        golden_tb_path: str | None = None,
        golden_rtl_blackbox_path: str | None = None,
        testbench_num: int = 1,
        rtl_num: int = 1,
    ) -> Tuple[bool, str]:
        self.golden_tb_path = golden_tb_path
        self.golden_rtl_blackbox_path = golden_rtl_blackbox_path
        log_dir_per_run = f"{self.log_path}/{benchmark_type_name}_{task_id}"
        self.output_dir_per_run = f"{self.output_path}/{benchmark_type_name}_{task_id}"
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_dir_per_run, exist_ok=True)
        set_log_dir(log_dir_per_run)
        print(
            f"Running instance with testbench_num: {testbench_num} and rtl_num: {rtl_num}"
        )
        if self.redirect_log:
            with open(f"{log_dir_per_run}/mage_rtl.log", "w") as f:
                # sys.stdout = f
                # sys.stderr = f
                print(
                    f"Running instance with testbench_num: {testbench_num} and rtl_num: {rtl_num}"
                )
                result = self._run(spec, testbench_num=testbench_num, rtl_num=rtl_num)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        else:
            result = self._run(spec, testbench_num=testbench_num, rtl_num=rtl_num)
        # Redirect log contains format with rich text.
        # Provide a rich-free version for log parsing or less viewing.
        if self.redirect_log:
            with open(f"{log_dir_per_run}/mage_rtl.log", "r") as f:
                content = f.read()
            content = re.sub(r"\[.*?m", "", content)
            with open(f"{log_dir_per_run}/mage_rtl_rich_free.log", "w") as f:
                f.write(content)
        return result
