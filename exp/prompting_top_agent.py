import argparse
import json
import os
from datetime import datetime

import python_call as py
from classify_circuit_type import CircuitTypeClassifier
from gen_tb import TB_Generator
from gen_tb_scenario import TB_Generator_Scenario
from mage.gen_config import Config
from mage.log_utils import get_logger, set_log_dir, switch_log_to_file
from pychecker import PyChecker

logger = get_logger(__name__)


args_dict = {
    # "model": "deepseek-reasoner",
    # "model": "gpt-4o-2024-08-06",
    # "model": "gpt-4o-mini-2024-07-18",
    # "model": "gemini-2.0-flash",
    "model": "claude-3-5-sonnet-v2@20241022",
    # "model_fixer": "models/gemini-2.0-flash",
    # "model_fixer": "claude-3-5-sonnet-20241022",
    # "model_fixer": "gpt-4o-2024-08-06",
    # "provider": "anthropic",
    # "provider": "openai",
    # "provider_fixer": "anthropic",
    # "provider_fixer": "openai",
    "temperature": 1,
    "top_p": 0.85,
    # "model": "claude-3-7-sonnet@20250219",
    # "model": "claude-3-5-sonnet-v2@20241022",
    "provider": "vertexanthropic",
    "provider_fixer": "vertexanthropic",
    "task_numbers": [11, 122],
    # "filter_instance": "Prob051|Prob052|Prob053|Prob054|Prob055|Prob101|Prob102|Prob103|Prob104|Prob105",
    # "filter_instance": "Prob092",
    # "filter_instance": "",
    "folder_path": "../verilog-eval/HDLBits/HDLBits_data_backup0304.jsonl",
    "run_identifier": "gen_tb",
    "key_cfg_path": "../key.cfg",
    "use_golden_ref": True,
    "max_trials": 3,
}


def get_prob_spec(file_path, task_number):
    # give the file path and task_number, return the problem specification and the header in verilog-eval/HDLBits/HDLBits_data_backup0304.jsonl

    with open(file_path, "r") as f:
        print(f"file_path: {file_path}", f"task_number: {task_number}")
        for line in f:
            data = json.loads(line)

            if data["task_number"] == task_number:
                # print(f"data: {data}")
                return data["description"], data["header"]
    return None, None


def main():
    args = argparse.Namespace(**args_dict)
    Config(args.key_cfg_path)
    switch_log_to_file()
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = f"output_tb_{args.run_identifier}_{timestamp}"
    log_dir = f"log_tb_{args.run_identifier}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    for task_number in args.task_numbers:

        task_id = task_number
        output_dir_per_task = f"{output_dir}/{task_id}"
        log_dir_per_task = f"{log_dir}/{task_id}"
        os.makedirs(output_dir_per_task, exist_ok=True)
        os.makedirs(log_dir_per_task, exist_ok=True)
        set_log_dir(log_dir_per_task)

        input_spec, header = get_prob_spec(args.folder_path, task_number)

        output_results = []
        tb_scenarios_path = os.path.join(output_dir_per_task, f"TB_scenarios.txt")
        stimulus_python_path = os.path.join(output_dir_per_task, f"stimulus.py")
        tb_generator_scenario = TB_Generator_Scenario(
            model=args.model,
            max_token=8192,
            provider=args.provider,
            cfg_path=args.key_cfg_path,
            tb_scenarios_path=tb_scenarios_path,
        )
        tb_genarator = TB_Generator(
            model=args.model,
            max_token=8192,
            provider=args.provider,
            cfg_path=args.key_cfg_path,
            stimulus_python_path=stimulus_python_path,
        )

        py_checker = PyChecker(
            model=args.model,
            max_token=8192,
            provider=args.provider,
            cfg_path=args.key_cfg_path,
        )
        circuit_type_classifier = CircuitTypeClassifier(
            model=args.model,
            max_token=8192,
            provider=args.provider,
            cfg_path=args.key_cfg_path,
        )
        circuit_type_output_json_obj = circuit_type_classifier.run(input_spec)
        circuit_type = circuit_type_output_json_obj["classification"]

        tb_generator_scenario.run(input_spec, header, circuit_type)
        tb_scenario_description = open(tb_scenarios_path, "r").read()
        print(f"tb_scenario_description: {tb_scenario_description}")
        _ = tb_genarator.run(
            input_spec,
            header,
            tb_scenario_description,
            circuit_type,
        )
        # subproc_call(f"cd {output_dir_per_task}", timeout=120)
        stimulus_result = py.python_call_and_save(
            f"{output_dir_per_task}/stimulus.py", silent=True
        )
        print(f"stimulus_result: {stimulus_result}")

        for i in range(args.max_trials):
            python_path = os.path.join(output_dir_per_task, f"pychecker_{i}.py")
            print(f"python_path: {python_path}")
            py_checker.run(input_spec, header, python_path, circuit_type)

            # subproc_call(f"cd {output_dir_per_task}", timeout=120)
            # subproc_call(f"cd {output_dir_per_task}", timeout=120)
            output_results.append(
                py.python_call_and_save(
                    f"{output_dir_per_task}/pychecker_{i}.py", silent=True, timeout=120
                )
            )

        # 将列表中的结果合并为一个字符串
        try:
            output_str = "\n".join(str(result) for result in output_results)
            output_file_path = os.path.join(output_dir_per_task, f"our_output.txt")
            with open(output_file_path, "w") as output_file:
                output_file.write(output_str)
        except Exception as e:
            logger.error(f"Error writing output file: {e}")
            logger.error(f"Output results: {output_results}")

    # summary.sort()
    # with open(summary_file_path, "a") as summary_file:
    #    summary_file.writelines(summary)


if __name__ == "__main__":
    main()
