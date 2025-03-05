import argparse
import json
import os
import re
from datetime import datetime

from classify_circuit_type import circuit_type_classifier
from mage.gen_config import Config
from mage.log_utils import get_logger, set_log_dir, switch_log_to_file

logger = get_logger(__name__)

args_dict = {
    # "model": "deepseek-reasoner",
    # "model": "gpt-4o-2024-08-06",
    # "model": "gpt-4o-mini-2024-07-18",
    # "model": "gemini-2.0-flash",
    # "model": "claude-3-5-sonnet-20241022",
    # "model_fixer": "models/gemini-2.0-flash",
    # "model_fixer": "claude-3-5-sonnet-20241022",
    # "model_fixer": "gpt-4o-2024-08-06",
    # "provider": "anthropic",
    # "provider": "openai",
    # "provider_fixer": "anthropic",
    # "provider_fixer": "openai",
    "temperature": 0,
    "top_p": 1,
    "model": "claude-3-7-sonnet@20250219",
    # "model": "claude-3-5-sonnet-v2@20241022",
    "provider": "vertexanthropic",
    "provider_fixer": "vertexanthropic",
    "filter_instance": "Prob131|Prob134|Prob135",
    # "filter_instance": "Prob051|Prob052|Prob053|Prob054|Prob055|Prob101|Prob102|Prob103|Prob104|Prob105",
    # "filter_instance": "Prob092",
    # "filter_instance": "",
    "folder_path": "../verilog-eval/dataset_spec-to-rtl",
    "run_identifier": "run_test",
    "key_cfg_path": "../key.cfg",
    "use_golden_ref": True,
}


def main():
    args = argparse.Namespace(**args_dict)
    Config(args.key_cfg_path)  # 仅用于验证配置文件
    switch_log_to_file()

    classifier = circuit_type_classifier(
        model=args.model,
        max_token=8192,
        provider=args.provider,
        cfg_path=args.key_cfg_path,
    )

    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = f"output_circuit_type_{args.run_identifier}_{timestamp}"
    log_dir = f"log_circuit_type_{args.run_identifier}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    summary_file_path = os.path.join(output_dir, "summary_circuit_type_casestudy.txt")
    summary = []

    count = 0

    for root, _dirs, files in os.walk(args.folder_path):
        for file in files:
            if re.match(f"({args.filter_instance}).*_prompt\\.txt$", file):
                print(f"Processing problem {count + 1}: {file}")
                task_id = file.replace("_prompt.txt", "")
                output_dir_per_task = f"{output_dir}/{task_id}"
                log_dir_per_task = f"{log_dir}/{task_id}"
                os.makedirs(output_dir_per_task, exist_ok=True)
                os.makedirs(log_dir_per_task, exist_ok=True)
                set_log_dir(log_dir_per_task)

                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    input_spec = f.read()

                output_json_obj = classifier.run(input_spec)

                classification = output_json_obj["classification"]
                reasoning = output_json_obj["reasoning"]
                output_file_path = os.path.join(
                    output_dir_per_task, f"check_circuit_type.json"
                )
                with open(output_file_path, "w") as output_file:
                    json.dump(output_json_obj, output_file, indent=4)
                summary.append(
                    f"Task: {task_id}, Circuit Type: {classification}, Reasoning: {reasoning}\n"
                )
                print(f"Task: {task_id}, Circuit Type: {classification}")

    summary.sort()
    with open(summary_file_path, "a") as summary_file:
        summary_file.writelines(summary)


if __name__ == "__main__":
    main()
