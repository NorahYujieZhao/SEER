import argparse
import json
import os
import re
from datetime import datetime

from classify_ambiguous import ambiguous_classifier
from fix_ambiguous import ambiguous_fixer
from mage.gen_config import Config
from mage.log_utils import get_logger, set_log_dir, switch_log_to_file

logger = get_logger(__name__)

args_dict = {
    "temperature": 0,
    "top_p": 1,
    "model": "claude-3-7-sonnet@20250219",
    "provider": "vertexanthropic",
    "provider_fixer": "vertexanthropic",
    "filter_instance": "Prob131|Prob134|Prob135",
    "folder_path": "../verilog-eval/dataset_spec-to-rtl",
    "run_identifier": "run_test",
    "key_cfg_path": "../key.cfg",
    "use_golden_ref": True,
}


def main():
    args = argparse.Namespace(**args_dict)
    Config(args.key_cfg_path)  # 仅用于验证配置文件
    switch_log_to_file()

    classifier = ambiguous_classifier(
        model=args.model,
        max_token=8192,
        provider=args.provider,
        cfg_path=args.key_cfg_path,
    )

    fixer = ambiguous_fixer(
        model=args.model,
        max_token=8192,
        provider=args.provider_fixer,
        cfg_path=args.key_cfg_path,
        use_golden_ref=args.use_golden_ref,
    )

    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = f"output_ambiguous_{args.run_identifier}_{timestamp}"
    log_dir = f"log_ambiguous_{args.run_identifier}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    summary_file_path = os.path.join(output_dir, "summary_ambiguous_casestudy.txt")
    summary = []

    total_spec = 0
    ambiguous_spec = 0
    fixed_spec = 0
    golden_ref = None

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

                if args.use_golden_ref:
                    golden_ref_file_path = os.path.join(root, f"{task_id}_ref.sv")
                    with open(golden_ref_file_path, "r") as f:
                        golden_ref = f.read()

                output_json_obj = classifier.run(input_spec)

                classification = output_json_obj["classification"]
                reasoning = output_json_obj["reasoning"]
                output_file_path = os.path.join(
                    output_dir_per_task, f"check_ambiguous.json"
                )
                with open(output_file_path, "w") as output_file:
                    json.dump(output_json_obj, output_file, indent=4)

                total_spec += 1
                if classification == "ambiguous":
                    ambiguous_spec += 1

                summary.append(
                    f"Task: {task_id}, Ambiguity: {classification}, Reasoning: {reasoning}\n"
                )
                print(f"Task: {task_id}, Ambiguity: {classification}")

                fix_iter = 0
                while classification == "ambiguous" and fix_iter < 5:
                    fix_iter += 1
                    input_spec = fixer.run(
                        input_spec, output_json_obj["reasoning"], golden_ref
                    )
                    logger.info(
                        f"Fixed spec, try to classify again -- trial {fix_iter}"
                    )
                    output_json_obj_fixed = classifier.run(input_spec)

                    classification = output_json_obj_fixed["classification"]

                    output_file_path_fixed = os.path.join(
                        output_dir_per_task, f"check_ambiguous_fixed.json"
                    )
                    with open(output_file_path_fixed, "w") as output_file_fixed:
                        json.dump(output_json_obj_fixed, output_file_fixed, indent=4)

                    summary.append(
                        f"Task: {task_id}, fix trial {fix_iter}, Classification: {classification}\n"
                    )
                    print(
                        f"Task: {task_id}, fix trial {fix_iter}, Classification: {classification}"
                    )
                if fix_iter > 0:
                    fixed_spec += 1
                    fixed_spec_file_path = os.path.join(
                        args.folder_path, f"{task_id}_prompt_fixed.txt"
                    )
                    with open(fixed_spec_file_path, "w") as fixed_spec_file:
                        fixed_spec_file.write(input_spec)

    summary.sort()
    summary.append(f"\nStatistics:\n")
    summary.append(f"Total Specs: {total_spec}\n")
    summary.append(f"Ambiguous Specs: {ambiguous_spec}\n")
    summary.append(f"Fixed Specs: {fixed_spec}\n")

    with open(summary_file_path, "a") as summary_file:
        summary_file.writelines(summary)

    print(f"\nFinal Statistics:")
    print(f"Total Specs: {total_spec}")
    print(f"Ambiguous Specs: {ambiguous_spec}")
    print(f"Fixed Specs: {fixed_spec}")


if __name__ == "__main__":
    main()
