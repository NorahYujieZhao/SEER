import argparse
import os
import json
import re
from typing import Dict

from classify_ambiguous import ambiguous_classifier
from fix_ambiguous import ambiguous_fixer

from mage.gen_config import Config
from mage.prompts import ORDER_PROMPT
from datetime import datetime

from mage.log_utils import get_logger, set_log_dir, switch_log_to_file, switch_log_to_stdout

logger = get_logger(__name__)

args_dict = {
    # "model": "deepseek-reasoner",
    #"model": "gpt-4o-2024-08-06",
    # "model": "gpt-4o-mini-2024-07-18",
    # "model": "gemini-2.0-flash",
     #"model": "claude-3-5-sonnet-20241022",
    # "model_fixer": "models/gemini-2.0-flash",
    #"model_fixer": "claude-3-5-sonnet-20241022",
    #"model_fixer": "gpt-4o-2024-08-06",
     #"provider": "anthropic",
    #"provider": "openai",
    #"provider_fixer": "anthropic",
    #"provider_fixer": "openai",
    'temperature': 0,
    'top_p': 1,
    
     "model": "claude-3-7-sonnet@20250219",
     "model_fixer": "claude-3-7-sonnet@20250219",
     "provider": "vertexanthropic",
     "provider_fixer": "vertexanthropic",

    # "filter_instance": "Prob011|Prob012|Prob013|Prob014|Prob015|Prob152|Prob153|Prob154|Prob155|Prob156",
    # "filter_instance": "Prob051|Prob052|Prob053|Prob054|Prob055|Prob101|Prob102|Prob103|Prob104|Prob105",
    # "filter_instance": "Prob14",
    "filter_instance": "Prob011|Prob012|Prob155|Prob156",
    # "filter_instance": "",
    # "folder_path": "../verilog-eval/dataset_code-complete-iccad2023",
    "folder_path": "../verilog-eval/dataset_spec-to-rtl",
    # "run_identifier": "check_gpt4o",
    # "run_identifier": "check_gpt4o_mini",
    # "run_identifier": "check_gemini2-0flash",
    # "run_identifier": "check_deepseekr1",
    # "run_identifier": "fix_claude3-5",
    "run_identifier":"run_test",
    # "base_url": "https://api.bianxie.ai/v1",
    "key_cfg_path": "../key.cfg",
    "use_golden_ref": True,
}

def main():
    args = argparse.Namespace(**args_dict)
    cfg = Config(args.key_cfg_path)

    output_path = f"./output_{args.run_identifier}"
    log_path = f"./log_{args.run_identifier}"
    switch_log_to_file()

    # if args.model.startswith("gpt"):
    #     api_key_classifier = cfg["OPENAI_API_KEY"]
    # elif args.model.startswith("claude"):
    #     api_key_classifier = cfg["ANTHROPIC_API_KEY"]
    # elif args.model.startswith("models/gemini"):
    #     api_key_classifier = cfg["GOOGLE_API_KEY"]
    # else:
    #     api_key_classifier = cfg["BIANXIE_API_KEY"]
    
    # if args.model_fixer.startswith("gpt"):
    #     api_key_fixer = cfg["OPENAI_API_KEY"]
    # elif args.model_fixer.startswith("claude"):
    #     api_key_fixer = cfg["ANTHROPIC_API_KEY"]
    # elif args.model_fixer.startswith("models/gemini"):
    #     api_key_fixer = cfg["GOOGLE_API_KEY"]
    # else:
        # api_key_fixer = cfg["BIANXIE_API_KEY"]
    
    classifier = ambiguous_classifier(model=args.model, max_token=8192, provider=args.provider, cfg_path=args.key_cfg_path)
    fixer = ambiguous_fixer(model=args.model_fixer, max_token=8192, provider=args.provider_fixer, cfg_path=args.key_cfg_path, use_golden_ref=args.use_golden_ref)

    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = f"output_{args.run_identifier}_{timestamp}"
    log_dir = f"log_{args.run_identifier}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    total_spec = 0
    ambiguous_spec = 0
    fixed_spec = 0
    summary_file_path = os.path.join(output_dir, "summary.txt")
    summary = []
    golden_ref = None

    for root, dirs, files in os.walk(args.folder_path):
        for file in files:
            if re.match(f"({args.filter_instance}).*_prompt\\.txt$", file):
            #if re.match(f".*_prompt\\.txt$", file):
                task_id = file.replace("_prompt.txt", "")
                output_dir_per_task = f"{output_dir}/{task_id}"
                log_dir_per_task = f"{log_dir}/{task_id}"
                os.makedirs(output_dir_per_task, exist_ok=True)
                os.makedirs(log_dir_per_task, exist_ok=True)
                set_log_dir(log_dir_per_task)

                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    input_spec = f.read()
                
                if args.use_golden_ref:
                    golden_ref_file_path = os.path.join(root, f"{task_id}_ref.sv")
                    with open(golden_ref_file_path, 'r') as f:
                        golden_ref = f.read()

                output_json_obj = classifier.run(input_spec)
                classification = output_json_obj["classification"]

                output_file_path = os.path.join(output_dir_per_task, f"check_ambiguous.json")
                with open(output_file_path, 'w') as output_file:
                    json.dump(output_json_obj, output_file, indent=4)

                total_spec += 1
                if classification == "ambiguous":
                    ambiguous_spec += 1
                summary.append(f"Task: {task_id}, Classification: {classification}\n")
                print(f"Task: {task_id}, Classification: {classification}")

                fix_iter = 0
                while classification == "ambiguous" and fix_iter < 5:
                    fix_iter += 1
                    input_spec = fixer.run(input_spec, output_json_obj["reasoning"], golden_ref)
                    logger.info(f"Fixed spec, try to classify again -- trial {fix_iter}")
                    output_json_obj_fixed = classifier.run(input_spec)

                    classification = output_json_obj_fixed["classification"]

                    output_file_path_fixed = os.path.join(output_dir_per_task, f"check_ambiguous_fixed.json")
                    with open(output_file_path_fixed, 'w') as output_file_fixed:
                        json.dump(output_json_obj_fixed, output_file_fixed, indent=4)

                    summary.append(f"Task: {task_id}, fix trial {fix_iter}, Classification: {classification}\n")
                    print(f"Task: {task_id}, fix trial {fix_iter}, Classification: {classification}")
                if fix_iter > 0 and classification == "unambiguous":
                    fixed_spec += 1

    summary.sort()
    summary.append(f"Total Spec: {total_spec}, Ambiguous Spec: {ambiguous_spec}\n")
    with open(summary_file_path, 'a') as summary_file:
        summary_file.writelines(summary)
    
    print(f"Total Spec: {total_spec}, Ambiguous Spec: {ambiguous_spec}, Fixed Spec: {fixed_spec}")

if __name__ == "__main__":
    main()