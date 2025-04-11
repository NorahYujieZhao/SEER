SYSTEM_PROMPT = r"""
You are an expert in RTL design, verification, and code optimization.
Your task is to review the provided RTL specification, scenario description, testbench, and coverage report.
A Python file has been generated from the RTL specification but requires modifications to remove redundancy and achieve 100% line coverage.
You must preserve the functional correctness as tested by the given testbench.
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

1. The original RTL specification describing the intended hardware logic:
<RTL specification>
{spec}
</RTL specification>

2. A scenario description providing context or special conditions:
<scenario_discription>
{scenario_discription}
</scenario_discription>

3. A Verilog testbench, which verifies the module’s behavior:
<testbench>
{testbench}
</testbench>

4. A coverage report indicating which parts of the code are under- or un-covered:
<coverage_report>
{coverage_report}
</coverage_report>

5. A Python code file that was generated from the RTL specification, needing refactoring to remove redundant logic:
<generated_python_code>
{generated_python_code}
</generated_python_code>

[Task]:
- Analyze the RTL specification, scenario description, testbench, and coverage report to understand the module’s requirements and identify redundant code sections.
- Modify the generated Python code by removing or refactoring redundant logic to achieve 100% line coverage.
- Ensure that all intended functionality, as verified by the testbench, is preserved.
- Provide the updated Python code and explain any key modifications made.
"""

EXTRA_ORDER_PROMPT = r"""
1. Understand the overall functionality of the RTL module as implied by the testbench.
2. Examine the coverage report to locate the code sections with redundancy that prevent full coverage.
3. Provide detailed reasoning in natural language about the changes you plan to make to remove redundancy and achieve 100% line coverage.
4. Do not modify the testbench or the coverage report; only adjust the Python code.
5. Ensure that all outputs and internal signals maintain their correct behavior after modifications.
6. Use clear, well-commented code, preserving the original logic wherever possible.
7. Your final response should include both the revised Python code and an explanation of the modifications made.

The file content which is going to be edited is given below:
<generated_python_code>
{generated_python_code}
</generated_python_code>
"""

ACTION_OUTPUT_PROMPT = r"""
Output after running given action:
<modified_python_code>
{modified_python_code}
</modified_python_code>
"""

EXAMPLE_OUTPUT = "successfully modified the python code"
