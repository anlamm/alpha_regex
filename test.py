from main import generate_gree_expression
import re
import json

path = "testcase"
SEP_LINE = "\n---------------------------------"

for i in range(1, 6):
    with open(f"{path}/scroll{i}.txt", "r", encoding="utf-8") as file:
        line = file.readline().strip()
        valid_strings = json.loads(line)
        line = file.readline().strip()
        invalid_strings = json.loads(line)
        line = file.readline().strip()
        sample_output = str(line)
        print(SEP_LINE)
        print(f"scroll {i}:\n")
        print("valid_strings:", valid_strings)
        print("invalid_strings:", invalid_strings)
        output = generate_gree_expression(valid_strings, invalid_strings)
        print("output:", output)
        print([re.match(output, string) for string in valid_strings])
        print([re.match(output, string) for string in invalid_strings])
        print("sample_output:", sample_output)
        print([re.match(sample_output, string) for string in valid_strings])
        print([re.match(sample_output, string) for string in invalid_strings])
