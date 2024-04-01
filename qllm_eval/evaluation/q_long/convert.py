import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="test_summary.txt")
parser.add_argument("--output", type=str, default="test_summary.csv")
args = parser.parse_args()

print(args)

# with open("qllm_eval/evaluation/q_long/test_summary.txt", "r") as file:
with open(args.file, "r") as file:
    text = file.read()

# Data extraction from the provided text
data = {'Test Dir': [], 'Plan ID': [], 'Accuracy': []}
for line in text.split("\n"):
    if "Command:" in line:
        command = line.split("python main_longeval_lut.py")[1]
        command_parts = command.split("--")
        test_dir = [part for part in command_parts if "test_dir" in part][0].split(" ")[1]
        lut_path = [part for part in command_parts if "lut_path" in part][0].split(" ")[1]
        plan_id = lut_path.split("/")[-1].split("_")[-1].split(".")[0]

        data['Test Dir'].append(test_dir)
        data['Plan ID'].append(plan_id)

    if "Result:" in line:
        # find the number after "accuracy"
        accuracy = re.findall(r"accuracy: (\d+\.\d+)", line)[0]

        data['Accuracy'].append(accuracy)

# Creating a DataFrame
df = pd.DataFrame(data)
print(df)

# save as csv
df.to_csv(args.output, index=False)