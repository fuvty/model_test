import re
import pandas as pd

with open("test_summary.txt", "r") as file:
    text = file.read()

# Data extraction from the provided text
data = []
for line in text.split("-------\n"):
    if "Command:" in line:
        command = line.split("python main_longeval_lut.py --")[1]
        command_parts = command.split(" --")
        test_dir = [part for part in command_parts if "test_dir" in part][0].split(" ")[1]
        lut_path = [part for part in command_parts if "lut_path" in part][0].split(" ")[1]
        plan_id = lut_path.split("/")[-1].split("_")[-1].split(".")[0]
        result = line.split("Result: ")[1].split(" ************")[0]
        accuracy = float(result.split("accuracy: ")[1])
        data.append([test_dir, plan_id, accuracy])

# Creating a DataFrame
df = pd.DataFrame(data, columns=["Test Dir", "Plan ID", "Accuracy"])
print(df)

# save as csv
df.to_csv("test_summary.csv", index=False)