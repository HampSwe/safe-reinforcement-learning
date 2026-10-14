from pathlib import Path
from tbparse import SummaryReader
import log_handler

# Inclusive interval [start_step, end_step]
#start_step = 0
#end_step = 10

start_step = 150
end_step = 200


project_dir = Path(__file__).resolve().parent
latest_exp = log_handler.latest_exp_dir(project_dir / "logs")

print(latest_exp)

print(f"Reading TensorBoard data from: {latest_exp}")

df = SummaryReader(str(latest_exp)).scalars

df = df[(df["step"] >= start_step) & (df["step"] <= end_step)]

table = df.pivot_table(
    index="step",
    columns="tag",
    values="value",
    aggfunc="last",
)

print(table.to_string())