import pandas as pd

# 1. Load your dat_data.csv
input_file = '/home/20234949/thesis/OSRL_continued/examples/eval/eval_suite/ccdt_buckets_cost_sweep_20260629_1709/raw_data.csv'
output_file = '/home/20234949/thesis/OSRL_continued/examples/eval/eval_suite/ccdt_buckets_cost_sweep_20260629_1709/raw_data_updated.csv'

df = pd.read_csv(input_file)

# 2. Multiply the specific column by 0.1
# This updates the 'Eval_Reward' column in place
df['Eval_Reward'] = df['Eval_Reward'] * 0.1

# 3. Save the result to a new file
df.to_csv(output_file, index=False)

print(f"✅ Success! Updated data saved to {output_file}")