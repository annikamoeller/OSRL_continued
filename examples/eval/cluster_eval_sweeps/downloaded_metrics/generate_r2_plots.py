import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. List of your file paths
file_paths = [
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/AntRun_Dist_a0.02_0Pre_20260622_181550_63918228-d708-4a29-ad65-0a068b9cd435.csv",
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/AntRun_Dist_a0.02_0Pre_20260622_185826_1e13f739-2d04-4d0f-98bb-3c7af4e56a99.csv",
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/AntRun_Thres_c10.0_0Pre_20260622_132654_1d968e3f-3342-467e-b61d-7f0c05bc7d83.csv",
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/AntRun_Thres_c10.0_0Pre_20260622_164159_25f279c3-e39c-47d3-87d2-9c2e93381576.csv",
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/CarCircle_Dist_a0.02_0Pre_20260622_164159_982eb4a4-5b30-4460-bafe-274e79df21ed.csv",
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/CarCircle_Dist_a0.02_0Pre_20260622_173133_11e7715b-78d3-4a8b-b7e3-59e353e89aa0.csv",
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/CarCircle_Thres_c10.0_0Pre_20260622_132653_65074bb7-47e2-421e-b5b9-916b73bc0036.csv",
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/CarCircle_Thres_c10.0_0Pre_20260622_132654_255ee4f8-9811-42a6-afe9-5d08cd39afce.csv",
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/DroneRun_Dist_a0.02_0Pre_20260622_194357_75a03523-d5e6-4946-9486-507a0887ced2.csv",
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/DroneRun_Dist_a0.02_0Pre_20260622_201348_be2b5aa6-c920-446a-8aaf-3b947af2b454.csv",
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/DroneRun_Thres_c10.0_0Pre_20260622_190340_d8ae3d3b-c4d6-4350-91ac-d1a12f139e7a.csv",
    "/home/20234949/thesis/OSRL_continued/examples/eval/cluster_eval_sweeps/downloaded_metrics/DroneRun_Thres_c10.0_0Pre_20260622_190742_9f1942c2-b02c-4e56-bdb8-9851fbc20c36.csv"
]

# 2. Parse filenames into a DataFrame
data_list = []
for path in file_paths:
    filename = os.path.basename(path)
    # Filename structure: Env_Method_...
    parts = filename.split('_')
    env = parts[0]
    method = parts[1] # "Dist" or "Thres"
    
    df = pd.read_csv(path)
    df['Env'] = env
    df['Method'] = method
    data_list.append(df)

master_df = pd.concat(data_list, ignore_index=True)

# 3. Plotting Logic
sns.set_theme(style="whitegrid")

for method_name in ["Dist", "Thres"]:
    # Filter data for the specific method
    plot_df = master_df[master_df['Method'] == method_name]
    
    # Create the plot
    g = sns.relplot(
        data=plot_df,
        x="_step", 
        y="eval/linear_probe_r2", 
        col="Env", 
        kind="line",
        height=4, 
        aspect=1.2,
        facet_kws={'sharey': True}
    )
    
    g.set_titles("{col_name}")
    g.set_axis_labels("Training Step", "Linear Probe $R^2$")
    plt.subplots_adjust(top=0.85)
    g.figure.suptitle(f"Method: {method_name} Evolution", fontsize=16, fontweight='bold')
    
    # Save or show
    plt.savefig(f"r2_evolution_{method_name}.png", dpi=300)
    print(f"Saved: r2_evolution_{method_name}.png")

plt.show()