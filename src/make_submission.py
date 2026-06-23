import pandas as pd
import os

RESULTS_PATH = "../results/competition/balance_ratio"

for f in os.listdir(RESULTS_PATH):
    if not f.endswith(".csv") or "MUCnoHARM" in f:
        continue
    df = pd.read_csv(os.path.join(RESULTS_PATH,f))
    out = df[["id","predicted_label"]]
    out = out.rename(columns={"predicted_label":"DEF"})

    out.to_csv(os.path.join(RESULTS_PATH,f"MUCnoHARM_def_{f}.csv"),index=False,sep=";")


