import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

results = pd.DataFrame(columns=["description","DEF","predicted_label","reply","model","prompt_mode","demonstration_mode","demonstration_size","embedding_mode","thinking_mode"])
score_table = pd.DataFrame(columns=["model","prompt_mode","demo_mode","demo_size","embedding_mode","retrieval_mode","thinking_mode","accuracy","f1_macro","precision","recall"])

for f in os.listdir("../results/"):
    if not f.endswith(".csv") or f.startswith("score_table"):
        continue
    df = pd.read_csv(f"../results/{f}")
    params = f.strip(".csv").split("_")

    df["model"] = [params[0]]*len(df)
    df["prompt_mode"] = [params[1]]*len(df)
    df["demo_mode"] = [params[2]]*len(df)
    df["demo_size"] = [params[3]]*len(df)
    df["embedding_mode"] = [params[4]]*len(df)
    df["retrieval_mode"] = [params[5]] * len(df)
    if len(params) == 6:
        df["thinking_mode"] = [False]*len(df)
    elif len(params) == 7 and params[6] == "thinking":
        df["thinking_mode"] = [True]*len(df)
    else:
        raise Exception("Thinking mode and number of parameters is off.")
    
    results = pd.concat([results,df],ignore_index=True)

    mask = pd.notna(df["predicted_label"])
    y_test = df.loc[mask, "DEF"].astype(bool)
    y_pred = df.loc[mask, "predicted_label"].astype(bool)
    if (~mask).sum() > 0:
        print(f)
        print(f"Scoring on {mask.sum()}/{len(mask)} samples ({(~mask).sum()} abstained)")

    acc = round(accuracy_score(y_test, y_pred), 2)
    f1 = round(f1_score(y_test, y_pred,average="macro"), 2)
    prec = round(precision_score(y_test, y_pred), 2)
    rec = round(recall_score(y_test, y_pred), 2)

    score_table.loc[len(score_table)] = [params[0],params[1],params[2],params[3],params[4],params[5],(False if len(params)==6 else True),acc,f1,prec,rec]

    #print(confusion_matrix(df["DEF"],df["predicted_label"]))

#print only the first two decimals of the scores
print(score_table.sort_values(by=["f1_macro"]))
#score_table.to_csv("../results/score_table.csv", index=False, float_format="%.3f")
