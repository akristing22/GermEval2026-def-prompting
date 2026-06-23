import promptolution
import importlib
importlib.reload(promptolution)
from promptolution.exemplar_selectors import RandomSearchSelector
from promptolution.tasks import ClassificationTask
from promptolution.llms import LocalLLM
from promptolution.predictors import FirstOccurrencePredictor
import pandas as pd
import nest_asyncio
import torch
import yaml

nest_asyncio.apply()  # Required for notebook environments

torch.cuda.empty_cache()

df = pd.read_csv("/data/def_train.csv",sep=";")
df["DEF"] = [str(x).lower() for x in df["DEF"].values]
df.rename(columns={"description":"x","DEF":"y"},inplace=True)

task_description = ("The dataset contains posts labelled as criminally relevant or not with respect to §185 StGB (Insult)."
                   "The task is to classify the posts as either criminally relevant or not."
                   "The class labels are 'true' for criminally relevant and 'false' for not criminally relevant.")

init_prompts = [
    "Is the following post criminally relevant with respect to §§185-187 StGB (Defamatory Offense)?\n Answer only with 'true' (criminally relevant according to §§185-187 StGB) or 'false' (not criminally relevant according to §§185-187 StGB).",
    "Step 1: A defamatory object can be a living individual, a group of persons that is distinguishable from the general public such that every member of that group could feel their honor is infringed or collective entities such as governments or press companies with a recognised social role and who act with a collective, single will.\n\
    Is there a defamatory object in this statement?\n\
    No → STOP (NOT criminally relevant according to §§185-187 StGB)\n\
    Yes → Step 2\n\
    Step 2: A disparaging statement is a statement which interferes with the potential victim's right to honour. It is already fulfilled by expressing contempt or disrespect through the allegation of shortcomings that could reduce the victim’s social standing.\n\
    Is there a disparaging statement directed towards the defamatory object?\n\
    No → STOP (NOT criminally relevant according to §§185-187 StGB)\n\
    Yes → Step 3\n\
    Step 3: A factual claim is a stetement that can be proven to be true or untrue in front of a court. A value judgment constitutes and expression of personal opinions.\n\
    Is it primarily a value judgement and NOT primarily a factual claim?\n\
    No → STOP (NOT criminally relevant according to §§185-187 StGB)\n\
    Yes → Step 4\n\
    Step 4: An abusive insult is a statement that constitutes breaking a taboo by itself and intends only the defamation of the victim without any substantiated contribution.\n\
    Is the statement an abusive insult?\n\
    Yes → STOP (criminally RELEVANT according to §§185-187 StGB)\n\
    No → Step 5\n\
    Step 5: A statement is of public interest if it contains a contribution to the public discourse with respect to a particular relevant topic of public interest.\n\
    Is the statement of public interest?\n\
    Yes → STOP (NOT criminally relevant according to §§185-187 StGB)\n\
    No → Step 6\n\
    Step 6: A statement is considered abusive criticism if it goes beyond plausible criticism by primarily intending to abusively offend the victim, hereby neglecting a substantiated contribution.\n\
    Is the statement abusive criticism?\n\
    Yes → STOP (criminally RELEVANT according to §§185-187 StGB)\n\
    No → STOP (NOT criminally relevant according to §§185-187 StGB)\n\n\
    According to this decision schema, decide whether the following text is criminally revant according to §§185-187 StGB (Defamatory Offense). Answer only with 'true' (criminally relevant according to §§185-187 StGB) or 'false' (not criminally relevant according to §§185-187 StGB)."]

#system_prompt = "You are a legal expert for defamatory offenses according the the German Criminal Code (§§185-187 StGB). Help the user decide whether given posts fall within in the scope of these articles."

with open("config.yaml") as stream:
    config = yaml.safe_load(stream)

task = ClassificationTask(df, task_description=task_description)

llm=LocalLLM(model_id=config["model_name"],batch_size=8)

predictor = FirstOccurrencePredictor(llm=llm,classes=["true","false"])
selector=RandomSearchSelector(task = task, predictor=predictor)

final_selection= [selector.select_exemplars(prompt=p,n_examples=config["demonstration_size"]) for p in init_prompts]

for element in final_selection:
    print(element.instruction)
    print(element.few_shots)