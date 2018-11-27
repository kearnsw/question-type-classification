import pandas as pd

crosswalk = pd.read_csv("crosswalk.csv", header=0, index_col=0)
raw = pd.read_csv("cleaned.tsv", header=0, sep="\t", names=["uri", "Question"])
data = pd.merge(raw, crosswalk, how="inner", on="uri")
data.to_csv("Yahoo_HealthQA.tsv", sep="\t")

