import pandas

df = pandas.read_table("",
                       name=["column_1", "column_2"],
                       sep="\t")

df["column_1"] = df.label.map({'ham':0, 'spam':1})

from sklearn.feature_extraction import text

df["column_2"] = df.label.map(text.CountVectorizer())

