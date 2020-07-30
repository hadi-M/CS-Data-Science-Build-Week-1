from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from ipdb import set_trace as st
import plotly.express as px
from datastructures import *

df = pd.read_csv("./temp_data.csv")


df.head()
X_train = df["chest_pain"]
y_train = df["chest_pain"]


clf = DecisionTreeClassifier(random_state=0)

# st()
# fig = px.scatter(df, x='chest_pain', y='heart_disease')
# fig.show()


# 0,0,0
# 0,0,0
# 0,0,1
# 0,1,1
# 0,0,1
# 1,1,1
# 0,0,2
# 0,0,0
# 0,1,0