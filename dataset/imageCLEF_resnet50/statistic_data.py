import pandas as pd

data = pd.read_csv("c_c.csv").values
print(data.shape)
print(len(data[0]))