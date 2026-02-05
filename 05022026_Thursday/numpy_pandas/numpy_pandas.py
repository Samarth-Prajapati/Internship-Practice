import pandas as pd
import numpy as np

# Numpy

arr = np.array([1,2,3,4,56,7])
# print(arr)

# print(np.zeros((2,3)))

# print(np.ones([4,3]))

# print(np.arange(1,20,2))

# print(np.linspace(1,16,4)) # Makes 4 equal proportion from start(1) to end(16)

arr1 = np.array([1,2,3,4,5,11])
arr2 = np.array([6,7,8,9,10,12])

# print(arr1)
# print(arr1[0])
# print(arr1[-1])
# print(arr1[1:4:2])
# print(arr1[arr1>5])

# print(np.shape(arr1))
# print(np.reshape(arr1,(2,3)))
# print(np.concatenate([arr1,arr2,arr1]))
# print(np.vstack([arr1,arr2])) # Arranges the array one after another
# print(np.hstack([arr1,arr2,arr1])) # Arrange the stack in one line

#Pandas

index = np.arange(15)
data1 = np.random.randint(low=1, high=20, size=15,dtype="int64")
data2 = np.random.randint(low=1, high=20, size=15,dtype="int64")
dict1 = {"data1":data1, "data2":data2}
# print(dict1)

# df = pd.DataFrame(dict1, index=index)
# print(df)

# print(np.mean(df["data1"]), np.mean(df["data2"]))
# print(np.median(df["data2"]))
# print(np.std(df["data1"]))
# print(np.sum(df["data2"]))
# print(np.min(df["data2"]))
# print(np.max(df["data2"]))
# print(np.dot(df["data2"], df["data1"]))

data = {
    "sepal_length": [5.1, 4.9, 6.2, 5.9, 6.7, 6.5, 5.0, 6.0, 6.9, 5.5, 4.7, 5.3, 6.1, 5.6, 6.8, 7.1, 4.8, 5.7, 6.4, 6.3],
    "sepal_width": [3.5, 3.0, 2.9, 3.0, 3.1, 3.0, 3.4, 2.7, 3.2, 2.6, 3.2, 3.7, 2.8, 2.9, 3.0, 3.0, 3.4, 2.8, 2.8, 3.3],
    "petal_length": [1.4, 1.4, 4.3, 4.2, 5.6, 5.8, 1.5, 5.1, 5.7, 4.4, 1.3, 1.5, 4.0, 3.6, 5.5, 5.9, 1.6, 4.5, 5.6, 6.0],
    "petal_width": [0.2, 0.2, 1.3, 1.5, 2.4, 2.2, 0.2, 1.6, 2.3, 1.2, 0.2, 0.2, 1.3, 1.3, 2.1, 2.1, 0.2, 1.3, 2.2, 2.5],
    "species": ["setosa", "setosa", "versicolor", "versicolor", "virginica", "virginica", "setosa", "versicolor", "virginica", "versicolor", "setosa", "setosa", "versicolor", "ersicolor", "virginica", "virginica", "setosa", "versicolor", "virginica", "virginica"]
}

# df = pd.DataFrame(data)
# df.to_csv("iris.csv")

df = pd.read_csv("iris.csv")
# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.shape)
# print(df.describe())

# print(df["sepal_length"])
# print(df[["sepal_length","sepal_width"]])

# print(df.loc[2,"petal_length"])
# print(df.iloc[1,3]) # Last index is excluded
# print(df.iloc[1:3,1:3])

# df["sepal_length"][1] = 20
df.iloc[3,1] = np.nan
df.iloc[0,3] = np.nan
df.iloc[0,0] = 0.0

# print(df.isna().sum())
# print(df.fillna(20,inplace=True))
# print(df.isna().sum())

df.rename(columns={"Unnamed: 0":"serial no."},inplace=True)
# print(df)

df.drop(columns=["serial no."],inplace=True)
# print(df)

def add2(x):
    return x+2

df["sepal_length"] = df["sepal_length"].apply(add2)
# print(df)

df["sepal_length"] = df["sepal_length"].astype("int64")
# print(df)

df2 = pd.concat([df["sepal_length"],df["sepal_width"]],axis=1)
# print(df2)

# print(df.sort_values("sepal_length",inplace=True))

