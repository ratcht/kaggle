from data import DataLoader
from model import KNN
import pandas as pd

dataset = DataLoader(file_name="train.csv", y_tag="Transported", tags_to_drop=["Name"], drop_na=True)


# turn age into groups
dataset.df["Age"] = dataset.df["Age"].apply(lambda x: round(x, -1))


# one hot split
one_hot = pd.get_dummies(dataset.df[["HomePlanet", "Destination"]])
dataset.df.drop(labels=["HomePlanet", "Destination"], axis=1, inplace=True)

dataset.df = dataset.df.merge(one_hot ,left_index=True, right_index=True)


# split passenger Id
dataset.df[["GroupId", "PassengerNumber"]] = dataset.df["PassengerId"].str.split("_", expand=True)
dataset.df.drop("PassengerId", axis=1, inplace=True)

# split cabin
cabin_split = dataset.df["Cabin"].str.split("/", expand=True)
cabin_split[["PortSide", "StarboardSide"]] = pd.get_dummies(cabin_split[2])
cabin_split.drop(labels=[2], inplace=True, axis=1)

cabin_split.rename(columns={"P": "PortSide", "S": "StarboardSide", 0: "Deck", 1: "CabinNumber"}, inplace=True)
cabin_split["Deck"] = cabin_split["Deck"].apply(lambda x: ord(x))

dataset.df = dataset.df.merge(cabin_split ,left_index=True, right_index=True).drop(labels=["Cabin"], axis=1)


# turn bools into nums
dataset.df.replace({False: 0, True: 1}, inplace=True)

# ensure everything is a number
dataset.df = dataset.df.astype(int)



x_train, x_test, y_train, y_test = dataset.train_test_split()


knn_model = KNN(x_train, y_train)
knn_model.fit()
print(knn_model.score(x_test, y_test))
