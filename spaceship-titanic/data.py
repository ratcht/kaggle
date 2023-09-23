import os
import pandas as pd
from sklearn.model_selection import train_test_split
from enum import Enum
from typing import Tuple


class DataLoader:
  def __init__(self, file_name: str, data_dir="data", y_tag:None|str = None, tags_to_drop:list[str] = [], drop_na:bool = False, random_state = 0):
    self.df: pd.DataFrame = pd.read_csv(os.path.join(data_dir, file_name))
    self.y_tag = y_tag

    self.df.drop(labels=tags_to_drop, axis=1, inplace=True)

    self.random_state = random_state

    if drop_na:
      self.df.dropna(inplace=True)

    
  def train_test_split(self, train_split:int = 0.7, test_split:int = 0.3) -> Tuple:
    if self.y_tag == None:
      raise Exception("No y tag Provided")

    y = self.df[self.y_tag].values
    x = self.df.drop(self.y_tag, axis=1).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_split, test_size=test_split, random_state=self.random_state)

    return x_train, x_test, y_train, y_test
  

  def get_unique(self, col:str):
    return self.df[col].unique()
  

  def get_column_names(self):
    return self.df.columns
    


def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:

  # turn age into groups
  df["Age"] = df["Age"].apply(lambda x: round(x, -1))


  # one hot split
  one_hot = pd.get_dummies(df[["HomePlanet", "Destination"]])
  df.drop(labels=["HomePlanet", "Destination"], axis=1, inplace=True)

  df = df.merge(one_hot ,left_index=True, right_index=True)


  # split passenger Id
  df[["GroupId", "PassengerNumber"]] = df["PassengerId"].str.split("_", expand=True)
  df.drop("PassengerId", axis=1, inplace=True)

  # split cabin
  cabin_split = df["Cabin"].str.split("/", expand=True)
  cabin_split[["PortSide", "StarboardSide"]] = pd.get_dummies(cabin_split[2])
  cabin_split.drop(labels=[2], inplace=True, axis=1)

  cabin_split.rename(columns={"P": "PortSide", "S": "StarboardSide", 0: "Deck", 1: "CabinNumber"}, inplace=True)
  cabin_split["Deck"] = cabin_split["Deck"].apply(lambda x: ord(x))

  df = df.merge(cabin_split ,left_index=True, right_index=True).drop(labels=["Cabin"], axis=1)


  # turn bools into nums
  df.replace({False: 0, True: 1}, inplace=True)

  # ensure everything is a number
  df = df.astype(int)

  return df