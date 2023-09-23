from data import DataLoader, filter_dataset
from model import KNN
import pandas as pd





dataset_train = DataLoader(file_name="train.csv", y_tag="Transported", tags_to_drop=["Name"], drop_na=True)
dataset_train.df = filter_dataset(dataset_train.df)


x_train, x_test, y_train, y_test = dataset_train.train_test_split()


knn_model = KNN(x_train, y_train)
knn_model.fit()
print(knn_model.score(x_test, y_test))



# load test data
dataset_test = DataLoader(file_name="test.csv", tags_to_drop=["Name"])
ids = dataset_test.df[["PassengerId"]]
dataset_test.df = filter_dataset(dataset_test.df)

results = knn_model.predict(dataset_test.df.values, ids, "Transported").astype(bool)


final = ids.merge(results, left_index=True, right_index=True)

final.to_csv("results.csv", index=False)