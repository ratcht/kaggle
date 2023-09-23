from sklearn.neighbors import KNeighborsClassifier

class KNN:
  def __init__(self, x_train, y_train, n_neighbors=5):
    self.x_train = x_train
    self.y_train = y_train

    self.neigh_model = KNeighborsClassifier(n_neighbors=n_neighbors)
  
  def fit(self):
    self.neigh_model.fit(self.x_train, self.y_train)

  def score(self, x_test, y_test) -> int:
    return self.neigh_model.score(x_test, y_test)
  
  def predict(self, x):
    predicted = self.neigh_model.predict(x)

    print(predicted)