from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

seperator = f"\n\n{'--' * 50}\n\n"

class ConvolutionalNeuralNetwork:
  """Implementing CNN on MNIST Dataset"""

  def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

  def load_data(self):
      """Loading Dataset"""

      try:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        print("Dataset Loaded Successfully", end = seperator)

      except Exception as error:
        print(error)

  def preprocess_data(self):
      """Preprocessing Data"""

      self.load_data()

      try:
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

        self.X_train = self.X_train.reshape(-1, 28, 28, 1)
        self.X_test = self.X_test.reshape(-1, 28, 28, 1)

        print("Train Shape : ", self.X_train.shape)
        print("Test Shape : ", self.X_test.shape, end = seperator)

      except Exception as error:
        print(error)

  def build_model(self):
      """Build and Compile model"""

      self.preprocess_data()

      try:
        self.model = Sequential([
            Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation = 'relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation = 'relu'),
            Dense(10, activation = 'softmax')
        ])

        self.model.compile(
            optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )
        print(self.model.summary(), end = seperator)

      except Exception as error:
        print(error)

  def train_model(self):
      """Training Model"""

      self.build_model()

      try:
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs = 5,
            validation_data = (self.X_test, self.y_test)
        )

      except Exception as error:
        print(error)

  def evaluate_model(self):
      """Evaluate Model"""

      self.train_model()

      try:
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print("Test Accuracy : ", accuracy, end = seperator)

      except Exception as error:
        print(error)

if __name__ == "__main__":
    cnn = ConvolutionalNeuralNetwork()
    cnn.evaluate_model()