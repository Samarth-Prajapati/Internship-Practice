import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

sentiment_analysis = {
    'Review': [
        'I love this product',
        'This is amazing',
        'Very bad experience',
        'I hate this item',
        'Excellent quality',
        'Worst purchase ever',
        'Really happy with this',
        'Not good at all',
        'Superb performance',
        'Terrible service'
    ],
    'Sentiment': [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
}

seperator = f"\n\n{'--' * 50}\n\n"

class SentimentLSTM:
    """Implementation of LSTM RNN"""

    def __init__(self, data):
        self.data = data
        self.df = None
        self.texts = None
        self.X = None
        self.y = None
        self.sequences = None
        self.tokenizer = Tokenizer()
        self.model = None
        self.max_len = 5

    def load_data(self):
        """Loading data to Dataframe"""

        try:
          self.df = pd.DataFrame(self.data)
          print(self.data, end = seperator)

        except Exception as error:
          print(error)

    def preprocess_data(self):
        """Preprocess data to vector embeddings"""

        self.load_data()

        try:
          self.texts = self.data['Review']
          self.y = self.data['Sentiment']

          # tokenizer converts a dict of vocabulary where all unique words are assigned a key.
          self.tokenizer.fit_on_texts(self.texts)

          # this creates a list of sequence with tokens of sentence
          self.sequences = self.tokenizer.texts_to_sequences(self.texts)

          # this adds paddings to the sequence by adding 0 in prefix
          self.X = pad_sequences(
              self.sequences,
              maxlen = self.max_len
          )

          print("Tokenized Data : ")
          print(self.X, end = seperator)

        except Exception as error:
          print(error)

    def build_model(self):
        """Build and Compile LSTM model"""

        self.preprocess_data()

        try:
          vocab_size = len(self.tokenizer.word_index) + 1

          self.model = Sequential()
          self.model.add(Embedding(
              input_dim = vocab_size,
              output_dim = 8,
              input_length = self.max_len
          ))
          self.model.add(LSTM(16))
          self.model.add(Dense(
              1,
              activation = 'sigmoid'
          ))

          self.model.compile(
              optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy']
          )
          print("Model Built Successfully", end = seperator)

        except Exception as error:
          print(error)

    def train_model(self):
        """Training model"""

        self.build_model()

        try:
          self.model.fit(
              self.X,
              self.y,
              epochs = 20,
              verbose = 1
          )
          print("Model trained successfully", end = seperator)

        except Exception as error:
          print(error)

    def predict(self):
        """Predict and test model"""

        self.train_model()

        try:
          test_text = ['I really love this']

          seq = self.tokenizer.texts_to_sequences(test_text)
          padded = pad_sequences(
              seq,
              maxlen = self.max_len
          )
          prediction = self.model.predict(padded)

          print("Prediction : ")
          print(prediction)

          if prediction > 0.5:
              print("Positive Sentiment", end = seperator)
          else:
              print("Negative Sentiment", end = seperator)

        except Exception as error:
          print(error)

lstm = SentimentLSTM(sentiment_analysis)
lstm.predict()