class InitialHidden(tf.keras.Model):

  def __init__(self, units):
    super(InitialHidden, self).__init__()
    
    self.fc = tf.keras.layers.Dense(units, activation='tanh')

  def __call__(self, x):

    x = tf.math.reduce_mean(x, axis=1)
    return self.fc(x)


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(Encoder output) shape == (batch_size, L, 512)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, L, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, L, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class Decoder(tf.keras.Model):

  def __init__(self, units, embedding_dim, vocab_size):
    super(Decoder, self).__init__()
    
    self.Wout = tf.keras.layers.Dense(vocab_size, use_bias=False)
    self.o_t = tf.keras.layers.Dense(units, use_bias=False, activation='tanh')
    self.lstm = tf.keras.layers.LSTM(units, return_state=True)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    self.attention = BahdanauAttention(units)
  

  def __call__(self, features, hidden, previous_y, previous_out):

    # embedding_previous_y shape == (batch_size, embedding_dim)
    embedding_previous_y = self.embedding(previous_y)

    # previous_out shape == (batch_size, units)
    # lstm_input shape == (batch_sise, embedding_dim + units)
    lstm_input = tf.concat([embedding_previous_y, previous_out], axis=1)
    lstm_input = tf.expand_dims(lstm_input, axis=1)

    # hidden is a list of hidden state and carry state respectively
    # each element has a shape == (batch_size, units)
    # actually output and state are the same tensors
    output, state, carry = self.lstm(lstm_input, initial_state=hidden)

    # context_vector shape == (batch_size, hidden_size)
    # attention_weights == (batch_size, L, 1)
    context_vector, attention_weights = self.attention(features, state)

    # calculating the output
    out = self.o_t(tf.concat([output, context_vector], axis=1))
    output = self.Wout(out)

    return output, out, attention_weights, [state, carry]