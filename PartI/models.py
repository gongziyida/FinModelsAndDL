import tensorflow as tf


# A feedforward model that takes (z, k) as input and outputs (I, V)
class SharedLayerNN(tf.keras.Model):
  def __init__(self, hidden_dim, delta):
    super().__init__()
    self.delta = delta

    self.concat = tf.keras.layers.Concatenate(axis=-1)
    self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
    self.dense2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
    self.dense3 = tf.keras.layers.Dense(2, activation='linear')

  def call(self, z, k):
    x = self.concat([z[:,None], tf.math.log(k[:,None])])
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)
    # I, v = (tf.sigmoid(x[:,0]) * 2 - 1 + self.delta) * k, x[:,1]
    I, v = (tf.exp(x[:,0]) - 1 + self.delta) * k, x[:,1]
    return I, v 

class SepLayerNN(tf.keras.Model):
  def __init__(self, hidden_dim, delta):
    super().__init__()
    self.delta = delta

    self.concat = tf.keras.layers.Concatenate(axis=-1)
    
    self.dense_I1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
    self.dense_I2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
    self.dense_I3 = tf.keras.layers.Dense(1, activation='linear')

    self.dense_v1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
    self.dense_v2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
    self.dense_v3 = tf.keras.layers.Dense(1, activation='linear')

  def call(self, z, k):
    x = self.concat([z[:,None], tf.math.log(k[:,None])])
    x_I = self.dense_I1(x)
    x_I = self.dense_I2(x_I)
    x_I = tf.squeeze(self.dense_I3(x_I), axis=-1)
    # I = (tf.sigmoid(x_I) * 2 - 1 + self.delta) * k
    I = (tf.exp(x_I) - 1 + self.delta) * k

    x_v = self.dense_v1(x)
    x_v = self.dense_v2(x_v)
    v = tf.squeeze(self.dense_v3(x_v), axis=-1)

    return I, v