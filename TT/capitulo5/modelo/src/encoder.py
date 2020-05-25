def get_positional_encoding_2d(height, width, d_model):
    
    positional_encodings = np.zeros((height, width, d_model))
    d_model = int(d_model / 2)
    h_vector = np.arange(height)
    w_vector = np.expand_dims(np.arange(width), axis=1)

    div_term = np.arange(d_model) // 2
    div_term = np.exp((-2*div_term / d_model) * np.log(10000.0))
    
    positional_encodings[:, :, 0:d_model:2] = np.sin((positional_encodings[:, :, 0:d_model:2] + div_term[0::2]) * w_vector)
    positional_encodings[:, :, 1:d_model:2] = np.cos((positional_encodings[:, :, 1:d_model:2] + div_term[1::2]) * w_vector)
    positional_encodings[:, :, d_model::2] = np.sin((positional_encodings[:, :, d_model::2] + div_term[0::2]) * h_vector[:, np.newaxis, np.newaxis])
    positional_encodings[:, :, d_model+1::2] = np.cos((positional_encodings[:, :, d_model+1::2] + div_term[0::2]) * h_vector[:, np.newaxis, np.newaxis])

    return positional_encodings


class Encoder(tf.keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self.pooling1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')
        self.pooling2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')
        self.pooling3 = tf.keras.layers.MaxPool2D(pool_size=(2,1), strides=(2,1), padding='same')
        self.pooling4 = tf.keras.layers.MaxPool2D(pool_size=(1,2), strides=(1,2), padding='same')

        self.conv1 = tf.keras.layers.Conv2D(64,  (3,3), (1,1), 'same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(128, (3,3), (1,1), 'same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(256, (3,3), (1,1), 'same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(256, (3,3), (1,1), 'same', activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(512, (3,3), (1,1), 'same', activation='relu')
        self.conv6 = tf.keras.layers.Conv2D(512, (3,3), (1,1), 'valid', activation='relu')

        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.batch_normalization3 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        
        y = self.conv1(x)
        y = self.pooling1(y)
        y = self.conv2(y)
        y = self.pooling2(y)
        y = self.conv3(y)
        y = self.batch_normalization1(y, training)
        y = self.conv4(y)
        y = self.pooling3(y)
        y = self.conv5(y)
        y = self.batch_normalization2(y, training)
        y = self.pooling4(y)
        y = self.conv6(y)
        y = self.batch_normalization3(y, training)

        y += get_positional_encoding_2d(y.shape[1], y.shape[2], y.shape[3])

        y = tf.reshape(y, (y.shape[0], -1, y.shape[3]))

        return y