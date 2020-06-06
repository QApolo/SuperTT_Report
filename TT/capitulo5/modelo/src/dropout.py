class CNN_Layer(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, padding, apply_batch_normalization=False, apply_dropout=False, dropout_rate=0.5):
        super(CNN_Layer, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, 
		                                   strides=(1,1), padding=padding, use_bias=False)
        
        self.batch_normalization = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.apply_batch_normalization = apply_batch_normalization
        self.apply_dropout = apply_dropout

    def __call__(self, x, training=False):
        
        y = self.conv(x)
        if self.apply_batch_normalization:
          y = self.batch_normalization(y, training)
        y = tf.nn.relu(y)
        if self.apply_dropout:
          y = self.dropout(y, training)

        return y
        

class CNN_Encoder(tf.keras.Model):

    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        
        self.pooling1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
        self.pooling2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')
        self.pooling3 = tf.keras.layers.MaxPool2D(pool_size=(2,1), strides=(2,1), padding='same')
        self.pooling4 = tf.keras.layers.MaxPool2D(pool_size=(1,2), strides=(1,2), padding='same')

        self.conv1 = CNN_Layer(64, (3,3), 'valid', apply_dropout=True, dropout_rate=0.2)
        self.conv2 = CNN_Layer(128, (3,3), 'valid')
        self.conv3 = CNN_Layer(256, (3,3), 'valid', apply_batch_normalization=True)
        self.conv4 = CNN_Layer(256, (3,3), 'valid', apply_dropout=True)
        self.conv5 = CNN_Layer(512, (3,3), 'valid', apply_batch_normalization=True, apply_dropout=True)
        self.conv6 = CNN_Layer(512, (3,3), 'same', apply_batch_normalization=True)

    def call(self, x, training=False):
        
        y = self.conv1(x, training)
        y = self.pooling1(y)
        y = self.conv2(y, training)
        y = self.pooling2(y)
        y = self.conv3(y, training)
        y = self.conv4(y, training)
        y = self.pooling3(y)
        y = self.conv5(y, training)
        y = self.pooling4(y)
        y = self.conv6(y, training)

        y += get_positional_encoding_2d(y.shape[1], y.shape[2], y.shape[3])
        y = tf.reshape(y, (y.shape[0], -1, y.shape[3]))

        return y