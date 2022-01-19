import tensorflow as tf
import tensorflow.keras as keras

class Residual(keras.layers.Layer) : 
    def __init__(self, filters = 32, strides = 1, use_1x1_conv=True) :
        super(Residual, self).__init__()
        self.use_1x1_conv = use_1x1_conv
        self.conv1 = keras.layers.Conv2D(filters, padding ='same', kernel_size = 3, strides = strides)
        self.conv2 = keras.layers.Conv2D(filters, padding ='same', kernel_size = 3)
        self.conv3 = None

        if use_1x1_conv : 
            self.conv3 = keras.layers.Conv2D(filters, kernel_size=1, strides = strides)
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, X) : 
        Y = self.conv1(X)
        Y = self.bn1(Y)
        Y = keras.activations.relu(Y)
        Y = self.conv2(Y)
        Y = self.bn2(Y)

        if self.conv3 is not None : 
            X = self.conv3(X)
        Y+=X

        return keras.activations.relu(Y)

class ResBlock(keras.layers.Layer) : 
    def __init__(self, channels, num_residuals, first_block = False, **kwargs) : 
        super(ResBlock, self).__init__(**kwargs)
        self.residuals = list()

        for i in range(num_residuals) : 
                if i == 0 and not first_block : 
                    self.residuals.append( Residual(filters=channels, strides = 2, use_1x1_conv = True) )
                else :
                    self.residuals.append( Residual(filters=channels, strides = 1 ) )
    def call(self, X) : 
        for layer in self.residuals.layers : 
            X = layer(X)
        return X

class ResNet18(keras.models.Model) :
    def __init__(self, num_classes : int, **kwargs) :
        super(ResNet18, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.relu1 = keras.layers.Activation('relu')
        self.max_pool1 = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.resblocks = [
            ResBlock(64, 2, first_block=True),
            ResBlock(128, 2),
            ResBlock(256, 2),
            ResBlock(512, 2)
        ]
        self.gap = keras.layers.GlobalAveragePooling2D()
        self.classifier = keras.layers.Dense(units = num_classes)

    def call(self, X) : 
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        X = self.max_pool1(X)

        for block in self.resblocks : 
            X = block(X)

        X = self.gap(X)
        X = self.classifier(X)
        X = keras.activations.softmax(X)
        return X

if __name__ =='__main__' : 
    X = tf.random.uniform(shape=(1, 224, 224, 1))
    for layer in ResNet18(10).layers:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)