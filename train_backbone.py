from pickletools import optimize
import tensorflow as tf
import tensorflow.keras as keras
from models.simple_backbone import ResNet18

model = ResNet18(100)
inputs = keras.layers.Input(shape=(224,224,3))
model.build(input_shape = (None,224,224,3))
model.compile(
	loss = 'categorical_crossentropy',
	metrics = keras.metrics.categorical_accuracy,
	optimizer = keras.optimizers.Adam()
)
model.summary()