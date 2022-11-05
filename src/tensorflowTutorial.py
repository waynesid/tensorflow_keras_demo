import tensorflow as tf

print(f'TF version {tf.version.VERSION}')

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f'Xtrain shape {x_train.shape}')
print(f'ytrain shape {y_train.shape}')
print(f'Xtest shape {x_test.shape}')
print(f'ytest shape {y_test.shape}')

x_train = x_train/255
x_test = x_test/255
print(f'Xtrain shape {x_train.shape}')
print(f'Xtest shape {x_test.shape}')

onetensor = x_train[4:5]
print(f'onetensor {onetensor}')

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy']
)

model.summary()

model.fit(x_train, y_train, epochs=3)

model.evaluate(x_test, y_test, verbose=2)

model.save('./models/mnist_model.model')

