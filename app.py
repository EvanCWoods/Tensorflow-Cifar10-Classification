import tensorflow as tf
import matplotlib.pyplot as plt
import kerastuner
from kerastuner.tuners import RandomSearch

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()

train_data = train_data / 255
test_data = test_data / 255
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

def create_model(hp):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=train_data.shape[1:]))

    for i in range(hp.Int('Conv layers', min_value=1, max_value=5)):
        model.add(tf.keras.layers.Conv2D(hp.Choice(f'layers {i} filters', [16,32,64,128]), kernel_size=3, activation='relu'))

    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Flatten())

    for i in range(hp.Int('Dense layers', min_value=1, max_value=5)):
        model.add(tf.keras.layers.Dense(hp.Choice(f'layers {i}, units', [16,32,64,128]), activation='relu'))
    
    model.add(tf.keras.layers.Dropout(rate=hp.Choice('dropout', [0.1,0.2,0.3,0.4,0.5])))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(lr=hp.Choice('learning rate', [1e-2,1e-3,1e-4])),
                metrics=['accuracy'])
    
    return model


tuner=RandomSearch(
    create_model,
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=1,
    directory='trials_dir',
    project_name='cifar10_tests'
)

tuner.search_space_summary()

tuner.search(train_data, train_labels, epochs=5, validation_data=(test_data, test_labels))


model.save('/saved_model')
