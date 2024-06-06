import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Reshape
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.utils import plot_model  # Add this import

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

import warnings
warnings.filterwarnings("ignore")

# Define custom Fuzzy Logic layer
class FuzzyLayer(layers.Layer):
    def __init__(self, num_membership_functions, num_rules, **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.num_membership_functions = num_membership_functions
        self.num_rules = num_rules

    def build(self, input_shape):
        self.membership_functions = self.add_weight(shape=(input_shape[-1], self.num_membership_functions),
                                                     initializer='random_normal',
                                                     trainable=True,
                                                     name='membership_functions')
        self.rule_weights = self.add_weight(shape=(self.num_membership_functions, self.num_rules),
                                                     initializer='random_normal',
                                                     trainable=True,
                                                     name='rule_weights')
        super(FuzzyLayer, self).build(input_shape)

    def call(self, inputs):
        # Apply fuzzy logic computation
        fuzzy_output = tf.matmul(inputs, self.membership_functions)
        fuzzy_output = tf.nn.softmax(fuzzy_output, axis=-1)  # Softmax for normalization
        
        # Rule evaluation: Apply rule weights to membership degrees
        rule_outputs = tf.matmul(fuzzy_output, self.rule_weights)
        
        # Aggregation: Max aggregation for Mamdani
        aggregated_output = tf.reduce_max(rule_outputs, axis=-1)
        
        return aggregated_output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_rules,)

# Define the 1D CNN with Fuzzy Logic
def create_cnn_with_fuzzy_logic(input_shape, num_classes, num_membership_functions, num_rules):
    inp = Input(shape=input_shape)
    # Adding three convolutional layers
    fuzzy_layers = layers.Conv1D(32, 3, activation='relu')(inp)
    fuzzy_layers = layers.Conv1D(64, 3, activation='relu')(fuzzy_layers)
    fuzzy_layers = layers.Conv1D(128, 3, activation='relu')(fuzzy_layers)
    fuzzy_layers = layers.MaxPooling1D(2)(fuzzy_layers)
    
    # Fuzzy layer added after third convolution
    fuzzy_layers = FuzzyLayer(num_membership_functions, num_rules)(fuzzy_layers)
    fuzzy_layers = Reshape((-1, 1))(fuzzy_layers)  # Reshape to 2D tensor
    
    # Original architecture
    fuzzy_layers = layers.Conv1D(256, 3, activation='relu')(fuzzy_layers)
    fuzzy_layers = layers.MaxPooling1D(2)(fuzzy_layers)
    fuzzy_layers = layers.Flatten()(fuzzy_layers)
    fuzzy_layers = Dense(64, activation='relu')(fuzzy_layers)
    output = Dense(num_classes, activation='softmax')(fuzzy_layers)
    
    model = models.Model(inputs=inp, outputs=output)
    return model

# Load data
df_train = pd.read_csv("eeg3.csv", header=None)
idx = df_train.shape[1] - 1
Y = np.array(df_train[idx].values).astype(np.int8) - 1
X = np.array(df_train[list(range(idx - 1))].values)[..., np.newaxis]

# Determine the number of classes
num_classes = len(np.unique(Y))

# Define input shape and parameters
input_shape = (X.shape[1], 1)  # Adjust input shape
num_membership_functions = 3
num_rules = 3

# Data Augmentation
def augment_data(X, Y):
    # Random time shifting
    shift = np.random.randint(0, X.shape[1])
    X_augmented = np.roll(X, shift=shift, axis=1)
    return X_augmented, Y

# Create the CNN model with Fuzzy Logic
model = create_cnn_with_fuzzy_logic(input_shape, num_classes, num_membership_functions, num_rules)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display the model architecture
model.summary()

# Define callbacks
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=25, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=25, verbose=2)
callbacks_list = [early, redonplat]  # early

# Train the model
history= model.fit(X, Y, epochs=25, verbose=1, callbacks=callbacks_list, validation_split=0.2)

def plot_metrics(history):
    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.show()

    # Accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()

plot_metrics(history)

# Plot model architecture
plot_model(model, show_shapes=True, expand_nested=True)