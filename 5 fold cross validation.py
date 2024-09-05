New! Keyboard shortcuts … Drive keyboard shortcuts have been updated to give you first-letters navigation
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

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

def create_cnn_with_fuzzy_logic(input_shape, num_classes, num_membership_functions, num_rules):
    inp = Input(shape=input_shape)
    conv_layers = Conv1D(32, 3, activation='relu')(inp)
    conv_layers = MaxPooling1D(2)(conv_layers)
    conv_layers = Conv1D(64, 3, activation='relu')(conv_layers)
    conv_layers = MaxPooling1D(2)(conv_layers)
    conv_layers = Flatten()(conv_layers)
    fuzzy_logic_layer = FuzzyLayer(num_membership_functions, num_rules)(conv_layers)
    reshaped = Reshape((-1, 1))(fuzzy_logic_layer)  # Reshape to 2D tensor
    dense_layer = Dense(64, activation='relu')(reshaped)
    output = Dense(num_classes, activation='softmax')(dense_layer)

    model = models.Model(inputs=inp, outputs=output)
    return model

# Load data
df_train = pd.read_csv("eeg3.csv", header=None)
idx = df_train.shape[1] - 1
Y = np.array(df_train.iloc[:, idx].values).astype(np.int8) - 1
X = np.array(df_train.iloc[:, :idx].values)[..., np.newaxis]

# Determine the number of classes
num_classes = len(np.unique(Y))

# Define input shape and number of membership functions and rules for fuzzy layer
input_shape = (X.shape[1], 1)
num_membership_functions = 3
num_rules = 3

# Define model and compile
model = create_cnn_with_fuzzy_logic(input_shape, num_classes, num_membership_functions, num_rules)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model architecture
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=25, verbose=1)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=15, verbose=2)
callbacks_list = [early_stopping, reduce_lr_on_plateau]

# Stratified K-Folds cross-validator
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Cross-validation loop
fold_no = 1
scores = []
histories = []

for train_index, test_index in kfold.split(X, Y):
    print(f"Training on fold {fold_no}...")
    # Generate batches from indices
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = Y[train_index], Y[test_index]

   # Fit data to model
    history = model.fit(
        x_train_fold, y_train_fold,
        batch_size=32,
        epochs=50,  # Select appropriate number of epochs
        verbose=0,
        validation_data=(x_test_fold, y_test_fold),
        callbacks=callbacks_list
    )

    # Append the history to the histories list for later use.
    histories.append(history.history)

    # Generate generalization metrics
    predictions = model.predict(x_test_fold)
    y_pred = np.argmax(predictions, axis=-1)
    y_true = y_test_fold

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Fold {fold_no} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")
    fold_no += 1

    scores.append((accuracy, precision, recall, f1))

# Displaying the mean scores across all folds
mean_scores = np.mean(scores, axis=0)
print(f"Mean scores: Accuracy={mean_scores[0]}, Precision={mean_scores[1]}, Recall={mean_scores[2]}, F1 Score={mean_scores[3]}")

# Plotting the learning curves
for fold_number, history in enumerate(histories, 1):
    plt.figure(figsize=(12, 5))

    # Subplot for the accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy for Fold {fold_number}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Subplot for the loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss for Fold {fold_number}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    # Plotting the mean scores
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
plt.bar(metrics, mean_scores)
plt.ylim(0, 1)  # Assuming your scores are between 0 and 1
plt.title('Mean Scores Across All Folds')
plt.ylabel('Scores')
for i, v in enumerate(mean_scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.show()