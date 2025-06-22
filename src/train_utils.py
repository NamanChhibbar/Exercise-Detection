import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .pose_utils import LandmarkClassifier


def create_datasets(
  x: list[np.ndarray],
  y: list[int],
  test_size: float = .2,
  val_size: float = .2,
  seed: int | None = None
  ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
  '''
  Create train, validation, and test datasets from the given data.

  Parameters:
    x (list[np.ndarray]): List of input data (landmarks).
    y (list[int]): List of labels corresponding to the input data.
    test_size (float): Proportion of the dataset to include in the test split.
    val_size (float): Proportion of the training data to include in the validation split.
    seed (int | None): Random seed for reproducibility.

  Returns:
    tuple: train_ds, val_ds, test_ds
  '''
  # Model input and output specifications
  input_spec = tf.TensorSpec(shape=(None, 99), dtype=tf.float32)
  output_spec = tf.TensorSpec(shape=(), dtype=tf.int32)
  # Split into train + validation and test
  x_trainval, x_test, y_trainval, y_test = train_test_split(
    x, y,
    test_size=test_size,
    random_state=seed
  )
  # Split into train and validation
  x_train, x_val, y_train, y_val = train_test_split(
    x_trainval, y_trainval,
    test_size=val_size,
    random_state=seed
  )
  # Train dataset
  train_ds = tf.data.Dataset.from_generator(
    lambda: ((x, y) for x, y in zip(x_train, y_train)),
    output_signature=(input_spec, output_spec)
  ).shuffle(len(x_train)).batch(1).prefetch(tf.data.AUTOTUNE)
  # Validation dataset
  val_ds = tf.data.Dataset.from_generator(
    lambda: ((x, y) for x, y in zip(x_val, y_val)),
    output_signature=(input_spec, output_spec)
  ).batch(1).prefetch(tf.data.AUTOTUNE)
  # Test dataset
  test_ds = tf.data.Dataset.from_generator(
    lambda: ((x, y) for x, y in zip(x_test, y_test)),
    output_signature=(input_spec, output_spec)
  ).batch(1).prefetch(tf.data.AUTOTUNE)
  return train_ds, val_ds, test_ds


def train_evaluate(
  model: LandmarkClassifier,
  train_ds: tf.data.Dataset,
  val_ds: tf.data.Dataset,
  test_ds: tf.data.Dataset,
  epochs: int = 10,
  learning_rate: float = 0.001,
  save_path: str = 'classifier.keras'
) -> tuple[dict, float, float]:
  '''
  Trains and evaluates the given model on the provided datasets.
  Saves the best model during training based on validation loss.

  Parameters:
    model (LandmarkClassifier): The model to train.
    train_ds (tf.data.Dataset): Training dataset.
    val_ds (tf.data.Dataset): Validation dataset.
    test_ds (tf.data.Dataset): Test dataset.
    epochs (int): Number of epochs to train for.
    learning_rate (float): Learning rate for the optimizer.
    save_path (str): Path to save the trained model. Must be of type ".keras".

  Returns:
    tuple: history, test_loss, test_accuracy
  '''
  # Set the model to training mode
  model.training(True)
  # Compile the model with optimizer, loss function, and metrics
  model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
  )
  # Train the model with the training and validation datasets
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[
      # Reduce learning rate on plateau scheduler
      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
      # Early stopping to prevent overfitting
      tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
      # Model checkpoint to save the best model
      tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_best_only=True, monitor='val_loss')
    ]
  )
  # Evaluate the model on the test dataset
  test_loss, test_accuracy = model.evaluate(test_ds)
  # Set the model back to inference mode
  model.training(False)
  return history.history, test_loss, test_accuracy
