import os

import numpy as np
import tensorflow as tf


class SequenceClassifier(tf.keras.Model):
  '''
  An Energy-Based Model (EBM) for classifying sequences.
  Uses stacked LSTM layers followed by a feedforward network (stacked dense layers) to classify sequences.
  Uses energy thresholding to determine if an input is out-of-distribution (OOD).
  The model outputs the class index for in-distribution inputs and -1 for OOD inputs.
  '''

  def __init__(
    self,
    stacked_lstm_units: list[int],
    ffn_layer_sizes: list[int],
    num_classes: int,
    input_dimension: int,
    activation: str = 'relu',
    temperature: float = 1.0,
    threshold: float = float('inf'),
    class_names: list[str] | None = None,
    **kwargs
  ) -> None:
    '''
    Parameters:
      stacked_lstm_units (list[int]): List of integers containing the number of units in the stacked LSTM layer.
      ffn_layer_sizes (list[int]): List of integers containing the sizes of dense layers in the FFN.
      num_classes (int): Number of output classes.
      input_dimension (int): Dimension of input vectors in sequences.
      activation (str = 'relu'): Activation function to use in feedforward layers.
      temperature (float = 1.0): Temperature parameter for energy calculation.
      threshold (float = inf): Threshold for energy to determine if an input is out-of-distribution is valid.
      class_names (list[str] | None = None): List of class names corresponding to the output classes.
      **kwargs: Additional keyword arguments for super class initialization.
    '''
    super().__init__(**kwargs)
    self.__config = {
      'stacked_lstm_units': stacked_lstm_units,
      'ffn_layer_sizes': ffn_layer_sizes,
      'num_classes': num_classes,
      'input_dimension': input_dimension,
      'activation': activation,
      'temperature': temperature,
      'threshold': threshold,
      'class_names': class_names,
      **kwargs
    }
    if class_names is not None:
      # Check if length class_names matches num_classes
      if len(class_names) != num_classes:
        raise ValueError(f'Number of class names ({len(class_names)}) does not match number of classes ({num_classes}).')
      # Add 'Other' to class names for out-of-distribution detection
      class_names.append('Other')
    # Stacked LSTM layer
    self.stacked_lstm = tf.keras.Sequential([
      tf.keras.layers.LSTM(units=size, return_sequences=True)
      for size in stacked_lstm_units[:-1]
    ])
    # Last LSTM layer without return sequences
    self.stacked_lstm.add(tf.keras.layers.LSTM(units=stacked_lstm_units[-1], return_sequences=False))
    # Feedforward network (list of Dense layers)
    self.ffn = tf.keras.Sequential([
      tf.keras.layers.Dense(units=size, activation=activation)
      for size in ffn_layer_sizes
    ])
    # Output layer
    self.output_layer = tf.keras.layers.Dense(units=num_classes)
    # Build the layers with shape of landmark vectors
    super().build((None, None, input_dimension))
    self.stacked_lstm.build((None, None, input_dimension))
    self.ffn.build(self.stacked_lstm.output_shape)
    self.output_layer.build(self.ffn.output_shape)

  def get_config(self) -> dict[str, any]:
    '''
    Returns the configuration of the model as a dictionary.
    This is used to automatically save the model configuration while saving the model.
    '''
    return self.__config

  @classmethod
  def from_config(cls, config: dict[str, any]) -> 'SequenceClassifier':
    '''
    Creates an instance from a configuration dictionary.
    This is used to automatically load the model configuration while loading the model.
    '''
    return cls(**config)

  @staticmethod
  def load_model(model_path: str) -> 'SequenceClassifier':
    '''
    Loads a saved model from a given path.
    The model is expected to be saved in the TensorFlow SavedModel format.
    '''
    if not os.path.exists(model_path):
      raise FileNotFoundError(f'Model file {model_path} does not exist.')
    return tf.keras.models.load_model(model_path, custom_objects={'SequenceClassifier': SequenceClassifier})

  def set_temperature(self, temperature: float) -> None:
    '''
    Sets the temperature for energy calculation.
    Warning: This will affect the energy calculation and you may need to call `set_threshold` again.

    Parameters:
      temperature (float): The temperature value to set.
    '''
    self.__config['temperature'] = temperature

  def energy(self, logits: tf.Tensor) -> tf.Tensor:
    '''
    Calculates energy of given logits.
    '''
    temperature = self.__config['temperature']
    return -temperature * tf.reduce_logsumexp(logits / temperature, axis=-1)

  def set_threshold(
    self,
    data: list[tf.Tensor | np.ndarray],
    rejection_rate: float = 0.05
  ) -> float:
    '''
    Sets the energy threshold for out-of-distribution detection from the given in-distribution data.

    Parameters:
      data (list[tf.Tensor]): A dataset containing samples to compute the threshold.
      rejection_rate (float = 0.05): The proportion of samples to reject as out-of-distribution.
    '''
    # Calculate energies for the given data
    energies = []
    for sample in data:
      # Convert sample to tensor if it's not already
      tensor_input = tf.convert_to_tensor(sample, dtype=tf.float32)[tf.newaxis, ...]
      # Get logits
      logits = self(tensor_input, return_logits=True)
      # Calculate energy and append to the list
      energy = self.energy(logits)[0].numpy()
      energies.append(energy)
    # Calculate the threshold based on the rejection rate
    threshold = np.quantile(energies, 1 - rejection_rate)
    self.__config['threshold'] = threshold
    return threshold

  def call(self, x: tf.Tensor, return_logits: bool = False) -> tf.Tensor:
    '''
    Forward pass of the model.
    
    Parameters:
      x (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_features).
      return_logits (bool): If True, returns logits instead of softmax probabilities.
    Returns:
      tf.Tensor: Output tensor of shape (batch_size, num_classes) containing class probabilities or logits.
    '''
    x = self.stacked_lstm(x)
    x = self.ffn(x)
    logits = self.output_layer(x)
    if return_logits:
      return logits
    return tf.nn.softmax(logits, axis=-1)
  
  def predict(self, x: tf.Tensor) -> list[int] | list[str]:
    '''
    Predicts the class index (or class name if class_names is not None) for the given input sequence.
    If the input is out-of-distribution, it returns -1 (or "Other" if class_names is not None).

    Parameters:
      x (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_features).

    Returns:
      (list[int] | list[str]): Predicted class indices or class names.
    '''
    logits = self.call(x, return_logits=True)
    energies = self.energy(logits)
    classes = tf.argmax(logits, axis=-1, output_type=tf.int32)
    # Set classes to -1 if energy exceeds threshold
    threshold = self.__config['threshold']
    classes = tf.where(energies > threshold, -1, classes)
    class_names = self.__config.get('class_names')
    if class_names is not None:
      # Convert class indices to class names
      return [class_names[i] for i in classes]
    return classes.numpy().tolist()
