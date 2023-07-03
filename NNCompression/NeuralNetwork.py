import numpy as np
import math
import random
import time
import tensorflow as tf
import os
from tensorflow.keras import mixed_precision

#Multiplo de 8.Divide el archivo en N lotes y los procesar� en paralelo.
batch_size = 512
#Multiplo de 8.El n�mero de unidades a usar dentro de cada capa LSTM.
rnn_units =  4
#Tama�o de la capa de incrustaci�n.
embedding_size=1024

def build_model(vocab_size):
  """Builds the model architecture.

    Args:
      vocab_size: Int, size of the vocabulary.
  """
  policy = mixed_precision.Policy('float16')
  mixed_precision.set_global_policy(policy)
  #Input layer
  inputs = []
  inputs.append(tf.keras.Input(batch_input_shape=[batch_size, 1]))
  inputs.append(tf.keras.Input(shape=(None,)))
  inputs.append(tf.keras.Input(shape=(None,)))
  embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)(inputs[0])
  #Hidden Layer
  predictions, state_h, state_c = tf.keras.layers.LSTM(rnn_units,
                          return_sequences=True,
                          return_state=True,
                          recurrent_initializer='glorot_uniform',
                          )(embedding, initial_state=[
                          tf.cast(inputs[1], tf.float16),
                          tf.cast(inputs[2], tf.float16)])

  layer_input = tf.slice(predictions, [0, 0, 0], [batch_size, 1, rnn_units])
  #Output Layer
  dense = tf.keras.layers.Dense(vocab_size, name='dense_logits')(layer_input)

  output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(dense)
  
  outputs = []
  outputs.append(output)
  outputs.append(state_h)
  outputs.append(state_c)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model

def get_symbol(index, length, freq, coder, compress, data):
  """Runs arithmetic coding and returns the next symbol.

  Args:
    index: Int, position of the symbol in the file.
    length: Int, size limit of the file.
    freq: ndarray, predicted symbol probabilities.
    coder: this is the arithmetic coder.
    compress: Boolean, True if compressing, False if decompressing.
    data: List containing each symbol in the file.
  
  Returns:
    The next symbol, or 0 if "index" is over the file size limit.
  """
  symbol = 0
  if index < length:
    if compress:
      symbol = data[index]
      coder.write(freq, symbol)
    else:
      symbol = coder.read(freq)
      data[index] = symbol
  return symbol

def train(pos, tensorInput, length, vocab_size, coder, model, optimizer, isCompressing, data, states):
  """Runs one training step.

  Args:
    pos: Int, position in the file for the current symbol for the *first* batch.
    tensorInput: Tensor, containing the last inputs for the model.
    length: Int, size limit of the file.
    vocab_size: Int, size of the vocabulary.
    coder: this is the arithmetic coder.
    model: the model to generate predictions.
    optimizer: optimizer used to train the model.
    isCompressing: Boolean, True if compressing, False if decompressing.
    data: List containing each symbol in the file.
    states: List containing state information for the layers of the model.
  
  Returns:
    tensorInput: Tensor, containing the last inputs for the model.
    cross_entropy: cross entropy numerator.
    denom: cross entropy denominator.
  """
  loss = cross_entropy = denom = 0
  split = math.ceil(length / batch_size)
  with tf.GradientTape() as tape:
    # Las entradas del modelo contienen tanto tensorInput como los estados de cada capa.
    inputs = states.pop(0)
    inputs.insert(0, tensorInput)
    # Ejecutamos el modelo para obtener las predicciones.
    outputs = model(inputs)
    predictions = outputs.pop(0)
    states.append(outputs)
    p = predictions.numpy()
    symbols = []
    # Cuando el �ltimo lote llega al final del archivo, comenzamos a darle "0" como entrada. Usamos una m�scara para evitar que esto influya.
    mask = []
    # Repasamos el lote ajustando la probabilidad de cada caracter en función de la predicción preparando la siguiente entrada.
    for i in range(batch_size):
      freq = np.cumsum(p[i][0] * 10000000 + 1)
      index = pos + 1 + i * split
      symbol = get_symbol(index, length, freq, coder, isCompressing, data)
      symbols.append(symbol)
      if index < length:
        prob = p[i][0][symbol]
        if prob <= 0:
          # Seteamos un valor peque�o para evitar errores.
          prob = 0.000001
        cross_entropy += math.log2(prob)
        denom += 1
        mask.append(1.0)
      else:
        mask.append(0.0)
    # "input_one_hot" se usar� tanto para la funci�n de p�rdida como para la siguiente entrada.
    input_one_hot = tf.expand_dims(tf.one_hot(symbols, vocab_size), 1)
    loss = tf.keras.losses.categorical_crossentropy(
        input_one_hot, predictions, from_logits=False) * tf.expand_dims(
            tf.convert_to_tensor(mask), 1)
    scaled_loss = optimizer.get_scaled_loss(loss)
    # Removemos la entrada mas antigua y concatenamos la nueva.
    tensorInput = tf.slice(tensorInput, [0, 1], [batch_size, 0])
    tensorInput = tf.concat([tensorInput, tf.expand_dims(symbols, 1)], 1)
  # Corremos backwards pass para actualizar los pesos del modelo.
  scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
  grads = optimizer.get_unscaled_gradients(scaled_gradients)
  capped_grads = [tf.clip_by_norm(grad, 4) for grad in grads]
  optimizer.apply_gradients(zip(capped_grads, model.trainable_variables))
  return (tensorInput, cross_entropy, denom)

def reset_seed():
  """Initializes various random seeds to help with determinism."""
  SEED = 1234
  os.environ['PYTHONHASHSEED']=str(SEED)
  random.seed(SEED)
  np.random.seed(SEED)
  tf.random.set_seed(SEED)

def process(isCompressing, length, vocab_size, coder, stringToCompress):
  """This runs compression/decompression.
  Args:
    isCompressing: Boolean, True if compressing, False if decompressing.
    length: Int, size limit of the file.
    vocab_size: Int, size of the vocabulary.
    coder: this is the arithmetic coder.
    stringToCompress: List containing each symbol in the file.
  """
  start = time.time()
  reset_seed()
  # Instanciamos red neuronal
  model = build_model(vocab_size = vocab_size)
  model.reset_states()

  # Dividimos el archivo en lotes de tama�o batch_size.
  split = math.ceil(length / batch_size)

  #Tasa de aprendizaje para Adam Optimizer (0.0005 por cada lote).
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      0.0005,
      split,
      0.0005,
      power=1.0)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, beta_1=0, beta_2=0.9999, epsilon=1e-5)
  optimizer = mixed_precision.LossScaleOptimizer(optimizer)

  # Usamos una distribuci�n uniforme para predecir el primer lote de s�mbolos.
  freq = np.cumsum(np.full(vocab_size, (1.0 / vocab_size)) * 10000000 + 1)
  # Construimos el primer lote para el entrenamiento.
  symbols = []
  for i in range(batch_size):
    symbols.append(get_symbol(i*split, length, freq, coder, isCompressing, stringToCompress))
  tensorInput = tf.expand_dims(symbols, 1)
  pos = cross_entropy = denom = 0
  template = 'Tiempo de procesamiento: {:0.2f} segs'
  # Inicializamos con ceros el estado inicial de la red neuronal.
  states = []
  states.append([tf.zeros([batch_size, rnn_units])] * 2)
  # entrenamos la red repitiendo el entrenamiento por cada lote hasta llegar al final del archivo.
  # a medida avanza el procesamiento de los lotes la red mejora su conocimiento sobre el archivo
  # aumentando su % de compresión. Podríamos hacer lotes mas chicos mejorando la tasa de compresión
  # pero aumentaría el consumo de hardware.
  while pos < split:
    tensorInput, ce, d = train(pos, tensorInput, length, vocab_size, coder, model, optimizer, isCompressing, stringToCompress, states)
    cross_entropy += ce
    denom += d
    pos += 1
  if isCompressing:
    coder.finish()
  print(template.format(time.time() - start))

