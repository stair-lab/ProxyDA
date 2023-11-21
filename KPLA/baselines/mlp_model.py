import tensorflow as tf


DEFAULT_LOSS = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def mlp(num_classes, width, input_shape, learning_rate,
        loss=DEFAULT_LOSS, metrics=[]):
  """Multilabel Classification."""
  model_input = tf.keras.Input(shape=input_shape)
  # hidden layer
  if width:
    x = tf.keras.layers.Dense(
        width, use_bias=True, activation='relu'
    )(model_input)
  else:
    x = model_input
  model_outuput = tf.keras.layers.Dense(num_classes,
                                        use_bias=True,
                                        activation="linear")(x)  # get logits
  opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
  model = tf.keras.models.Model(model_input, model_outuput)
  model.build(input_shape)
  model.compile(loss=loss, optimizer=opt, metrics=metrics)

  return model