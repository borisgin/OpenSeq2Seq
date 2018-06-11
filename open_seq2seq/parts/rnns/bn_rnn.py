"""
Custom RNN Cell definition.
Default RNNCell in TensorFlow throws errors when
variables are re-used between devices.
"""
import tensorflow as tf

from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.python.util import nest
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

class BatchNormRNNCell(BasicRNNCell):
    """
    RNN Cell with sequential batch normalization:
       output = new_state = activation(BN(W * input) + U * state + B).
           state dim: batch_size * num_units
           input dim: batch_size * feature_size
           W: feature_size * num_units
           U: num_units * num_units
    """

    def __init__(self,
                 num_units,
                 #input_size=None,
                 reuse= None,
                 name=None,
                 activation=tf.nn.relu,
                 momentum=0.9,
                 epsilon=0.001,
                 *args,
                 **kwargs
    ):
      super(BasicRNNCell, self).__init__(_reuse=reuse, name=name)
      self._num_units = num_units
      self._activation = activation
      self._name = name
      self._momentum=momentum
      self._epsilon = epsilon
      self._training = kwargs["training"]

    def build(self, scope, input_size,num_units, training):
      num_units=self._num_units
      if training:
        reuse=tf.AUTO_REUSE
      else:
        reuse = True

      with tf.variable_scope(scope or "bn_rnn", reuse=reuse):
          mean = tf.get_variable(name='mean',
                                 shape=[num_units],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer(),
                                 regularizer=None,
                                 trainable=False)
          variance = tf.get_variable(name='variance',
                                 shape=[num_units],
                                 dtype=tf.float32,
                                 initializer=tf.ones_initializer(),
                                 trainable=False)

          offset = tf.get_variable(name='offset',
                                 shape=[num_units],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)

          scale = tf.get_variable(name='scale',
                                 shape=[num_units],
                                 dtype=tf.float32,
                                 initializer=tf.ones_initializer(),
                                 trainable=True)

      return mean, variance, offset, scale



    def __call__(self,
                 inputs,
                 state,
                 scope=None):

      dtype=inputs.dtype
      with tf.variable_scope(scope or "bn_rnn"):
        print( "rnn cell input size: ", inputs.get_shape().as_list())
        print( "rnn cell state size: ", state.get_shape().as_list())
        input_size = inputs.get_shape()[1]
        num_units=self._num_units
        state_size = state.get_shape()[1]

        mean, variance, offset, scale = self.build(scope, input_size, num_units,
                                                   training=self._training)

        W = tf.get_variable(name='W', shape=[input_size, num_units])
        U = tf.get_variable(name='U', shape=[num_units, num_units] )
        B = tf.get_variable(name='B', shape=[num_units],
                            initializer= tf.zeros_initializer())

        #---------------------------------------------
        w_out = tf.matmul(inputs, W, transpose_a=False, transpose_b=False)
        bn, mean, variance = seq_batch_norm(w_out,
                                 scale, offset, mean, variance,
                                 momentum = self._momentum,
                                 epsilon  = self._epsilon,
                                 training = self._training,
                                 fused= True)
        with tf.control_dependencies([mean, variance]):
          bn=tf.identity(bn)

        # if not self._training:
        #   mean=tf.Print(mean, [mean], "after eval mean")

        u_out = tf.matmul(state, U, transpose_a=False, transpose_b=False)
        res = tf.add_n([bn, u_out])
        res= tf.add(res, B)
        output = self._activation(res)
        state = output

      return output, state

def seq_batch_norm(x,
                 scale, offset, mean, variance,
                 scope=None,
                 momentum=0.9, epsilon=0.001,
                 training=True,
                 fused= True):

  """
  sequence batch normalization, input N * D
  """
  # inputs_shape = x.get_shape()
  # param_shape = inputs_shape[-1]

  if fused:
    #expand [T,C] -> [T,C, 1, 1]
    x1 = tf.expand_dims(x, axis=-1)
    x1 = tf.expand_dims(x1, axis=-1)

    # FIXME: currently always workig in train mode. FIX if u can
    training= True
    # end of FIXME-----------------------------------------------
    if training:
      y1, batch_mean, batch_var= tf.nn.fused_batch_norm(x=x1,
                         scale=scale,
                         offset=offset,
                         mean=None,
                         variance=None,
                         epsilon=epsilon,
                         data_format='NCHW',
                         is_training=training)

      mean = tf.add(momentum * mean, (1.0 -momentum) *  batch_mean)
      variance = tf.add(momentum * variance, (1.0 -momentum) * batch_var)
      # mean= ema.average_name(batch_mean)
      # variance = ema.average_name(batch_var)

      with tf.control_dependencies([mean, variance]):
        y1=tf.identity(y1)

      # mean=tf.Print(mean, [mean], "train mean")
      # variance=tf.Print(variance, [variance], "train variance")

    else:
      # mean=tf.Print(mean, [mean], "eval mean")
      # variance=tf.Print(variance, [variance], "eval variance")

      y1, _, _ = tf.nn.fused_batch_norm(x=x1,
                         scale=scale,
                         offset=offset,
                         mean=mean,
                         variance=variance,
                         epsilon=epsilon,
                         data_format='NCHW',
                         is_training=training)

    # squeeze back to B,C
    y2 = tf.squeeze(y1, axis=-1)
    output = tf.squeeze(y2, axis=-1)

  else:
    x1 = tf.expand_dims(x, axis=-1)
    x1 = tf.expand_dims(x1, axis=-1)
    y1 = tf.nn.batch_normalization(x1, mean, variance, offset, scale, epsilon)
    print(y1.shape)
    y2 = tf.squeeze(y1, axis=-1)
    output = tf.squeeze(y2, axis=-1)

  return output, mean, variance

#========================================================================

class SeqNormRNNCell(BasicRNNCell):
    """
    RNN Cell with sequential batch normalization (ver 2.):
       output = new_state = activation(BN(W * input) + U * state + B).
           state dim: batch_size * num_units
           input dim: batch_size * feature_size
           W: feature_size * num_units
           U: num_units * num_units
    """

    def __init__(self,
                 num_units,
                 # input_size=None,
                 reuse=None,
                 name=None,
                 activation=tf.nn.relu,
                 momentum=0.5,
                 epsilon=0.001,
                 *args,
                 **kwargs
                 ):
      super(BasicRNNCell, self).__init__(_reuse=reuse, name=name)
      self._num_units = num_units
      self._activation = activation
      self._name = name
      self._momentum = momentum
      self._epsilon = epsilon
      self._training = kwargs["training"]

    def __call__(self,
                 inputs,
                 state,
                 scope=None):
      dtype = inputs.dtype
      with tf.variable_scope(scope or "bn_rnn"):
        print("rnn cell input size: ", inputs.get_shape().as_list())
        print("rnn cell state size: ", state.get_shape().as_list())
        input_size = inputs.get_shape()[1]
        num_units = self._num_units
        state_size = state.get_shape()[1]

        W = tf.get_variable( name='W', shape=[input_size, num_units])
        U = tf.get_variable(name='U', shape=[state_size, num_units])
        B = tf.get_variable(name='B', shape=[num_units],
                            initializer=tf.zeros_initializer())

        # ---------------------------------------------

        w_out = tf.matmul(inputs, W, transpose_a=False, transpose_b=False)

        # w_out = tf.expand_dims(w_out, axis=-1)
        # w_out = tf.expand_dims(w_out, axis=-1)
        bn= tf.layers.batch_normalization (w_out,
                  momentum=self._momentum,
                  epsilon=self._epsilon,
                  training= True, #self._training,
            )
        # bn = tf.squeeze(bn, axis=-1)
        # bn = tf.squeeze(bn, axis=-1)

        # if not self._training:
        #   mean=tf.Print(mean, [mean], "after eval mean")

        u_out = tf.matmul(state, U, transpose_a=False, transpose_b=False)
        res = tf.add_n([bn, u_out])
        res = tf.add(res, B)
        output = self._activation(res)
        state = output

      return output, state


