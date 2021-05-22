import tensorflow as tf


initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                     mode='FAN_AVG',
                                                                     uniform=True,
                                                                     dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


def normalize(inputs, epsilon=1e-8, scope='normalize', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        axis = [-1]
        shape = [inputs.shape[i] for i in axis]

        scale = tf.get_variable(name='scale', shape=shape, dtype=tf.float32, initializer=tf.ones_initializer())
        bias = tf.get_variable(name='bias', shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=axis, keep_dims=True)

        norm = (inputs - mean) * tf.rsqrt(variance + epsilon)
        return scale * norm + bias


def conv(inputs, output_size, kernel_size=[1,2,3,4], bias=None, activation=None, name="conv", isNormalize=False, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        conv_features = []
        shapes = inputs.shape.as_list()
        for k in kernel_size:
            filter_shape = [k, shapes[-1], output_size]
            bias_shape = [1,1,output_size]
            strides = 1
            kernel_ = tf.get_variable("kernel_%s"%k,
                                      filter_shape,
                                      dtype = tf.float32,
                                      regularizer=regularizer,
                                      initializer = initializer())
            feature = tf.nn.conv1d(inputs, kernel_, strides, "SAME")
            if bias:
                feature += tf.get_variable("bias_%s"%k,
                                           bias_shape,
                                           regularizer=regularizer,
                                           initializer = tf.zeros_initializer())
            if activation is not None:
                feature = activation(feature)
            conv_features.append(feature)
        output = tf.concat(conv_features, axis=-1)
        if isNormalize:
            output = normalize(output, 1e-8, "normalize", reuse) 
        return output


def self_attention(queries, keys, num_units,
                   query_masks=None, key_masks=None,
                   num_blocks=6, num_heads=1,
                   dropout_rate=0, causality=False,
                   use_linear=False, use_residual=True,
                   use_feed=True, attention_flag='full',
                   is_training=False, scope=None, reuse=None,
                   queries_hist=None, keys_hist=None):
    '''Applies singlehead attention.
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    hiddens =[]
    hiddens.append(queries)
    with tf.variable_scope(scope or "self_attention", reuse=reuse):        
        # Linear projections
        if use_linear:
            queries = tf.layers.dense(queries, num_units, activation=tf.nn.relu, name="dense_q")  # (N, T_q, C)
            keys = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name="dense_k")  # (N, T_k, C)
            values = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name="dense_v")  # (N, T_k, C)
        else:
            values = keys

        if attention_flag=='dot':
            if queries_hist==None:
                outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # (N, T_q, T_k)
            else:
                outputs = tf.matmul(tf.concat([queries, queries_hist], axis=-1), tf.transpose(tf.concat([keys, keys_hist], axis=-1), [0, 2, 1]))  # (N, T_q, T_k)
        else:
            if queries_hist==None:
                outputs = full_attention(queries, keys) # fully aware attention
            else:
                outputs = full_attention(tf.concat([queries, queries_hist], axis=-1), tf.concat([keys, keys_hist], axis=-1)) # fully aware attention

        # Scale
        scale = tf.maximum(1.0, keys.get_shape().as_list()[-1] ** 0.5)
        outputs = outputs / scale

        # Key Masking
        if key_masks is None:
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

        # For mask_softmax activation
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (N, T_q, T_k)
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        if query_masks is None:
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, values)  # ( h*N, T_q, C/h)

        if use_residual:
            # Residual connection
            outputs += queries
            # Normalize
            outputs = normalize(outputs)  # (N, T_q, C)

        # Feed Forward
        if use_feed:
            outputs = feedforward(outputs, num_units=[num_units, num_units], scope='feed_forward')

        hiddens.append(outputs)
    return hiddens


def feedforward(inputs, num_units=[2048, 512], scope="feed_forward", use_dense=True, reuse=None):
    '''Point-wise feed forward net.
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if use_dense:
            outputs = tf.layers.dense(inputs, num_units[0], activation = tf.nn.relu, 
                                            # kernel_initializer = tf.contrib.keras.initializers.he_normal(), 
                                            use_bias=True, name="dense_1")  # (N, T_q, C)
            outputs = tf.layers.dense(outputs, num_units[1], activation=None, 
                                            # kernel_initializer = tf.contrib.layers.xavier_initializer(), 
                                            use_bias=True, name="dense_2")  # (N, T_q, C)  
        else:          
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = normalize(outputs)
            
    return outputs


def full_attention(utt_how, resp_how, dim=None, scope="full_attention", reuse=None):
    '''Fully aware attention
    Args:
      utt_how: [batch, len_utt, dim] 
      resp_how: [batch, len_res, dim]
      scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as response
    '''
    if dim==None:
        dim = utt_how.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        U = tf.get_variable('Weight_U', shape=[dim, dim], dtype=tf.float32)

        I = tf.eye(dim)
        D = tf.get_variable('Weight_D', shape=[dim, dim], dtype=tf.float32)
        D = tf.multiply(D, I)  # restrict to diagonal

        f1 = tf.nn.relu(tf.einsum('aij,jk->aik', utt_how, U), name='utt_how_relu') # [batch, len_utt, dim]
        f2 = tf.nn.relu(tf.einsum('aij,jk->aik', resp_how, U), name='resp_how_relu') # [batch, len_res, dim]
        S = tf.einsum('aij,jk->aik', f1, D)  # [batch, len_utt, dim]
        S = tf.einsum('aij,akj->aik', S, f2) # [batch, len_utt,len_res]
    return S
    

def bilstm_layer_cudnn(input_data, num_layers, rnn_size, keep_prob=1.):
    """Multi-layer BiLSTM cudnn version, faster
    Args:
        input_data: float32 Tensor of shape [seq_length, batch_size, dim].
        num_layers: int64 scalar, number of layers.
        rnn_size: int64 scalar, hidden size for undirectional LSTM.
        keep_prob: float32 scalar, keep probability of dropout between BiLSTM layers
    Return:
        output: float32 Tensor of shape [seq_length, batch_size, dim * 2]
    """
    input_data = tf.transpose(input_data, [1, 0, 2])
    with tf.variable_scope("bilstm", reuse=tf.AUTO_REUSE):
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=num_layers,
            num_units=rnn_size,
            input_mode="linear_input",
            direction="unidirectional",
            dropout=1 - keep_prob)
        outputs, output_states = lstm(inputs=input_data)
        
    outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs