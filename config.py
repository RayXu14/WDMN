import tensorflow as tf


def get_args():

    tf.flags.DEFINE_string('data_path', '../../data/ubuntu_data', 'Path to dataset. ')
    
    ''' Model arguments '''
    tf.flags.DEFINE_integer("num_layer", 4, "Number of IR blocks")
    tf.flags.DEFINE_boolean("init_dict", True, "Use initialized word2vec")
    tf.flags.DEFINE_integer("vocab_size", 439760, "Size of vocabulary")
    tf.flags.DEFINE_integer("max_turn", 10, "Max length of context")
    tf.flags.DEFINE_integer("max_utterance_len", 50, "Max length of utterance") 

    tf.flags.DEFINE_integer("embed_dim", 200, "Dimensionality of embedding")
    tf.flags.DEFINE_integer("hidden_dim", 200, "Dimensionality of rnn")

    tf.flags.DEFINE_boolean('use_globalLoss', False, 'Whether to use an unify loss for all IR blocks ') 
    tf.flags.DEFINE_boolean('use_loss_decay', False, 'Whether to decay loss along IR block chain') 

    ''' Training arguments '''
    tf.flags.DEFINE_string('optimizer', 'adam', 'Which optimization method to use') # adam 0.001  adadelta
    tf.flags.DEFINE_float('lr', 0.0005, 'Learning rate')
    tf.flags.DEFINE_boolean("lr_decay", True, 'Whether ti decay learning rate during training')
    tf.flags.DEFINE_float('decay_rate', 0.9, 'Learning rate decay speed')  
    tf.flags.DEFINE_integer('decay_steps', 5000, 'Learning rate decay steps')  
    tf.flags.DEFINE_float('lr_minimal', 0.00005, 'Minimal learning rate') # 0.00002
    tf.flags.DEFINE_float('clip_value', 10.0, 'Clip value')
    tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability") 
    
    tf.flags.DEFINE_integer("batch_size", 20, 'Batch size')
    tf.flags.DEFINE_integer("num_epochs", 2000000, "Number of training epochs")
    tf.flags.DEFINE_integer("print_every", 50, "Print the results after this many steps")
    tf.flags.DEFINE_integer("eval_every", 50000, "Evaluate model after this many step")
    tf.flags.DEFINE_integer("checkpoint_every", 1250000, "Save model after this many step")

    tf.flags.DEFINE_boolean("reload_model", False, "Allow reload the model")
    tf.flags.DEFINE_string('log_root', 'debug/', 'Root directory for all logging.')

    ''' GPU arguments '''
    tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    return tf.flags.FLAGS
