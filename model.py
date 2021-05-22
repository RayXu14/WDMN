import os
import sys

import numpy as np
import tensorflow as tf

from layers import *


class WDMN(object):
    def __init__(self, FLAGS, pretrained_word_embeddings=None):
        self.FLAGS = FLAGS
        embed_dim = FLAGS.embed_dim
        vocab_size = FLAGS.vocab_size
        hidden_dim = FLAGS.hidden_dim
        max_turn = FLAGS.max_turn
        max_word_len = FLAGS.max_utterance_len

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.context = tf.placeholder(tf.int32, [None, 10, 50], name='context')
        self.response = tf.placeholder(tf.int32, [None, 50], name='response')
        self.target = tf.placeholder(tf.int32, [None, ], name='target')

        self.context_mask = tf.cast(tf.not_equal(self.context, 0), tf.float32)
        self.response_mask = tf.cast(tf.not_equal(self.response, 0), tf.float32)

        self.expand_response_mask = tf.tile(tf.expand_dims(self.response_mask, 1),
                                            [1, max_turn, 1]) 
        self.expand_response_mask = tf.reshape(self.expand_response_mask,
                                               [-1, max_word_len])  
        self.parall_context_mask = tf.reshape(self.context_mask, [-1, max_word_len])  

        self.y_pred = 0.0
        self.loss = 0.0
        self.loss_list = []

        with tf.variable_scope("word_embeddings"):
            word_embeddings = tf.get_variable('word_embeddings_v',
                                              shape=(vocab_size, embed_dim),
                                              dtype=tf.float32,
                                              trainable=True)
            if pretrained_word_embeddings is not None:
                self.embedding_init = word_embeddings.assign(pretrained_word_embeddings)
                
            self.context_embeddings = tf.nn.embedding_lookup(word_embeddings,
                                                             self.context)  
            self.response_embeddings = tf.nn.embedding_lookup(word_embeddings,
                                                              self.response)  
            self.context_embeddings = tf.layers.dropout(self.context_embeddings,
                                                        rate=1.0-self.dropout_keep_prob)
            self.response_embeddings = tf.layers.dropout(self.response_embeddings,
                                                         rate=1.0-self.dropout_keep_prob)
            self.context_embeddings = tf.multiply(self.context_embeddings,
                                                  tf.expand_dims(self.context_mask,
                                                                 axis=-1))  
            self.response_embeddings = tf.multiply(self.response_embeddings,
                                                   tf.expand_dims(self.response_mask,
                                                                  axis=-1)) 

        self.expand_response_embeddings = tf.tile(tf.expand_dims(self.response_embeddings, 1),
                                                  [1, max_turn, 1, 1]) 
        self.expand_response_embeddings = tf.reshape(self.expand_response_embeddings,
                                                     [-1, max_word_len, embed_dim]) 
        self.parall_context_embeddings = tf.reshape(self.context_embeddings,
                                                    [-1, max_word_len, embed_dim])
        
        # Initial basic representation
        context_rep = self.parall_context_embeddings
        response_rep = self.expand_response_embeddings
        c_seq_base = self.parall_context_embeddings
        r_seq_base = self.expand_response_embeddings
        c_local_base = self.parall_context_embeddings
        r_local_base = self.expand_response_embeddings
        c_self_base = self.parall_context_embeddings
        r_self_base = self.expand_response_embeddings
        
        losses_list = []
        y_pred_list = []
        logits_list=[]
        for n_layer in range(FLAGS.num_layer):
            with tf.variable_scope('layer_{}'.format(n_layer + 1)):
                # Sequence Encoding
                context_seq_rep = bilstm_layer_cudnn(c_seq_base,
                                                     num_layers=1,
                                                     rnn_size=hidden_dim)
                response_seq_rep = bilstm_layer_cudnn(r_seq_base,
                                                      num_layers=1,
                                                      rnn_size=hidden_dim)
                c_seq_base, r_seq_base, seq_loss, seq_pred, seq_logits = \
                    self.IRblock(context_seq_rep, response_seq_rep, 
                    c_seq_base, r_seq_base, n_layer, 'sequential')
                    
                # Local Encoding
                conv_dim = 50
                kernels = [1,2,3,4]
                with tf.variable_scope("local_rep"):
                    context_local_rep = conv(c_local_base, hidden_dim,
                                            kernel_size=kernels, bias=True,
                                            activation=tf.nn.relu, isNormalize=True,
                                            reuse=tf.AUTO_REUSE)
                    response_local_rep = conv(r_local_base, hidden_dim,
                                             kernel_size=kernels, bias=True,
                                             activation=tf.nn.relu, isNormalize=True,
                                             reuse=tf.AUTO_REUSE)

                c_local_base, r_local_base, local_loss, local_pred, local_logits = \
                    self.IRblock(context_local_rep, response_local_rep,
                    c_local_base, r_local_base, n_layer, 'local')
            
                # Self-attended Encoding
                context_self_rep = self_attention(c_self_base, c_self_base, embed_dim, 
                                                    query_masks=self.parall_context_mask, 
                                                    key_masks=self.parall_context_mask, 
                                                    num_blocks=1, num_heads=1, 
                                                    dropout_rate=1.0-self.dropout_keep_prob,
                                                    use_residual=True, use_feed=True, 
                                                    scope='self_attention')[1]  # [batch*turn, len_utt, embed_dim, 2]
                response_self_rep = self_attention(r_self_base, r_self_base, embed_dim, 
                                                    query_masks=self.expand_response_mask, 
                                                    key_masks=self.expand_response_mask, 
                                                    num_blocks=1, num_heads=1, 
                                                    dropout_rate=1.0-self.dropout_keep_prob, 
                                                    use_residual=True, use_feed=True, 
                                                    scope='self_attention', reuse=True)[1]  # [batch*turn, len_res, embed_dims, 2]

                c_self_base, r_self_base, self_loss, self_pred, self_logits = \
                    self.IRblock(context_self_rep, response_self_rep,
                    c_self_base, r_self_base, n_layer, 'self')
                
            losses_list.append(seq_loss   + local_loss   + self_loss) 
            y_pred_list.append(seq_pred   + local_pred   + self_pred) 
            logits_list.append(seq_logits + local_logits + self_logits)

        if FLAGS.use_loss_decay:
            self.loss = sum([((idx + 1) / float(FLAGS.num_layer)) * item for idx, item in enumerate(losses_list)])
        else:
            self.loss = sum(losses_list)
        self.loss_list = losses_list

        self.y_pred = sum(y_pred_list)

        if FLAGS.use_globalLoss:
            logits_sum = tf.add_n(logits_list)
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=logits_sum)
            self.loss = tf.reduce_mean(tf.clip_by_value(self.loss, -FLAGS.clip_value, FLAGS.clip_value))
            self.loss_list = [self.loss]
            self.y_pred = tf.nn.softmax(logits_sum) 


        self.correct = tf.equal(tf.cast(tf.argmax(self.y_pred, axis=1), tf.int32), tf.to_int32(self.target))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))

        
    def IRblock(self, c_rep, r_rep, c_base, r_base, n_layer, scope):   
        c_mask            = self.parall_context_mask
        r_mask            = self.expand_response_mask
        c_origin          = self.parall_context_embeddings
        r_origin          = self.expand_response_embeddings
        embed_dim         = self.FLAGS.embed_dim
        max_turn          = self.FLAGS.max_turn
        max_word_len      = self.FLAGS.max_utterance_len
        target            = self.target
        dropout_keep_prob = self.dropout_keep_prob
            
        with tf.variable_scope(scope):       
            # Word Level Alignment
            inter_feat = tf.matmul(c_rep,
                                   tf.transpose(r_rep, perm=[0, 2, 1])) / float(embed_dim)
            inter_feat_base = tf.matmul(c_base,
                                   tf.transpose(r_base, perm=[0, 2, 1])) / float(embed_dim)
                                   
            c_score = tf.nn.softmax(inter_feat, axis=-1)
            r_score = tf.nn.softmax(tf.transpose(inter_feat, [0, 2, 1]), axis=-1)
            c_cross_rep = tf.matmul(c_score, r_rep)
            r_cross_rep = tf.matmul(r_score, c_rep)
            c_cross_rep *= tf.expand_dims(c_mask, axis=-1)
            r_cross_rep *= tf.expand_dims(r_mask, axis=-1)
            
            c_score_base = tf.nn.softmax(inter_feat_base, axis=-1)
            r_score_base = tf.nn.softmax(tf.transpose(inter_feat_base, [0, 2, 1]), axis=-1)
            c_cross_base = tf.matmul(c_score_base, r_base)
            r_cross_base = tf.matmul(r_score_base, c_base)
            c_cross_base *= tf.expand_dims(c_mask, axis=-1)
            r_cross_base *= tf.expand_dims(r_mask, axis=-1)
                 
            # Sentence Level Alignment
            c_stc_level = tf.reduce_max(c_cross_rep, axis=1)
            r_stc_level = tf.reduce_max(r_cross_rep, axis=1)
            stc_level = tf.concat([c_stc_level, r_stc_level,
                                   c_stc_level * r_stc_level,
                                   c_stc_level - r_stc_level], axis=-1)
            stc_level_dense = tf.layers.dense(stc_level, embed_dim, 
                                              activation=tf.nn.tanh,
                                              name='stc_level_dense')
                                   
            c_stc_level_base = tf.reduce_max(c_cross_base, axis=1)
            r_stc_level_base = tf.reduce_max(r_cross_base, axis=1)
            stc_level_base = tf.concat([c_stc_level_base, r_stc_level_base,
                                   c_stc_level_base * r_stc_level_base,
                                   c_stc_level_base - r_stc_level_base], axis=-1)
            stc_level_base_dense = tf.layers.dense(stc_level_base, embed_dim, 
                                              activation=tf.nn.tanh,
                                              name='stc_level_base_dense')

            # Calculate updating
            c_cross_inner = c_rep * c_cross_rep
            r_cross_inner = r_rep * r_cross_rep
            c_base_inner = c_base * c_cross_base
            r_base_inner = r_base * r_cross_base

            c_cat_rep = tf.concat([c_base, c_rep, c_cross_base, c_cross_rep, c_base_inner, c_cross_inner], axis=-1) 
            r_cat_rep = tf.concat([r_base, r_rep, r_cross_base, r_cross_rep, r_base_inner, r_cross_inner], axis=-1)

            c_cat_dense_rep = tf.layers.dense(c_cat_rep, embed_dim,
                                              activation=tf.nn.relu, use_bias=True,
                                              name='c_update_dense') 
            c_cat_dense_rep = tf.layers.dropout(c_cat_dense_rep,
                                                rate=1.0-dropout_keep_prob)

            r_cat_dense_rep = tf.layers.dense(r_cat_rep, embed_dim,
                                              activation=tf.nn.relu, use_bias=True, 
                                              name='r_update_dense')
            r_cat_dense_rep = tf.layers.dropout(r_cat_dense_rep,
                                                rate=1.0-dropout_keep_prob)


            inter_feat_collection = [inter_feat_base, inter_feat]
            matching_feat = tf.stack(inter_feat_collection, axis=-1)
            
            # Update representation
            if n_layer == 0:
                c_base = tf.add(c_base, c_cat_dense_rep)
                r_base = tf.add(r_base, r_cat_dense_rep)
            else:
                c_base = tf.add_n([c_origin, c_base, c_cat_dense_rep])
                r_base = tf.add_n([r_origin, r_base, r_cat_dense_rep])

            c_base = normalize(c_base, scope='c_update_norm')
            r_base = normalize(r_base, scope='r_update_norm')

            c_base = tf.multiply(c_base, tf.expand_dims(c_mask, axis=-1))
            r_base = tf.multiply(r_base, tf.expand_dims(r_mask, axis=-1))

            with tf.variable_scope('aggregation'):
                # Word Level
                conv1 = tf.layers.conv2d(matching_feat,
                                         filters=32, kernel_size=(3, 3), strides=(1, 1),
                                         padding='same', activation=tf.nn.relu,
                                         name='conv1')
                pool1 = tf.layers.max_pooling2d(conv1, (3, 3), strides=(3, 3),
                                                padding='same',
                                                name='max_pooling1')
                
                conv2 = tf.layers.conv2d(pool1,
                                         filters=16, kernel_size=(3, 3), strides=(1, 1),
                                         padding='same', activation=tf.nn.relu,
                                         name='conv2')
                pool2 = tf.layers.max_pooling2d(conv2,
                                                (3, 3), strides=(3, 3),
                                                padding='same',
                                                name='max_pooling2')   
                                                
                flatten = tf.contrib.layers.flatten(pool2)
                flatten = tf.layers.dropout(flatten, rate=1.0 - dropout_keep_prob)

                matching_vector = tf.layers.dense(flatten, embed_dim,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.tanh,
                                    name='dense_feat') 
                matching_vector = tf.reshape(matching_vector, [-1, max_turn, embed_dim])

                final_word_level_cell = tf.contrib.rnn.GRUCell(embed_dim, kernel_initializer=tf.orthogonal_initializer())
                _, last_word_level_hidden = tf.nn.dynamic_rnn(final_word_level_cell,
                                                              matching_vector,
                                                              dtype=tf.float32,
                                                              scope='final_word_level_rnn')
                # Sentence Level
                stc_level_dense = tf.reshape(stc_level_dense,
                                             [-1, max_turn, embed_dim])
                stc_level_base_dense = tf.reshape(stc_level_base_dense,
                                             [-1, max_turn, embed_dim])
                                             
                final_stc_level_cell = tf.contrib.rnn.GRUCell(embed_dim, kernel_initializer=tf.orthogonal_initializer())
                _, last_stc_level_hidden = tf.nn.dynamic_rnn(final_stc_level_cell,
                                                tf.concat([stc_level_dense, stc_level_base_dense], 
                                                axis=-1),
                                                dtype=tf.float32,
                                                scope='final_stc_level_rnn')
                # Merge
                last_hidden = tf.concat([last_word_level_hidden, last_stc_level_hidden],
                                        axis=-1)
                logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='final_merge')

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
            loss = tf.reduce_mean(tf.clip_by_value(loss, -self.FLAGS.clip_value, self.FLAGS.clip_value))
            y_pred = tf.nn.softmax(logits)
        
        return c_base, r_base, loss, y_pred, logits
