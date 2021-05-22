import os
import sys
import random
import pickle
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from config import get_args
from data_utils import CRMatchingDataset
from metrics import recall_2at1, recall_at_k, precision_at_k, MRR, MAP
from model import WDMN as model


random.seed(1234)
np.random.seed(1234) 

FLAGS = get_args()
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))

    
if __name__ == "__main__":

    ''' Output directory for checkpoints and predictions '''
    out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.log_root))
    print("Writing to {}\n".format(out_dir))

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if FLAGS.init_dict:
        ''' Load pretrained word embeddings from disk '''
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Loading pretrained word embeddings ... | {time_str}")
        init_embeddings_path = '%s/vocab_and_embeddings.pkl'%(FLAGS.data_path)
        with open(init_embeddings_path, 'rb') as f:
            vocab, embeddings = pickle.load(f)
        pretrained_word_embeddings = np.array(embeddings)
        FLAGS.vocab_size = pretrained_word_embeddings.shape[0]
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'loaded vocab size {pretrained_word_embeddings.shape[0]} | {time_str}')
    else:            
        pretrained_word_embeddings = None
        
    ''' Loading dataset '''
    train_file = '%s/train.pkl'%(FLAGS.data_path)
    dev_file = '%s/dev.pkl'%(FLAGS.data_path)
    test_file = '%s/test.pkl'%(FLAGS.data_path)
    
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("Creating dataset... | %s " % time_str)
    
    with open(train_file, 'rb') as f:
        train_contexts, train_responses, train_labels = pickle.load(f)
    with open(dev_file, 'rb') as f:
        dev_contexts, dev_responses, dev_labels = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_contexts, test_responses, test_labels = pickle.load(f)
    trainset = CRMatchingDataset(train_contexts, train_responses, train_labels, shuffle=True)
    devset = CRMatchingDataset(dev_contexts, dev_responses, dev_labels, shuffle=False)
    testset = CRMatchingDataset(test_contexts, test_responses, test_labels, shuffle=False)

    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("Created dataset. | %s " % time_str)

    ''' Init tensorflow session'''
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)

    with sess.as_default():

        ''' Init WDMN model '''
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("Creating WDMN model... | %s " % time_str)
        model = model(FLAGS, pretrained_word_embeddings)
        print('++++++++++++++\nprint model parameters\n++++++++++++++')
        total_cnt = 0
        for v in tf.global_variables():
          print(v)
          try:
            total_cnt += np.prod([int(e) for e in v.get_shape()])
          except:
            pass
        print(f'++++++++++++++\nTotal number of parameters = {total_cnt}\n++++++++++++++')
        
        ''' Init training'''
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(model.loss, global_step=global_step) 
        
        ''' Init saver '''
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Initing Saver | {time_str} ")
        saver = tf.train.Saver(max_to_keep=1)
        if FLAGS.reload_model:
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Reloading model from {checkpoint_dir} | {time_str}")
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        else:
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Init parameters | {time_str}")
            sess.run(tf.global_variables_initializer())
            if FLAGS.init_dict:
                sess.run(model.embedding_init)

                
        def train_step(dataset):
            """
            A single training step
            """
            train_step = tf.train.global_step(sess, global_step)
            
            ''' Learning_rate decaying '''
            if FLAGS.lr_decay:
                current_lr = max(FLAGS.lr * np.power(FLAGS.decay_rate, (train_step/FLAGS.decay_steps)), FLAGS.lr_minimal)
            else:
                current_lr = FLAGS.lr
                
            ''' Training step '''
            contexts, responses, labels = dataset.next()
            feed_dict = {
                learning_rate: current_lr,
                model.context: contexts,
                model.response: responses,
                model.target: labels,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, model.loss, model.accuracy], feed_dict)

            ''' visualization '''
            if step == 0 or step % FLAGS.print_every == 0:
                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("Step: %d \t| loss: %.3f \t| acc: %.3f \t| lr: %.5f \t| %s" %
                      (step, loss, accuracy, current_lr, time_str))

                      
        def eval(dataset, split):
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Evaluating {split} set")
            acc = []
            losses = []
            pred_scores = []
            true_scores = []
            count = 0
            
            ''' Inferencing '''
            for _ in range(dataset.batches()):
                contexts, responses, labels = dataset.next()
                feed_dict = {
                    model.context: contexts,
                    model.response: responses,
                    model.target: labels,
                    model.dropout_keep_prob: 1.0
                }
                step, loss, accuracy, y_pred, target = sess.run(
                    [global_step, model.loss, model.accuracy, model.y_pred, model.target], feed_dict)
                    
                acc.append(accuracy)
                losses.append(loss)
                pred_scores += list(y_pred[:, 1])
                true_scores += list(target)

                count += 1
                if count % 2500 == 0:
                    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Evaluated {count} batches | {time_str}")

            assert dataset.index == 0
            
            ''' Writing predictions '''
            MeanAcc = sum(acc) / len(acc)
            MeanLoss = sum(losses) / len(losses)
            if len(pred_scores) % 10 != 0:
                print(f'Warning: eval {len(pred_scores)} cases cannot be divided by 10, will cut remainder')
                pred_scores = pred_scores[:int(len(pred_scores) / 10) * 10]
                true_scores = true_scores[:int(len(true_scores) / 10) * 10]
            
            with open(os.path.join(out_dir, 'predScores-iter-%s.txt'%(step)), 'w') as f:
                for score1, score2 in zip(pred_scores, true_scores):
                    f.writelines(str(score1) + '\t' + str(score2) + '\n')

            ''' Calculating metrics'''
            num_sample = int(len(pred_scores) / 10)
            score_list = np.split(np.array(pred_scores), num_sample, axis=0)
            recall_2_1 = recall_2at1(score_list, k=1)
            recall_at_1 = recall_at_k(np.array(true_scores),  np.array(pred_scores), 1) 
            recall_at_2 = recall_at_k(np.array(true_scores),  np.array(pred_scores), 2)
            recall_at_5 = recall_at_k(np.array(true_scores),  np.array(pred_scores), 5)
            precision_at_1 = precision_at_k(np.array(true_scores),  np.array(pred_scores), 1)
            map10 = MAP(np.array(true_scores),  np.array(pred_scores))
            mrr10 = MRR(np.array(true_scores),  np.array(pred_scores))
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("**********************************")
            print(f'{split} pred_scores: {len(pred_scores)}')
            print("Step: %d \t| loss: %.3f \t| acc: %.3f \t|  %s" %(step, MeanLoss, MeanAcc, time_str))
            print("recall_2_1:  %.3f" % (recall_2_1))
            print("recall_at_1: %.3f" % (recall_at_1))
            print("recall_at_2: %.3f" % (recall_at_2))
            print("recall_at_5: %.3f" % (recall_at_5))
            print("precision_at_1: %.3f" % (precision_at_1))
            print("MAP: %.3f" % (map10))
            print("MRR: %.3f" % (mrr10))
            print("**********************************")

            return MeanLoss, recall_2_1 + recall_at_1
            
        if FLAGS.reload_model:
            ''' Evaluating reloaded model '''
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(time_str + '\tEvaluating Reloaded model...')
            meanLoss, metrics = eval(devset, 'dev')
            _, _ = eval(testset, 'test')
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(time_str + '\tEvaluated Reloaded model.')
       
       
        ''' Training procedure '''
        optimal_metrics = 0.0
        optimal_step = 0
        
        for i in range(FLAGS.num_epochs):
            train_step(trainset)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.eval_every == 0:
                print('Evaluating...')
                meanLoss, metrics = eval(devset, 'dev')
                _, _ = eval(testset, 'test')
                
                ''' Save best model '''
                if metrics > optimal_metrics:
                    optimal_metrics = metrics
                    optimal_step = current_step
                    print("opt_step: %d \t| opt_metric: %.3f" %(optimal_step, optimal_metrics))
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
