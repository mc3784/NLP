#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cbof import TextCBOF
from tensorflow.contrib import learn
from sys import exit
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Prob of drop out")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

createFile = False
splitPercentage = 0.1
timestamp = str(int(time.time()))
output_file = 'results.txt.' +timestamp

# Files Header 
with open(output_file, 'a') as out:
    out.write("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        out.write("{}={}".format(attr.upper(), value))
        out.write("\n")
    out.write("step,train_loss,train_acc,test_loss,test_acc"+ '\n')
loss_list=[]
earlyStopping = True
notImproving = 0
maxNotImprovingTimes = 5
# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(createFile,"train")
print("Total number of samples: {}".format(len(x_text))) 
numberTestSamples = int(splitPercentage*int(len(x_text)))
print("Number of test samples: {}".format(numberTestSamples)) 

# Build vocabulary
l = [len(x.split(" ")) for x in x_text]
max_document_length = reduce(lambda x, y: x + y, l) / len(l)
print("max_document_length: {} ".format(max_document_length)) 
max_document_length = max([len(x.split(" ")) for x in x_text])
#max_document_length = 50

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

x = np.array(x_text)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]


# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-numberTestSamples], x_shuffled[-numberTestSamples:]
y_train, y_dev = y_shuffled[:-numberTestSamples], y_shuffled[-numberTestSamples:]

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


vocabulary = data_helpers.create_vocabulary(x_train.tostring().split())
vocabulary_file='vocabulary.txt'+timestamp
with open(vocabulary_file, 'w') as thefile:
    for item in vocabulary:
        thefile.write("%s\n" % item)

x_train = data_helpers.substitute_oov(x_train,vocabulary)
x_dev = data_helpers.substitute_oov(x_dev,vocabulary)


#print x_train[0]
x_train = np.array(list(vocab_processor.fit_transform(x_train)))
x_dev = np.array(list(vocab_processor.fit_transform(x_dev)))

#print x_train[0]

#exit()


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cbof = TextCBOF(
            sequence_length=x_train.shape[1],
            num_classes=2,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            n_hidden=256,
            #num_filters=FLAGS.num_filters,
            dropout_keep_prob = FLAGS.dropout_keep_prob,
            l2_reg_lambda=FLAGS.l2_reg_lambda
            )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cbof.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cbof.loss)
        acc_summary = tf.scalar_summary("accuracy", cbof.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)


        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cbof.input_x: x_batch,
              cbof.input_y: y_batch
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cbof.loss, cbof.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #save value for plot
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                with open(output_file, 'a') as out:
                    out.write("{},{:g},{:g}".format(step, loss, accuracy) + ',')
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            global notImproving
            feed_dict = {
              cbof.input_x: x_batch,
              cbof.input_y: y_batch
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cbof.loss, cbof.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
	    #Save value for plot:
            with open(output_file, 'a') as out:
                out.write("{:g},{:g}".format(loss, accuracy) + '\n')
            if writer:
                writer.add_summary(summaries, step)
            #Early stopping condition
            if len(loss_list) > 0 and loss > loss_list[-1]:
               notImproving+=1 
               print("NOT IMPROVING FROM PREVIOUS STEP")
            else:
               notImproving = 0
            if earlyStopping and notImproving > maxNotImprovingTimes:
               print(loss_list)
               sess.close()
               exit()
            loss_list.append(loss) 
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation: notImproving: {}".format(notImproving))
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
                #print(loss_list)
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
