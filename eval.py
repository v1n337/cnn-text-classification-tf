#! /usr/bin/env python

import sys
import csv
import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.metrics import confusion_matrix, f1_score

import config
import data_helpers

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-dir", type=str, required=True)
args_namespace = parser.parse_args(sys.argv[1:])
command_line_args = vars(args_namespace)
checkpoint_dir = command_line_args['checkpoint_dir']

x_raw, y_test = data_helpers.load_data_and_labels(config.eval_positive_data_file, config.eval_negative_data_file)
y_test = np.argmax(y_test, axis=1)

# Map data into vocabulary
vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_conf = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=config.allow_soft_placement,
        log_device_placement=config.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), config.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
    print("F1-Score: {:g}".format(f1_score(y_true=y_test, y_pred=all_predictions)))

    print("Confusion matrix:\n")
    print(confusion_matrix(y_true=y_test, y_pred=all_predictions))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
print("Execution complete")
