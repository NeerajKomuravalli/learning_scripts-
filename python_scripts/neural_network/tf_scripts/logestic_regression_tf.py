import tensorflow as tf
import pandas as pd
import random

import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
# tensorboard --logdir summary_logs/


'''
df = pd.read_csv('~/Downloads/adult.csv', na_values='?')

df.dropna(axis=0, how='any', inplace=True)

# categorical_variables = ["workclass", "marital-status", "occupation", 
# "relationship", "race", "gender", "native-country", "age"]
# remove_columns = ["fnlwgt", "fnlwgt", "education"]
y = ["income"]

# df_cat = pd.get_dummies(df[categorical_variables])
y_gt = pd.get_dummies(df[y])
# df.drop(categorical_variables, axis=1, inplace=True)
# df.drop(y, axis=1, inplace=True)
# df.drop(remove_columns, axis=1, inplace=True)
# df = pd.merge(df, df_cat, left_index=True, right_index=True)

df = df[["hours-per-week"]]

train_ind = random.sample(range(df.shape[0]),int(df.shape[0]*80/float(100)))
test_ind = []
for ind in range(df.shape[0]):
	if ind not in train_ind:
		test_ind.append(ind)

x_train = df.iloc[train_ind]
y_train = y_gt.iloc[train_ind]

x_test = df.iloc[test_ind]
y_test = y_gt.iloc[test_ind]
'''

iris = load_iris()
iris_X, iris_y = iris.data[:-1, :], iris.target[:-1]
iris_y = pd.get_dummies(iris_y).values
x_train, x_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

# Defining graph
# place holders for data
x_plc = tf.placeholder(tf.float32, [None, x_train.shape[1]], name='x_input_placeholder')
y_plc = tf.placeholder(tf.float32, [None, y_train.shape[1]], name='y_input_placeholder')

# Weight
weight = tf.Variable(tf.random_normal([x_train.shape[1], y_train.shape[1]], mean=0, stddev=0.01, name='weight'))
# bias
bias = tf.Variable(tf.random_normal([1, y_train.shape[1]], mean=0, stddev=0.01, name='bias'))

# operations
weight_mul = tf.matmul(x_plc, weight, name='matmul')
add_bias = tf.add(weight_mul, bias, name='add_bias')
sigmoid = tf.nn.sigmoid(add_bias, name='activation')

# training
learning_rate = tf.train.exponential_decay(learning_rate=0.0008,
                                           global_step=1,
                                           decay_steps=x_train.shape[0],
                                           decay_rate=0.95,
                                           staircase=True)

# Defining error
squared_error = tf.nn.l2_loss(sigmoid - y_plc, name="squared_error_cost")
# Optimizing 
training = tf.train.GradientDescentOptimizer(learning_rate).minimize(squared_error)

saver = tf.train.Saver()

with tf.Session() as sess:
    # initializing variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Correct predictions
    correct_preds = tf.equal(tf.argmax(sigmoid, 1), tf.argmax(y_plc, 1))
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"))

    # Summary op for regression output
    activation_summary_OP = tf.summary.histogram("output", sigmoid)

    # Summary op for accuracy
    accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy)

    # Summary op for cost
    cost_summary_OP = tf.summary.scalar("cost", squared_error)

    # Summary ops to check how variables (W, b) are updating after each iteration
    weightSummary = tf.summary.histogram("weights", weight.eval(session=sess))
    biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))

    # Merge all summaries
    merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])

    # Summary writer
    writer = tf.summary.FileWriter("summary_logs", sess.graph)

    # We train now
    no_of_epochs = 700
    accuracy_list = []
    cost_list = []
    for epoch in range(no_of_epochs):
        print("Running training on : ", epoch)
        step = sess.run(training, feed_dict={x_plc: x_train, y_plc: y_train})

        accuracy_step, cost_step, summary = sess.run([accuracy, squared_error, merged], feed_dict={x_plc: x_train, y_plc: y_train})
        accuracy_list.append(accuracy_step)
        cost_list.append(cost_step)

        writer.add_summary(summary, (epoch+1))

        saver.save(sess, 'logestic/logestic_regression_' + str(epoch))

        print("Cost : ", cost_step)

    # For test find accuracy
    test_accuracy = sess.run(accuracy, feed_dict={x_plc: x_train, y_plc: y_train})

    print("Here is the test accuracy acheived : ", test_accuracy)
