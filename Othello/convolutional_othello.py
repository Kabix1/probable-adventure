import tensorflow as tf
import numpy as np
# from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
from matplotlib import pyplot as plt
import math
import othello

from input_data import input_data

tf.__version__
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.
fc_size = 128             # Number of neurons in fully-connected layer.


# data.test.cls = np.argmax(data.test.labels, axis=1)
board_size = 8
board_size_flat = board_size * board_size
board_shape = (board_size, board_size)
num_channels = 1
num_classes = 2


train = input_data("data/NN_saved_games.dat")

# images = data.test.images[0:9]
# cls_true = data.test.cls[0:9]
# plot_images(images=images, cls_true=cls_true)
# replace with boards from saved file

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)
    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)
    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases
    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)
    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.
    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


x = tf.placeholder(tf.float32, shape=[None, board_size_flat], name='x')
x_image = tf.reshape(x, [-1, board_size, board_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
layer_conv1
layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
layer_conv2
layer_flat, num_features = flatten_layer(layer_conv2)
layer_flat
num_features
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc1
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
layer_fc2
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session = tf.Session()
session.run(tf.initialize_all_variables())
train_batch_size = 64
total_iterations = 0


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations
    # Start-time used for printing time-usage below.
    start_time = time.time()
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = train.next_batch(train_batch_size)
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            # Print it.
            print(msg.format(i + 1, acc))
    # Update the total number of iterations performed.
    total_iterations += num_iterations
    # Ending time.
    end_time = time.time()
    # Difference between start and end-times.
    time_dif = end_time - start_time
    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


test_batch_size = 256


# def predict_image(image):
#     feed_dict = {x: [image],
#             y_true: data.test.labels[0:0]}
#     cls_pred = session.run(y_pred_cls, feed_dict = feed_dict)
#     return cls_pred[0]


optimize(num_iterations=10000)


def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.
    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}
    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)
    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]
    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)
    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]
            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def TF_eval(sess, y):
    def TF_weighted(player, board):
        game_x = othello2TF(board)
        y_pred = sess.run(tf.nn.softmax(y), feed_dict={x: game_x})[0]
        player = 1 if player == "o" else 0
        return y_pred[player]
    return TF_weighted


def othello2TF(board):
    x = np.array(board)
    x = np.reshape(np.reshape(x, (10, 10))[1:9,1:9], 64)
    x = np.array([1 if sq == "o" else -1 if sq == "@" else 0 for sq in x], np.float32)
    x = np.reshape(x, (-1, 64))
    return x


def TF2othello(x):
    board = np.zeros((10, 10))
    board[:] = 15
    x = np.reshape(x, (8, 8))
    board[1:9, 1:9] = x
    board = list(np.reshape(board, 100))
    board = ["o" if sq == 1 else "@" if sq == -1 else "?" if sq == 15 else  "." for sq in board]
    return board


def play_testgame():
    conv_player = othello.alphabeta_searcher(2, TF_eval(session, y_pred))
    opponent = othello.alphabeta_searcher(4, othello.weighted_score(othello.SQUARE_WEIGHTS))
    game, winner = othello.play_game(conv_player, opponent)
    game_xs = []
    for board in game:
        game_xs.append(othello2TF(board))
    game_xs = np.reshape(game_xs, (-1, 64))
    y = session.run(tf.nn.softmax(y_pred), feed_dict={x: game_xs})
    for board, prediction in zip(game, y):
        print(othello.print_board(board))
        print("TF-graph predicts {:.2f}% chance of win".format(prediction[0]*100))
        input()

play_testgame()
session.close()
