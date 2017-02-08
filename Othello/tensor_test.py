# General imports
import numpy as np
from input_data import input_data
import tensorflow as tf

# Used for pretty output
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Othello specific
import othello


def plot_output(boards, winners, cls_pred=None):
    assert len(boards) == len(winners) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    string_boards = []
    for board in boards:
        temp = TF2othello(board)
        temp = ["O" if sq == "o" else "X" if sq == "@" else sq for sq in temp]
        string_boards.append(temp)

    for i, ax in enumerate(axes.flat):
        # Create image
        printable_board = othello.print_board(string_boards[i])
        im = Image.new('RGB', (105, 130), "white")
        d = ImageDraw.Draw(im)
        font = ImageFont.truetype("UbuntuMono-B.ttf", 12)
        d.text((2, 0), printable_board, "black", font)
        # Plot image.
        ax.imshow(im)

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "Winner: {0}".format(winners[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(winners[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


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


def TF_eval(sess, y):
    def TF_weighted(player, board):
        game_x = othello2TF(board)
        y_pred = sess.run(tf.nn.softmax(y), feed_dict={x: game_x})[0]
        player = 1 if player == "o" else 0
        return y_pred[player]
    return TF_weighted


# Othello board with 8*8 squares
data_size = 64

# Win or lose
num_categories = 2

test = input_data("data/NN_saved_games.dat")
train = input_data("data/NN_saved_games.dat")

batch_size = 100
num_networks = 4

# Load test data
test_boards, test_winners = test.next_batch(100)


x = tf.placeholder(tf.float32, [None, data_size])

W_list = [tf.Variable(tf.zeros([data_size, num_categories])) for _ in range(num_networks)]
b_list = [tf.Variable(tf.zeros([num_categories])) for _ in range(num_networks)]

y_list = []

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, num_categories])

for W, b in zip(W_list, b_list):
    y_list.append(tf.matmul(x, W) + b)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for y in y_list:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # Train
    for _ in range(100):
        batch_xs, batch_ys = train.next_batch(batch_size)
        batch_xs = np.reshape(batch_xs, (-1, data_size))
        batch_ys = np.reshape(batch_ys, (-1, num_categories))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    # y = x*W + b; y_ and x from data
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: test_boards,y_: test_winners}))

for W, b in zip(W_list, b_list):
    val = W.eval(sess)
    val = val[:,1]
    val = np.reshape([round(v, 2) for v in val], (8,8))
    np.set_printoptions(precision=3)
    print("Bias of {} and following weight matrix:".format(b.eval(sess)))
    for row in val:
        print(row)
    print()


othello_strategies = []

for y, W, b in zip(y_list, W_list, b_list):
    w = (W.eval(sess)[:,1] + b.eval(sess)[1])*100
    w = np.reshape(w, (8,8))
    weights = np.zeros((10, 10))
    weights[1:9,1:9] = w
    weights = np.reshape(weights, 100)
    print(weights)
    othello_strategies.append(othello.alphabeta_searcher(4, othello.weighted_score(weights)))

for i in range(len(othello_strategies)):
    for j in range(i, len(othello_strategies)):
        # Play a test game
        game, winner = othello.play_game(othello_strategies[i], othello_strategies[j])

        game_xs = []
        for board in game:
            game_xs.append(othello2TF(board))
        game_xs = np.reshape(game_xs, (-1, 64))
        y_pred = sess.run(tf.nn.softmax(y_list[i]), feed_dict={x: game_xs})
        for board, prediction in zip(game, y_pred):
            print(othello.print_board(board))
            print("TF-graph predicts {:.2f}% chance of win".format(prediction[0]*100))
            input()
        # print(othello.print_board(TF2othello(test_data[0])))
        # plot_output(test_data, y_pred)


