import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import pickle


def generateSeq(inputDim, outputDim, structure):
    inputs = keras.Input(shape=(inputDim,), name="digits")
    x = layers.Dense(structure[0], activation="relu", name="dense_0")(inputs)
    for i in range(1, len(structure)):
        x = layers.Dense(structure[i], activation="relu", name="dense_{}".format(i))(x)
    outputs = layers.Dense(outputDim, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def rnn(x, y, structure, test_x, test_y, batchsize=10000):
    tf.reset_default_graph()
    tf_X = tf.placeholder(tf.float32, (None, x.shape[1]))  # =>X
    tf_y = tf.placeholder(tf.float32, (None, y.shape[1]))  # =>y
    tf.add_to_collection("inputs", tf_X)
    tf.add_to_collection("inputs", tf_y)
    output = tf.layers.dense(tf_X, structure[0], tf.nn.relu, name="hidden0")
    for i in range(1, len(structure)):
        output = tf.layers.dense(output, structure[i], tf.nn.relu, name="hidden{}".format(i))
    output = tf.layers.dense(output, 1, name='output')  # one output layer
    tf.add_to_collection("outputs", output)

    # loss_ = tf.losses.mean_squared_error(tf_y,output) # mse
    # loss_ = tf.reduce_mean(tf.sqrt(tf.pow(tf_y - output, 2))) #rmse
    loss_ = tf.reduce_mean(tf.keras.losses.mean_squared_logarithmic_error(tf_y, output)) #mlse
    tf.add_to_collection("loss", loss_)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss_)
    max_step = 1000
    global_step = 0
    test_errs = []
    train_errs = []
    last_pred = None
    last_idx = None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        last_err = None
        for step in range(1, max_step+1):
            global_step = step
            idx = np.random.choice(np.arange(len(x)), batchsize, replace=False)
            feed_x = x[idx]
            feed_y = y[idx]
            _, err, pred = sess.run([train_op, loss_, output], feed_dict={tf_X: feed_x, tf_y: feed_y})
            idx = np.random.choice(np.arange(len(test_x)), batchsize, replace=False)
            feed_x_test = test_x[idx]
            feed_y_test = test_y[idx]
            test_err, pred_test = sess.run([loss_, output], feed_dict={tf_X: feed_x_test, tf_y: feed_y_test})
            test_errs.append(test_err)
            train_errs.append(err)
            if step % 100 == 0:
                print('step:', step, 'loss:', err)
            if step == max_step or np.isnan(err):
                test_err, last_pred = sess.run([loss_, output], feed_dict={tf_X: test_x, tf_y: test_y})
                last_idx = idx
            # if last_err is not None and abs(last_err - np.mean(train_errs[-10:])) < 1e-6:
            #     break
            last_err = err
        saver = tf.train.Saver()
        model_type_name = '_'.join(str(x) for x in structure)
        saver.save(sess, '../models/rnn_{}.model'.format(model_type_name), global_step=global_step)
    print('final_result:', last_err)
    plt.figure(figsize=(10, 10))
    x = [i[0] for i in test_y]
    y = [i[0] for i in last_pred]

    plt.scatter(x, y, s=5, alpha=0.3)

    plt.plot([i for i in range(0, 6000)], [i for i in range(0,6000)])
    plt.ylabel('Predicted duration')
    plt.xlabel('True duration')
    plt.show()
    return train_errs, test_errs, global_step


def train(train_data, model):
    x_train, y_train = train_data
    # x_val, y_val = test_data
    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.MAE,
        # List of metrics to monitor
        metrics=[keras.metrics.MAE, keras.metrics.MSLE],
    )
    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=2,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_split=0.1,
    )
    print(history.history)
    return model


def evaluate(test, model):
    x_test, y_test = test
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)


def evaluateRnn(x, y, batchsize=10000):
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../models/rnn_128_128_128.model-1000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../models/'))
        graph = sess.graph
        tf_X, tf_y = graph.get_collection_ref("inputs")
        loss_ = graph.get_collection_ref("loss")[0]
        idx = np.random.choice(np.arange(len(x)), batchsize, replace=False)
        feed_x_test = x[idx]
        feed_y_test = y[idx]
        loss = sess.run([loss_], feed_dict={tf_X: feed_x_test, tf_y: feed_y_test})
    return loss

def predict(x):
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../models/rnn_128_128_128.model-1000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../models/'))
        graph = sess.graph
        tf_X, tf_y = graph.get_collection_ref("inputs")
        loss_ = graph.get_collection_ref("loss")[0]
        output_ = graph.get_collection_ref("outputs")[0]
        print(x.shape)
        y = np.array([[1] for _ in range(len(x))])
        print(y.shape)
        pred = sess.run([output_], feed_dict={tf_X: x, tf_y: y})
    return pred


if __name__ == '__main__':
    X = h5py.File('../data/temp4_imputation.H5')
    y = h5py.File('../data/temp4_y.H5')
    X_test = np.array(X['df_test'])
    X_train = np.array(X['df_train'])
    y_test = np.array(y['y_test'])
    y_train = np.array(y['y_train'])
    X_p = h5py.File('../data/pred_df.H5')
    X_pred = np.array(X_p['pred_df'])
    Seq = False
    Rnn = False
    RnnEva = False
    predictRnn = True
    if Seq:
        model = generateSeq(X_train.shape[1], 1, [64, 64])
        model = train((X_train, y_train), model)
        model.save('../models/Seq')
    if Rnn:
        graph_name = '59 feature'
        train_errs, test_errs, global_step = rnn(X_train, np.array([y_train]).T, [128,128,128], X_test, np.array([y_test]).T)
        # plt.plot([i for i in range(global_step)], train_errs, label='train')
        # plt.plot([i for i in range(global_step)], test_errs, label='test')
        # plt.ylabel('Training Steps')
        # plt.xlabel('Loss')
        # plt.legend()
        # plt.savefig('../results/{}.png'.format(graph_name))
    if RnnEva:
        print(evaluateRnn(X_test, np.array([y_test]).T))
        exit()
        x = np.linspace(-1, 1, 100)[:, np.newaxis]  # <==>x=x.reshape(100,1)
        noise = np.random.normal(0, 0.1, size=x.shape)
        y = np.power(x, 2) + x + noise  # y=x^2 + x+ noise
        print(x.shape, y.shape)
        print(X_test.shape, np.array([y_test]).T.shape)
    if predictRnn:
        pred = predict(X_pred)
        with open("../results/predictions.txt", "wb") as fp:  # Pickling
            pickle.dump(pred, fp)
