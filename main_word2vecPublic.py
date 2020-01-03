import argparse
import tensorflow as tf
import pickle
import math
import numpy as np
FLAGS = None


# Converts the label to one-hot code
def transform_one_hot(labels):
    one_hot = np.eye(3)[labels]
    return one_hot

def load_data():
    """
    Load data13 from pickle
    :return: Arrays
    """
    with open(FLAGS.source_data, 'rb') as f:
        data_x = pickle.load(f)
        data_y = pickle.load(f)
        test_x = pickle.load(f)
        test_y = pickle.load(f)
        dev_x = pickle.load(f)
        dev_y = pickle.load(f)
        texts = pickle.load(f)
        # id2word = pickle.load(f)
        word2id = pickle.load(f)
        tag2id = pickle.load(f)
        id2tag = pickle.load(f)
        test_weight = pickle.load(f)
        train_weight = pickle.load(f)
        dev_weight = pickle.load(f)
        train_sentence_len = pickle.load(f)
        test_sentence_len = pickle.load(f)
        dev_sentence_len = pickle.load(f)
        return data_x, data_y, test_x, test_y, dev_x, dev_y, texts, word2id, tag2id, id2tag, test_weight, train_weight, dev_weight, \
               train_sentence_len, test_sentence_len, dev_sentence_len


# randomly initial weight
def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial, name='w')

# randomly initial bias
def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial, name='b')

# initial lstm
def lstm_cell(num_units, keep_prob=1):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)


def main():
    # Load data13
    train_x, train_y, test_x, test_y, dev_x, dev_y, texts, word2id, tag2id, id2tag, test_weight, train_weight, dev_weight,\
        train_sentence_len, test_sentence_len, dev_sentence_len= load_data()
    train_weight = train_weight.astype(np.float64)
    test_weight = test_weight.astype(np.float64)
    dev_weight = dev_weight.astype(np.float64)

    train_loss_w = transform_one_hot(train_y)
    dev_loss_w = transform_one_hot(dev_y)
    test_loss_w = transform_one_hot(test_y)

    print('train_x_size:', len(train_sentence_len))
    train_steps = math.ceil(train_x.shape[0] / FLAGS.train_batch_size)
    print("train_x.shape[0]:", train_x.shape[0], "train_steps:", train_steps, "FLAGS.train_batch_size",
          FLAGS.train_batch_size)
    dev_steps = math.ceil(dev_x.shape[0] / FLAGS.dev_batch_size)
    print("dev_x.shape[0]:", dev_x.shape[0], "dev_steps:", dev_steps, "FLAGS.dev_batch_size",
          FLAGS.dev_batch_size)
    test_steps = math.ceil(test_x.shape[0] / FLAGS.test_batch_size)
    print("test_x.shape[0]:", test_x.shape[0], "test_steps:", test_steps, "FLAGS.test_batch_size",
          FLAGS.test_batch_size)
    vocab_size = len(word2id) + 1
    print('Vocab Size', vocab_size)

    # global_step = tf.Variable(-1, trainable=False, name='global_step')
    global_step = tf.Variable(-1, trainable=True, name='global_step')

    # Train and dev dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    # train_dataset = tf.data13.Dataset.from_tensor_slices(train_x)
    train_dataset = train_dataset.batch(FLAGS.train_batch_size)

    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x, dev_y))
    dev_dataset = dev_dataset.batch(FLAGS.dev_batch_size)


    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.batch(FLAGS.test_batch_size)

    train_weight_dataset = tf.data.Dataset.from_tensor_slices(train_weight)
    train_weight_dataset = train_weight_dataset.batch(FLAGS.train_batch_size)

    test_weight_dataset = tf.data.Dataset.from_tensor_slices(test_weight)
    test_weight_dataset = test_weight_dataset.batch(FLAGS.test_batch_size)

    dev_weight_dataset = tf.data.Dataset.from_tensor_slices(dev_weight)
    dev_weight_dataset = dev_weight_dataset.batch(FLAGS.dev_batch_size)

    train_len_dataset = tf.data.Dataset.from_tensor_slices(train_sentence_len)
    train_len_dataset = train_len_dataset.batch(FLAGS.train_batch_size)

    test_len_dataset = tf.data.Dataset.from_tensor_slices(test_sentence_len)
    test_len_dataset = test_len_dataset.batch(FLAGS.test_batch_size)

    dev_len_dataset = tf.data.Dataset.from_tensor_slices(dev_sentence_len)
    dev_len_dataset = dev_len_dataset.batch(FLAGS.dev_batch_size)

    train_lossW_dataset = tf.data.Dataset.from_tensor_slices(train_loss_w)
    train_lossW_dataset = train_lossW_dataset.batch(FLAGS.train_batch_size)

    test_lossW_dataset = tf.data.Dataset.from_tensor_slices(test_loss_w)
    test_lossW_dataset = test_lossW_dataset.batch(FLAGS.test_batch_size)

    dev_lossW_dataset = tf.data.Dataset.from_tensor_slices(dev_loss_w)
    dev_lossW_dataset = dev_lossW_dataset.batch(FLAGS.test_batch_size)

    # A reinitializable iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    iterator_w = tf.data.Iterator.from_structure(train_weight_dataset.output_types, train_weight_dataset.output_shapes)
    iterator_len = tf.data.Iterator.from_structure(train_len_dataset.output_types, train_len_dataset.output_shapes)
    iterator_lossW = tf.data.Iterator.from_structure(train_lossW_dataset.output_types, train_lossW_dataset.output_shapes)
    with tf.name_scope('train_dataset_initial'):
        train_initializer = iterator.make_initializer(train_dataset)
    with tf.name_scope('dev_dataset_initial'):
        dev_initializer = iterator.make_initializer(dev_dataset)
    with tf.name_scope('test_dataset_initial'):
        test_initializer = iterator.make_initializer(test_dataset)
    with tf.name_scope('train_weight_dataset_initial'):
        tw_initializer = iterator_w.make_initializer(train_weight_dataset)

    with tf.name_scope('test_weight_dataset_initial'):
        te_initializer = iterator_w.make_initializer(test_weight_dataset)

    with tf.name_scope('dev_weight_dataset_initial'):
        de_initializer = iterator_w.make_initializer(dev_weight_dataset)

    train_len_initializer = iterator_len.make_initializer(train_len_dataset)
    dev_len_initializer = iterator_len.make_initializer(dev_len_dataset)
    test_len_initializer = iterator_len.make_initializer(test_len_dataset)

    train_lossW_initializer = iterator_lossW.make_initializer(train_lossW_dataset)
    dev_lossW_initializer = iterator_lossW.make_initializer(dev_lossW_dataset)
    test_lossW_initializer = iterator_lossW.make_initializer(test_lossW_dataset)
    # Input Layer
    # with tf.variable_scope('inputs'):
    with tf.name_scope('inputs'):
        x, y_label = iterator.get_next()
        tw = iterator_w.get_next()
        sentence_len = iterator_len.get_next()
        lossW = iterator_lossW.get_next()
        print("tw:", tw)

    x = tf.cast(x, dtype=tf.float32)
    # the input of the network
    inputs = x
    # Variables
    keep_prob = tf.placeholder(tf.float32, [])
    is_train = tf.placeholder(tf.bool)
    # st = tf.placeholder(tf.int32, [])
    with tf.name_scope('biLSTM_Cell_Layer'):
        # RNN Layer
        # cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)])
        # cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)])
        with tf.name_scope("LSTM_Cell_fw"):
            cell_fw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
        with tf.name_scope("LSTM_Cell_bw"):
            cell_bw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]

        inputs = tf.unstack(inputs, FLAGS.time_step, axis=1)
        output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)

        output = tf.stack(output, axis=1)
        print('Output', output)
        output = tf.reshape(output, [-1, FLAGS.num_units * 2])
        output = tf.tanh(output)
        # output = tf.layers.batch_normalization(output, training= is_train)
        print('Output Reshape', output)
    with tf.name_scope('hidden1'):
        w1 = weight([FLAGS.num_units * 2, 128])
        b1 = bias([128])
        hidden1 = tf.matmul(output, w1) + b1
        hidden1 = tf.layers.batch_normalization(hidden1, training=is_train)
    # Output Layer
    # with tf.variable_scope('outputs'):
    with tf.name_scope('hidden2'):
        w2 = weight([128, FLAGS.num_units])
        b2 = bias([FLAGS.num_units])
        hidden = tf.matmul(hidden1, w2) + b2
        tw = tf.cast(tw, dtype=tf.float32)
        # word_weight = tw
        word_weight = tf.reshape(tw, [-1, 50])
        print("word_weight:", word_weight)
        hidden1 = tf.multiply(hidden, word_weight)
        hidden2 = tf.layers.batch_normalization(hidden1, training=is_train)  # 标准化
        print("hidden2:", hidden2)
    with tf.name_scope('outputs'):
        w4 = weight([50, FLAGS.category_num])
        b4 = bias([FLAGS.category_num])
        y = tf.matmul(hidden2, w4) + b4
        y = tf.layers.batch_normalization(y, training= is_train)
        y_ = tf.nn.softmax(y)
        print("Y:", y)
        y_predict = tf.cast(tf.argmax(y_, axis=1), tf.int32)  # tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引
        print('Output Y', y_predict)
    tf.summary.histogram('y_predict', y_predict)
    # 改变正在训练的数据中标签的维度，使其成为一维列向量
    y_label_reshape = tf.cast(tf.reshape(y_label, [-1]), tf.int32)
    # Change the dimension of the training sentence length to make it a one-dimensional vector
    # 改变正在训练句子长度的维度，使其成为一维向量
    sentence_len_reshape = tf.cast(tf.reshape(sentence_len, [-1]), tf.int32)
    # The length of the sentence is mapped to a bool matrix, and the number of true is the number of Chinese characters in the sentence
    # 将句子长度映射成为bool矩阵，true的个数为句中汉字的个数
    loss_mask = tf.sequence_mask(tf.to_int32(sentence_len_reshape), tf.to_int32(FLAGS.time_step))
    # The bool matrix is transformed into a numerical matrix to eliminate the error loss caused by tail filling
    # 将bool矩阵转化为数值矩阵，目的是消除尾填充造成的损失误差
    loss_mask = tf.cast(tf.reshape(loss_mask, [-1]), tf.float32)
    print('loss_mask:', loss_mask)
    # The predicted value of the tail fill is not considered
    y_predict = tf.cast(y_predict, tf.float32) * loss_mask
    # y_predict = tf.cast(y_predict, tf.float32)
    yy_label_reshape = tf.cast(y_label_reshape, tf.float32) * loss_mask
    correct_prediction = tf.cast(tf.equal(y_predict, yy_label_reshape), tf.float32)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # accuracy = correct_sum / mask_sum
        tf.summary.scalar('accuracy', accuracy)
    ww = tf.constant(50, dtype=tf.float32)
    # Loss
    with tf.name_scope('loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,
                                                                logits=tf.cast(y, tf.float32))*loss_mask

        loss_sum = tf.reduce_sum(loss)

        mask = tf.reduce_sum(loss_mask)    # 统计一句话中的实际的字数
        cross_entropy = loss_sum/mask * ww
        tf.summary.scalar('loss', cross_entropy)
    print("y_type,y_label.type", type(y.shape), type(y_label_reshape.shape))
    print('Prediction', correct_prediction, 'Accuracy', accuracy)

    # Train
    train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)

    # Summaries
    # 合并所有的summary
    summaries = tf.summary.merge_all()

    # Saver
    saver = tf.train.Saver()

    # Iterator
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Global step
    gstep = 0
    # writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'train'),
    #                                sess.graph)

    writer = tf.summary.FileWriter(FLAGS.summaries_dir,
                                   sess.graph)


    if FLAGS.train:

        if tf.gfile.Exists(FLAGS.summaries_dir):
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)

        for epoch in range(FLAGS.epoch_num):
            print('epoch:', epoch, "epoch_num:", FLAGS.epoch_num)
            tf.train.global_step(sess, global_step_tensor=global_step)

            # Train 获取训练的数据
            sess.run(train_initializer)
            sess.run(tw_initializer)
            sess.run(train_len_initializer)
            sess.run(train_lossW_initializer)
            # yy_label_shape, yy = sess.run([y_label_reshape, y_predict], feed_dict={keep_prob: FLAGS.keep_prob})
            # print("yy_label_shape:",yy_label_shape)
            # print("yy:", yy)
            for step in range(int(train_steps)):
                smrs, loss, acc, gstep, _ = sess.run([summaries, cross_entropy, accuracy, global_step, train],
                                                     feed_dict={keep_prob: FLAGS.keep_prob, is_train:True})
                # Print log
                if step % FLAGS.steps_per_print == 0:
                    print('Global Step', gstep, 'Step', step, 'Train Loss', loss, 'Accuracy', acc)


                # Summaries for tensorboard
                if gstep % FLAGS.steps_per_summary == 0:
                    writer.add_summary(smrs, gstep)
                    print('Write summaries to', FLAGS.summaries_dir)
                if loss <= 0.025:
                    saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)
                    return

            # 验证数据训练
            if epoch % FLAGS.epochs_per_dev == 0:
                # Dev
                sess.run(dev_initializer)
                sess.run(de_initializer)
                sess.run(dev_len_initializer)
                sess.run(dev_lossW_initializer)
                for step in range(int(dev_steps)):
                    if step % FLAGS.steps_per_print == 0:
                        print('Dev Accuracy', sess.run(accuracy,  feed_dict={keep_prob: FLAGS.keep_prob, is_train:True}),
                              'Step', step)

            # Save model

            if epoch % FLAGS.epochs_per_save == 0:
                saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)


        # plot_learning_curves(accuracy)

    else:
        ckpt = tf.train.get_checkpoint_state('ckpt4')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from', ckpt.model_checkpoint_path)
        sess.run(test_initializer)
        sess.run(te_initializer)
        sess.run(test_len_initializer)
        sess.run(test_lossW_initializer)
        for step in range(int(test_steps)):
            x_results, y_predict_results, acc, y_label_results, hidden_ = sess.run([x, y_predict, accuracy, y_label_reshape, hidden],
                                                                          feed_dict={keep_prob:FLAGS.keep_prob, is_train:True})
            print('Test step', step, 'Accuracy', acc)
            y_predict_results = np.reshape(y_predict_results, (x_results.shape[0], x_results.shape[1]))
            y_label_results = np.reshape(y_label_results, (x_results.shape[0], x_results.shape[1]))
            f = 0
            TE = 0
            SE = 0    # of sentences the evaluated system reported to have errors
            DC = 0    # of sentences with correctly detected results
            DE = 0    # of sentences with correctly detected errors
            FPE = 0   # of sentences with false positive errors
            ANE = 0   # of testing sentences without errors
            ALL = 0   # of all testing sentences
            AWE = 0   # of testing sentences with errors
            CLD = 0   # of sentences with correct location detection
            CEL = 0   # of sentences with correct error locations

            for i in range(len(x_results)):
                # print('hidden:', hidden_[i])
                y_predict_result, y_label_result = list(filter(lambda x: x, y_predict_results[i])), list(filter(lambda x: x, y_label_results[i]))
                y_predict_text, y_label_text = ''.join(id2tag[y_predict_result].values), \
                                               ''.join(id2tag[y_label_result].values)
                index = step * len(x_results) +  i
                if y_predict_text == y_label_text:
                    f += 1
                print(texts[index])
                print(y_predict_text, "  ", y_label_text, ' f', f)
                ALL += 1
                if 'e' in y_predict_text:
                    SE += 1
                if 'e' in y_label_text:
                    TE += 1                    # of testing sentences with errors
                if ('e' not in y_label_text and 'e' not in y_predict_text) or ('e' in y_label_text and 'e' in y_predict_text):
                    DC += 1
                if 'e' in y_label_text and 'e' in y_predict_text:
                    DE += 1
                # if ('e' in y_label_text and 'e' not in y_predict_text) or ('e' in y_predict_text and 'e' not in y_label_text):
                if ('e' in y_predict_text and 'e' not in y_label_text):
                    FPE += 1
                if 'e' not in y_label_text:
                    ANE += 1
                if 'e' in y_label_text:
                    AWE += 1
                if y_predict_text == y_label_text:
                    CLD += 1
                if y_predict_text == y_label_text and 'e' in y_predict_text:
                    CEL += 1
            print('SE:', SE, 'TE:', TE, 'DE:', DE, 'DC:', DC, 'FPE:', FPE, 'ANE:', ANE,'ALL:', ALL,  'AWE:', AWE, 'CLD:', CLD, 'CEL:', CEL)
            FAR = FPE / ANE
            DA  =  DC / ALL
            DP = DE / SE
            DR = DE / TE
            DF1 = 2 * DP * DR / (DP + DR)
            ELA = CLD / ALL
            ELP = CEL / SE
            ELR = CEL / TE
            ELF1 = 2 * ELP * ELR / (ELP + ELR)
            print('FAR:', FAR, 'DA:', DA, 'DP:', DP, 'DR:', DR, 'DF1:', DF1, 'ELA:', ELA, 'ELP:', ELP, 'ELR:', ELR, 'ELF1:', ELF1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BI LSTM')
    parser.add_argument('--train_batch_size', help='train batch size', default=15)
    parser.add_argument('--dev_batch_size', help='dev batch size', default=50)
    parser.add_argument('--test_batch_size', help='test batch size', default=6409)  # The number of samples per test on the model
    parser.add_argument('--source_data', help='source size', default='./data/data_word3vec.pkl')
    parser.add_argument('--num_layer', help='num of layer', default=2, type=int)
    parser.add_argument('--num_units', help='num of units', default=50, type=int)  #num_units is the number of the lstm cell
    # parser.add_argument('--time_step', help='time steps', default=32, type=int)
    parser.add_argument('--time_step', help='time steps', default=50, type=int)   # the length of the input sample
    parser.add_argument('--embedding_size', help='time steps', default=128, type=int)
    parser.add_argument('--category_num', help='category num', default=3, type=int) # number of categories
    parser.add_argument('--learning_rate', help='learning rate', default=0.01, type=float)
    parser.add_argument('--epoch_num', help='num of epoch', default=50, type=int)
    parser.add_argument('--epochs_per_test', help='epochs per test', default=10, type=int)
    parser.add_argument('--epochs_per_dev', help='epochs per dev', default=2, type=int)
    parser.add_argument('--epochs_per_save', help='epochs per save', default=2, type=int)
    parser.add_argument('--steps_per_print', help='steps per print', default=100, type=int)
    parser.add_argument('--steps_per_summary', help='steps per summary', default=100, type=int)
    parser.add_argument('--keep_prob', help='train keep prob dropout', default=1, type=float)
    # parser.add_argument('--keep_prob', help='train keep prob dropout', default=0.6, type=float)
    parser.add_argument('--checkpoint_dir', help='checkpoint dir', default='ckpt4/model.ckpt', type=str)
    parser.add_argument('--summaries_dir', help='summaries dir', default='summaries/', type=str)
    # Eliminating comments here can test the model performance
    parser.add_argument('--train', help='train', default=False, type=bool)
    # Eliminating comments here can train the model
    # parser.add_argument('--train', help='train', default=True, type=bool)
    FLAGS, args = parser.parse_known_args()
    main()
