from __future__ import print_function

from flip_gradient import flip_gradient
from utils import *
from tensorflow import set_random_seed
import argparse

INPUT_DIM = 2048
NUM_CLASS = 65


class OfficeModel:

    def __init__(self, input_feature_size, z_hidden_size):
        self.input_shape = input_feature_size
        self.z_hidden_size = z_hidden_size

    def influence(self, x, is_reuse, l, is_training, dropout):
        with tf.variable_scope("feature_extractor", reuse=is_reuse):
            inner_code = self.feature_encoder(x=x, is_training=is_training, dropout_rate=dropout)

        with tf.variable_scope("classifier_1", reuse=is_reuse):
            output = self.label_prediction(feature=inner_code, is_reuse=is_reuse)
            softmax_outputs = tf.nn.softmax(output, axis=1)

        with tf.variable_scope("classifier_2", reuse=is_reuse):
            features_adv = flip_gradient(inner_code, l)
            outputs_adv = self.label_prediction(feature=features_adv, is_reuse=is_reuse)

        return inner_code, output, softmax_outputs, outputs_adv, tf.nn.softmax(outputs_adv)

    def feature_encoder(self, x, is_training, dropout_rate):
        W_feature_exact_0 = weight_variable(shape=[self.input_shape, 2000],
                                            name="feature_extractor_weight_1")
        b_feature_exact_0 = bias_variable(shape=[2000], name="feature_extractor_biases_1")
        h_feature_exact_0 = tf.nn.relu(tf.matmul(x, W_feature_exact_0) + b_feature_exact_0)
        h_feature_exact_0 = tf.layers.dropout(h_feature_exact_0, training=is_training, rate=dropout_rate)

        W_feature_exact_1 = weight_variable(shape=[2000, self.z_hidden_size], name="feature_extractor_weight_2")
        b_feature_exact_1 = bias_variable(shape=[self.z_hidden_size], name="feature_extract_biases_2")
        h_feature_exact_1 = tf.nn.relu(tf.matmul(h_feature_exact_0, W_feature_exact_1) + b_feature_exact_1)
        h_feature_exact_1 = tf.layers.dropout(h_feature_exact_1, training=is_training, rate=dropout_rate)

        return h_feature_exact_1

    def label_prediction(self, feature, is_reuse):
        with tf.variable_scope('label_predictor', reuse=is_reuse):
            W_fc0 = weight_variable(shape=[self.z_hidden_size, NUM_CLASS], name="fc0_w")
            b_fc0 = bias_variable(shape=[NUM_CLASS], name="fc0_b")
            # label_logits = tf.nn.relu(tf.matmul(feature, W_fc0) + b_fc0)
            label_logits = tf.matmul(feature, W_fc0)

            # W_fc1 = weight_variable(shape=[100, NUM_CLASS], name="fcd_w1")
            # b_fc1 = bias_variable(shape=[NUM_CLASS], name="fcd_b1")
            # label_logits = tf.matmul(h_fc0, W_fc1) + b_fc1

        return label_logits


def get_one_hot_label(y):
    n_values = np.max(y) + 1
    one_hot_label = np.eye(n_values)[y]
    return one_hot_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lizijian/workspace/pytorch_da_pipeline/feature_dataset", type=str)
    parser.add_argument("--data_dir", default="Office-Home_resnet50", type=str)
    parser.add_argument("--exp_name", default="officehome", type=str)
    parser.add_argument("--src", default="Art", type=str)
    parser.add_argument("--tgt", default="Clipart", type=str)
    parser.add_argument("--domain_num", default=2, type=int)
    parser.add_argument("--z_hidden_state", default=256, type=int)  # fixed
    parser.add_argument("--dropout_rate", default=0.7, type=float)  # fixed
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--l2_weight", default=5e-4, type=float)  # 5e-3, 1e-3, 3e-3, 7e-3, 1e-2
    parser.add_argument("--learning_rate", default=0.025, type=float)
    parser.add_argument("--srcweight", default=0.4, type=float)
    parser.add_argument("--momentum", default=0.8, type=float)
    args = parser.parse_args()

    data_dir = args.data_dir
    source_name = args.src
    target_name = args.tgt
    set_random_seed(1)
    source_train_input, source_train_label, \
    target_train_input, target_train_label, \
    target_test_input, target_test_label = \
        load_resnet_office(args.root, source_name=source_name, target_name=target_name, data_folder=data_dir)

    source_train_y = get_one_hot_label(source_train_label)
    target_train_y = get_one_hot_label(target_train_label)
    target_test_y = get_one_hot_label(target_test_label)

    print(source_train_input.shape)
    print(source_train_label.shape)
    print(target_train_input.shape)
    print(target_train_label.shape)
    print(target_test_input.shape)
    print(target_test_label.shape)

    batch_size = 256
    num_steps = 120000

    graph = tf.get_default_graph()
    set_random_seed(args.seed)

    with graph.as_default():
        model = OfficeModel(input_feature_size=INPUT_DIM, z_hidden_size=args.z_hidden_state)

        source_input = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM])
        source_label = tf.placeholder(dtype=tf.int32, shape=[None, NUM_CLASS])
        target_input = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM])
        target_label = tf.placeholder(dtype=tf.int32, shape=[None, NUM_CLASS])

        learning_rate = tf.placeholder(tf.float32, [])

        alpha = tf.placeholder(tf.float32, [])
        train_mode = tf.placeholder(tf.bool, [])

        # influence(self, x, is_reuse, l, is_training, dropout):
        # inner_code, output, softmax_outputs, tf.nn.softmax(outputs_adv)
        _, src_outputs, src_softmax_outputs, src_outputs_adv, src_softmax_outputs_adv = \
            model.influence(x=source_input, is_reuse=False, l=alpha,
                            is_training=train_mode, dropout=args.dropout_rate)

        _, tgt_outputs, tgt_softmax_outputs, tgt_outputs_adv, tgt_softmax_outputs_adv = \
            model.influence(x=target_input, is_reuse=True, l=alpha,
                            is_training=train_mode, dropout=args.dropout_rate)

        classifier_loss = \
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=src_outputs, labels=source_label))

        target_adv_src = tf.reduce_max(src_outputs, axis=1)
        target_adv_tgt = tf.reduce_max(tgt_outputs, axis=1)
        # print(src_outputs)
        # print(source_label)
        # print("================")
        # print(src_outputs_adv)
        # print(target_adv_src)
        # exit()

        classifier_loss_adv_src = \
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=src_outputs_adv,
                                                                      labels=tf.one_hot(tf.cast(target_adv_src, tf.int32),
                                                                                        NUM_CLASS)))

        # negative likelihood
        dist = tf.distributions.Categorical(tf.nn.softmax(tgt_softmax_outputs_adv, axis=1))
        classifier_loss_adv_tgt = - tf.reduce_sum(dist.log_prob(target_adv_tgt))

        transfer_loss = args.srcweight * classifier_loss_adv_src + 0.1 * classifier_loss_adv_tgt

        total_loss = classifier_loss + transfer_loss
        dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.80).minimize(total_loss)

        target_correct_label = tf.equal(tf.argmax(target_label, 1), tf.argmax(tgt_outputs, 1))
        target_label_acc = tf.reduce_mean(tf.cast(target_correct_label, tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            tf.global_variables_initializer().run()

            gen_source_batch = batch_generator([source_train_input, source_train_y], batch_size)
            gen_target_batch = batch_generator([target_train_input, target_train_y], batch_size)

            best_result = 0.0
            lr = args.learning_rate
            for global_steps in range(num_steps):
                p = float(global_steps) / num_steps
                L = min(2 / (1. + np.exp(-10. * p)) - 1, 0.20)
                lr = max(args.learning_rate / (1. + 10 * p) ** 0.55, 0.001)

                X0, y0 = next(gen_source_batch)
                X1, y1 = next(gen_target_batch)

                _, batch_classifier_loss_adv_src, batch_classifier_loss_adv_tgt, batch_classifier_loss\
                    = session.run([dann_train_op, classifier_loss_adv_src, classifier_loss_adv_tgt, classifier_loss],
                                  feed_dict={source_input: X0, source_label: y0, target_input: X1, learning_rate: lr,
                                             alpha: L, train_mode: True})

                if global_steps % 100 == 0:

                    target_label_accuracy = session.run(target_label_acc,
                                                        feed_dict={target_input: target_test_input,
                                                                   target_label: target_test_y,
                                                                   train_mode: False})

                    if target_label_accuracy > best_result:
                        best_result = target_label_accuracy

                print("global steps:", global_steps, " result", target_label_accuracy, "best result: ", best_result)

