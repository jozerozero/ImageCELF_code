from __future__ import print_function

from flip_gradient import flip_gradient
from utils import *
from tensorflow import set_random_seed
import argparse
import tensorboardX as tb
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_xent_two
from tensorbayes.layers import instance_norm
from PIL import Image
from plot_utils import plot_feature_tsne, plot_tsne

INPUT_DIM = 2048
NUM_CLASS = 12


class OfficeModel:

    def __init__(self, input_feature_size, d_input_size, d_hidden_size, z_hidden_size):
        self.input_shape = input_feature_size
        self.d_input_size = d_input_size
        self.d_hidden_size = d_hidden_size
        self.z_hidden_size = z_hidden_size

    def inference(self, x, is_reuse, l, is_training, domain, dropout_rate, only_classifier=False):
        with tf.variable_scope("inference", reuse=is_reuse):
            # x = tf.expand_dims(tf.expand_dims(x, axis=-1), axis=-1)
            # x = instance_norm(x)
            # x = tf.squeeze(tf.squeeze(x))
            h_feature_exact_1 = self.feature_encoder(x=x, is_training=is_training,
                                                     dropout_rate=dropout_rate, is_reuse=is_reuse)
            label_logits = self.label_prediction(feature=h_feature_exact_1, is_reuse=is_reuse)

            if only_classifier:
                return label_logits, h_feature_exact_1

        with tf.variable_scope("decode", reuse=is_reuse):
            domain_code = self.domain_encoder(d=domain, is_reuse=is_reuse)
            domain_code = tf.tile(tf.expand_dims(domain_code, axis=1), [1, self.z_hidden_size, 1])
            rec = self.decoder(latent_variable=h_feature_exact_1, domain_code=domain_code, is_reuse=is_reuse,
                               is_training=is_training, dropout_rate=dropout_rate)

            tran_domain_code = self.domain_encoder(d=tf.ones_like(domain)-domain, is_reuse=True)
            tran_domain_code = tf.tile(tf.expand_dims(tran_domain_code, axis=1), [1, self.z_hidden_size, 1])
            trans_rec = self.decoder(latent_variable=h_feature_exact_1, domain_code=tran_domain_code,
                                     is_reuse=True, is_training=is_training, dropout_rate=dropout_rate)

        with tf.variable_scope("trans", reuse=is_reuse):
            rec_h_feature_exact_1 = self.feature_encoder(x=rec, is_training=is_training,
                                                         dropout_rate=dropout_rate, is_reuse=is_reuse)
            rec_label_logits = self.label_prediction(feature=rec_h_feature_exact_1, is_reuse=is_reuse)

            rec_h_feature_exact_1 = self.feature_encoder(x=trans_rec, is_training=is_training,
                                                         dropout_rate=dropout_rate, is_reuse=True)
            trans_label_logits = self.label_prediction(feature=rec_h_feature_exact_1, is_reuse=True)

        with tf.variable_scope("domain_disc", reuse=is_reuse):
            domain_logits = self.domain_prediction(feature=h_feature_exact_1, is_reuse=is_reuse, l=l,
                                                   is_training=is_training)

        return label_logits, rec, trans_rec, rec_label_logits, trans_label_logits, domain_logits, h_feature_exact_1

    def decoder(self, latent_variable, domain_code, is_reuse, is_training, dropout_rate):
        with tf.variable_scope("feature_decoder", reuse=is_reuse):
            latent_variable = tf.expand_dims(latent_variable, axis=-1)
            feature = tf.concat([latent_variable, domain_code], axis=-1)

            feature = tf.reshape(feature, [-1, self.z_hidden_size + self.d_hidden_size * self.z_hidden_size])

            decoder_w0 = weight_variable(shape=[self.z_hidden_size + self.d_hidden_size * self.z_hidden_size,
                                                self.input_shape],
                                         name="decoder_weight_0")
            decoder_b0 = bias_variable(shape=[self.input_shape], name="decoder_biases_0")
            decoder_h0 = tf.matmul(feature, decoder_w0) + decoder_b0
            decoder_h0 = tf.nn.relu(decoder_h0)
            decoder_h0 = tf.layers.dropout(decoder_h0, training=is_training, rate=dropout_rate)
            decoder_h0 = tf.contrib.layers.layer_norm(decoder_h0, scale=True)

        return decoder_h0

    def domain_encoder(self, d, is_reuse):
        with tf.variable_scope("domain_encoder", reuse=is_reuse):
            W_d_feature_extract_0 = weight_variable(shape=[self.d_input_size, self.d_hidden_size],
                                                    name="doamin_feature_extractor_weight_1")
            b_d_feature_extract_0 = bias_variable(shape=[self.d_hidden_size], name="domain_feature__extract_biases_1")

            d_feature_extract_0 = tf.matmul(d, W_d_feature_extract_0) + b_d_feature_extract_0

        return d_feature_extract_0

    def feature_encoder(self, x, is_training, dropout_rate, is_reuse):
        with tf.variable_scope("feature_encoder", reuse=is_reuse):
            W_feature_exact_0 = weight_variable(shape=[self.input_shape, self.z_hidden_size],
                                                name="feature_extractor_weight_1")
            # b_feature_exact_0 = bias_variable(shape=[self.z_hidden_size], name="feature_extractor_biases_1")
            h_feature_exact_0 = tf.nn.relu(tf.matmul(x, W_feature_exact_0))
            h_feature_exact_0 = tf.layers.dropout(h_feature_exact_0, training=is_training, rate=dropout_rate)

            # W_feature_exact_1 = weight_variable(shape=[2000, self.z_hidden_size], name="feature_extractor_weight_2")
            # b_feature_exact_1 = bias_variable(shape=[self.z_hidden_size], name="feature_extract_biases_2")
            # h_feature_exact_1 = tf.nn.relu(tf.matmul(h_feature_exact_0, W_feature_exact_1) + b_feature_exact_1)
            # h_feature_exact_1 = tf.layers.dropout(h_feature_exact_1, training=is_training, rate=dropout_rate)

            return h_feature_exact_0

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

    def domain_prediction(self, feature, is_reuse, l, is_training):
        with tf.variable_scope("domain_predictor", reuse=is_reuse):
            feat = flip_gradient(feature, l)

            d_W_fc0 = weight_variable(shape=[self.z_hidden_size, 100], name="fcd_w0")
            d_b_fc0 = bias_variable(shape=[100], name="fcd_b0")
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)
            # d_h_fc0 = tf.layers.dropout(d_h_fc0, training=is_training)

            d_W_fc1 = weight_variable(shape=[100, 2], name="fcd_w1")
            d_b_fc1 = bias_variable(shape=[2], name="fcd_b1")
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

        return d_logits

    def normalize_perturbation(self, d, scope=None):
        with tf.name_scope(scope, 'norm_pert'):
            output = tf.nn.l2_normalize(d, axis=1)
        return output
        pass

    def pertub_image(self, x, logits, is_training, dropout_rate, radius):
        eps = 5e-6 * self.normalize_perturbation(tf.random_normal(shape=tf.shape(x)))

        # def inference(self, x, is_reuse, l, is_training, domain, dropout_rate, only_classifier=False):
        eps_p, _ = self.inference(x=x+eps, is_reuse=True, l=None, is_training=is_training, domain=None,
                                  dropout_rate=dropout_rate, only_classifier=True)
        loss = softmax_xent_two(labels=logits, logits=eps_p)

        eps_adv = tf.gradients(loss, [eps], aggregation_method=2)[0]
        eps_adv = self.normalize_perturbation(eps_adv)

        x_adv = tf.stop_gradient(x + radius * eps_adv)

        return x_adv

    def vta_loss(self, input, logits, is_training, dropout_rate, radius):
        x_adv = self.pertub_image(x=input, logits=logits, is_training=is_training,
                                  dropout_rate=dropout_rate, radius=radius)
        p_adv, _ = self.inference(x=x_adv, is_reuse=True, l=None, is_training=is_training, domain=None,
                                  dropout_rate=dropout_rate, only_classifier=True)
        loss = tf.reduce_mean(softmax_xent_two(labels=tf.stop_gradient(logits), logits=p_adv))
        return loss


def get_one_hot_label(y):
    n_values = np.max(y) + 1
    one_hot_label = np.eye(n_values)[y]
    return one_hot_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--root", default="/home/lizijian/workspace/additional/DA_JAE/data", type=str)
    parser.add_argument("--root", default="dataset", type=str)
    parser.add_argument("--data_dir", default="imagecelf_mat", type=str)
    parser.add_argument("--exp_name", default="officehome", type=str)
    parser.add_argument("--src", default="p", type=str)
    parser.add_argument("--tgt", default="c", type=str)
    parser.add_argument("--domain_num", default=2, type=int)
    parser.add_argument("--d_hidden_size", default=1, type=int)    # fixed
    parser.add_argument("--z_hidden_state", default=2048, type=int)    # fixed
    parser.add_argument("--dropout_rate", default=0.7, type=float)    #fixed
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--lambda_rec", default=0.1, type=float)
    parser.add_argument("--lambda_mi", default=0.001, type=float)
    parser.add_argument("--l2_weight", default=1e-4, type=float)  # 5e-3, 1e-3, 3e-3, 7e-3, 1e-2
    parser.add_argument("--learning_rate", default=0.04, type=float)
    # parser.add_argument("--tw", default=1e-2, type=float)
    parser.add_argument("--tw", default=1.0, type=float)
    parser.add_argument("--momentum", default=0.75, type=float)
    parser.add_argument("--radius", default=6.5, type=float)

    args = parser.parse_args()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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

    batch_size = 64
    num_steps = 25000

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    exp_folder = os.path.join(args.exp_name, "%s_%s" % (args.src, args.tgt))
    exp_name = "lr_%g_momentum_%g_seed_%d_lambda_rec_%g_lambda_mi_%g_dropout_%g_l2_%g_tw_%g_radius_%g" % \
               (args.learning_rate, args.momentum, args.seed, args.lambda_rec,
                args.lambda_mi, args.dropout_rate, args.l2_weight, args.tw, args.radius)
    exp_path = os.path.join("logs", exp_folder, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    if not os.path.exists(os.path.join("logs", exp_folder, "record.txt")):
        record_file = open(os.path.join("logs", exp_folder, "record.txt"), mode="w")
    else:
        record_file = open(os.path.join("logs", exp_folder, "record.txt"), mode="a")

    writer = tb.SummaryWriter(exp_path)

    graph = tf.get_default_graph()
    set_random_seed(args.seed)
    with graph.as_default():
        model = OfficeModel(input_feature_size=INPUT_DIM, d_input_size=args.domain_num,
                            d_hidden_size=args.d_hidden_size, z_hidden_size=args.z_hidden_state)

        source_input = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM])
        source_label = tf.placeholder(dtype=tf.int32, shape=[None, NUM_CLASS])
        target_input = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM])
        target_label = tf.placeholder(dtype=tf.int32, shape=[None, NUM_CLASS])

        source_domain = tf.placeholder(tf.float32, [None, 2])
        target_domain = tf.placeholder(tf.float32, [None, 2])
        learning_rate = tf.placeholder(tf.float32, [])

        alpha = tf.placeholder(tf.float32, [])
        train_mode = tf.placeholder(tf.bool, [])

        src_label_logits, src_rec, trans_rec, src_rec_label_logits, \
        trans_label_logits, src_domain_logits, src_inner_code = \
            model.inference(x=source_input, is_reuse=False, l=alpha, is_training=train_mode, domain=source_domain,
                            dropout_rate=args.dropout_rate)

        tgt_lable_logits, tgt_rec, _, _, _, tgt_domain_logits, tgt_inner_code = \
            model.inference(x=target_input, is_reuse=True, l=alpha, is_training=train_mode, domain=target_domain,
                            dropout_rate=args.dropout_rate)

        # def vta_loss(self, input, logits, is_training, dropout_rate, radius):
        loss_src_vat = model.vta_loss(input=source_input, logits=src_label_logits, is_training=train_mode,
                                      dropout_rate=args.dropout_rate, radius=args.radius)
        loss_tgt_vat = model.vta_loss(input=target_input, logits=tgt_lable_logits, is_training=train_mode,
                                      dropout_rate=args.dropout_rate, radius=args.radius)

        vat_loss = loss_src_vat + args.tw * loss_tgt_vat

        src_label_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=src_label_logits, labels=source_label))

        label_loss = src_label_loss

        # rec_label_logits = tf.concat([src_rec_label_logits, trans_label_logits], axis=0)
        # double_src_label = tf.concat([source_label, source_label], axis=0)
        # label_loss_2 = \
        #     tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=rec_label_logits,
        #                                                               labels=double_src_label))
        rec_src_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=src_rec_label_logits,
                                                                                 labels=source_label))
        trans_src_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=trans_label_logits,
                                                                                 labels=source_label))
        label_loss_2 = rec_src_loss + trans_src_loss

        # label_loss_2 = loss_rec_src_class + loss_trans_src_class

        domain_logits = tf.concat([src_domain_logits, tgt_domain_logits], 0)
        source_target_domain_label = tf.concat([source_domain, target_domain], 0)

        domain_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=domain_logits, labels=source_target_domain_label))

        # rec_logits = tf.concat([src_rec, tgt_rec], axis=0)
        # data_input = tf.concat([source_input, target_input], axis=0)

        src_rec_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(src_rec - source_input), axis=-1)))
        tgt_rec_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tgt_rec - target_input), axis=-1)))
        rec_loss = src_rec_loss + tgt_rec_loss

        infer_var_list = [v for v in tf.trainable_variables() if v.name.split("/")[0] == "inference"]
        #
        greg_loss = args.l2_weight * tf.reduce_mean([tf.nn.l2_loss(x) for x in infer_var_list if "w" in x.name])
        #
        loss_trg_cent = tf.reduce_mean(softmax_xent_two(labels=tgt_lable_logits, logits=tgt_lable_logits))

        # tgt_pred = tf.nn.softmax(tgt_lable_logits, axis=-1)
        # condition_entropy = 2 + tf.reduce_mean(tf.reduce_mean(tf.log(tgt_pred) * tgt_pred, axis=-1))
        # print(condition_entropy)
        # exit()

        # total_loss = label_loss + domain_loss + greg_loss + args.lambda_rec * rec_loss + \
        #              args.lambda_mi * label_loss_2 + args.tw * loss_trg_cent + vat_loss
        total_loss = label_loss + domain_loss + greg_loss + args.lambda_rec * rec_loss + \
                     args.lambda_mi * label_loss_2

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            dann_train_op = tf.train.MomentumOptimizer(learning_rate, args.momentum).minimize(total_loss)

            dann_train_op_2 = tf.train.MomentumOptimizer(learning_rate, args.momentum).minimize(vat_loss)

        domain_correct_label = tf.equal(tf.argmax(source_target_domain_label, 1), tf.argmax(domain_logits, 1))
        domain_acc = tf.reduce_mean(tf.cast(domain_correct_label, tf.float32))

        target_correct_label = tf.equal(tf.argmax(target_label, 1), tf.argmax(tgt_lable_logits, 1))
        target_label_acc = tf.reduce_mean(tf.cast(target_correct_label, tf.float32))

        source_correct_label = tf.equal(tf.argmax(source_label, 1), tf.argmax(src_label_logits, 1))
        source_label_acc = tf.reduce_mean(tf.cast(source_correct_label, tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            tf.global_variables_initializer().run()

            gen_source_batch = batch_generator([source_train_input, source_train_y], batch_size)
            gen_target_batch = batch_generator([target_train_input, target_train_y], batch_size)

            source_domain_input = np.tile([1., 0.], [batch_size, 1])
            target_domain_input = np.tile([0., 1.], [batch_size, 1])

            best_result = 0.0
            lr = args.learning_rate
            for global_steps in range(num_steps):
                p = float(global_steps) / num_steps
                L = min(2 / (1. + np.exp(-10. * p)) - 1, 0.20)
                lr = max(args.learning_rate / (1. + 10 * p) ** 0.55, 0.01)
                # lr = args.learning_rate

                X0, y0 = next(gen_source_batch)
                X1, y1 = next(gen_target_batch)

                # if global_steps < num_steps - 0:
                #     train_op = dann_train_op
                # else:
                #     train_op = dann_train_op

                _, batch_total_loss, batch_domain_loss, batch_label_loss, \
                batch_src_rec_loss, batch_tgt_rec_loss, batch_rec_src_loss, \
                batch_trans_src_loss, batch_domain_acc = session.run(
                    [dann_train_op, total_loss, domain_loss, label_loss, src_rec_loss, tgt_rec_loss,
                     rec_src_loss, trans_src_loss, domain_acc],
                    feed_dict={source_input: X0, source_label: y0, target_input: X1, learning_rate: lr, alpha: L,
                               source_domain: source_domain_input, target_domain: target_domain_input,
                               train_mode: True})

                writer.add_scalar("loss_rec_source", batch_src_rec_loss, global_steps)
                writer.add_scalar("loss_rec_target", batch_tgt_rec_loss, global_steps)
                writer.add_scalar("loss_rec_src_class", batch_rec_src_loss, global_steps)
                writer.add_scalar("loss_trans_src_class", batch_trans_src_loss, global_steps)
                writer.add_scalar("loss_total", batch_total_loss, global_steps)
                writer.add_scalar("train_domain_acc", batch_domain_acc, global_steps)

                if global_steps > 5000:
                    print("best_result: ", best_result)
                    break

                if global_steps % 20 == 0:
                    target_domain_test_input = np.tile([0., 1.], [target_test_input.shape[0], 1])

                    feed_dict = {target_input: target_test_input,
                                 target_label: target_test_y,
                                 train_mode: False,
                                 target_domain: target_domain_test_input}

                    test_tgt_acc, test_tgt_rec, test_tgt_feat\
                        = session.run([target_label_acc, tgt_rec, tgt_inner_code], feed_dict=feed_dict)

                    source_domain_test_input = np.tile([1., 0.], [source_train_input.shape[0], 1])

                    feed_dict = {source_input: source_train_input,
                                 source_label: source_train_y,
                                 train_mode: False,
                                 source_domain: source_domain_test_input}

                    test_src_acc, test_src_rec, test_src_feat, test_trans_rec \
                        = session.run([source_label_acc, src_rec, src_inner_code, trans_rec], feed_dict=feed_dict)

                    print("Steps: %d \t target acc: %g \t source acc: %g" %
                          (global_steps, test_tgt_acc, test_src_acc))

                    if best_result < test_tgt_acc:
                        best_result = test_tgt_acc

                    writer.add_scalar("source_acc", test_src_acc, global_steps)
                    writer.add_scalar("target_acc", test_tgt_acc, global_steps)

                    plot_domain_label = ["source"] * len(test_src_feat) + ["target"] * len(test_tgt_feat)

                    # if global_steps % 1000 == 0:
                    #     src_img_path, tgt_img_path, domain_img_path = \
                    #         plot_tsne(src_feature=test_src_feat, tgt_feature=test_tgt_feat,
                    #                   src_label=source_train_label.tolist(),
                    #                   tgt_label=target_test_label.tolist(), domain_label=plot_domain_label,
                    #                   epoch=global_steps//1000,
                    #                   src_idx=len(test_src_feat), folder=exp_path)
                    #     src_img = Image.open(src_img_path)
                    #     tgt_img = Image.open(tgt_img_path)
                    #     domain_img = Image.open(domain_img_path)
                    #     writer.add_image("1_source tsne", np.transpose(np.array(src_img), [2, 0, 1]),
                    #                      global_step=global_steps//1000)
                    #     writer.add_image("2_target tsne", np.transpose(np.array(tgt_img), [2, 0, 1]),
                    #                      global_step=global_steps//1000)
                    #     writer.add_image("3_domain tsne", np.transpose(np.array(domain_img), [2, 0, 1]),
                    #                      global_step=global_steps//1000)
                    #
                    #     src_path, tgt_path, tran_path = \
                    #         plot_feature_tsne(src_original_feature=source_train_input,
                    #                           tgt_origin_feature=target_test_input,
                    #                           src_rec_feature=test_src_rec, tgt_rec_feature=test_tgt_rec,
                    #                           trans_feature=test_trans_rec, epoch=global_steps // 1000,
                    #                           folder=exp_path)
                    #     src_origin_feature_img = Image.open(src_path)
                    #     tgt_origin_feature_img = Image.open(tgt_path)
                    #     trans_origin_feature_img = Image.open(tran_path)
                    #     writer.add_image("4_src_reconstrction tsne",
                    #                      np.transpose(np.array(src_origin_feature_img), [2, 0, 1]),
                    #                      global_step=global_steps//1000)
                    #     writer.add_image("5_tgt_reconstrction tsne",
                    #                      np.transpose(np.array(tgt_origin_feature_img), [2, 0, 1]),
                    #                      global_step=global_steps//1000)
                    #     writer.add_image("6_trans_reconstrction tsne",
                    #                      np.transpose(np.array(trans_origin_feature_img), [2, 0, 1]),
                    #                      global_step=global_steps//1000)
    print(exp_name, file=record_file)
    print("best_result: %g\n" % best_result, file=record_file)
