import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os


def plot_feature_tsne(src_original_feature, tgt_origin_feature, src_rec_feature,
                      tgt_rec_feature, trans_feature, epoch, folder):

    feature = np.concatenate([src_original_feature, tgt_origin_feature, src_rec_feature,
                              tgt_rec_feature, trans_feature], axis=0)
    src_original_label = ["source_origin"] * len(src_original_feature)
    tgt_original_label = ["target_rec"] * len(tgt_origin_feature)
    src_rec_label = ["source_rec"] * len(src_rec_feature)
    tgt_rec_label = ["target_rec"] * len(tgt_rec_feature)
    trans_label = ["trans_rec"] * len(trans_feature)

    label = src_original_label + tgt_original_label + src_rec_label + tgt_rec_label + trans_label
    tsne = TSNE(n_components=2, random_state=0, n_jobs=8)
    tsne_obj = tsne.fit_transform(feature)

    index = len(src_original_feature)
    trans_src_original_feature = tsne_obj[: index]

    trans_tgt_origin_feature = tsne_obj[index: index + len(tgt_origin_feature)]
    index += len(tgt_origin_feature)

    trans_src_rec_feature = tsne_obj[index: index + len(src_rec_feature)]
    index += len(src_rec_feature)

    trans_tgt_rec_feature = tsne_obj[index: index + len(tgt_rec_feature)]
    index += len(tgt_rec_feature)

    trans_trans_feature = tsne_obj[index: index+len(trans_feature)]

    # plot original and reconstruction of source
    src_path = plot_domain_level_img(feature=np.concatenate([trans_src_original_feature, trans_src_rec_feature], axis=0),
                                     label=src_original_label + src_rec_label, folder=folder, epoch=epoch,
                                     prefix="src_and_rec")

    # plot original and reconstruction of target
    tgt_path = plot_domain_level_img(feature=np.concatenate([trans_tgt_origin_feature, trans_tgt_rec_feature], axis=0),
                                     label=["target_origin"] * len(trans_tgt_origin_feature) +
                                           ["target_rec"] * len(trans_tgt_rec_feature), folder=folder, epoch=epoch,
                                     prefix="tgt_and_rec")

    # plot trans_source and rec_tgt
    tran_path = plot_domain_level_img(feature=np.concatenate([trans_trans_feature, trans_tgt_rec_feature], axis=0),
                                      label=["trans_feature"] * len(trans_trans_feature) +
                                            ["target_rec"] * len(trans_tgt_rec_feature), folder=folder, epoch=epoch,
                                      prefix="trans_and_rec")

    print(src_path)
    print(tgt_path)
    print(tran_path)

    return src_path, tgt_path, tran_path


def plot_tsne(src_feature, tgt_feature, src_label, tgt_label, domain_label, epoch, src_idx, folder):

    feature = np.concatenate([src_feature, tgt_feature], axis=0)
    label = np.concatenate([src_label, tgt_label], axis=0)
    tsne = TSNE(n_components=2, random_state=0, n_jobs=8)
    tsne_obj = tsne.fit_transform(feature)

    # plot source fig
    src_path = plot_class_level_img(feature=tsne_obj[: src_idx], label=label[: src_idx],
                                    folder=folder, epoch=epoch, prefix="src")
    # plt.figure(figsize=(10, 10))
    # src_df = pd.DataFrame({'X': tsne_obj[:src_idx, 0],
    #                         'Y': tsne_obj[:src_idx, 1],
    #                         'digit': label[:src_idx]})
    #
    # img = sns.scatterplot(x="X", y="Y",
    #                 hue="digit",
    #                 palette=['purple', 'red', 'orange', 'brown', 'blue',
    #                          'dodgerblue', 'green', 'lightgreen', 'darkcyan', 'black'],
    #                 legend='full',
    #                 data=src_df)
    # plt.savefig(os.path.join("../logs", folder, "src_%d.png" % epoch))

    # plot target fig
    tgt_path = plot_class_level_img(feature=tsne_obj[src_idx: ], label=label[src_idx:],
                                    folder=folder, epoch=epoch, prefix="tgt")
    # src_df = pd.DataFrame({'X': tsne_obj[src_idx:, 0],
    #                        'Y': tsne_obj[src_idx:, 1],
    #                        'digit': label[src_idx:]})
    #
    # img = sns.scatterplot(x="X", y="Y",
    #                       hue="digit",
    #                       palette=['purple', 'red', 'orange', 'brown', 'blue',
    #                                'dodgerblue', 'green', 'lightgreen', 'darkcyan', 'black'],
    #                       legend='full',
    #                       data=src_df)
    # plt.savefig(os.path.join("../logs", folder, "tgt_%d.png" % epoch))

    # plot domain fig
    domain_path = plot_domain_level_img(feature=tsne_obj, label=domain_label, folder=folder, epoch=epoch)
    # domain_df = pd.DataFrame({'X': tsne_obj[:, 0],
    #                          'Y': tsne_obj[:, 1],
    #                          'digit': domain_label})
    # img = sns.scatterplot(x="X", y="Y",
    #                       hue="digit",
    #                       palette=['blue', 'red'],
    #                       # legend='full',
    #                       data=domain_df)
    # plt.savefig(os.path.join("../logs", folder, "domain_%d.png" % epoch))

    # return os.path.join("../logs", folder, "src_%d.png" % epoch), \
    #        os.path.join("../logs", folder, "tgt_%d.png" % epoch), \
    #        os.path.join("../logs", folder, "tgt_%d.png" % epoch)

    print(src_path)
    print(tgt_path)
    print(domain_path)

    return src_path, tgt_path, domain_path

import random
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color



def plot_class_level_img(feature, label, folder, epoch, prefix):
    plt.figure(figsize=(10, 10))
    src_df = pd.DataFrame({'X': feature[:, 0],
                           'Y': feature[:, 1],
                           'digit': label})



    sns.scatterplot(x="X", y="Y",
                    hue="digit",
                    palette=[randomcolor() for _ in range(65)],
                    legend='full',
                    data=src_df)
    save_path = os.path.join(folder, "%s_%d.png" % (prefix, epoch))
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_domain_level_img(feature, label, folder, epoch, prefix=""):
    domain_df = pd.DataFrame({'X': feature[:, 0],
                              'Y': feature[:, 1],
                              'digit': label})
    sns.scatterplot(x="X", y="Y",
                    hue="digit",
                    palette=['blue', 'red'],
                    legend='full',
                    data=domain_df)
    save_path = os.path.join(folder, prefix+"domain_%d.png" % epoch)
    plt.savefig(save_path)
    plt.close()

    return save_path

