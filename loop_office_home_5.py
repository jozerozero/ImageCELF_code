import os
import argparse

# lr_list = [0.05, 0.025]
# momentum_list = [0.7, 0.8]
# lambda_rec_list = [0.1]
# lambda_mi_list = [0.001, 0.003, 0.005]
# dropout_list = [0.7]
# # l2_list = [5e-4, 1e-4]
# l2_list = [1e-4]
# tw_list = [1e-2, 5e-2]

lr_list = [0.05, 0.025]
momentum_list = [0.5, 0.7, 0.8]  # 0.5, 0.7, 0.9
lambda_rec_list = [0.1, 0.3, 0.5]
lambda_mi_list = [0.001, 0.003, 0.005]  #
dropout_list = [0.7, 0.8]  # 0.8
# l2_list = [5e-4, 1e-4]
l2_list = [1e-4]  # 5e-5
tw_list = [1e-2, 5e-2]  # 5e-2
radius_list = [1.5, 3.5, 5]    # 5, 3.5

# best result
# lr_0.05_momentum_0.8_seed_1_lambda_rec_0.1w_lambda_mi_0.001_dropout_0.7_l2_0.0005

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="RealWorld", type=str)
parser.add_argument("--tgt", default="Clipart", type=str)
parser.add_argument("--is_use_theta_in_encoder", default=True, type=bool)
parser.add_argument("--is_use_theta_in_decoder", default=True, type=bool)
parser.add_argument("--dirtt", default=True, type=bool)
parser.add_argument("--pseudo_label", default=True, type=bool)
args = parser.parse_args()

command_list = []
command_format = "python ours_office_home_all_trick.py --src SOURCE --tgt TARGET " \
                 "--learning_rate %g --momentum %g --lambda_rec %g --lambda_mi %g --dropout_rate %g --l2_weight %g " \
                 "--tw %g --radius %g --is_use_theta_in_encoder USE_IN_ENCODER --is_use_theta_in_decoder USE_IN_DECODER " \
                 "--dirtt DIRTT --pseudo_label PSUEDO_LABEL"

command_format = command_format.replace("SOURCE", args.src).replace("TARGET", args.tgt)
command_format = command_format.replace("USE_IN_ENCODER", str(args.is_use_theta_in_encoder)).replace("USE_IN_DECODER", str(args.is_use_theta_in_decoder))
command_format = command_format.replace("DIRTT", str(args.dirtt)).replace("PSUEDO_LABEL", str(args.pseudo_label))

for radius in radius_list:
    for tw in tw_list:
        for lr in lr_list:
            for momentum in momentum_list:
                for lambda_rec in lambda_rec_list:
                    for lambda_mi in lambda_mi_list:
                        for dropout in dropout_list:
                            for l2 in l2_list:
                                # command_list.append(
                                #     command_format % (lr, momentum, lambda_rec, lambda_mi, dropout, l2, tw, radius))
                                os.system(command_format % (lr, momentum, lambda_rec, lambda_mi, dropout, l2, tw, radius))

# parall_num = 8
#
# for i in range(0, len(command_list), parall_num):
#     begin = i
#     if begin + parall_num > len(command_list):
#         end = len(command_list)
#     else:
#         end = begin + parall_num
#
#     middle = (begin + end) // 2
#
#     card_1_list = ["CUDA_VISIBLE_DEVICES=0 " + c for c in command_list[begin: middle]]
#     card_2_list = ["CUDA_VISIBLE_DEVICES=0 " + c for c in command_list[middle: end]]
#     total_list = card_1_list + card_2_list
#
#     command = " & ".join(total_list)
#     os.system(command)
