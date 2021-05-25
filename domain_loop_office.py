import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--is_use_theta_in_encoder", default=True, type=bool)
parser.add_argument("--is_use_theta_in_decoder", default=True, type=bool)
args = parser.parse_args()

domain_list = ["Art", "Clipart", "Product", "RealWorld"]

command_list = list()
command_format = "python loop_office_home_1.py --src %s --tgt %s --is_use_theta_in_encoder %s --is_use_theta_in_decoder %s"

for src in domain_list:
    for tgt in domain_list:
        if src != tgt:
            command = command_format % (src, tgt, str(args.is_use_theta_in_encoder), str(args.is_use_theta_in_decoder))
            os.system(command)

# parall_num = 12

# total_command = " & ".join(command_list)
# os.system(total_command)
