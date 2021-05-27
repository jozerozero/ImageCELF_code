import os

domain_list = ["Art", "Clipart", "Product", "RealWorld"]

command_list = list()
command_format = "python office_home_abalation.py --src %s --tgt %s"

for src in domain_list:
    for tgt in domain_list:
        if src != tgt:
            command = command_format % (src, tgt)
            os.system(command)