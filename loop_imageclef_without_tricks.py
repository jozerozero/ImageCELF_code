import os

domain_list = ["c", "i", "p"]

command_list = list()
command_format = "python da_imageclef_without_tricks.py --src %s --tgt %s"

for src in domain_list:
    for tgt in domain_list:
        if src != tgt:
            command = command_format % (src, tgt)
            os.system(command)