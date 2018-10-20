import glob
import os

filenames = glob.glob("*")

unique = list(set([x[0:6] for x in filenames]))

for fn in filenames:
    os.system("mv {} {:04d}{}".format(fn, unique.index(fn[0:6]), fn[6:]))
