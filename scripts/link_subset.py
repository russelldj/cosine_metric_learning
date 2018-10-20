import os
import glob

NUM_IN_FIRST_15 = 6076

for i in range(NUM_IN_FIRST_15):
    fns = glob.glob('../all_bounding_boxes/{:04d}_c1s1_*_00.jpg'.format(i))
    for fn in fns:
        print(os.system('ln -s {} {}'.format(fn, fn.replace('../all_bounding_boxes', '.'))))

