import glob
import pandas as pd

sum_ = 0

for i in range(1, 16):
    print("object_annot_P_{:02d}.txt".format(i))
    data = pd.read_csv("object_annot_P_{:02d}.txt".format(i), delimiter=' ', names=['ID', 'x1', 'y1', 'x2', 'y2', 'frame', 'active', 'class'], index_col=False)
    sum_ += data['ID'].max()

print(sum_)
