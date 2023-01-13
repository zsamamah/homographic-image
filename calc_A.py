import numpy as np

empty9 = [0] * 9
A1 = np.matrix([empty9, empty9, empty9, empty9, empty9, empty9, empty9, empty9])

# 0, 0  350, 0
# 0, 300  350, 300

point_cor1 = [[0, 0, 788, 190], [350, 0, 1090, 213], [0, 300, 765, 877], [350, 300, 1055, 770]]

i = 0
for (x, y, x_new, y_new) in point_cor1:
    temp = [x, y, 1, 0, 0, 0, -x_new * x, -x_new * y, -x_new]
    for j in range(9):
        A1[2 * i, j] = temp[j]
    temp = [0, 0, 0, x, y, 1, -y_new * x, -y_new * y, -y_new]
    for j in range(9):
        A1[2 * i + 1, j] = temp[j]
    i = i + 1

print(np.array2string(A1,separator=','))
