# Test matMul: matrix-matrix multiply

# 2x2 identity * 2x2 matrix = same matrix
@I [[1.0, 0.0], [0.0, 1.0]]
@M [[3.0, 4.0], [5.0, 6.0]]
@IM (matMul I M)
assert (eq (nth (nth IM 0) 0) 3.0)
assert (eq (nth (nth IM 0) 1) 4.0)
assert (eq (nth (nth IM 1) 0) 5.0)
assert (eq (nth (nth IM 1) 1) 6.0)

# 2x2 general: [[1,2],[3,4]] * [[5,6],[7,8]]
# = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
# = [[19, 22], [43, 50]]
@A [[1.0, 2.0], [3.0, 4.0]]
@B [[5.0, 6.0], [7.0, 8.0]]
@C (matMul A B)
assert (eq (nth (nth C 0) 0) 19.0)
assert (eq (nth (nth C 0) 1) 22.0)
assert (eq (nth (nth C 1) 0) 43.0)
assert (eq (nth (nth C 1) 1) 50.0)

# 2x3 * 3x2 -> 2x2
# [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]]
# row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
# row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
@P [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
@Q [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
@R (matMul P Q)
assert (eq (len R) 2)
assert (eq (len (nth R 0)) 2)
assert (eq (nth (nth R 0) 0) 58.0)
assert (eq (nth (nth R 0) 1) 64.0)
assert (eq (nth (nth R 1) 0) 139.0)
assert (eq (nth (nth R 1) 1) 154.0)
