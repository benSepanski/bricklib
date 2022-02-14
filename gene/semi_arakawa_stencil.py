from st.expr import Index, ConstRef, If, FunctionOfLocalVectorIndex
from st.grid import Grid

# Rely on vecscatter to pass in brick-dimension and vector fold
for expected_var in ['brick_dim', 'fold']:
    if not expected_var in locals():
        raise ValueError(f"Expected bricks code generation to put '{expected_var}'")
print("Brick-Dim: ", brick_dim)
print("Fold: ", fold)

i, j, k, l, m, n = map(Index, range(6))

# Declare grid
input = Grid("bIn", 6, complex_valued=True)
output = Grid("bOut", 6, complex_valued=True)
coeffs = []
for stencil_idx in range(13):
    def f(i, k, l, m, n, stencil_idx=stencil_idx, fold=fold, brick_dim=brick_dim):
        brick_idx = f"coeff.step * bCoeffIndex"

        idx_in_fold = "threadIdx.x"
        # Divide out j extent
        if fold[1] > 1:
            idx_in_fold_terms = []
            if fold[0] * fold[1] < 32:
                idx_in_fold_terms.append(f"({idx_in_fold} / {fold[0] * fold[1]}) * {fold[0]}")
            if fold[0] > 1:
                idx_in_fold_terms.append(f"{idx_in_fold} % {fold[0]}")
            idx_in_fold = " + ".join(idx_in_fold_terms)
        # Add up idx of vector (as compile-time known part + runtime-known part)
        runtime_part_of_idx_of_vec = []
        compile_time_part_of_idx_of_vec = 0
        stride = 1
        for vec_idx_d, d in zip([i, k, l, m, n], [0, 2, 3, 4, 5]):
            try:
                vec_idx_d_val = int(eval(repr(vec_idx_d)))
                compile_time_part_of_idx_of_vec += vec_idx_d_val * stride
            except ValueError:
                runtime_part_of_idx_of_vec.append(vec_idx_d)
                if stride > 1:
                    runtime_part_of_idx_of_vec[-1] = f"({runtime_part_of_idx_of_vec[-1]}) * {stride}"
            stride *= brick_dim[d] // fold[d]

        vec_len = fold[0] * fold[2] * fold[3] * fold[4] * fold[5]
        if len(runtime_part_of_idx_of_vec) > 0:
            idx_of_vec = f"{13 * vec_len} * ({' + '.join(runtime_part_of_idx_of_vec)})"
            if compile_time_part_of_idx_of_vec != 0:
                idx_of_vec = str(13 * vec_len * compile_time_part_of_idx_of_vec) + " + " + idx_of_vec
        else:
            idx_of_vec = str(13 * vec_len * compile_time_part_of_idx_of_vec)

        idx_in_brick = f"{idx_of_vec} + {stencil_idx} * {vec_len} + {idx_in_fold}"

        return f"coeff.dat[{brick_idx} + {idx_in_brick}]"
    coeffs.append(FunctionOfLocalVectorIndex(f, (i, k, l, m, n)))

# Express computation
calc = coeffs[ 0] * input(i, j, k + 0, l - 2, m, n) + \
       coeffs[ 1] * input(i, j, k - 1, l - 1, m, n) + \
       coeffs[ 2] * input(i, j, k + 0, l - 1, m, n) + \
       coeffs[ 3] * input(i, j, k + 1, l - 1, m, n) + \
       coeffs[ 4] * input(i, j, k - 2, l + 0, m, n) + \
       coeffs[ 5] * input(i, j, k - 1, l + 0, m, n) + \
       coeffs[ 6] * input(i, j, k + 0, l + 0, m, n) + \
       coeffs[ 7] * input(i, j, k + 1, l + 0, m, n) + \
       coeffs[ 8] * input(i, j, k + 2, l + 0, m, n) + \
       coeffs[ 9] * input(i, j, k - 1, l + 1, m, n) + \
       coeffs[10] * input(i, j, k + 0, l + 1, m, n) + \
       coeffs[11] * input(i, j, k + 1, l + 1, m, n) + \
       coeffs[12] * input(i, j, k + 0, l + 2, m, n)

output(i, j, k, l, m, n).assign(calc)

STENCIL = [output]
NEIGHBOR_DIMS = [2, 3]
