from typing import Tuple, Callable, List

from functools import partial

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
for d in range(5):
    coeffs.append(ConstRef(f"const_i_deriv_coeff_dev[{d}]"))

fold_stride = [1]
for d in range(6):
    fold_stride.append(fold_stride[-1] * fold[d])


def eval_index_in_vector(dim: int, fold: Tuple[str] = fold, fold_stride: List[int] = fold_stride) -> str:
    index = "0"
    if fold_stride[dim] < 32 and fold[dim] > 1:
        index = "threadIdx.x"
        if fold_stride[dim] > 1:
            index = f"({index} / {fold_stride[dim]})"
        if fold[dim] > 1:
            index = f"({index} % {fold[dim]})"
    return index


index_in_vector = []
for d in range(6):
    index_in_vector.append(eval_index_in_vector(d))


def build_eval_pre_coeff(name: str, fold: Tuple[int] = fold, index_in_vector: List[str] = index_in_vector) -> Callable[
    [str, str, str, str, str], str]:
    def eval_pre_coeff(i: str, k: str, l: str, m: str, n: str):
        indexes = []
        for d, index in zip([0, 2, 3, 4, 5], [i, k, l, m, n]):
            if index.isdigit():
                index_as_int = int(index)
                index_as_int *= fold[d]
                index = str(index_as_int)
            else:
                if fold[d] > 1:
                    index = f"{fold[d]} * ({index})"
            idx_in_vec = index_in_vector[d]
            if idx_in_vec != "0":
                index += " + " + idx_in_vec
            indexes.append(index)
        return f"{name}[bCoeffIndex][{indexes[4]}][{indexes[3]}][{indexes[2]}][{indexes[1]}][{indexes[0]}]"

    return eval_pre_coeff


def eval_ikj(j: str, fold: Tuple[int] = fold,
             index_in_vector: List[str] = index_in_vector) -> str:
    return f"ikj[PADDING[1] + b_j * BRICK_DIM[1] + {j} * {fold[1]} + {index_in_vector[1]}]"


eval_p1 = partial(build_eval_pre_coeff(name="bP1"))
eval_p2 = partial(build_eval_pre_coeff(name="bP2"))
p1 = FunctionOfLocalVectorIndex(eval_p1, (i, k, l, m, n))
p2 = FunctionOfLocalVectorIndex(eval_p2, (i, k, l, m, n))
ikj = FunctionOfLocalVectorIndex(eval_ikj, (j,))

# Express computation
calc = p1 * (
        coeffs[0] * input(i - 2, j, k, l, m, n) +
        coeffs[1] * input(i - 1, j, k, l, m, n) +
        coeffs[2] * input(i + 0, j, k, l, m, n) +
        coeffs[3] * input(i + 1, j, k, l, m, n) +
        coeffs[4] * input(i + 2, j, k, l, m, n)
) + p2 * ikj * input(i, j, k, l, m, n)

output(i, j, k, l, m, n).assign(calc)

STENCIL = [output]
NEIGHBOR_DIMS = [0]
