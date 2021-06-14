from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input_grid = Grid("bIn", 3, complex_valued=True)
output_grid = Grid("bOut", 3, complex_valued=True)
param = [ConstRef(f"zCoeff[{i}]", complex_valued=True) for i in range(7)]

# Express computation
# output[i, j, k] is assumed
calc = param[0] * input_grid(i, j, k) + \
       param[1] * input_grid(i + 1, j, k) + \
       param[2] * input_grid(i - 1, j, k) + \
       param[3] * input_grid(i, j + 1, k) + \
       param[4] * input_grid(i, j - 1, k) + \
       param[5] * input_grid(i, j, k + 1) + \
       param[6] * input_grid(i, j, k - 1)
output_grid(i, j, k).assign(calc)

STENCIL = [output_grid]
