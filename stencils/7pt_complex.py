from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
DIM = 3
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input = Grid("bIn", 3, complex_valued=True)
output = Grid("bOut", 3, complex_valued=True)
param = [ConstRef("coeff[0]", complex_valued=True), ConstRef("coeff[1]", complex_valued=True),
         ConstRef("coeff[2]", complex_valued=True), ConstRef("coeff[3]", complex_valued=True),
         ConstRef("coeff[4]", complex_valued=True), ConstRef("coeff[5]", complex_valued=True),
         ConstRef("coeff[6]", complex_valued=True)]

# Express computation
# output[i, j, k] is assumed
calc = param[0] * input(i, j, k) + \
       param[1] * input(i + 1, j, k) + \
       param[2] * input(i - 1, j, k) + \
       param[3] * input(i, j + 1, k) + \
       param[4] * input(i, j - 1, k) + \
       param[5] * input(i, j, k + 1) + \
       param[6] * input(i, j, k - 1)
output(i, j, k).assign(calc)

STENCIL = [output]
