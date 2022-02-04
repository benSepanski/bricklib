from st.expr import Index, ConstRef, If
from st.grid import Grid
from st.func import Func

# Declare indices
i, j, k, l, m, n = map(Index, range(6))

# Declare grid
input = Grid("bIn", 6, complex_valued=True)
output = Grid("bOut", 6, complex_valued=True)
param = []
for d in range(13):
    param.append(ConstRef(f"c[{d}]"))

# Express computation
calc = None
idx = 0
for dl in range(-2, 3):
    max_dk = 2 - abs(dl)
    for dk in range(-max_dk, max_dk+1):
        term = ConstRef(f"c[{idx}]") * input(i, j, k + dk, l + dl, m, n)
        idx += 1
        if calc is None:
            calc = term
        else:
            calc += term
output(i, j, k, l, m, n).assign(calc)

STENCIL = [output]
