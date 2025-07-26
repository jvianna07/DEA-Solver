# DEA-Solver
Software to evaluate the efficiency of a group of similar decisions making units. 

```python
import pandas as pd
from DEA_Solver import DEA_CCR

datas=pd.read_csv("datas.csv", sep=";", index_col=False)
xi, xo= datas.values[:,1:2].astype(np.float32), datas.values[:,2:].astype(np.float32)
datas
```
Load `xi` and `xo` as two dimension array.

```python
res=DEA_CCR(xi, xo)
```
You can solve the model by using the `solve` method:

```python
res.solve()
```
