# DEA-Solver
Software to evaluate the efficiency of a group of similar decisions making units (DMUs). 

```python
import pandas as pd
from DEA_Solver import DEA_CCR

datas=pd.read_csv("sample datas/datas.csv", sep=";", index_col=False)
datas
```
Load `datas` as `pd.Dataframe`. Indicate the input and output columns by passing a tuple. You can use `None` to indicate zero index column and last column.


```python
res = DEA_CCR(csv_file, input_cols=(1,2), output_cols=(2,None))
```

To view selected inputs and outpus use `show_inputs()` and `show_outputs()` methods.

You can solve the model by using the `solve` method. This will return a list of Z in objective function for each DMU.

```python
res.solve()
```

To show the ranked table use the `rank` method:

```python
res.rank(sort='no')
```
The `sort` parameter is used to sort dataframe by score. You can choose between `['ascending'/'asc', 'descending'/'desc' and 'no']`. Default is `no`.


<h2 style="color:red">Notes about Linear Programming</h2>
O DEA Solver usa o linprog para ordenar as DMUs mais eficientes através de um ranking que varia de 0-1.

O lingprog resolve o problema de minimização da forma:

$\text{min}\quad C^Tx$\
st.\
$A_{ub} x \leq b_{ub}$\
$A_{eq} x = b_{eq}$\
$l\leq x \leq u$

Onde:
- $C^T$ --- obj (objective function)
- $A_{ub}$ --- lhs_ineq (left hand side inequality constraint)
- $b_{ub}$ --- rhs_ineq (right hand side inequality constraint)
- $A_{eq}$ --- lhs_eq (left hand side equality constraint)
- $b_{eq}$ --- rhs_eq (right hand side equality constraint) 

Todas as restrições com sinal de maior ou igual ($\geq$) devem ser convertidas para menor ou igual ($\leq$) multiplicando por -1. 

Para problema de maximizacao basta multiplicar a função objetivo por -1.

Outra coisa que deve ser levada em consideração são as variáveis de decisão de não negatividade. Para definir isso, crie tuplas para cada variável. Por exemplo, se uma variável $x_1$ tem que ser maior ou igual à zero, use `x_1=(0,None)`. Pode trocar o `None` por `np.inf` ou `float('inf')`. Se a variável $x_1$ é livre, pode escrever `x_1=(None,None)`. Em suma, para todas variáveis `x`, deverá indicar o lower e o upper `x=(l,u)`, mesmo que ambas sejam `None`.

A sintaxe para linprog é:

```python
linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=(0, None))
```

Na variável `bounds` deverá mandar uma lista com todas variáveis definidas. Exemplo: `bounds=[x_0, x_1, x_2]`.


Documentação:
* [Documentação do lingprog no Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)
* [Exemplos com explicação](https://realpython.com/linear-programming-python/)
* [Outros Exemplos](https://www.cuemath.com/algebra/linear-programming/)

