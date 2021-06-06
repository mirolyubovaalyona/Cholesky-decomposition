import numpy as np
from math import sqrt


def cholesky(A):
    L = [[0.0] * len(A) for _ in range(len(A))]
    for i, (Ai, Li) in enumerate(zip(A, L)):
        for j, Lj in enumerate(L[:i+1]):
            s = sum(Li[k] * Lj[k] for k in range(j))
            Li[j] = sqrt(Ai[i] - s) if (i == j) else \
                      (1.0 / Lj[j] * (Ai[j] - s))
    return L


a= [[10, 2,  1],
          [2, 10,  3,],
          [1, 3, 10]]

b=[12, 13, 14]

n=len(b)

t=np.array((cholesky(a)))
tt=t.transpose()

y=[0]*n
x=[0]*n

y[0]=b[0]/t[0][0]
for i in range(1, n):
    s = 0;
    for k in range(0, i):
        s  += t[i][k]*y[k]
    y[i] = (b[i] - s)/t[i][i]


x[n-1]=y[n-1]/tt[n-1][n-1]
for i in range(n-2, -1, -1):
    ss=0
    for k in range(n-1, i, -1):
        ss = ss+ tt[i][k]*x[k]
    x[i] = (y[i] - ss)/tt[i][i]

print("x:")
for i in range(n):
    print("x", i, "=", x[i])

print(np.linalg.solve(a, b))
