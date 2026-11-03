import numpy as np, sympy as sp
def F_numeric(beta, rho, K=4000):
    # exact expectation under: EMA momentum, AR(1) grads (corr rho), pathwise orthogonality <W_s,g_s>=0
    G = 1.0
    a = np.arange(K)
    if abs(rho - beta) < 1e-12:
        first = (a + 1) * beta**a
    else:
        first = (rho**(a + 1) - beta**(a + 1)) / (rho - beta)
    f = (1 - beta) * G * (first + beta**a * beta * rho / (1 - beta * rho))   # E<u_{s+a}, g_s>
    cumf = np.concatenate([[0.0], np.cumsum(f)])                              # sum_{a<k} f(a)
    k = np.arange(K)
    EWu = -(1 - beta) * np.sum(beta**k * cumf[k])                             # E<W_t,u_t>/eta
    Eu2 = (1 - beta)**2 * G / (1 - beta**2) * (1 + beta * rho) / (1 - beta * rho)
    return -2 * EWu + Eu2                                                     # E Delta ||W||^2 / eta^2
for rho in [0.0, 0.5, 0.8]:
    print(rho, [round(F_numeric(b, rho), 3) for b in [0.0, 0.9]])
b, r = sp.symbols('beta rho', positive=True)
# closed form via symbolic sums
a, k = sp.symbols('a k', integer=True, nonnegative=True)
f = (1 - b) * ((r**(a + 1) - b**(a + 1)) / (r - b) + b**a * b * r / (1 - b * r))
cum = sp.simplify(sp.summation(f, (a, 0, k - 1)))
EWu = -(1 - b) * sp.summation(sp.expand(b**k * cum), (k, 0, sp.oo))
Eu2 = (1 - b)**2 / (1 - b**2) * (1 + b * r) / (1 - b * r)
Fsym = sp.simplify(sp.factor(-2 * EWu + Eu2))
print("closed form F(beta,rho) =", Fsym)
print("check:", [float(Fsym.subs({b: 0.9, r: rr})) for rr in [0.5, 0.8]])
