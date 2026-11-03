import numpy as np
rng = np.random.default_rng(1)
d, eta, steps = 2000, 1e-3, 40000
print(" rho  beta   growth/(eta^2 G)   theory S(0)/G=(1+rho)/(1-rho)")
for rho in [0.0, 0.5, 0.8]:
    for beta in [0.0, 0.9]:
        W = rng.normal(size=d); W *= 10/np.linalg.norm(W)
        m = np.zeros(d); z = rng.normal(size=d); inc = 0.0; G = 0.0; n = 0
        for t in range(steps):
            z = rho*z + np.sqrt(1-rho**2)*rng.normal(size=d)       # AR(1) raw gradient
            g = z - (z@W)/(W@W)*W                                   # orthogonal to current W
            G += g@g
            m = beta*m + (1-beta)*g
            Wn = W - eta*m
            if t >= steps//4: inc += Wn@Wn - W@W; n += 1
            W = Wn
        G /= steps
        print(f" {rho:.1f}  {beta:.1f}      {inc/(n*eta**2*G):8.3f}          {(1+rho)/(1-rho):8.3f}")
