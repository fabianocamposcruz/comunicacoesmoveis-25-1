# Reexecutando o código após reset do ambiente

import numpy as np
import matplotlib.pyplot as plt
from fGeraCanal import gera_canal
from fEstimaCanal import f_estima_canal
from scipy.stats import nakagami, rice, rayleigh, weibull_min

# Parâmetros
sPar = {
    'd0': 5,
    'P0': 0,
    'nPoints': 50000,
    'totalLength': 100,
    'n': 4,
    'sigma': 6,
    'shadowingWindow': 200,
    'm': 4,
    'txPower': 0,
    'nCDF': 40,
    'dW': 100,
    'chFileName': 'Prx_sintetico'
}

sPar['dMed'] = sPar['totalLength'] / sPar['nPoints']

# Geração do canal sintético
vtDist, vtPathLoss, vtShadCorr, vtFading, vtPrxdBm = gera_canal(sPar)

print("Canal sintético:")
print(f"   Média do sombreamento: {np.mean(vtShadCorr):.5f}")
print(f"   Std do sombreamento: {np.std(vtShadCorr):.5f}")
print(f"   Janela de correlação do sombreamento: {sPar['shadowingWindow']} amostras")
print(f"   Expoente de path loss: {sPar['n']}")
print(f"   m de Nakagami: {sPar['m']}")

# Estimação
vtW = [10, 50, 150, 200]
sOut = []
print("\nEstimação do Fading para várias janelas (estudo numérico sem conhecimento a priori do canal)")
print("Resultados com SciPy (equivalente ao fitdist do MATLAB)")

for iw, w in enumerate(vtW):
    sPar['dW'] = w
    out = f_estima_canal(sPar)
    sOut.append(out)
    env_norm = out['vtEnvNorm']
    print(f"Janela W = {w}")

    # Ajuste Nakagami
    m_naka, _, omega_naka = nakagami.fit(env_norm, floc=0)
    print(f"  Nakagami: m = {m_naka:.4f}, omega = {omega_naka:.5f}")

    # Ajuste Rice
    s_rice, _, sigma_rice = rice.fit(env_norm, floc=0)
    K_rice = s_rice**2 / (2*(sigma_rice** 2))
    print(f"  Rice: K = {K_rice:.4f}")

    # Ajuste Rayleigh
    _, sigma_rayleigh = rayleigh.fit(env_norm, floc=0)
    print(f"  Rayleigh: sigma = {sigma_rayleigh:.5f}")

    # Ajuste Weibull
    k_wei, _, lambda_wei = weibull_min.fit(env_norm, floc=0)
    print(f"  Weibull: k = {k_wei:.4f}, lambda = {lambda_wei:.5f}\n")
    print(rice.fit(env_norm, floc=0))
