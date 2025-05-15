import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.stats import nakagami
from scipy.special import gamma

# ---------- Parâmetros para geração do canal sintético ----------
sPar = {
    'd0': 5,                         # distância de referência d0
    'P0': 0,                         # Potência na distância de referência (dBm)
    'nPoints': 50000,               # Número de amostras
    'totalLength': 100,             # Distância total da rota
    'n': 4,                          # Expoente de perda de percurso
    'sigma': 6,                     # Desvio padrão do shadowing (dB)
    'shadowingWindow': 200,         # Tamanho da janela de correlação
    'm': 4,                          # Parâmetro de Nakagami
    'txPower': 0,                   # Potência de transmissão (dBm)
    'nCDF': 40,
    'chFileName': 'Prx_sintetico'
}

sPar['dMed'] = sPar['totalLength'] / sPar['nPoints']

# ---------- Geração do vetor de distâncias ----------
vtDist = np.arange(sPar['d0'], sPar['totalLength'], sPar['dMed'])
nSamples = len(vtDist)

# ---------- Perda de percurso determinística ----------
vtPathLoss = sPar['P0'] + 10*sPar['n']*np.log10(vtDist / sPar['d0'])

# ---------- Geração do sombreamento ----------
nShadowSamples = mt.floor(nSamples / sPar['shadowingWindow'])
vtShadowing = sPar['sigma'] * np.random.randn(nShadowSamples)

# Amostras para a última janela
restShadowing = sPar['sigma'] * np.random.randn(1) * np.ones(nSamples % sPar['shadowingWindow'])

# Repetição do mesmo valor durante a janela de correlação

vtShadowing = np.tile(vtShadowing, (sPar['shadowingWindow'], 1))
vtShadowing = vtShadowing.T.flatten()
vtShadowingVec = np.concatenate((vtShadowing, restShadowing))

# ---------- Filtragem do sombreamento (média móvel) ----------
jan = sPar['shadowingWindow'] // 2
vtShadCorr = []

for i in range(jan, nSamples - jan):
    vtShadCorr.append(np.mean(vtShadowing[i - jan:i + jan + 1]))
vtShadCorr = np.array(vtShadCorr)

# ---------- Ajuste de desvio padrão após filtragem ----------
vtShadCorr = vtShadCorr * np.std(vtShadowingVec) / np.std(vtShadCorr)
vtShadCorr = vtShadCorr - np.mean(vtShadCorr) + np.mean(vtShadowingVec)

# ---------- Geração do desvanecimento de pequena escala (Nakagami) ----------
# PDF de Nakagami para amostragem
m = sPar['m']
fpNakaPdf = lambda x: ((2* (m**m))/gamma(m))*x**(2*m-1)*np.exp(-m*x**2)

# Amostras com distribuição de Nakagami (envelope normalizado)
# Usando scipy.stats.nakagami para gerar diretamente
vtNakagamiNormEnvelope = nakagami.rvs(m, size=nSamples)

# Fading em dB
vtNakagamiSampdB = 20 * np.log10(vtNakagamiNormEnvelope)

# ---------- Cálculo da potência recebida ----------
# Ajuste nos vetores devido à filtragem do shadowing
vtTxPower = sPar['txPower'] * np.ones(nSamples)
vtTxPower = vtTxPower[jan:nSamples - jan]
vtPathLoss = vtPathLoss[jan:nSamples - jan]
vtFading = vtNakagamiSampdB[jan:nSamples - jan]
vtDist = vtDist[jan:nSamples - jan]

vtPrx = vtTxPower - vtPathLoss + vtShadCorr + vtFading

# ---------- Salvamento dos dados em arquivo .npz ----------
np.savez(f"{sPar['chFileName']}.npz",
         vtDist=vtDist,
         vtPathLoss=vtPathLoss,
         vtShadCorr=vtShadCorr,
         vtFading=vtFading,
         vtPrx=vtPrx)

# ---------- Exibição de informações ----------
print('Canal sintético:')
print(f'   Média do sombreamento: {np.mean(vtShadCorr):.4f}')
print(f'   Std do sombreamento: {np.std(vtShadCorr):.4f}')
print(f'   Janela de correlação do sombreamento: {sPar["shadowingWindow"]} amostras')
print(f'   Expoente de path loss: {sPar["n"]}')
print(f'   m de Nakagami: {sPar["m"]}')

# ---------- Gráfico da potência recebida ----------
plt.figure()
log_distancia = np.log10(vtDist)
plt.plot(log_distancia, vtPrx, label='Prx canal completo')
plt.plot(log_distancia, sPar['txPower'] - vtPathLoss, label='Prx (somente path loss)', linewidth=1)
plt.plot(log_distancia, sPar['txPower'] - vtPathLoss + vtShadCorr, label='Prx (path loss + shadowing)', linewidth=1, color='yellow')
plt.xlabel('log10(d)')
plt.ylabel('Potência [dBm]')
plt.title('Canal sintético: Potência recebida vs log da distância')
plt.legend()
plt.xlim([0.7, 1.6])
plt.grid(True)

# ---------- Gráfico do desvanecimento Nakagami ----------
plt.figure()
f_hist, x_hist = np.histogram(vtNakagamiNormEnvelope, bins=100, density=True)
x_centers = (x_hist[:-1] + x_hist[1:]) / 2
plt.bar(x_centers, f_hist, width=(x_hist[1] - x_hist[0]), alpha=0.6, label='Histograma normalizado')
x_pdf = np.linspace(0, np.max(vtNakagamiNormEnvelope), 1000)
plt.plot(x_pdf, fpNakaPdf(x_pdf), 'r', label='PDF teórica')
plt.title('Canal sintético - desvanecimento de pequena escala')
plt.legend()
plt.grid(True)

plt.show()
