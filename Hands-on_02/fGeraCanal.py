import numpy as np
import math as mt
from scipy.special import gamma
from scipy.stats import nakagami
import matplotlib.pyplot as plt

def gera_canal(sPar):
    nPoints = sPar['nPoints']
    totalLength = sPar['totalLength']
    P0 = sPar['P0']
    d0 = sPar['d0']
    n = sPar['n']
    sigma = sPar['sigma']
    shadowingWindow = sPar['shadowingWindow']
    m = sPar['m']
    dMed = sPar['dMed']
    txPower = sPar['txPower']

    # Distância do transmissor
    d = np.arange(d0, totalLength, dMed)
    nSamples = len(d)

    # Perda de percurso
    vtPathLoss = P0 + 10*n*np.log10(d/d0)

    # Sombreamento
    nShadowSamples = mt.floor(nSamples / shadowingWindow)
    shadowing = sigma * np.random.randn(nShadowSamples)
    restShadowing = sigma * np.random.randn(1) * np.ones(nSamples % shadowingWindow)
    shadowing = np.tile(shadowing, (shadowingWindow, 1))
    shadowing = shadowing.T.flatten()
    shadowing = np.concatenate((shadowing, restShadowing))

    # Filtro de média móvel
    jan = shadowingWindow // 2
    vtShadCorr = []
    for i in range(jan, nSamples - jan):
        vtShadCorr.append(np.mean(shadowing[i - jan:i + jan + 1]))
    vtShadCorr = np.array(vtShadCorr)

    # Ajuste do desvio padrão
    vtShadCorr *= np.std(shadowing) / np.std(vtShadCorr)
    vtShadCorr += np.mean(shadowing) - np.mean(vtShadCorr)

    # Nakagami PDF
    #def nakagami_pdf(x):
     #   return (2 * (m ** m) / gamma(m)) * (x ** (2 * m - 1)) * np.exp(-m * x ** 2)

    # Amostragem via rejeição
    #x_vals = np.linspace(0, 3, 5000)
   # pdf_vals = nakagami_pdf(x_vals)
   # max_pdf = np.max(pdf_vals)
    #nakagami_samples = []
    #while len(nakagami_samples) < nSamples:
     #   x_try = np.random.uniform(0, 3)
      #  y_try = np.random.uniform(0, max_pdf)
       # if y_try < nakagami_pdf(x_try):
        #    nakagami_samples.append(x_try)
    #nakagami_samples = np.array(nakagami_samples[:nSamples])
    #nakagamiSamp = 20 * np.log10(nakagami_samples)
    fpNakaPdf = lambda x: ((2* (m**m))/gamma(m))*x**(2*m-1)*np.exp(-m*x**2)
    # Amostras com distribuição de Nakagami (envelope normalizado)
    # Usando scipy.stats.nakagami para gerar diretamente
    vtNakagamiNormEnvelope = nakagami.rvs(m, size=nSamples)
    # Fading em dB
    vtNakagamiSampdB = 20 * np.log10(vtNakagamiNormEnvelope)

    # Ajuste de tamanho por causa da filtragem
    txPowerVec = txPower * np.ones(nSamples)
    txPowerVec = txPowerVec[jan:nSamples - jan]
    vtPathLoss = vtPathLoss[jan:nSamples - jan]
    vtFading = vtNakagamiSampdB[jan:nSamples - jan]
    vtDist = d[jan:nSamples - jan]

    # Potência recebida
    vtPrxdBm = txPowerVec - vtPathLoss + vtShadCorr + vtFading
    # ---------- Salvamento dos dados em arquivo .npz ----------
    np.savez(f"{sPar['chFileName']}.npz",
         vtDist=vtDist,
         vtPathLoss=vtPathLoss,
         vtShadCorr=vtShadCorr,
         vtFading=vtFading,
         vtPrxdBm=vtPrxdBm)
    
    return vtDist, vtPathLoss, vtShadCorr, vtFading, vtPrxdBm
