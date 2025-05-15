import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
from scipy.stats import nakagami, kstest
from fGeraCanal import gera_canal

# Parâmetros para geração do canal sintético
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

# Gera o canal sintético
vtDist, vtPathLoss, vtShadCorr, vtFading, vtPrxdBm = gera_canal(sPar)

# Transforma potência de dBm para mW
vtPtrxmW = 10 ** (vtPrxdBm / 10)
nSamples = len(vtPtrxmW)

# Cálculo do desvanecimento lento e rápido
dMeiaJanela = round((sPar['dW'] - 1) / 2)
vtDesLarga = []
vtDesPequeEst = []

for ik in range(dMeiaJanela +1, nSamples - dMeiaJanela):
    media_janela = np.mean(vtPtrxmW[ik - dMeiaJanela:ik + dMeiaJanela])
    des_larga = 10 * np.log10(media_janela)
    vtDesLarga.append(des_larga)
    des_peq = vtPrxdBm[ik] - des_larga
    vtDesPequeEst.append(des_peq)

# Envoltória normalizada
indexes = range(dMeiaJanela+1, nSamples - dMeiaJanela)
vtPtrxmWNew = 10 ** (vtPrxdBm[indexes] / 10)
desLarga_Lin = 10 ** (np.array(vtDesLarga) / 10)
envNormal = np.sqrt(vtPtrxmWNew) / np.sqrt(desLarga_Lin)

# Ajuste nos vetores
vtDistEst = vtDist[dMeiaJanela+1:nSamples - dMeiaJanela]
vtPrxdBm = vtPrxdBm[dMeiaJanela+1:nSamples - dMeiaJanela]

# Estimativa da perda de percurso
vtDistLog = np.log10(vtDist)
vtDistLogEst = np.log10(vtDistEst)
dCoefReta = np.polyfit(vtDistLogEst, vtPrxdBm, 1)
dNEst = -dCoefReta[0] / 10
print(f'   Estimação dos parâmetros de larga escala (W = {sPar["dW"]}):')
print(f'   Expoente de perda de percurso estimado n = {dNEst:.4f}')

vtPathLossEst = np.polyval(dCoefReta, vtDistLogEst)
vtShadCorrEst = np.array(vtDesLarga) - vtPathLossEst
stdShad = np.std(vtShadCorrEst)
meanShad = np.mean(vtShadCorrEst)
print(f'   Desvio padrão do sombreamento estimado = {stdShad:.4f}')
print(f'   Média do sombreamento estimado = {meanShad:.4f}')

vtPathLossEst = -vtPathLossEst
vtPrxEst = sPar['txPower'] - vtPathLossEst + vtShadCorrEst + np.array(vtDesPequeEst)

# Estimativa da CDF do fading
vtn = np.arange(1, sPar['nCDF'])
xCDF = 1.2 ** (vtn - 1) * 0.01
cdffn = np.array([np.sum(envNormal <= x) for x in xCDF])
yccdfEst = cdffn / cdffn[-1]
xccdfEst = 20 * np.log10(xCDF)

# Gráficos
plt.figure()
plt.plot(vtDistLogEst, vtPrxEst, label='Prx canal completo', linewidth=1)
plt.plot(vtDistLogEst, sPar['txPower'] - vtPathLossEst, label='Prx (somente path loss)', color='red', linewidth=1)
plt.plot(vtDistLogEst, sPar['txPower'] - vtPathLossEst + vtShadCorrEst, label='Prx (path loss + shadowing)', color='yellow', linewidth=1)
plt.xlabel('log10(d)')
plt.ylabel('Potência [dBm]')
plt.title('Prx original vs estimada')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(vtDistLog, -vtPathLoss, label='Path Loss original', linewidth=1)
plt.plot(vtDistLogEst, -vtPathLossEst, label='Path Loss estimado', linewidth=1)
plt.title('Perda de percurso original vs estimada')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(vtDistLog, vtShadCorr, label='Shadowing original', linewidth=1)
plt.plot(vtDistLogEst, vtShadCorrEst, label='Shadowing estimado', linewidth=1)
plt.title('Sombreamento original vs estimado')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(vtDistLog, vtFading, label='Fading original', linewidth=1)
plt.plot(vtDistLogEst, vtDesPequeEst, label='Fading estimado', linewidth=1)
plt.title('Fading original vs estimado')
plt.legend()
plt.grid(True)

# CDFs comparativas com Nakagami
plt.figure()
plt.plot(xccdfEst, yccdfEst, '--', label='CDF das amostras')
vtm = [1, 2, 4, 6]
xCDF_lin = 10 ** (xccdfEst / 20)
for m in vtm:
    #cdfnaka = gammainc(m * xCDF_lin ** 2, m)
    cdfnaka = gammainc(m, m*xCDF_lin**2)
    plt.plot(20 * np.log10(xCDF_lin), cdfnaka, label=f'm = {m}', linewidth=1)
plt.axis([-30, 10, 1e-5, 1])
plt.xlabel('x [dB]')
plt.ylabel('F(x)')
plt.title('Estudo do fading com conhecimento da distribuição')
plt.legend()
plt.grid(True)

plt.show()


data = envNormal
print("Teste de Kolmogorov-Smirnov para diferentes m da distribuição Nakagami:")
print("-------------------------------------------------------------")
for m in vtm:
    # Cria uma distribuição Nakagami teórica com parâmetro m
    dist = nakagami(m)
    
    
    # Aplica o teste KS: compara os dados com a CDF da distribuição Nakagami(m)
    D, p_value = kstest(data, dist.cdf)
    
    print(f"m = {m:.1f} | KS-stat = {D:.4f} | p-valor = {p_value:.4f}")