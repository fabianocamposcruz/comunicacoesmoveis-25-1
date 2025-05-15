import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from distfit import distfit
from sklearn.metrics import mean_squared_error
from fGeraCanal import gera_canal
from fEstimaCanal import f_estima_canal

# Parâmetros para geração do canal sintético
sPar = {
    'd0': 5,                        # Distância de referência d0
    'P0': 0,                        # Potência medida na distância de referência d0 (em dBm)
    'nPoints': 50000,               # Número de amostras da rota de medição
    'totalLength': 100,              # Distância final da rota de medição
    'n': 4,                          # Expoente de perda de percurso
    'sigma': 6,                      # Desvio padrão do shadowing em dB
    'shadowingWindow': 200,          # Tamanho da janela de correlação do shadowing
    'm': 4,                          # Parâmetro de Nakagami
    'txPower': 0,                    # Potência de transmissão em dBm
    'nCDF': 40,                      # Número de pontos da CDF normalizada
    'dW': 100,                       # Janela de estimação do sombreamento
    'chFileName': 'Prx_sintetico',   # Nome do arquivo de canal
    'dMed': 100 / 50000              # Distância entre pontos de medição
}

# Gera o canal sintético
vtDist, vtPathLoss, vtShadCorr, vtFading, vtPrxdBm = gera_canal(sPar)

# Exibe informações do canal sintético
print('Canal sintético:')
print(f'   Média do sombreamento: {np.mean(vtShadCorr)}')
print(f'   Std do sombreamento: {np.std(vtShadCorr)}')
print(f'   Janela de correlação do sombreamento: {sPar["shadowingWindow"]} amostras')
print(f'   Expoente de path loss: {sPar["n"]}')
print(f'   m de Nakagami: {sPar["m"]}')

# Várias janelas de filtragem para testar a estimação
vtW = [10, 50, 150, 200]
vtMSEShad = []
vtMSEFad = []
plt.figure()
chMarkers = ['o-', 'x-', 's-', 'd-', '>-', '^-', '-.']
for iw in range(len(vtW)):
    sPar['dW'] = vtW[iw]
    sOut = f_estima_canal(sPar)
    
    vtDistEst = sOut['vtDistEst']
    vtPathLossEst = sOut['vtPathLossEst']
    dNEst = sOut['dNEst']
    vtShadCorrEst = sOut['vtShadCorrEst']
    dStdShadEst = sOut['dStdShadEst']
    dStdMeanShadEst = sOut['dStdMeanShadEst']
    vtDesPequeEst = sOut['vtDesPequeEst']
    vtPrxEst = sOut['vtPrxEst']
    vtXCcdfEst = sOut['vtXCcdfEst']
    vtYCcdfEst = sOut['vtYCcdfEst']
    vtDistLogEst =np.log10(vtDistEst)
    vtDistLog = np.log10(vtDist)
    # MSE com Shadowing conhecido
    dMeiaJanela = round((sPar['dW'] - 1) / 2)
    len_min = min(len(vtShadCorr), len(vtShadCorrEst))
    mse_shad = mean_squared_error(vtShadCorr[dMeiaJanela+1:-dMeiaJanela], vtShadCorrEst)
    vtMSEShad.append(mse_shad)
    
        # MSE com Fading conhecido
    mse_fad = mean_squared_error(vtDesPequeEst, vtFading[dMeiaJanela+1:-dMeiaJanela])
    vtMSEFad.append(mse_fad)
    
    print(f'Estimação dos parâmetros de larga escala (W = {sPar["dW"]}):')
    print(f'   Expoente de perda de percurso estimado n = {sOut["dNEst"]}')
    print(f'   Desvio padrão do sombreamento estimado = {sOut["dStdShadEst"]}')
    print(f'   Média do sombreamento estimado = {sOut["dStdMeanShadEst"]}')
    print(f'   MSE Shadowing = {mse_shad}')
    print('----\n')
    plt.plot(sOut['vtXCcdfEst'], sOut['vtYCcdfEst'], chMarkers[iw % len(chMarkers)], label=f'W = {vtW[iw]}')
    

# Estudo na melhor janela de filtragem
print(f'Estudo na melhor janela de filtragem')
print(f'   Janelas utilizadas = {vtW}')
# Melhor janela com Shadowing conhecido
valBestShad = min(vtMSEShad)
posBestShad = vtMSEShad.index(valBestShad)
print(f'   Melhor MSE relativo aos valores reais do Shadowing (melhor janela):')
print(f'      Melhor janela W = {vtW[posBestShad]}: MSE Shadowing = {valBestShad}')
# Melhor janela com Fading conhecido
valBestFad = min(vtMSEFad)
posBestFad = vtMSEFad.index(valBestFad)
print(f'   Melhor MSE relativo aos valores reais do Fading:')
print(f'      Melhor janela W = {vtW[posBestFad]}: MSE Shadowing = {valBestFad}')
print('----------------------------------------------------------------------------------\n')
    
chLegendaW = [f'W = {w}' for w in vtW]
plt.legend(chLegendaW)

plt.xlabel('x')
plt.ylabel('F(x)')
plt.axis([-10, 10, 1e-5, 1])
plt.grid(True, which='both')
plt.title('Comparação entre CCDF estimadas e CDFs Nakagami teóricas')

# Plot das CDFs Nakagami teórica
vtm = sPar['m']
xCDF = 10 ** (sOut['vtXCcdfEst'] / 20)
tam_dist = len(xCDF)  # Tamanho da distribuição
for ik in range(vtm):
    im = vtm
    cdfnaka = gamma.cdf(im * xCDF ** 2, im)
    plt.plot(20 * np.log10(xCDF), cdfnaka, '--', linewidth=2)
        

# Cálculo do erro médio quadrático da CDF do Fading
print('MSE da CDF com várias janelas de filtragem com o conhecimento do Fading:')
vtm = [2,3,4,5]
mtMSEFad = np.zeros((len(vtm), len(vtW)))
for ik, m in enumerate(vtm):
    for il, w in enumerate(vtW):
        sPar['dW'] = w
        sOut = f_estima_canal(sPar)
        xCDF = 10 ** (sOut['vtXCcdfEst'] / 20)
        cdfnaka = gamma.cdf(m*xCDF**2, m)

        if np.any(np.isnan(sOut['vtYCcdfEst'])) or np.any(np.isnan(cdfnaka)):
            mse_fad_cdf = np.nan
        else:
            mse_fad_cdf = mean_squared_error(cdfnaka, sOut['vtYCcdfEst'])

        mtMSEFad[ik, il] = mse_fad_cdf
        print(f'  m = {m}: W = {w}: MSE Fading = {mse_fad_cdf}')
    print('----')

# Melhor m para cada W
vLinha = np.min(mtMSEFad, axis=0)          # menor MSE para cada W
posLinha = np.argmin(mtMSEFad, axis=0)     # melhor m (linha) para cada W

valCol = np.min(vLinha)                    # menor MSE global
posCol = np.argmin(vLinha)                 # W correspondente

bestLin = posLinha[posCol]                 # índice da melhor linha (m)
bestCol = posCol                           # índice da melhor coluna (W)
m_best = vtm[bestLin]
print(f'Melhor MSE relativo aos valores reais do fading:')
print(f'   W = {vtW[bestCol]} e m = {vtm[bestLin]}: MSE Fading = {valCol}')
print('----------------------------------------------------------------------------------\n')

plt.show()
