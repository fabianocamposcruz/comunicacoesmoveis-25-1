import numpy as np
import plotly.graph_objects as go
from fDrawDeploy import fDrawDeploy
from fDrawSector import fDrawSector

# Entrada de parâmetros
dR = 500  # Raio do Hexágono
dFc = 800  # Frequência da portadora MHz
dHMob = 1.5  # Altura do receptor em metros
dHBs = 32  # Altura do transmissor em metros
dPtdBm = 21  # EIRP em dBm (incluindo ganho e perdas)
dSensitivity = -90  # Sensibilidade do receptor

# Cálculos de outras variáveis que dependem dos parâmetros de entrada
dPasso = np.ceil(dR/16).astype(int)  # Resolução do grid: distância entre pontos de medição
dRMin = dPasso  # Raio de segurança
dIntersiteDistance = 2*np.sqrt(3/4)*dR  # Distância entre ERBs (somente para informação)
dDimX = 5*dR  # Dimensão X do grid
dDimY = 6*np.sqrt(3/4)*dR  # Dimensão Y do grid
dPtLinear = 10**(dPtdBm/10)*1e-3  # EIRP em escala linear Watts
#dAhm = 3.2 * (np.log10(11.75 * dHMob)) ** 2 - 4.97  # fator de correção do Modelo Okumura-Hata
# Inclusão de condição com o fator de correção do Modelo Okumura-Hata para poder ser aplicado em frequências abaixo de 300 MHz
if (dFc >= 300):
    dAhm = 3.2*(np.log10(11.75 * dHMob))**2 - 4.97
else:
    dAhm = 8.29*(np.log10(11.54 * dHMob))**2 - 1.1

# Vetor com posições das BSs (grid Hexagonal com 7 células, uma célula central e uma camada de células ao redor)
vtBs = [0] # ERB 1 na posição 0,0
dOffset = np.pi/6 # Ângulo relativo entre as ERBs com relação ao centro
for iBs in range(2, 8):
    vtBs.append(dR*np.sqrt(3)*np.exp(1j*((iBs-2)*np.pi/3 + dOffset))) # Acrescenta a outras ERBs ao redor da ERB 1
vtBs = np.array(vtBs) + (dDimX/2 + 1j*dDimY/2)  # Ajuste de posição das bases (posição relativa ao canto inferior esquerdo)

# Matriz de referência com posição de cada ponto do grid
dDimY = dDimY + np.mod(dDimY, dPasso)  # Ajuste de dimensão para medir toda a dimensão do grid
dDimX = dDimX + np.mod(dDimX, dPasso)  # Ajuste de dimensão para medir toda a dimensão do grid
mtPosx, mtPosy = np.meshgrid(np.arange(0, dDimX, dPasso), np.arange(0, dDimY, dPasso)) # O meshgrid gera uma matriz vetorizada e o arange retorna valores uniformemente espaçados (inicío, fim, step)

# Iniciação da Matriz de potência recebida máxima em cada ponto medido
mtPowerFinaldBm = -np.inf * np.ones_like(mtPosy)

# Calcular O REM de cada ERB e acumular a maior potência em cada ponto de medição
for iBsD in range(len(vtBs)):  # Loop nas 7 ERBs/ o vtBs é a posição (X,Y) de cada ERB
    # Matriz 3D com os pontos de medição de cada ERB
    mtPosEachBS = (mtPosx + 1j*mtPosy) - vtBs[iBsD] # Matriz com a diferença entre a posição da ERB em questão e os demais pontos
    mtDistEachBs = np.abs(mtPosEachBS)  # Distância entre cada ponto de medição e a ERB, ou seja, o valor absoluto da diferença anterior
    mtDistEachBs[mtDistEachBs < dRMin] = dRMin  # Implementação do raio de segurança
    
    # Okumura-Hata (cidade urbana) - dB
    mtPldB = 69.55 + 26.16*np.log10(dFc) + (44.9 - 6.55*np.log10(dHBs))*np.log10(mtDistEachBs/1e3) - 13.82*np.log10(dHBs) - dAhm
    mtPowerEachBSdBm = dPtdBm - mtPldB  # Potências recebidas em cada ponto de medição
    
    # Cálulo da maior potência em cada ponto de medição
    mtPowerFinaldBm = np.maximum(mtPowerFinaldBm, mtPowerEachBSdBm)

dOutRate = 100 * len(np.where(mtPowerFinaldBm < dSensitivity)[0]) / mtPowerFinaldBm.size

# Criando o gráfico interativo com plotly
fig = go.Figure()

# Adicionando a camada de potência final no gráfico
fig.add_trace(go.Heatmap(
    z=mtPowerFinaldBm,  # Potência final (média das ERBs)
    x=mtPosx[0, :],  # Posições em X
    y=mtPosy[:, 0],
    customdata=np.array(dOutRate), # Posições em Y
    colorscale='plasma',  # Escolha da paleta de cores
    colorbar=dict(title="Potência (dBm)"),
    hovertemplate='<b>Potência:</b> %{z} dBm <br>Outage: %{customdata} <br>X: %{x} <br>Y: %{y}<extra></extra>'  # Exibição do valor da potência ao passar o mouse
))

# Títulos e ajustes do gráfico
fig.update_layout(
    title="Plotagem da Potência Total das ERBs",
    xaxis_title="Posição X",
    yaxis_title="Posição Y",
    xaxis=dict(scaleanchor="y"),  # Para garantir que o gráfico seja proporcional
    yaxis=dict(scaleanchor="x"),
    showlegend=False
)

# Exibindo o gráfico
fDrawDeploy(dR, vtBs,fig)
fig.show()
print (dOutRate)
