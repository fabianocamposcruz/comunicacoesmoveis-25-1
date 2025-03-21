import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Entrada de parâmetros
dR = 5e3  # Raio do Hexágono
dFc = 800  # Frequência da portadora MHz

# Cálculos de outras variáveis que dependem dos parâmetros de entrada
dPasso = np.ceil(dR / 20).astype(int)  # Resolução do grid: distância entre pontos de medição
dRMin = dPasso  # Raio de segurança
dIntersiteDistance = 2 * np.sqrt(3 / 4) * dR  # Distância entre ERBs (somente para informação)
dDimX = 5 * dR  # Dimensão X do grid
dDimY = 6 * np.sqrt(3 / 4) * dR  # Dimensão Y do grid
dPtdBm = 57  # EIRP (incluindo ganho e perdas)
dPtLinear = 10 ** (dPtdBm / 10) * 1e-3  # EIRP em escala linear
dHMob = 5  # Altura do receptor
dHBs = 30  # Altura do transmissor
dAhm = 3.2 * (np.log10(11.75 * dHMob)) ** 2 - 4.97  # Modelo Okumura-Hata

def fDrawSector(dR, dCenter):
    # Criando um array para armazenar os pontos do hexágono
    vtHex = []
    
    # Calculando os 6 pontos que formam o hexágono
    for ie in range(1, 7):
        vtHex.append(dR * (np.cos((ie - 1) * np.pi / 3) + 1j * np.sin((ie - 1) * np.pi / 3)))
    
    # Adicionando o ponto central a cada coordenada
    vtHex = np.array(vtHex) + dCenter
    
    # Adicionando o primeiro ponto novamente no final para fechar a figura
    vtHexp = np.concatenate([vtHex, [vtHex[0]]])

    # Retornar os dados do hexágono para plotagem
    return vtHexp.real, vtHexp.imag

def fDrawDeploy(dR, vtBs):
    # Criando a figura
    #fig = go.Figure()

    # Desenhando os setores hexagonais
    for vtB in vtBs:
        x, y = fDrawSector(dR, vtB)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='black')))

    # Plotando as posições das bases (como círculos vermelhos)
    #vtBs = np.array(vtBs)
    #fig.add_trace(go.Scatter(x=vtBs.real, y=vtBs.imag, mode='markers', marker=dict(color='red', size=4)))

    # Atualizando o layout
    fig.update_layout(
        template="plotly_dark",
       # showlegend=False,
        xaxis=dict(scaleanchor="y"),  # Garante que a escala de x e y seja a mesma
        yaxis=dict(scaleanchor="x")
    )

    # Exibindo o gráfico
    #fig.show()

# Vetor com posições das BSs (grid Hexagonal com 7 células, uma célula central e uma camada de células ao redor)
vtBs = [0]
dOffset = np.pi / 6
for iBs in range(2, 8):
    vtBs.append(dR * np.sqrt(3) * np.exp(1j * ((iBs - 2) * np.pi / 3 + dOffset)))
vtBs = np.array(vtBs) + (dDimX / 2 + 1j * dDimY / 2)  # Ajuste de posição das bases (posição relativa ao canto inferior esquerdo)

# Matriz de referência com posição de cada ponto do grid
dDimY = dDimY + np.mod(dDimY, dPasso)  # Ajuste de dimensão para medir toda a dimensão do grid
dDimX = dDimX + np.mod(dDimX, dPasso)  # Ajuste de dimensão para medir toda a dimensão do grid
mtPosx, mtPosy = np.meshgrid(np.arange(0, dDimX, dPasso), np.arange(0, dDimY, dPasso))

# Iniciação da Matriz de potência recebida máxima em cada ponto medido
mtPowerFinaldBm = -np.inf * np.ones_like(mtPosy)

# Calcular O REM de cada ERB e acumular a maior potência em cada ponto de medição
for iBsD in range(len(vtBs)):  # Loop nas 7 ERBs
    # Matriz 3D com os pontos de medição de cada ERB
    mtPosEachBS = (mtPosx + 1j * mtPosy) - vtBs[iBsD]
    mtDistEachBs = np.abs(mtPosEachBS)  # Distância entre cada ponto de medição e a sua ERB
    mtDistEachBs[mtDistEachBs < dRMin] = dRMin  # Implementação do raio de segurança
    
    # Okumura-Hata (cidade urbana) - dB
    mtPldB = 69.55 + 26.16 * np.log10(dFc) + (44.9 - 6.55 * np.log10(dHBs)) * np.log10(mtDistEachBs / 1e3) - 13.82 * np.log10(dHBs) - dAhm
    mtPowerEachBSdBm = dPtdBm - mtPldB  # Potências recebidas em cada ponto de medição
    
    # Cálulo da maior potência em cada ponto de medição
    mtPowerFinaldBm = np.maximum(mtPowerFinaldBm, mtPowerEachBSdBm)

# Criando o gráfico interativo com plotly
fig = go.Figure()

# Adicionando a camada de potência final no gráfico
fig.add_trace(go.Heatmap(
    z=mtPowerFinaldBm,  # Potência final (média das ERBs)
    x=mtPosx[0, :],  # Posições em X
    y=mtPosy[:, 0],  # Posições em Y
    colorscale='plasma',  # Escolha da paleta de cores
    colorbar=dict(title="Potência (dBm)"),
    hovertemplate='Potência: %{z} dBm<extra></extra>'  # Exibição do valor da potência ao passar o mouse
))

# Títulos e ajustes do gráfico
fig.update_layout(
    title="Potência Final no Grid (Interativo)",
    xaxis_title="Posição X (metros)",
    yaxis_title="Posição Y (metros)",
    xaxis=dict(scaleanchor="y"),  # Para garantir que o gráfico seja proporcional
    yaxis=dict(scaleanchor="x"),
    showlegend=False
)

# Exibindo o gráfico
fDrawDeploy(dR, vtBs)
fig.show()
