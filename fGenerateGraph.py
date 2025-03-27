import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from fDrawDeploy import DrawDeploy

def GenerateGraph (dFc, dR, dHMob, dHBs, dPtdBm, dPtdBmMicro, vtBsMicro, dSensitivity):
           
    # Cálculos de outras variáveis que dependem dos parâmetros de entrada
    dPasso = np.ceil(dR/20).astype(int)  # Resolução do grid: distância entre pontos de medição
    dRMin = dPasso  # Raio de segurança
    dIntersiteDistance = 2*np.sqrt(3/4)*dR  # Distância entre ERBs (somente para informação)
    dDimX = 5*dR  # Dimensão X do grid
    dDimY = 6*np.sqrt(3/4)*dR  # Dimensão Y do grid
    dPtLinear = 10**(dPtdBm/10)*1e-3  # EIRP em escala linear Watts
    #dAhm = 3.2 * (np.log10(11.75 * dHMob)) ** 2 - 4.97  # fator de correção do Modelo Okumura-Hata
    # Inclusão de condição com o fator de correção do Modelo Okumura-Hata para poder ser aplicado em frequências abaixo de 400 MHz
    if (dFc >= 400):
        dAhm = 3.2*(np.log10(11.75 * dHMob))**2 - 4.97
    else:
        dAhm = 8.29*(np.log10(11.54 * dHMob))**2 - 1.1

    # Vetor com posições das BSs (grid Hexagonal com 7 células, uma célula central e uma camada de células ao redor)
    vtBs = [0] # ERB 1 na posição 0,0
    dOffset = np.pi/6 # Ângulo relativo entre as ERBs com relação ao centro
    for iBs in range(2, 8):
        vtBs.append(dR*np.sqrt(3)*np.exp(1j*((iBs-2)*np.pi/3 + dOffset))) # Acrescenta a outras ERBs ao redor da ERB 1
    teste = vtBs
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
    
    
    for iBsD in range(len(vtBsMicro)):  
        
        mtPosEachBSMicro = (mtPosx + 1j*mtPosy) - vtBsMicro[iBsD] 
        mtDistEachBsMicro = np.abs(mtPosEachBSMicro)  
        mtDistEachBsMicro[mtDistEachBsMicro < dRMin] = dRMin  
        
        # Okumura-Hata (cidade urbana) - dB
        mtPldBMicro = 55 + 38*np.log10(mtDistEachBsMicro/1e3) + (24.5 + (1.5*dFc)/925)*np.log10(dFc)
        mtPowerEachBSdBmMicro = dPtdBmMicro - mtPldBMicro  # Potências recebidas em cada ponto de medição
        
        # Cálulo da maior potência em cada ponto de medição
        mtPowerFinaldBm = np.maximum(mtPowerFinaldBm, mtPowerEachBSdBmMicro)

    dOutRate = 100 * len(np.where(mtPowerFinaldBm < dSensitivity)[0]) / mtPowerFinaldBm.size
    
    
    dOutRatePoint = np.where (mtPowerFinaldBm < dSensitivity, 0, 1)
    
    # Criando o gráfico interativo com plotly
    fig = go.Figure()
    #fig = make_subplots(rows=1, cols=2)

    # Adicionando a camada de potência final no gráfico
    fig.add_trace(go.Heatmap(
        #z=mtPowerFinaldBm,  # Potência final (média das ERBs)
        z=dOutRatePoint,
        x=mtPosx[0, :],  # Posições em X
        y=mtPosy[:, 0], # Posições em Y
        colorscale='inferno',
        opacity=1, # Escolha da paleta de cores
        hovertemplate='X: %{x} <br>Y: %{y}<extra></extra>'  # Exibição do valor da potência ao passar o mouse
        ))
    
    fig.add_trace(go.Heatmap(
        z=mtPowerFinaldBm,  # Potência final (média das ERBs)
        x=mtPosx[0, :],  # Posições em X
        y=mtPosy[:, 0], # Posições em Y
        colorscale='plasma',  # Escolha da paleta de cores
        colorbar=dict(title="Potência em (dBm)"),
        opacity=0,
        hovertemplate='<b>Potência:</b> %{z} dBm <br>X: %{x} <br>Y: %{y}<extra></extra>'  # Exibição do valor da potência ao passar o mouse
    ))

    fig.update_layout(
        title="Plotagem da Potência Total das ERBs",
        xaxis_title="Posição X",
        yaxis_title="Posição Y",
        xaxis=dict(scaleanchor="y"),  # Para garantir que o gráfico seja proporcional
        yaxis=dict(scaleanchor="x"),
        legend=dict(entrywidth=0),
        coloraxis_showscale=False,
        showlegend=False
     )
    
    # Exibindo o gráfico
    DrawDeploy(dR, vtBs, fig)
    fig.show()
    print (dOutRatePoint)
    
