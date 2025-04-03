import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Função para desenhar um setor hexagonal
def DrawSector(dR, dCenter):
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

def DrawDeploy(dR, vtBs, fig):
    
    dDimX = 5*dR  # Dimensão X do grid
    dDimY = 6*np.sqrt(3/4)*dR  # Dimensão Y do grid
    # Desenhando os setores hexagonais
    for vtB in vtBs:
        x, y = DrawSector(dR, vtB)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='darkorange'), opacity=0.6, hovertemplate='X: %{x} <br>Y: %{y}<extra></extra>', hoverinfo='none'))
    # Plotando as posições das bases (como círculos vermelhos)
    #vtBs = np.array(vtBs)
    fig.add_trace(go.Scatter(x=vtBs.real, y=vtBs.imag, mode='markers', marker=dict(color='red', size=10), hovertemplate='X: %{x} <br>Y: %{y}<extra></extra>', hoverinfo='none'))

# Entrada de parâmetros
dR = 5e3  # Raio do Hexágono
# Cálculos de outras variáveis que dependem dos parâmetros de entrada
dPasso = np.ceil(dR / 10)  # Resolução do grid: distância entre pontos de medição
dIntersiteDistance = 2 * np.sqrt(3 / 4) * dR  # Distância entre ERBs
dDimX = 5 * dR  # Dimensão X do grid
dDimY = 6 * np.sqrt(3 / 4) * dR  # Dimensão Y do grid
# Vetor com posições das BSs (grid Hexagonal com 7 células, uma célula central e uma camada de células ao redor)
vtBs = [0]
dOffset = np.pi / 6
for iBs in range(2, 8):
    vtBs.append(dR * np.sqrt(3) * np.exp(1j * ((iBs - 2) * np.pi / 3 + dOffset)))
vtBs = np.array(vtBs) + (dDimX / 2 + 1j * dDimY / 2)  # Ajuste de posição das bases

# Matriz de referência com posição de cada ponto do grid
dDimY = dDimY + np.mod(dDimY, dPasso)  # Ajuste de dimensão para medir toda a dimensão do grid
dDimX = dDimX + np.mod(dDimX, dPasso)  # Ajuste de dimensão para medir toda a dimensão do grid
mtPosx, mtPosy = np.meshgrid(np.arange(0, dDimX, dPasso), np.arange(0, dDimY, dPasso))

# Inicializa a lista que vai acumular os pontos filtrados
all_filtered_mtPos = []

# Calcular os pontos de medição relativos de cada ERB
for iBsD in range(len(vtBs)):  # Loop nas 7 ERBs
    # Matriz 2D com os pontos de medição relativos de cada ERB
    mtPosEachBS = (mtPosx + 1j * mtPosy) - vtBs[iBsD]
    
    # Calculando as distâncias corretamente
    distance = np.sqrt(np.square(np.real(mtPosEachBS)) + np.square(np.imag(mtPosEachBS)))
    
    # Máscara booleana para filtrar os pontos onde a distância é menor ou igual a dR
    mask = distance <= dR
    
    # Usando a máscara para armazenar os pontos de mtPosEachBS com distância <= dR
    filtered_mtPos = mtPosEachBS[mask]
    
    # Acumula os pontos filtrados em cada iteração
    all_filtered_mtPos.append(filtered_mtPos)
    
    # Plot da posição relativa dos pontos de medição de cada ERB individualmente
   # plt.figure()
    #plt.plot(np.real(filtered_mtPos), np.imag(filtered_mtPos), 'bo')  # Plot dos pontos filtrados
    #fDrawDeploy(dR, vtBs - vtBs[iBsD])
    #plt.axis('equal')
    #plt.title(f'ERB {iBsD + 1}')

# Concatena todos os pontos filtrados de todas as ERBs em um único array
all_filtered_mtPos = np.concatenate(all_filtered_mtPos)

fig = go.Figure()
# Mostrar os pontos filtrados acumulados no final
fig.add_trace(go.Scatter(
    x=np.real(all_filtered_mtPos),  # Posições em X
    y=np.imag(all_filtered_mtPos),  # Posições em Y
    mode='markers',  # Mostrando como pontos (sem linhas conectando)
    marker=dict(color='blue', size=5)  # Personalizando a cor e o tamanho dos pontos
))
fig.update_layout(
        template="simple_white",
        title=f"Campo com Outage: ",
        xaxis_title="Posição X",
        yaxis_title="Posição Y",
        xaxis=dict(scaleanchor="y"),  # Para garantir que o gráfico seja proporcional
        yaxis=dict(scaleanchor="x"),
        legend=dict(entrywidth=0),
        showlegend=False
     )
    
    # Exibindo o gráfico
DrawDeploy(dR, vtBs, fig)
fig.show()