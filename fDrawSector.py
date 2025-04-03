import numpy as np
import plotly.graph_objects as go

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


