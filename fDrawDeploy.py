import numpy as np
import plotly.graph_objects as go
from fDrawSector import DrawSector

def DrawDeploy(dR, vtBs, fig):
    
    # Desenhando os setores hexagonais
    for vtB in vtBs:
        x, y = DrawSector(dR, vtB)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='black')))

    # Plotando as posições das bases (como círculos vermelhos)
    #vtBs = np.array(vtBs)
    #fig.add_trace(go.Scatter(x=vtBs.real, y=vtBs.imag, mode='markers', marker=dict(color='red', size=4)))

    # Atualizando o layout
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(scaleanchor="y"),  # Garante que a escala de x e y seja a mesma
        yaxis=dict(scaleanchor="x")
    )
