import numpy as np
import plotly.graph_objects as go
from fDrawSector import DrawSector

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

