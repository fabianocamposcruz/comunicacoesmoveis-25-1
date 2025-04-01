import numpy as np
from fGenerateGraph import GenerateGraph

dFc = 800
dR = 500  # Raio do Hexágono
dHMob = 1.5  # Altura do receptor em metros
dHBs = 32  # Altura do transmissor em metros
dPtdBm = 21 # EIRP em dBm (incluindo ganho e perdas)
dPtdBmMicro = 20
dSensitivity = -90  # Sensibilidade do receptor
vtBsMicro = np.load('ListMicroCell.npy')
vtBsMicro =[]
np.save('ListMicroCell.npy', vtBsMicro)
GenerateGraph(dFc, dR, dHMob, dHBs, dPtdBm, dPtdBmMicro, vtBsMicro, dSensitivity)