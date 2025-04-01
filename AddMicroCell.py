import numpy as np
from fGenerateGraph import GenerateGraph

dFc = 800
X = 1500
Y = 1732.051
dR = 500  # Raio do Hexágono
dHMob = 1.5  # Altura do receptor em metros
dHBs = 32  # Altura do transmissor em metros
dPtdBm = 21  # EIRP em dBm (incluindo ganho e perdas)
dPtdBmMicro = 21
dSensitivity = -90  # Sensibilidade do receptor
vtBsMicro = np.load('ListMicroCell.npy')
NewvtBsMicro = X + 1j*Y
if (not (NewvtBsMicro in vtBsMicro)):
    vtBsMicro = np.append (vtBsMicro, NewvtBsMicro)
vtBsMicro = np.array (vtBsMicro)
np.save('ListMicroCell.npy', vtBsMicro)
GenerateGraph(dFc, dR, dHMob, dHBs, dPtdBm, dPtdBmMicro, vtBsMicro, dSensitivity)
