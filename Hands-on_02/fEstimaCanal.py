import numpy as np

def f_estima_canal(sPar):
    # Carrega os dados salvos no arquivo .npz
    data = np.load(sPar['chFileName']+".npz")
    vtPrxdBm = data['vtPrxdBm']
    vtDist = data['vtDist']

    vtPtrxmW = 10 ** (vtPrxdBm / 10)
    nSamples = len(vtPtrxmW)

    # Inicializações
    vtDesLarga = []
    vtDesPequeEst = []

    # Janela de média
    dMeiaJanela = round((sPar['dW'] - 1) / 2)

    for ik in range(dMeiaJanela+1, nSamples - dMeiaJanela):
        media_janela = np.mean(vtPtrxmW[ik - dMeiaJanela:ik + dMeiaJanela])
        des_larga = 10 * np.log10(media_janela)
        vtDesLarga.append(des_larga)
        des_peq = vtPrxdBm[ik] - des_larga
        vtDesPequeEst.append(des_peq)

    indexes = range(dMeiaJanela+1, nSamples - dMeiaJanela)
    vtPtrxmWNew = 10 ** (vtPrxdBm[indexes] / 10)
    desLarga_Lin = 10 ** (np.array(vtDesLarga) / 10)
    envNormal = np.sqrt(vtPtrxmWNew) / np.sqrt(desLarga_Lin)

    vtDistEst = vtDist[dMeiaJanela+1:nSamples - dMeiaJanela]
    vtPrxdBmEst = vtPrxdBm[dMeiaJanela+1:nSamples - dMeiaJanela]

    vtDistLog = np.log10(vtDist)
    vtDistLogEst = np.log10(vtDistEst)
    dCoefReta = np.polyfit(vtDistLogEst, vtPrxdBmEst, 1)
    dNEst = -dCoefReta[0] / 10
    vtPathLossEst = np.polyval(dCoefReta, vtDistLogEst)

    vtShadCorrEst = vtDesLarga - vtPathLossEst
    dStdShadEst = np.std(vtShadCorrEst)
    dStdMeanShadEst = np.mean(vtShadCorrEst)
    vtPathLossEst = -vtPathLossEst
    vtPrxEst = sPar['txPower'] - vtPathLossEst + vtShadCorrEst + vtDesPequeEst

    vtn = np.arange(1, sPar['nCDF'])
    xCDF = (1.2 ** (vtn - 1)) * 0.01
    cdffn = np.array([np.sum(envNormal <= x) for x in xCDF])
    yccdfEst = cdffn / cdffn[-1]
    xccdfEst = 20 * np.log10(xCDF)

    sOut = {
        'vtDistEst': vtDistEst,
        'vtPathLossEst': vtPathLossEst,
        'dNEst': dNEst,
        'vtShadCorrEst': vtShadCorrEst,
        'dStdShadEst': dStdShadEst,
        'dStdMeanShadEst': dStdMeanShadEst,
        'vtDesPequeEst': vtDesPequeEst,
        'vtPrxEst': vtPrxEst,
        'vtXCcdfEst': xccdfEst,
        'vtYCcdfEst': yccdfEst,
        'vtEnvNorm': envNormal 
    }

    return sOut
