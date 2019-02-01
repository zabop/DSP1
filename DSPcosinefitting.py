import numpy as np
data=np.genfromtxt('signal.data')

time=data[:,0]
signal=data[:,1]
signalerror=data[:,2]


import matplotlib.pyplot as plt
plt.figure()
plt.plot(time,signal)
plt.scatter(time,signal,s=5)
plt.show()



from gatspy.periodic import LombScargleFast
dmag=0.000005
nyquist_factor=40

model = LombScargleFast().fit(time, signal, dmag)
periods, power = model.periodogram_auto(nyquist_factor)

model.optimizer.period_range=(0.2, 10)
period = model.best_period
print(period)


def G(x, A_0,
         A_1, phi_1,
         A_2, phi_2,
         A_3, phi_3,
         A_4, phi_4,
         A_5, phi_5,
         A_6, phi_6,
         A_7, phi_7,
         A_8, phi_8,
         A_9, phi_9,
         A_10, phi_10,
         freq):
    return (A_0 + A_1 * np.sin(2 * np.pi * 1 * freq * x + phi_1) +
                  A_2 * np.sin(2 * np.pi * 2 * freq * x + phi_2) +
                  A_3 * np.sin(2 * np.pi * 3 * freq * x + phi_3) +
                  A_4 * np.sin(2 * np.pi * 4 * freq * x + phi_4) +
                  A_5 * np.sin(2 * np.pi * 5 * freq * x + phi_5) +
                  A_6 * np.sin(2 * np.pi * 6 * freq * x + phi_6) +
                  A_7 * np.sin(2 * np.pi * 7 * freq * x + phi_7) +
                  A_8 * np.sin(2 * np.pi * 8 * freq * x + phi_8) +
                  A_9 * np.sin(2 * np.pi * 9 * freq * x + phi_9) +
                  A_10 * np.sin(2 * np.pi * 10 * freq * x + phi_10))







def fitter(time, signal, signalerror, LSPfreq):

    from scipy import optimize

    pfit, pcov = optimize.curve_fit(lambda x, _A_0,
                                           _A_1, _phi_1,
                                           _A_2, _phi_2,
                                           _A_3, _phi_3,
                                           _A_4, _phi_4,
                                           _A_5, _phi_5,
                                           _A_6, _phi_6,
                                           _A_7, _phi_7,
                                           _A_8, _phi_8,
                                           _A_9, _phi_9,
                                           _A_10, _phi_10,
                                           _freqfit:

                                    G(x, _A_0, _A_1, _phi_1,
                                      _A_2, _phi_2,
                                      _A_3, _phi_3,
                                      _A_4, _phi_4,
                                      _A_5, _phi_5,
                                      _A_6, _phi_6,
                                      _A_7, _phi_7,
                                      _A_8, _phi_8,
                                      _A_9, _phi_9,
                                      _A_10, _phi_10,
                                      _freqfit),

                                    time, signal, p0=[11,  1, 0, #p0 is the initial guess for numerical fitting
                                                           0.1, 0,
                                                           0, 0,
                                                           0, 0,
                                                           0, 0,
                                                           0, 0,
                                                           0, 0,
                                                           0, 0,
                                                           0, 0,
                                                           0, 0,
                                                      LSPfreq],


                                    sigma=signalerror, absolute_sigma=True)

    error = []  # DEFINE LIST TO CALC ERROR
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i]) ** 0.5)  # CALCULATE SQUARE ROOT OF TRACE OF COVARIANCE MATRIX
        except:
            error.append(0.00)
    perr_curvefit = np.array(error)

    return pfit, perr_curvefit

LSPfreq=1/period
pfit, perr_curvefit = fitter(time, signal, signalerror, LSPfreq)

plt.figure()
model=G(time,pfit[0],pfit[1],pfit[2],pfit[3],pfit[4],pfit[5],pfit[6],pfit[7],pfit[8],pfit[8],pfit[10],pfit[11],pfit[12],pfit[13],pfit[14],pfit[15],pfit[16],pfit[17],pfit[18],pfit[19],pfit[20],pfit[21])
plt.scatter(time,model,marker='+')
plt.plot(time,model)
plt.plot(time,signal,c='r')
plt.show()
