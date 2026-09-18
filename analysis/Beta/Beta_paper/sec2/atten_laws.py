import numpy as np
import matplotlib.pyplot as plt

from caesar.pyloser.atten_laws import calzetti, cardelli, smc, lmc

wave = np.linspace(1200, 3000, 1000)

plt.plot(wave, calzetti(wave), label='Calzetti')
plt.plot(wave, cardelli(wave), label='MW (Cardelli)')
plt.plot(wave, smc(wave), label='SMC')
plt.plot(wave, lmc(wave), label='LMC')

plt.axvline(1500, linestyle='--', alpha=0.5)
plt.axvline(2300, linestyle='--', alpha=0.5)
plt.axvline(2800, linestyle='--', alpha=0.5)

plt.xlabel(r'Wavelength [$\AA$]')
plt.ylabel(r'$A_\lambda/A_V$')
plt.legend()

plt.tight_layout()
plt.savefig('dust_attenuation_curves.png', dpi=300, bbox_inches='tight')
plt.close()
