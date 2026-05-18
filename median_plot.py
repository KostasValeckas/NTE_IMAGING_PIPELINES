import numpy as np
import matplotlib.pyplot as plt

names = ["EP250108a_g", "GRB250327B_r", "GRB250129A_r", "GRB250404A_r", "GRB250404A_z"]

mags_gcn = [20.05, 17.13, 19.24, 20.43, 19.79]
uncertainties_gcn = [0.03, 0.04, 0.04, 0.05, 0.06]

mags_pipeline = [20.01, 17.36, 19.27, 20.43, 19.62]
uncertainties_pipeline = [0.06, 0.05, 0.09, 0.05, 0.07]

plt.figure(figsize=(10, 6))
plt.errorbar(names, mags_gcn, yerr=uncertainties_gcn, fmt='o', label='GCN Magnitudes', capsize=5, color='blue')
plt.errorbar(names, mags_pipeline, yerr=uncertainties_pipeline, fmt='x', label='Pipeline Magnitudes', capsize=5, color='orange')
plt.xlabel('Object Name')
plt.ylabel('Magnitude (AB)')
plt.title('Comparison of GCN and Pipeline Magnitudes')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()