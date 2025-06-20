import numpy as np
from scipy.stats import f_oneway, ttest_ind

np.random.seed(0)
algo1_no_tl = np.random.normal(0.70, 0.02, 1000)
algo1_tl = np.random.normal(0.70, 0.04, 1000)
algo2_no_tl = np.random.normal(0.68, 0.03, 1000)
algo2_tl = np.random.normal(0.70, 0.05, 1000)

f_stat, p_value = f_oneway(algo1_no_tl, algo1_tl, algo2_no_tl, algo2_tl)
print(f"F-stat: {f_stat:.3f}, p-value: {p_value:.4f}")

###### alpha 0.05 (https://datatab.net/tutorial/t-distribution)
# last print: hypothesis 0: all algo despite with or without TL have the same mean
# last print: p-value with 0.000... << 0.05, reject -> at least one group is different than at least one other group
# last print: F-stat 84.473, high value -> evidence of statistical differences, not by chance
######

t_stat, p = ttest_ind(algo1_no_tl, algo1_tl)
print(f"t-value: {t_stat:.3f}, p-value: {p:.4f}")

t_stat, p = ttest_ind(algo2_no_tl, algo2_tl)
print(f"t-value: {t_stat:.3f}, p-value: {p:.4f}")

t_stat, p = ttest_ind(algo2_no_tl, algo1_no_tl)
print(f"t-value: {t_stat:.3f}, p-value: {p:.4f}")

t_stat, p = ttest_ind(algo1_tl, algo2_tl)
print(f"t-value: {t_stat:.3f}, p-value: {p:.4f}")

###### alpha 0.05 (https://datatab.net/tutorial/t-distribution)
# last print: hypothesis 0: the mean between both algo after TL is equivalent
# last print: p-value with 0.46 >> 0.05, accept -> no evidence of statistical differences
# last print: t-value 0.738 << 1.962 accept (prove) -> no significant differences
######
