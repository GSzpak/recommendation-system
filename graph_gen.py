%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt


x = [1, 3, 5, 10, 20, 30, 40, 50]
i_adjusted_cosine = [1.049, 0.885, 0.836, 0.792, 0.772, 0.768, 0.771, 0.776]
u_median_centered = [1.013, 0.867, 0.829, 0.8, 0.788, 0.787, 0.788, 0.79]
u_msd = [0.981, 0.849, 0.813, 0.787, 0.778, 0.779, 0.782, 0.785]

g1 = zip(x, i_adjusted_cosine)
g2 = zip(x, u_median_centered)
g3 = zip(x, u_msd)

assert len(g1) == len(x)
assert len(g2) == len(x)
assert len(g3) == len(x)

plt.plot(x, u_median_centered, label='user-user, median_centered_corr')
plt.plot(x, i_adjusted_cosine, label='item-item, adjusted_cosine')
legend = plt.legend(loc='upper right')
plt.show()