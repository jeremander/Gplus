import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from ggplot import *

folder = "gplus0"
#folder = "gplus0_sample1M"

# degree histogram and power law fit
degree_dict = pickle.load(open(folder + '/data/degree_dict.pickle', 'rb'))
power_law = pickle.load(open(folder + '/data/degree_power_law.pickle', 'rb'))
degrees = list(degree_dict.values())
degree_counts = defaultdict(int)
for deg in degrees:
    degree_counts[deg] += 1
del(degree_counts[0])
log10C = np.mean(np.array([power_law.alpha * np.log10(deg) + np.log10(count) for (deg, count) in degree_counts.items()]))
power_law_df = pd.DataFrame(columns = ['degree', 'fittedCount'])
power_law_df['degree'] = 10 ** (np.arange(0, np.log10(max(degrees) * 1.2), 0.1))
power_law_df['fittedCount'] = power_law_df['degree'].map(lambda x : 10 ** (log10C - power_law.alpha * np.log10(x)))
power_law_df = power_law_df[power_law_df['fittedCount'] >= 0.1]
degree_df = pd.DataFrame(list(degree_dict.values()), columns = ['degree'])
degree_plot = ggplot(aes(x = 'degree'), data = degree_df) + geom_histogram(binwidth = 20., fill = 'royalblue', color = 'black') + scale_x_log10() + scale_y_log10() + ggtitle("Degree distribution\nPower law alpha = %.3f, log10(intercept) = %.3f" % (power_law.alpha, log10C)) + xlab("node degree") + ylab("")
degree_plot += geom_line(aes(x = 'degree', y = 'fittedCount'), data = power_law_df, size = 3, color = 'red') + xlim(low = 0.5, high = max(degrees) * 1.1) + ylim(low = 0.5, high = max(power_law_df['fittedCount'] * 1.1))

# component size rank plot
comp_size_df = pd.read_csv(folder + '/data/comp_sizes.csv').sort_values(by = 'componentSize', ascending = False)
comp_size_plot = ggplot(aes(x = list(range(len(comp_size_df))), y = comp_size_df['componentSize']), data = comp_size_df) + geom_point(size = 20, color = 'maroon') + scale_y_log10() + ggtitle("Component sizes") + scale_x_continuous(breaks = list(np.arange(0, 1200, 100))) + xlim(low = -25, high = 1175) + xlab("rank") + ylim(low = 0.5, high = 10**7) + ylab("comp size")

ggsave(folder + '/plots/degree_plot', degree_plot)
ggsave(folder + '/plots/component_size_plot', comp_size_plot)
