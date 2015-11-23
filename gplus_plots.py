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
power_law_df = pd.DataFrame(columns = ['log10Degree', 'log10FittedCount'])
power_law_df['log10Degree'] = np.arange(0.0, np.log10(max(degrees) * 1.2), 0.1)
power_law_df['log10FittedCount'] = power_law_df['log10Degree'].map(lambda x : log10C - power_law.alpha * x)
power_law_df = power_law_df[power_law_df['log10FittedCount'] >= 0]
degree_df = pd.DataFrame([np.log10(count) for count in degree_dict.values() if (count > 0)], columns = ['log10Degree'])
bins = np.linspace(0.0, np.log10(max(degrees) * 1.2), 30)[1:]
bin_labels = np.digitize(degree_df['log10Degree'], bins)
bin_counts = [0 for i in range(len(bins))]
for i in bin_labels:
    bin_counts[i] += 1
bin_df = pd.DataFrame(columns = ['log10Degree', 'log10Count'])
bin_df['log10Degree'] = bins
bin_df['log10Count'] = list(map(np.log10, bin_counts))
degree_plot = ggplot(aes(x = 'log10Degree', y = 'log10Count', width = 0.175), data = bin_df) + geom_bar(stat = 'identity', fill = 'royalblue', color = 'black') + ggtitle("Degree distribution\nPower law alpha = %.3f, log10(intercept) = %.3f" % (power_law.alpha, log10C)) + xlab("log10(degree)") + ylab("log10(count)")
degree_plot += geom_line(aes(x = 'log10Degree', y = 'log10FittedCount'), data = power_law_df, size = 3, color = 'red') + xlim(low = -1, high = np.log10(max(degrees) * 1.1)) + ylim(low = -1, high = np.log(max(power_law_df['log10FittedCount'] * 1.1)))

# component size rank plot
comp_size_df = pd.read_csv(folder + '/data/comp_sizes.csv').sort_values(by = 'componentSize', ascending = False)
comp_size_plot = ggplot(aes(x = list(range(len(comp_size_df))), y = comp_size_df['componentSize']), data = comp_size_df) + geom_point(size = 20, color = 'maroon') + scale_y_log10() + ggtitle("Component sizes") + scale_x_continuous(breaks = list(np.arange(0, 1200, 100))) + xlim(low = -25, high = 1175) + xlab("rank") + ylim(low = 0.5, high = 10**7) + ylab("comp size")

ggsave(folder + '/plots/degree_plot', degree_plot)
ggsave(folder + '/plots/component_size_plot', comp_size_plot)
