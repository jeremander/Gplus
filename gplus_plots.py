import pickle
import pandas as pd
import numpy as np
from ggplot import *

folder = "gplus0"
#folder = "gplus0_sample1M"

# degree histogram
degree_dict = pickle.load(open(folder + '/data/degree_dict.pickle', 'rb'))
degree_df = pd.DataFrame(list(degree_dict.values()), columns = ['degree'])
degree_hist = ggplot(aes(x = 'degree'), data = degree_df) + geom_histogram(binwidth = 20., fill = 'royalblue', color = 'black') + scale_x_log10() + scale_y_log10() + ggtitle("Degree distribution") + xlab("node degree")

# component size histogram
comp_size_df = pd.read_csv(folder + '/data/comp_sizes.csv').sort_values(by = 'componentSize', ascending = False)
comp_size_plot = ggplot(aes(x = list(range(len(comp_size_df))), y = comp_size_df['componentSize']), data = comp_size_df) + geom_point(size = 20, color = 'maroon') + scale_y_log10() + ggtitle("Component sizes") + scale_x_continuous(breaks = list(np.arange(0, 1200, 100))) + xlim(low = -25, high = 1175) + xlab("rank") + ylim(low = 0.5, high = 10**7) + ylab("comp size")


ggsave(folder + '/plots/degree_hist', degree_hist)
ggsave(folder + '/plots/component_size_plot', comp_size_plot)
