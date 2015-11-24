import pandas as pd
import subprocess
from collections import defaultdict
from ggplot import *

def unescape(val):
    return str(val).encode().decode('unicode-escape')

node_attr_filename = 'gplus0_lcc/data/node_attributes.csv'
attr_df = pd.read_csv(node_attr_filename, sep = ';')

attr_types = set(attr_df['attributeType'])
attr_vals = set([unescape(val) for val in attr_df['attributeVal']])
attr_freqs_by_type = dict((t, defaultdict(int)) for t in attr_types)
for (t, val) in zip(attr_df['attributeType'], attr_df['attributeVal']):
    attr_freqs_by_type[t][unescape(val)] += 1

num_unique_attrs_by_type = dict((t, len(attr_freqs_by_type[t])) for t in attr_types)
num_attr_instances_by_type = dict((t, sum(attr_freqs_by_type[t].values())) for t in attr_types)
sorted_attr_freqs_by_type = dict((t, sorted(attr_freqs_by_type[t].items(), key = lambda pair : pair[1], reverse = True)) for t in attr_types)

def attr_freq_df(rank_thresh = 1000):
    afdf = pd.DataFrame(columns = ['rank', 'freq', 'percentage', 'type'])
    for t in attr_types:
        df = pd.DataFrame(columns = afdf.columns)
        df['rank'] = list(range(rank_thresh))
        df['freq'] = [pair[1] for pair in sorted_attr_freqs_by_type[t][:rank_thresh]]
        df['percentage'] = 100 * np.cumsum(df['freq']) / num_attr_instances_by_type[t]
        df['type'] = t
        afdf = afdf.append(df)
    return afdf

afdf100 = attr_freq_df(100)
afdf1000 = attr_freq_df(1000)

rank_plot100 = ggplot(aes(x = 'rank', y = 'freq', color = 'type'), data = afdf100) + geom_line(size = 3) + ggtitle("Most frequent attributes by type") + xlab("rank") + xlim(low = -1, high = 101) + ylab("") + scale_y_log10() + scale_x_continuous(breaks = range(0, 105, 20))
rank_plot1000 = ggplot(aes(x = 'rank', y = 'freq', color = 'type'), data = afdf1000) + geom_line(size = 3) + ggtitle("Most frequent attributes by type") + xlab("rank") + xlim(low = -1, high = 1001) + ylab("") + scale_y_log10() + scale_x_continuous(breaks = range(0, 1050, 200))

cumulative_rank_plot100 = ggplot(aes(x = 'rank', y = 'percentage', color = 'type'), data = afdf100) + geom_line(size = 3) + ggtitle("Cumulative percentage of most frequent attributes") + xlim(low = -1, high = 101) + ylab("%") + scale_y_continuous(labels = range(0, 120, 20), limits = (0, 100)) + scale_x_continuous(breaks = range(0, 105, 20))
cumulative_rank_plot1000 = ggplot(aes(x = 'rank', y = 'percentage', color = 'type'), data = afdf1000) + geom_line(size = 3) + ggtitle("Cumulative percentage of most frequent attributes") + xlim(low = -1, high = 1001) + ylab("%") + scale_y_continuous(labels = range(0, 120, 20), limits = (0, 100)) + scale_x_continuous(breaks = range(0, 1050, 200))

# annoyingly, the legend formatting is screwed up, so may have to save these manually
ggsave("gplus0_lcc/plots/top100_attrs", rank_plot100)
ggsave("gplus0_lcc/plots/top1000_attrs", rank_plot1000)
ggsave("gplus0_lcc/plots/top100_attrs_cumulative", cumulative_rank_plot100)
ggsave("gplus0_lcc/plots/top1000_attrs_cumulative", cumulative_rank_plot1000)

def generate_report(rank_thresh = None):
    report = "Attributes in LCC of Google+ graph\n\n"
    report += "Total nodes in LCC:           4690159\n"
    report += "Nodes in LCC with attributes:  909314 (19.4%)\n\n"
    report += "        Type  #Values  #Occurrences\n-----------------------------------\n"
    total_attrs = 0
    total_instances = 0
    for t in attr_types:
        report += "%12s  %7d  %12d\n" % (t, num_unique_attrs_by_type[t], num_attr_instances_by_type[t])
        total_attrs += num_unique_attrs_by_type[t]
        total_instances += num_attr_instances_by_type[t]
    report += ('-' * 35) + '\n'
    report += "%12s  %7d  %12d\n" % ("total", total_attrs, total_instances)
    report += "\n\n"
    for t in attr_types:
        val_lengths = [len(pair[0]) for pair in sorted_attr_freqs_by_type[t]]
        rank_thresh = len(val_lengths) if (rank_thresh is None) else rank_thresh
        prec = min(int(np.percentile(val_lengths, 90)), max(val_lengths[:rank_thresh]))
        report += ('#' * (len(t) + 4)) + '\n'
        report += "# %s #\n" % t
        report += ('#' * (len(t) + 4)) + '\n\n'
        for i in range(rank_thresh):
            report += (("%%%ds" % prec) + "   %d\n") % (sorted_attr_freqs_by_type[t][i][0], sorted_attr_freqs_by_type[t][i][1])
        report += "\n\n"
    return report

open("gplus0_lcc/reports/attr_report_all.txt", 'w').write(generate_report(None))
open("gplus0_lcc/reports/attr_report_top1000.txt", 'w').write(generate_report(1000))
open("gplus0_lcc/reports/attr_report_top100.txt", 'w').write(generate_report(100))


def format_location_for_lookup(location):
    """Takes a location string and normalizes it via Shane Bergsma's Perl script."""
    p = subprocess.Popen('echo "%s" | ./string_proc/formatLocationsForLookup.pl' % location, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
    output, err = p.communicate()
    if err:
        raise OSError(err)
    return output.decode('utf-8').strip()



