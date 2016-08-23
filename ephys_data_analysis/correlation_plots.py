__author__ = "Olivia Haas"

# modules

import numpy
import matplotlib.pyplot as pl
import seaborn as sns
import hickle
import pandas as pd


# ____________________________________________________ getting arrays ________________________

data = hickle.load('/Users/haasolivia/Desktop/plots/correlation_info.hkl')


# ____________________________________________________ definitions ________________________


def plot_corr_matrix(data, text_info, fig_name, m_columns, m_rows):

    figM, axM = pl.subplots(1, 1, figsize=(10, 11))
    df = pd.DataFrame(data, columns=m_columns, index=m_rows)
    df_names = pd.DataFrame(text_info)

    sns.heatmap(df, annot=False, ax=axM, cmap="RdBu_r", vmin=-1, vmax=1)

    colors = numpy.empty(numpy.array(data).shape, dtype='string')

    for l in numpy.arange(len(colors)):
        for m in numpy.arange(len(colors[l])):
            if abs(data[l][m]) < 0.6:
                colors[l][m] = 'k'
            else:
                colors[l][m] = 'w'
    c = pd.DataFrame(colors)
    for x in range(df.shape[1]):
        for n, y in enumerate(numpy.fliplr([range(df.shape[0])])[0]):
            axM.text(x + 0.5, y + 0.5, df_names[x][n], ha='center', va='center', fontsize=16, color=c[x][n])
    axM.tick_params(labelsize=18)
    axM.xaxis.tick_top()
    print 'Saving figure under: /Users/haasolivia/Desktop/plots/'+fig_name+'.pdf'
    figM.savefig('/Users/haasolivia/Desktop/plots/'+fig_name+'.pdf', format='pdf')


# ____________________________________________________ main ________________________


plot_corr_matrix(data=data['bin_matrix'], text_info=data['extra_info'], fig_name=data['fig_name'],
                 m_columns=data['matrix_names'], m_rows=data['m_rows'])
plot_corr_matrix(data=data['bin_matrixS'], text_info=data['extra_infoS'], fig_name=data['fig_nameS'],
                 m_columns=data['matrix_names'], m_rows=data['m_rows'])
plot_corr_matrix(data=data['bin_matrixL'], text_info=data['extra_infoL'], fig_name=data['fig_nameL'],
                 m_columns=data['matrix_names'], m_rows=data['m_rows'])
