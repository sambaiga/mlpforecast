import matplotlib
import matplotlib_inline.backend_inline
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf", "retina")  # For export
from cycler import cycler
az.style.use(["science"])
from utils.data_processing import add_time_features
import altair as alt
alt.themes.enable("opaque")
alt.data_transformers.disable_max_rows()
import plotly.graph_objects as go
import plotly.express as px
from palettable.cmocean.diverging import Curl_20




def get_grouped_statics(data, value='NetLoad(kW)', column='YEAR'):
    grouped_stats = data.groupby(column)[value].agg([
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('Std', 'std'),
        ('MAD', lambda x: np.median(np.abs(x - np.median(x)))),
        ('Kurt', kurtosis),
        ('Skew', skew)
    ])

    table = grouped_stats.to_latex(index=True, formatters={"name": str.upper},
                      float_format="{:.2f}".format)
    return table

def create_heatmap(df, value, index='Time', column='date'):
    data_df = df.copy()
    data_df["date"] = df.index.date
    data_df["Time"] = df.index.time
    data_df["Month"] = df.index.month
    data_df["Day"] = df.index.dayofweek
    get_first = lambda x: x.iloc[0]
    # Pivot dates and times to create a two dimensional representation
    data = data_df.pivot_table(index=index, columns=column, values=value, aggfunc=get_first, dropna=True)
    return data


def get_3Dplot(fig, hm, ax_hm, cmap, label='Power (kW)'):
    # create a 3d figure
    ax = plt.subplot(111, projection='3d')

    # Create meshgrid
    X, Y = np.meshgrid(np.arange(hm.shape[1]), np.arange(hm.shape[0]))

    # Plot the surface
    plot = ax.plot_surface(X=X, Y=Y, Z=np.nan_to_num(hm.values), cmap=cmap)

    # set tick every three hours for the time axis
    ax.set_yticks(ticks=range(0, 97, 3 * 4))
    ax.set_yticklabels(labels=range(0, 25, 3))
    ax.set_ylabel('Time')

    # set ticks for the date axis based on the heatmap above
    ax.set_xticks(ax_hm.get_xticks()[::10])
    ax.set_xticklabels(ax_hm.get_xticklabels()[::10], rotation=5)
    ax.set_xlabel('Date')

    # Remove gray panes and axis grid
    ax.xaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.fill = False
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)
    # Remove z-axis
    ax.zaxis.line.set_lw(0.)
    ax.set_zlabel(label)
   
    #fig.colorbar(plot, ax=ax, label='Energy (kWh)')
    #fig.colorbar(plot, ax=ax, label=label, fraction=0.046)
    
    return fig
    

def visualise_timeseries_plotly(data, x_col, y_col):
    n_rows=1
    fig = px.scatter(data, x=x_col, y=y_col)
    fig.update_xaxes(matches=None, showticklabels=True, visible=True)
    fig.update_layout(margin=dict(l=60, r=10, t=20, b=50))
    fig.update_layout(template="plotly_white", font=dict(size=10))
    fig.update_annotations(font_size=10)
    fig.update_layout(autosize=True, height=150 * n_rows)
    fig.show()  
    
def visualise_timeseries_altair(data,  y_col, figure_path=None, y_label='Power (kW)'):
    colors=['#4e79a7', '#f28e2b', '#e15759', '#f69a48', '#00c0bf','#fdcd49','#8da798','#a19368','#525252','#a6761d','#7035b7','#cf166e']
    chart = alt.Chart(data.reset_index()).mark_point().encode(
            x=alt.X('timestamp:T', axis=alt.Axis(title='Date')),
            y = alt.X(f'{y_col[0]}:Q', title=y_label),
            color=alt.value(colors[0])
        )
    if len(y_col)>1:
        for i in range(1, len(y_col)):
            chart+=alt.Chart(data.reset_index()).mark_point().encode(
            x=alt.X('timestamp:T', axis=alt.Axis(title='Date')),
            y = alt.X(f'{y_col[i]}:Q', title=y_label),
            color=alt.value(colors[i])
        )
    chart=chart.configure_axis(
        grid=False,
        labelFontSize=12,
        titleFontSize=12
    ).configure_view(
        strokeOpacity=0
    ).properties(width=900,
                    height=100
    )
    
    return chart
    
    
    
    
## Correlation analysis
def get_correlation(ax, data, variable, 
                    column, 
                    threshold=0.2, 
                    cmap = sns.diverging_palette(240, 10, as_cmap=True)):
    non_zero_varlist=[variable]+column
    correlations =  data[non_zero_varlist].corr().unstack().sort_values(ascending=False) # Build correlation matrix
    correlations = pd.DataFrame(correlations).reset_index() # Convert to dataframe
    correlations.columns = ['col1', 'col2', 'correlation'] # Label it
    load_corr = correlations.query(f"col1 == '{str(variable)}' & col2 != '{str(variable)}'")
    load_corr[np.abs(load_corr['correlation'])>=threshold]
    load_corr=load_corr[np.abs(load_corr['correlation'])>=threshold]
    
    corr=load_corr.pivot('col2', 'col1', 'correlation')
    ax=sns.heatmap(corr,  linewidths=.5, cmap=cmap, center=0, annot=True, fmt=".1g")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment="right")
    ax.set_title("")
    ax.set_ylabel("")
    ax.set_xlabel("")
    return load_corr, ax

def plot_correlation(ax, data, variable, column):
    corr = data[variable+column].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Colors
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    ax=sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0, annot=True, fmt=".1g")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment="right")
    ax.set_title("Correlation Heatmap")
    return ax


def plot_distribution(ax, df, index_col='HOUR', val_col='WindSpeed', hue_col=None):
    spivot = pd.pivot_table(df, index=index_col, values=val_col, columns=hue_col,  aggfunc=np.mean)
    sdv = pd.pivot_table(df, index=index_col, values=val_col, columns=hue_col, aggfunc=np.std)
    spivot.plot(ax=ax)
    ax.fill_between(np.arange(24), (spivot.min(1)-sdv.min(1)), 
                        (spivot.max(1)+sdv.max(1)),  color="lightsteelblue", alpha=0.5)
    return ax
    
    
def scatter_plot_altair( df, index_col='HOUR', 
                    x_col='WindSpeed', 
                    y_col='WindGen(MWh)',
                    y_label='Energy(MWh)', 
                    width=250, 
                    height=350, 
                    limit=[0, 100],
                    figure_path=None):
    chart=alt.Chart(df).mark_circle(size=60).encode(
        y = alt.Y(f'{y_col}', scale=alt.Scale(domain=limit), title=y_label),
        x = alt.X(f'{x_col}', axis=alt.Axis( title=x_col)),
        color=alt.Color(f'{index_col}:N'),
    ).configure_axis(
        grid=False,
        labelFontSize=12,
        titleFontSize=12
    ).configure_view(
        strokeOpacity=0
    ).properties(width=width,
                    height=height
    )
    return chart

def plot_kde_(ax, data, x_col, hue_col, label):
    #sns.kdeplot(data, x=x_col, ax=ax, hue=hue_col,  palette='tab20')
    sns.histplot(data, x=x_col, ax=ax, hue=hue_col,  palette='tab20', kde=True)
    ax.autoscale()
    ax.set_xlabel(label)
    return ax

def plot_cdf_(ax, data, x_col, hue_col, label):
    sns.kdeplot(data, x=x_col, ax=ax, hue=hue_col, cumulative=True, common_norm=False, common_grid=False,  palette='tab20')
    ax.autoscale()
    ax.set_xlabel(label)
    ax.set_ylim(0, 1)
    return ax
       