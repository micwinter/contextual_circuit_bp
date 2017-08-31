import os
import numpy as np
from db import db
from db import credentials
from config import Config
from argparse import ArgumentParser
import pandas as pd
from utils.py_utils import get_dt_stamp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import plotly.plotly as py
import plotly.tools as tls


def plot_with_plotly(plotly_fig, chart):
    try:
        plot_url = py.plot(plotly_fig, auto_open=False)
        print 'Uploaded %s chart to: %s' % (chart, plot_url)
    except:
        pass


def main(
        experiment_name,
        im_ext='.pdf',
        log_transform_loss=True,
        colors='Paired',
        exclude=None):
    """Plot results of provided experiment name."""
    config = Config()
    pl_creds = credentials.plotly_credentials()
    py.sign_in(
        pl_creds['username'],
        pl_creds['api_key'])

    # Get experiment data
    perf = db.get_performance(experiment_name=experiment_name)
    if len(perf) == 0:
        raise RuntimeError('Could not find any results.')
    structure_names = [x['model_struct'].split('/')[-1] for x in perf]
    optimizers = [x['optimizer'] for x in perf]
    lrs = [x['lr'] for x in perf]
    datasets = [x['dataset'] for x in perf]
    loss_funs = [x['loss_function'] for x in perf]
    optimizers = [x['optimizer'] for x in perf]
    wd_types = [x['regularization_type'] for x in perf]
    wd_penalties = [x['regularization_strength'] for x in perf]
    steps = [float(x['training_step']) for x in perf]
    training_loss = [float(x['training_loss']) for x in perf]
    validation_loss = [float(x['validation_loss']) for x in perf]

    # Pass data into a pandas DF
    model_params = ['%s | %s | %s | %s | %s | %s | %s | %s' % (
        ipa,
        ipb,
        ipc,
        ipd,
        ipe,
        ipf,
        ipg,
        iph) for ipa, ipb, ipc, ipd, ipe, ipf, ipg, iph in zip(
            structure_names,
            optimizers,
            lrs,
            loss_funs,
            optimizers,
            wd_types,
            wd_penalties,
            datasets)]

    # DF and plot
    df = pd.DataFrame(
        np.vstack(
            (
                model_params,
                steps,
                training_loss,
                validation_loss
            )
        ).transpose(),
        columns=[
            'model parameters',
            'training iteration',
            'training loss',
            'validation loss'
            ]
        )
    df['training iteration'] = pd.to_numeric(
        df['training iteration'],
        errors='coerce')
    df['training loss'] = pd.to_numeric(df['training loss'], errors='coerce')
    if log_transform_loss:
        loss_label = 'Log loss'
        df['training loss'] = np.log(df['training loss'])
    else:
        loss_label = 'Normalized loss (x / max(x))'
        df['training loss'] /= df.groupby(
            'model parameters')['training loss'].transform(max)
    df['validation loss'] = pd.to_numeric(df['validation loss']) * 100.
    if exclude is not None:
        exclusion_search = df['model parameters'].str.contains(exclude)
        df = df[exclusion_search == False]
        print 'Removed %s rows.' % exclusion_search.sum()

    # Start plotting
    matplotlib.style.use('ggplot')
    plt.rc('font', size=6)
    plt.rc('legend', fontsize=8, labelspacing=3)
    f, axs = plt.subplots(2, figsize=(20, 30))
    ax = axs[1]
    for k in df['model parameters'].unique():
        tmp = df[df['model parameters'] == k]
        ax = tmp.plot(
            x='training iteration',
            y='training loss',
            label=k,
            kind='line',
            ax=ax,
            logy=False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Training')
    ax.set_ylabel(loss_label)
    # ax.legend_.remove()
    ax = axs[0]
    for k in df['model parameters'].unique():
        tmp = df[df['model parameters'] == k]
        ax = tmp.plot(
            x='training iteration',
            y='validation loss',
            label=k,
            kind='line',
            ax=ax,
            logy=False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Validation')
    ax.set_ylabel('Categorization accuracy (%)')
    # ax.legend_.remove()
    out_name = os.path.join(
        config.plots,
        '%s_%s%s' % (
            experiment_name, get_dt_stamp(), im_ext))
    plt.savefig(out_name)
    print 'Saved to: %s' % out_name
    plotly_fig = tls.mpl_to_plotly(f)
    plotly_fig['layout']['autosize'] = True
    # plotly_fig['layout']['showlegend'] = True
    plot_with_plotly(plotly_fig, 'bar')
    plt.close(f)

    # Plot max performance bar graph
    f = plt.figure()
    max_perf = df.groupby(
        ['model parameters'], as_index=False)['validation loss'].max()
    # max_perf['model parameters'] = max_perf['model parameters'].str.replace(
    #     '|', '\n')
    plt.rc('xtick', labelsize=2)
    ax = max_perf.plot.bar(x='model parameters', y='validation loss', legend=False)
    plt.tight_layout()
    ax.set_title('Max validation value')
    ax.set_ylabel('Categorization accuracy (%)')
    out_name = os.path.join(
        config.plots,
        '%s_%s_bar%s' % (
            experiment_name, get_dt_stamp(), im_ext))
    plt.savefig(out_name)
    print 'Saved to: %s' % out_name
    plotly_fig = tls.mpl_to_plotly(f)
    plot_with_plotly(plotly_fig, chart='bar')
    plt.close(f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--experiment',
        dest='experiment_name',
        type=str,
        default=None,
        help='Name of the experiment.')
    parser.add_argument(
        '--exclude',
        dest='exclude',
        type=str,
        default=None,
        help='Experiment exclusion keyword.')
    args = parser.parse_args()
    main(**vars(args))
