import os
import numpy as np
from db import db
from config import Config
from argparse import ArgumentParser
import pandas as pd
import seaborn as sns
from main import get_dt_stamp
from matplotlib import pyplot as plt


def main(experiment_name, im_ext='.pdf', val_score='val accuracy'):
    """Plot results of provided experiment name."""
    config = Config()

    # Get experiment data
    perf = db.get_performance(experiment_name=experiment_name)
    if len(perf) == 0:
        raise RuntimeError('Could not find any results.')
    structure_names = [x['model_struct'].split('/')[-1] for x in perf]
    optimizers = [x['optimizer'] for x in perf]
    lrs = [x['lr'] for x in perf]
    loss_funs = [x['loss_function'] for x in perf]
    optimizers = [x['optimizer'] for x in perf]
    wd_types = [x['wd_type'] for x in perf]
    wd_penalties = [x['wd_penalty'] for x in perf]
    steps = [float(x['training_step']) for x in perf]
    training_loss = [float(x['training_loss']) for x in perf]
    validation_loss = [float(x['validation_loss']) for x in perf]

    # Pass data into a pandas DF
    model_params = ['%s | %s | %s | %s | %s | %s | %s' % (
        ipa,
        ipb,
        ipc,
        ipd,
        ipe,
        ipf,
        ipg) for ipa, ipb, ipc, ipd, ipe, ipf, ipg in zip(
            structure_names,
            optimizers,
            lrs,
            loss_funs,
            optimizers,
            wd_types,
            wd_penalties)]

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
            val_score
            ]
        )
    df['training iteration'] = pd.to_numeric(df['training iteration'])
    df['training loss'] = pd.to_numeric(df['training loss'])
    df['validation loss'] = pd.to_numeric(df[val_score]) * 100.
    f, axs = plt.subplots(2, figsize=(20, 30))
    ax = sns.pointplot(
        x='training iteration',
        y='training loss',
        hue='model parameters',
        ci=None,
        estimator=np.sum,
        data=df,
        ax=axs[0],
        scale=.25)
    ax.set_title('Training')
    ax = sns.pointplot(
        x='training iteration',
        y=val_score,
        hue='model parameters',
        ci=None,
        estimator=np.sum,
        data=df,
        ax=axs[1],
        scale=.25)
    ax.set_title('Validation')
    out_name = os.path.join(
        config.plots,
        '%s_%s%s' % (
            experiment_name, get_dt_stamp(), im_ext))
    plt.savefig(out_name)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--experiment',
        dest='experiment_name',
        type=str,
        default=None,
        help='Name of the experiment.')
    args = parser.parse_args()
    main(**vars(args))
