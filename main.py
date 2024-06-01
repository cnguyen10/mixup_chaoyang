import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp
import tensorflow_probability.substrates.jax as tfp
import chex

import mlx.data as dx

import random
import os
import json
import collections
from tqdm import tqdm
import logging
import argparse

import h5py

import mlflow

from ResNet import ResNet18
from augmentation import augment_image


class TrainState(train_state.TrainState):
    """a Flax-based dataclass to store model and params"""
    batch_stats: dict


def parse_arguments() -> argparse.Namespace:
    """parse input arguments"""
    parser = argparse.ArgumentParser(description='Parse input arguments')

    parser.add_argument('--experiment-name', type=str, default='Chaoyang - mixup')
    parser.add_argument('--run-description', type=str, default=None)

    parser.add_argument(
        '--train-file',
        type=str,
        default='/sda2/datasets/chaoyang/train_multiple_labels.json',
        help='Path to files in the training set'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default='/sda2/datasets/chaoyang/test.json',
        help='Path to files in the testing set'
    )
    parser.add_argument('--majority-vote', dest='majority_vote', action='store_true')
    parser.add_argument('--no-majority-vote', dest='majority_vote', action='store_false')
    parser.set_defaults(majority_vote=True)

    parser.add_argument('--mixup', dest='mixup', action='store_true')
    parser.add_argument('--no-mixup', dest='mixup', action='store_false')
    parser.set_defaults(mixup=False)

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='Parameter of Beta distribution for mixup'
    )

    parser.add_argument('--num-classes', type=int, default=4, help='Number of classes')

    parser.add_argument('--pretrained-params-path', type=str, default=None)

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='The total number of epochs to run'
    )

    parser.add_argument('--run-id', type=str, default=None, help='Resume a run in MLFlow')

    parser.add_argument(
        '--jax-platform',
        type=str,
        default='cuda',
        help='cpu, cuda or tpu'
    )
    parser.add_argument(
        '--mem-frac',
        type=float,
        default=0.9,
        help='Percentage of GPU memory allocated for Jax'
    )

    parser.add_argument('--prefetch-size', type=int, default=8)
    parser.add_argument('--num-threads', type=int, default=4)

    parser.add_argument('--tqdm', dest='tqdm', action='store_true')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false')
    parser.set_defaults(tqdm=True)

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--logdir', type=str, default='./logdir')
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default='http://127.0.0.1:8080',
        help='MLFlow server'
    )

    return parser.parse_args()


def get_dataset(
    dataset_file: str,
    dset_root: str = None
) -> tuple[dx._c.Buffer, collections.Counter]:
    # load json data
    with open(file=dataset_file, mode='r') as f:
        # load a list of dictionaries
        json_data = json.load(fp=f)

    if dset_root is None:
        dset_root = ''

    cls_count = collections.Counter()

    # list of dictionaries, each dictionary is a sample
    data_dicts = [None] * len(json_data)
    for i, data_dict in enumerate(json_data):
        d = {}
        for key in data_dict:
            d[key] = data_dict[key]
        d['file'] = os.path.join(dset_root, data_dict['name']).encode('ascii')

        data_dicts[i] = d

        cls_count.update([d['label']])

    # load image dataset without batching nor shuffling
    dset = (
        dx.buffer_from_vector(data=data_dicts)
        .load_image(key='file', output_key='image')
        # .image_resize(key='image', w=64, h=64)
        .key_transform(key='image', func=lambda x: x.astype('float32') / 255)
        .key_transform(key='label', func=lambda y: y.astype('int32'))
    )

    for key in ('label_B', 'label_A', 'label_C'):
        if key not in data_dict:
            continue

        dset = dx.Buffer.key_transform(
            self=dset,
            key=key,
            func=lambda y: y.astype('int32')
        )

    return dset, cls_count


def preparare_dataset(dataset: dx._c.Buffer, shuffle: bool) -> dx._c.Stream:
    """
    """
    if shuffle:
        dset = dataset.shuffle()
    else:
        dset = dataset

    dset = (
        dset
        .to_stream()
        .batch(batch_size=args.batch_size)
        .prefetch(prefetch_size=args.prefetch_size, num_threads=args.num_threads)
    )

    return dset


def initialise_model(
    num_classes: int,
    lr: float,
    image_shape: tuple[int, int, int],
    key: jax.random.PRNGKey,
    decay_steps: int = 100_000,
    pretrained_params_path: str = None
) -> TrainState:
    key1, key2 = jax.random.split(key=key, num=2)

    lr_schedule_fn = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=decay_steps
    )

    model = ResNet18(num_classes=num_classes)
    params = model.init(
        rngs=key1,
        x=jax.random.uniform(key=key2, shape=(1,) + image_shape),
        train=False
    )
    tx = optax.sgd(learning_rate=lr_schedule_fn, momentum=0.9)

    # region LOADING PRE-TRAINED PARAMS
    if pretrained_params_path is not None:
        # loaded_params = h5py.File(name='/sda2/pretrained_models/resnet18_weights.h5')
        loaded_params = h5py.File(name=pretrained_params_path)

        params['params']['conv_init']['kernel'] = jnp.asarray(
            a=loaded_params['conv1']['weight'],
            dtype=jnp.float32
        )

        params['params']['bn_init']['scale'] = jnp.asarray(
            a=loaded_params['bn1']['scale'],
            dtype=jnp.float32
        )
        params['params']['bn_init']['bias'] = jnp.asarray(
            a=loaded_params['bn1']['bias'],
            dtype=jnp.float32
        )
        params['batch_stats']['bn_init']['mean'] = jnp.asarray(
            a=loaded_params['bn1']['mean'],
            dtype=jnp.float32
        )
        params['batch_stats']['bn_init']['var'] = jnp.asarray(
            a=loaded_params['bn1']['var'],
            dtype=jnp.float32
        )

        for i in range(8):
            params['params']['ResNetBlock_{:d}'.format(i)]['Conv_0']['kernel'] = jnp.asarray(
                a=loaded_params['layer{:d}'.format(i//2 + 1)]['block{:d}'.format(i % 2)]['conv1']['weight'],
                dtype=jnp.float32
            )

            params['params']['ResNetBlock_{:d}'.format(i)]['BatchNorm_0']['scale'] = jnp.asarray(
                a=loaded_params['layer{:d}'.format(i//2 + 1)]['block{:d}'.format(i % 2)]['bn1']['scale'],
                dtype=jnp.float32
            )
            params['params']['ResNetBlock_{:d}'.format(i)]['BatchNorm_0']['bias'] = jnp.asarray(
                a=loaded_params['layer{:d}'.format(i//2 + 1)]['block{:d}'.format(i % 2)]['bn1']['bias'],
                dtype=jnp.float32
            )
            params['batch_stats']['ResNetBlock_{:d}'.format(i)]['BatchNorm_0']['mean'] = jnp.asarray(
                a=loaded_params['layer{:d}'.format(i//2 + 1)]['block{:d}'.format(i % 2)]['bn1']['mean'],
                dtype=jnp.float32
            )
            params['batch_stats']['ResNetBlock_{:d}'.format(i)]['BatchNorm_0']['var'] = jnp.asarray(
                a=loaded_params['layer{:d}'.format(i//2 + 1)]['block{:d}'.format(i % 2)]['bn1']['var'],
                dtype=jnp.float32
            )

            params['params']['ResNetBlock_{:d}'.format(i)]['Conv_1']['kernel'] = jnp.asarray(
                a=loaded_params['layer{:d}'.format(i//2 + 1)]['block{:d}'.format(i % 2)]['conv2']['weight'],
                dtype=jnp.float32
            )

            params['params']['ResNetBlock_{:d}'.format(i)]['BatchNorm_1']['scale'] = jnp.asarray(
                a=loaded_params['layer{:d}'.format(i//2 + 1)]['block{:d}'.format(i % 2)]['bn2']['scale'],
                dtype=jnp.float32
            )
            params['params']['ResNetBlock_{:d}'.format(i)]['BatchNorm_1']['bias'] = jnp.asarray(
                a=loaded_params['layer{:d}'.format(i//2 + 1)]['block{:d}'.format(i % 2)]['bn2']['bias'],
                dtype=jnp.float32
            )
            params['batch_stats']['ResNetBlock_{:d}'.format(i)]['BatchNorm_1']['mean'] = jnp.asarray(
                a=loaded_params['layer{:d}'.format(i//2 + 1)]['block{:d}'.format(i % 2)]['bn2']['mean'],
                dtype=jnp.float32
            )
            params['batch_stats']['ResNetBlock_{:d}'.format(i)]['BatchNorm_1']['var'] = jnp.asarray(
                a=loaded_params['layer{:d}'.format(i//2 + 1)]['block{:d}'.format(i % 2)]['bn2']['var'],
                dtype=jnp.float32
            )

            if 'conv_proj' in params['params']['ResNetBlock_{:d}'.format(i)]:
                params['params']['ResNetBlock_{:d}'.format(i)]['conv_proj']['kernel'] = jnp.asarray(
                    a=loaded_params['layer{:d}'.format(i//2 + 1)]['block0']['downsample']['conv']['weight'],
                    dtype=jnp.float32
                )
                params['params']['ResNetBlock_{:d}'.format(i)]['norm_proj']['scale'] = jnp.asarray(
                    a=loaded_params['layer{:d}'.format(i//2 + 1)]['block0']['downsample']['bn']['scale'],
                    dtype=jnp.float32
                )
                params['params']['ResNetBlock_{:d}'.format(i)]['norm_proj']['bias'] = jnp.asarray(
                    a=loaded_params['layer{:d}'.format(i//2 + 1)]['block0']['downsample']['bn']['bias'],
                    dtype=jnp.float32
                )
                params['batch_stats']['ResNetBlock_{:d}'.format(i)]['norm_proj']['mean'] = jnp.asarray(
                    a=loaded_params['layer{:d}'.format(i//2 + 1)]['block0']['downsample']['bn']['mean'],
                    dtype=jnp.float32
                )
                params['batch_stats']['ResNetBlock_{:d}'.format(i)]['norm_proj']['var'] = jnp.asarray(
                    a=loaded_params['layer{:d}'.format(i//2 + 1)]['block0']['downsample']['bn']['var'],
                    dtype=jnp.float32
                )
    # endregion

    state = TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        batch_stats=params['batch_stats'],
        tx=tx
    )

    return state


@jax.jit
def train_step(
    x: chex.Array,
    y: chex.Array,
    state: TrainState,
    reweight: chex.Array
) -> tuple[TrainState, chex.Scalar]:
    """
    """
    def loss_fn(
        params: flax.core.frozen_dict.FrozenDict
    ) -> tuple[chex.Scalar, flax.core.frozen_dict.FrozenDict]:
        """calculate the loss"""
        logits, batch_stats_new = state.apply_fn(
            variables={'params': params, 'batch_stats': state.batch_stats},
            x=x,
            train=True,
            mutable=['batch_stats']
        )
        loss = optax.losses.softmax_cross_entropy(
            logits=logits,
            labels=y
        )
        loss = loss * reweight
        # loss = jnp.mean(a=loss, axis=0)
        loss = jnp.sum(a=loss, axis=0)

        return loss, batch_stats_new

    grad_value_fn = jax.value_and_grad(fun=loss_fn, argnums=0, has_aux=True)
    (loss, batch_stats), grads = grad_value_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=batch_stats['batch_stats'])

    return state, loss


@jax.jit
def train_mixup_step(
    x: chex.Array,
    y: chex.Array,
    y_permutated: chex.Array,
    state: TrainState,
    mixup_coefficient: chex.Scalar
) -> tuple[TrainState, chex.Scalar]:
    """
    """
    def loss_fn(
        params: flax.core.frozen_dict.FrozenDict
    ) -> tuple[chex.Scalar, flax.core.frozen_dict.FrozenDict]:
        """calculate the loss"""
        logits, batch_stats_new = state.apply_fn(
            variables={'params': params, 'batch_stats': state.batch_stats},
            x=x,
            train=True,
            mutable=['batch_stats']
        )
        loss_original = optax.losses.softmax_cross_entropy(
            logits=logits,
            labels=y
        )
        loss_permutated = optax.losses.softmax_cross_entropy(
            logits=logits,
            labels=y_permutated
        )
        loss = mixup_coefficient * loss_original + (1 - mixup_coefficient) * loss_permutated
        loss = jnp.mean(a=loss, axis=0)

        return loss, batch_stats_new

    grad_value_fn = jax.value_and_grad(fun=loss_fn, argnums=0, has_aux=True)
    (loss, batch_stats), grads = grad_value_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=batch_stats['batch_stats'])

    return state, loss


def train(dataset: dx._c.Buffer, state: TrainState) -> tuple[TrainState, chex.Scalar]:
    dset = preparare_dataset(dataset=dataset, shuffle=True)

    loss_accum = nnx.metrics.Average()

    for samples in tqdm(
        iterable=dset,
        desc='train',
        total=len(dataset)//args.batch_size + 1,
        position=2,
        leave=False,
        disable=not args.tqdm
    ):
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)
        if args.majority_vote:
            y = jax.nn.one_hot(x=samples['label'], num_classes=args.num_classes)

            reweight_vec = jnp.array(
                object=[args.reweight_dict[samples['label'][i]] for i in range(len(y))],
                dtype=jnp.float32
            )
        else:  # soft-label
            y = jnp.zeros(shape=(len(x), args.num_classes), dtype=jnp.float32)
            for key in ('label_B', 'label_A', 'label_C'):
                labels = jnp.asarray(a=samples[key])
                y += jax.nn.one_hot(x=labels, num_classes=args.num_classes)

            y /= 3

            reweight_vec = jnp.ones(shape=(len(y),), dtype=jnp.float32)

        reweight_vec /= jnp.sum(a=reweight_vec, axis=0)


        # perform data augmentation
        keys = jax.random.split(
            key=jax.random.key(seed=random.randint(a=0, b=1_000)),
            num=len(x)
        )
        x = augment_image(keys, x)

        if args.mixup:
            mixup_coefficient = tfp.distributions.Beta(
                concentration0=args.alpha,
                concentration1=args.alpha
            ).sample(sample_shape=(1,), seed=keys[0])

            x_permutated = jax.random.permutation(key=keys[0], x=x, axis=0)
            y_permutated = jax.random.permutation(key=keys[0], x=y, axis=0)

            x = mixup_coefficient * x + (1 - mixup_coefficient) * x_permutated
            y = mixup_coefficient * y + (1 - mixup_coefficient) * y_permutated

            state, loss = train_mixup_step(x, y, y_permutated, state, mixup_coefficient)
        else:
            state, loss = train_step(x, y, state, reweight_vec)

        loss_accum.update(values=loss)

    return state, loss_accum.compute()


def evaluate(dataset: dx._c.Buffer, state: TrainState) -> chex.Scalar:
    """evaluate on test set"""
    dset = preparare_dataset(dataset=dataset, shuffle=False)

    accuracy = nnx.metrics.Accuracy()
    for samples in tqdm(
        iterable=dset,
        desc='test',
        total=len(dataset)//args.batch_size + 1,
        position=2,
        leave=False,
        disable=not args.tqdm
    ):
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)
        y = jnp.asarray(a=samples['label'], dtype=jnp.int32)

        logits, _ = state.apply_fn(
            variables={'params': state.params, 'batch_stats': state.batch_stats},
            x=x,
            train=True,
            mutable=['batch_stats']
        )

        accuracy.update(logits=logits, labels=y)

    return accuracy.compute()


def main() -> None:
    """
    """
    # region DATA
    logging.info(msg='Loading datasets...')
    dset_train, cls_count = get_dataset(
        dataset_file=args.train_file,
        dset_root='/sda2/datasets/chaoyang'
    )
    dset_test, _ = get_dataset(
        dataset_file=args.test_file,
        dset_root='/sda2/datasets/chaoyang'
    )

    # update class-reweight to deal with im-balanced data
    args.reweight_dict = {key: cls_count[key] / len(dset_train) for key in cls_count}
    # endregion

    # region MODEL
    logging.info(msg='Loading model(s)...')
    state = initialise_model(
        num_classes=args.num_classes,
        lr=args.lr,
        image_shape=dset_train[0]['image'].shape,
        key=jax.random.key(seed=random.randint(a=0, b=1_000)),
        decay_steps=(args.num_epochs + 200) * (len(dset_train) // args.batch_size + 1),
        pretrained_params_path=args.pretrained_params_path
    )

    # checkpoint manager to store and save parameters
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        max_to_keep=1,
        step_format_fixed_length=3
    )
    # endregion

    # enable MLFlow tracking
    logging.info(msg='Enable Mlflow...')
    mlflow.set_tracking_uri(uri=args.tracking_uri)
    mlflow.set_experiment(experiment_name=args.experiment_name)
    mlflow.set_system_metrics_sampling_interval(interval=60)
    mlflow.set_system_metrics_samples_before_logging(samples=1)
    with mlflow.start_run(run_id=args.run_id, log_system_metrics=True) as mlflow_run:
        logging.info(msg='Start run id {:s}'.format(mlflow_run.info.run_id))

        if args.run_id is None:
            # log hyper-parameters if this is a new run
            mlflow.log_params(
                params={key: args.__dict__[key] \
                    for key in args.__dict__ \
                        if isinstance(args.__dict__[key], (int, bool, str, float))}
            )

        # create directory to store checkpoints
        if not os.path.exists(
            path=os.path.join(os.path.dirname(p=__file__), args.logdir)
        ):
            os.mkdir(path=os.path.join(os.path.dirname(p=__file__), args.logdir))

        with ocp.CheckpointManager(
            directory=ocp.test_utils.erase_and_create_empty(
                directory=os.path.join(
                    os.path.dirname(p=__file__),
                    args.logdir,
                    mlflow_run.info.run_id
                )
            ),
            options=ckpt_options
        ) as ckpt_mngr:
            start_epoch_id = 0

            # resume running
            if args.run_id is not None:
                logging.info(msg='Resume previous run...')
                # load checkpoint
                state_ = ckpt_mngr.restore(
                    step=ckpt_mngr.latest_step(),
                    args=ocp.args.StandardRestore(item=state)
                )

                state = state.replace(
                    step=state_['steps'],
                    params=state_['params'],
                    batch_stats=state_['batch_stats'],
                    opt_state=state_['opt_state']
                )

                start_epoch_id = ckpt_mngr.latest_step() - 1

                del state_

            logging.info(msg='Start training...')
            for epoch_id in tqdm(
                iterable=range(start_epoch_id, args.num_epochs, 1),
                desc='progress',
                leave=True,
                position=1,
                disable=not args.tqdm
            ):
                state, loss = train(dataset=dset_train, state=state)
                accuracy = evaluate(dataset=dset_test, state=state)

                mlflow.log_metrics(
                    metrics={
                        'loss': loss,
                        'accuracy': accuracy
                    },
                    step=epoch_id + 1
                )

                ckpt_mngr.save(step=epoch_id + 1, args=ocp.args.StandardSave(state))

    logging.info('Training is done.')
    return None


if __name__ == '__main__':
    args = parse_arguments()

    # set jax memory allocation
    jax.config.update(name='jax_platforms', val=args.jax_platform)
    assert args.mem_frac < 1. and args.mem_frac > 0.
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(args.mem_frac)

    # disable mlflow's tqdm
    os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'false'

    main()
