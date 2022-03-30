import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmcls import __version__
from mmcls.apis import set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--wandb-prefix', type=str, default='', help='prefix for wandb')
    parser.add_argument('--wandb-postfix', type=str, default='', help='postfix for wandb')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--corruption', type=str, default='')
    parser.add_argument('--level', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(corruption='gaussion_noise', severity=5, name=""):
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_classifier(cfg.model)
    model.init_weights()

    if args.corruption:
        corruption = args.corruption
    if args.level:
        severity = args.level
    if isinstance(corruption, list):
        args.wandb_prefix = 'multi'
    else:
        args.wandb_prefix = corruption+str(severity)
    # args.wandb_prefix = "_".join(corruption) if isinstance(corruption, list) else corruption
    data = [cfg, cfg.data.train, cfg.data.val, cfg.data.test]
    for d in data:
        d.corruption = corruption
        d.severity = severity

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmcls version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)

    if args.local_rank == 0:
        for hook in cfg.log_config.hooks:
            if hook['type'] == 'WandbLoggerHook':
                hook['init_kwargs']['name'] = args.wandb_prefix + hook['init_kwargs']['name'] + args.wandb_postfix + name
                hook['init_kwargs']['config'] = copy.deepcopy(cfg)
            elif hook['type'] == 'NeptuneLoggerHook':
                hook['init_kwargs']['name'] = args.wandb_prefix + hook['init_kwargs']['name'] + args.wandb_postfix + name
                hook['init_kwargs']['config'] = copy.deepcopy(cfg)

    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        device='cpu' if args.device == 'cpu' else 'cuda',
        meta=meta)


if __name__ == '__main__':
    corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 
                    'defocus_blur', 'glass_blur', 'motion_blur', 
                    'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 
                    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    severity = [1, 2, 3, 4, 5]

    multi_corruptions = [
        ['gaussian_noise', 'shot_noise'], 
        ['gaussian_noise', 'pixelate'], 
        ['gaussian_noise', 'shot_noise', 'pixelate'],
        ['shot_noise', 'glass_blur','jpeg'],
        ['impulse_noise', 'glass_blur', 'elastic_transform']
    ]
    multi_severity = [[5,5], [5,5], [5,5,5], [5,5,5]]

    # main(corruption = corruptions, severity = 5)

    # corruption = [
    #         "diffusion_shot_noise_5", "diffusion_glass_blur_5", 
    #         "diffusion_impulse_noise_5", "diffusion_elastic_transform_5"
    #     ]
    # main(corruption = corruption, severity = 1)
    # corruptions = ['shot_noise', 'glass_blur', 'impulse_noise', 'elastic_transform']
    
    # l = [str(i) for i in range(50, 501, 50)][7]
    # corruption = ["cifar10c_q_sample_output/" + "q_" + l + "/" + c + "_5_samples_10000x32x32x3_q_" + l for c in corruptions]
    # main(corruption = corruption, severity = 1, name='_t' + l + '')
    # cs = [
    #     ['gaussian_noise', 'shot_noise', 'impulse_noise', 'glass_blur','jpeg_compression'],
    #     ['elastic_transform', 'pixelate', 'defocus_blur',  'motion_blur', 'zoom_blur', 'snow', 'frost', 'brightness',],
    #     ['contrast', 'fog']
    # ]
    # i = 2
    # corruption = ["N2_T300/" + c + "_level_5_samples_10000x32x32x3"  for c in cs[0]+cs[1]+cs[2]]
    # corruption_p = cs[2]
    # main(corruption = corruption, severity = 1, name='_'+chr(ord('A')+i))
    main(corruption = corruptions, severity = 5)
    # main(corruption = corruption, severity = 1, name='_ABC')


    # cs = ['gaussian_noise', 'shot_noise', 'impulse_noise', 
    #     'defocus_blur', 'glass_blur', 'motion_blur', 
    #     'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 
    #     'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    # corruption = ['cifar10c_N1_t400/' + c + '_level_5_samples_10000x32x32x3' for c in cs]
    # main(corruption = corruption, severity = 1)

    # ex = ['N2_t300/', 'N2_t400/', 'N4_t300/', 'N4_t400/']
    # cs = ['shot_noise', 'impulse_noise', 'glass_blur', 'elastic_transform']
    # for e in ex:
    #     co = [e + c + '_level_5_samples_10000x32x32x3' for c in cs]
    #     main(corruption = co, severity = 1)

    # cs = ['shot_noise', 'impulse_noise', 'glass_blur', 'elastic_transform']
    # ss = ['49', '99']
    # for s in ss:
    #     co = ['N1_t400/' + c + '_level_5_step_' + s + '_samples_10000x32x32x3' for c in cs]
    #     main(corruption = co, severity = 1)