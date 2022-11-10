# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=True, **eval_kwargs):
        warnings.warn(
            'DeprecationWarning: EvalHook and DistEvalHook in mmcls will be '
            'deprecated, please install mmcv through master branch.')
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.by_epoch = by_epoch

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str, optional): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=True,
                #  vote = False,
                #  patient_gt_csv = None,
                #  wandb_name=None,
                #  wandb_entity=None,
                #  wandb_project=None,
                 **eval_kwargs):
        warnings.warn(
            'DeprecationWarning: EvalHook and DistEvalHook in mmcls will be '
            'deprecated, please install mmcv through master branch.')
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs
        self.vote = vote
        self.patient_gt_csv = patient_gt_csv
        self.wandb_name = wandb_name,
        self.wandb_entity = wandb_entity,
        self.wandb_project = wandb_project

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        # from mmcls.apis import multi_gpu_test, multi_gpu_test_vote
        # patient_label_dict = dict()
        # if self.vote:

        #     results,patient_label_dict = multi_gpu_test_vote(
        #         runner.model,
        #         self.dataloader,
        #         tmpdir=osp.join(runner.work_dir, '.eval_hook'),
        #         gpu_collect=self.gpu_collect,
        #         patient_gt_csv = self.patient_gt_csv,
        #         wandb_name=self.wandb_name,
        #         wandb_entity=self.wandb_entity,
        #         wandb_project=self.wandb_project)
        # else:
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            # print(patient_label_dict)
            # import pandas as pd
            # from sklearn.metrics import roc_auc_score, roc_curve, auc
            # patient_gt = pd.read_csv(self.patient_gt_csv)
            # right_count = 0
            # wrong_count = 0
            # all_patient_count = len(patient_gt)
            # patient_pred=[]
            # patient_gtlabel=[]
            # for k, v in patient_label_dict.items():
            #     patient_pred.append(v)
            #     lab = patient_gt.loc[patient_gt['case_id']==k,['label']].values.tolist()[0][0]
            #     print(lab)
            #     real_v=0
            #     if lab == 'None':
            #         real_v=1
            #         patient_gtlabel.append(1)
            #     else:
            #         patient_gtlabel.append(0)
            #     if v == real_v:
            #         right_count+=1
            #     else:
            #         wrong_count+=1
            # auc = roc_auc_score(patient_pred,patient_gtlabel)
            # print("auc",auc)
            # print("right count:",right_count)
            # print("patient_level accuracy:",right_count/all_patient_count)
            self.evaluate(runner, results)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
