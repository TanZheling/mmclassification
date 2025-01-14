# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time
from collections import Counter
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import wandb
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

def single_gpu_test_vote(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    **show_kwargs):
    model.eval()
    # torch.save(model.state_dict(), '/home/sjtu/scratch/zltan/pretrained_models/load_timm_models.pth')
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    patient_vote = dict()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        for j in range(len(result)):
            # print(type(result), result[j].shape, result[j])
            if data['img_metas'].data[0][j]['patient'] not in patient_vote:
                patient_vote[data['img_metas'].data[0][j]['patient']] = [result[j][0]]
            else:
                patient_vote[data['img_metas'].data[0][j]['patient']] += [result[j][0]]
        batch_size = len(result)
        results.extend(result)

        if show or out_dir:
            scores = np.vstack(result)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [model.CLASSES[lb] for lb in pred_label]

            img_metas = data['img_metas'].data[0]
            imgs = tensor2imgs(data['img'], **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                result_show = {
                    'pred_score': pred_score[i],
                    'pred_label': pred_label[i],
                    'pred_class': pred_class[i]
                }
                model.module.show_result(
                    img_show,
                    result_show,
                    show=show,
                    out_file=out_file,
                    **show_kwargs)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    patient_label_dict = dict()
    for p in patient_vote.keys():
        mean_ = sum(patient_vote[p]) / len(patient_vote[p])
        patient_label_dict[p] = 1 if mean_ > 0.5 else 0 #分类为第一类则为1，否则为0
    # for i in range(len(results)):
    #     # print(data_loader.dataset[i])
    #     p = data_loader.dataset[i]['img_metas'].data['patient']
    #     results[i] = [patient_label_dict[p], 1-patient_label_dict[p]]
    return results, patient_label_dict


def multi_gpu_test_vote(model, data_loader, tmpdir=None, 
    gpu_collect=False, patient_gt_csv=None,wandb_name=None,
    wandb_entity=None,wandb_project=None,tmppat=None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    # wandb.init(
    #             name=wandb_name, 
    #             entity=wandb_entity, project=wandb_project)
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
            raise OSError((f'The tmpdir {tmpdir} already exists.',
                           ' Since tmpdir will be deleted after testing,',
                           ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    patient_vote = dict()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        for j in range(len(result)):
            # print(type(result), result[j].shape, result[j])
            if data['img_metas'].data[0][j]['patient'] not in patient_vote:
                patient_vote[data['img_metas'].data[0][j]['patient']] = [result[j][0]]
            else:
                patient_vote[data['img_metas'].data[0][j]['patient']] += [result[j][0]]

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    patient_label_dict = dict()
    for p in patient_vote.keys():
        mean_ = sum(patient_vote[p]) / len(patient_vote[p])
        patient_label_dict[p] = 1 if mean_ > 0.5 else 0 #分类为第一类则为1，否则为0
    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
        patient_label_dict = collect_results_gpu(patient_label_dict, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
        patient_label_dict = collect_results_cpu(patient_label_dict, len(dataset),tmppat)
    # patient_gt = pd.read_csv(patient_gt_csv)
    # right_count = 0
    # wrong_count = 0
    # all_patient_count = len(patient_gt)
    # patient_pred=[]
    # patient_gtlabel=[]
    # for k, v in patient_label_dict.items():
    #     patient_pred.append(v)
    #     lab = patient_gt.loc[patient_gt['case_id']==k,['label']].values.tolist()[0][0]
    #     # print(lab)
    #     real_v=0
    #     if lab == 'msi':
    #         real_v=1
    #         patient_gtlabel.append(1)
    #     else:
    #         patient_gtlabel.append(0)
    #     if v == real_v:
    #         right_count+=1
    #     else:
    #         wrong_count+=1
    # auc = roc_auc_score(patient_pred,patient_gtlabel)
    # wandb.log({"patient auc":auc})
    # print("patient auc",auc)
    # # for i in range(len(patient_gt)):
    # #     if patient_label_dict[patient_gt.iloc[i,0]]>0.5:
    # #         if patient_gt.iloc[i,2]=='msi':
    # #             right_count+=1
    # #         else:
    # #             wrong_count+=1
    # #     else:
    # #         if patient_gt.iloc[i,2]=='msi':
    # #             wrong_count+=1
    # #         else:
    # #             right_count+=1
    # #assert right_count+wrong_count==all_patient_count
    # print("right count:",right_count)
    # print("patient_level accuracy:",right_count/all_patient_count)
    # wandb.log({"patient_level accuracy":right_count/all_patient_count})
    return results,patient_label_dict


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = mmcv.load(part_file)
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
