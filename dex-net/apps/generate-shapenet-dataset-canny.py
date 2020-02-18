#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 20/05/2018 2:45 PM 
# File Name  : generate-dataset-canny.py
import numpy as np
import sys
import pickle
import re
from dexnet.grasping.quality import PointGraspMetrics3D
from dexnet.grasping import GaussianGraspSampler, AntipodalGraspSampler, UniformGraspSampler, GpgGraspSampler
from dexnet.grasping import RobotGripper, GraspableObject3D, GraspQualityConfigFactory, PointGraspSampler
import dexnet
from autolab_core import YamlConfig
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
import os
import multiprocessing
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # for the convenient of run on remote computer
import logging
import argparse

log_filename = os.path.join('generate-shapenet.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
hdlr = logging.FileHandler(log_filename)
hdlr.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(hdlr)




def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        if root.count('/') == file_dir_.count('/') + 1:
            file_list.append(root)
    file_list.sort()
    return file_list


def do_job(i):
    object_name = file_list_all[i]['object_name']
    logger.info("start job ", object_name)
    good_grasp = multiprocessing.Manager().list()
    p_set = [multiprocessing.Process(target=worker, args=(i, 100, 20, good_grasp)) for _ in
             range(50)]  # grasp_amount per friction: 20*40
    [p.start() for p in p_set]
    [p.join() for p in p_set]
    good_grasp = list(good_grasp)

    good_grasp_file_name = "./generated_grasps/{}_{}_{}".format(filename_prefix, str(object_name), str(len(good_grasp)))
    with open(good_grasp_file_name + '.pickle', 'wb') as f:
        pickle.dump(good_grasp, f, protocol=2)

    tmp = []
    for grasp in good_grasp:
        grasp_config = grasp[0].configuration
        score_friction = grasp[1]
        score_canny = grasp[2]
        tmp.append(np.concatenate([grasp_config, [score_friction, score_canny]]))
    np.save(good_grasp_file_name + '.npy', np.array(tmp))
    logger.info("finished job ", object_name)


def worker(i, sample_nums, grasp_amount, good_grasp):
    object_name = file_list_all[i]['object_name']
    logger.info('a worker of task {} start'.format(object_name))

    yaml_config = YamlConfig(home_dir + "/code/grasp-pointnet/dex-net/test/config.yaml")
    gripper_name = 'robotiq_85'
    gripper = RobotGripper.load(gripper_name, home_dir + "/code/grasp-pointnet/dex-net/data/grippers")
    grasp_sample_method = "antipodal"
    if grasp_sample_method == "uniform":
        ags = UniformGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "gaussian":
        ags = GaussianGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "antipodal":
        ags = AntipodalGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "gpg":
        ags = GpgGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "point":
        ags = PointGraspSampler(gripper, yaml_config)
    else:
        raise NameError("Can't support this sampler")
    logger.info("Log: do job", i)

    if os.path.exists(str(file_list_all[i]['obj_file'])):
        of = ObjFile(str(file_list_all[i]['obj_file']))
        sf = SdfFile(str(file_list_all[i]['sdf_file']))
    else:
        logger.info("can't find any obj or sdf file!")
        raise NameError("can't find any obj or sdf file!")
    mesh = of.read()
    sdf = sf.read()
    obj = GraspableObject3D(sdf, mesh)
    logger.info("Log: opened object", i + 1, object_name)

    force_closure_quality_config = {}
    canny_quality_config = {}
    fc_list_sub1 = np.arange(2.0, 0.75, -0.4)
    fc_list_sub2 = np.arange(0.5, 0.36, -0.05)
    fc_list = np.concatenate([fc_list_sub1, fc_list_sub2])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        yaml_config['metrics']['force_closure']['friction_coef'] = value_fc
        yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc

        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            yaml_config['metrics']['force_closure'])
        canny_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            yaml_config['metrics']['robust_ferrari_canny'])

    good_count_perfect = np.zeros(len(fc_list))
    count = 0
    minimum_grasp_per_fc = grasp_amount
    while np.sum(good_count_perfect < minimum_grasp_per_fc) != 0:
        grasps = ags.generate_grasps(obj, target_num_grasps=sample_nums, grasp_gen_mult=10,
                                     vis=False, random_approach_angle=True)
        count += len(grasps)
        for j in grasps:
            for ind_, value_fc in [f for f in enumerate(fc_list)][::-1]:
                is_force_closure = PointGraspMetrics3D.grasp_quality(j, obj, force_closure_quality_config[value_fc], vis=False)
                if is_force_closure and good_count_perfect[ind_] < minimum_grasp_per_fc:
                    canny_quality = PointGraspMetrics3D.grasp_quality(j, obj, canny_quality_config[value_fc], vis=False)
                    good_grasp.append((j, value_fc, canny_quality))
                    good_count_perfect[ind_] += 1
                    break
        logger.info('Object:{} GoodGrasp:{}'.format(object_name, good_count_perfect))

    object_name_len = len(object_name)
    object_name_ = str(object_name) + " " * (25 - object_name_len)
    if count == 0:
        good_grasp_rate = 0
    else:
        good_grasp_rate = len(good_grasp) / count
    logger.info('Gripper:{} Object:{} Rate:{:.4f} {}/{}'.
          format(gripper_name, object_name_, good_grasp_rate, len(good_grasp), count))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename_prefix = sys.argv[1]
    else:
        filename_prefix = "default"
    home_dir = os.environ['HOME']

    parser = argparse.ArgumentParser(description='Generate Grasps for Dex-Net')
    parser.add_argument('-ds', '--dataset', type=str, required=True)
    args = parser.parse_args()

    file_list_available = []
    ds = np.loadtxt(args.dataset, delimiter=' ', dtype=np.str)
    file_list_available = ds

    # check if file already processed

    already_generated = [re.sub('_\d+\.pickle', '', f) for f in os.listdir('./generated_grasps/') if '.pickle' in f]
    file_list_all = []
    for object_name, obj_file, sdf_file in file_list_available:
        good_grasp_file_name = "{}_{}".format(filename_prefix, str(object_name))
        if good_grasp_file_name not in already_generated:
            file_list_all.append({"object_name": object_name, "obj_file": obj_file, "sdf_file": sdf_file})

    print(file_list_all)
    object_numbers = file_list_all.__len__()

    job_list = np.arange(object_numbers)
    job_list = list(job_list)
    pool_size = 1  # number of jobs did at same time
    assert (pool_size <= len(job_list))
    # Initialize pool
    pool = []
    for _ in range(pool_size):
        job_i = job_list.pop(0)
        pool.append(multiprocessing.Process(target=do_job, args=(job_i,)))
    [p.start() for p in pool]
    # refill
    while len(job_list) > 0:
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                job_i = job_list.pop(0)
                p = multiprocessing.Process(target=do_job, args=(job_i,))
                p.start()
                pool.append(p)
                break
    logger.info('All job done.')
