import argparse
from glob import glob
import re
import csv
from collections import OrderedDict
import torch
import os
from Common.pc_util import *

from upsampling.loss.chamfer_dist import ChamferFunction as chamfer_3DDist
from sklearn.neighbors import NearestNeighbors
import math
from time import time

parser = argparse.ArgumentParser()
parser.add_argument("--pred", type=str, required=False,default="/pred", help=".xyz")  # 网络输出数据所在的位置
parser.add_argument("--gt", type=str, required=False, default="/gt", help=".xyz")  # GT所在的位置
FLAGS = parser.parse_args()
PRED_DIR = os.path.abspath(FLAGS.pred)  # 拿到测试数据所在的文件夹绝对路径
GT_DIR = os.path.abspath(FLAGS.gt)
print(PRED_DIR)
# NAME = FLAGS.name
print(GT_DIR)
chamfer_distance = chamfer_3DDist.apply
gt_paths = glob(os.path.join(GT_DIR, '*.xyz'))  # 拿到所有的测试数据路径
pred_paths = glob(os.path.join(PRED_DIR, '*.xyz'))
print(gt_paths)
print(pred_paths)

gt_names = [os.path.basename(p)[:-4] for p in gt_paths]
pred_names = [os.path.basename(p)[:-4] for p in gt_paths]

gt = load(gt_paths[0])[:, :3]  # 当前只取第一个数据评估
pred = load(pred_paths[0])[:, :3]

pred_tensor, centroid, furthest_distance = normalize_point_cloud(gt)
gt_tensor, centroid, furthest_distance = normalize_point_cloud(pred)

cd_forward, cd_backward = chamfer_distance(torch.from_numpy(gt_tensor).cuda(), torch.from_numpy(pred_tensor).cuda())

cd_forward = cd_forward[0, :].cpu().numpy()
cd_backward = cd_backward[0, :].cpu().numpy()

precentages = np.array([0.004, 0.006, 0.008, 0.01, 0.012])
# precentages = np.array([0.004, 0.006])

def cal_nearest_distance(queries, pc, k=2):
    """
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    dis, knn_idx = knn_search.kneighbors(queries, return_distance=True)
    return dis[:, 1]


def analyze_uniform(idx_file, radius_file, map_points_file):
    start_time = time()
    points = load(map_points_file)[:, 4:]
    radius = np.loadtxt(radius_file)
    print('radius:', radius)
    with open(idx_file) as f:
        lines = f.readlines()

    sample_number = 1000
    rad_number = radius.shape[0]

    uniform_measure = np.zeros([rad_number, 1])

    densitys = np.zeros([rad_number, sample_number])

    expect_number = precentages * points.shape[0]
    print(expect_number, rad_number)
    expect_number = np.reshape(expect_number, [rad_number, 1])

    for j in range(rad_number):
        uniform_dis = []

        for i in range(sample_number):

            density, idx = lines[i * rad_number + j].split(':')
            densitys[j, i] = int(density)
            coverage = np.square(densitys[j, i] - expect_number[j]) / expect_number[j]

            num_points = re.findall("(\d+)", idx)

            idx = list(map(int, num_points))
            if len(idx) < 5:
                continue

            idx = np.array(idx).astype(np.int32)
            map_point = points[idx]

            shortest_dis = cal_nearest_distance(map_point, map_point, 2)
            disk_area = math.pi * (radius[j] ** 2) / map_point.shape[0]
            expect_d = math.sqrt(2 * disk_area / 1.732)  ##using hexagon

            dis = np.square(shortest_dis - expect_d) / expect_d
            dis_mean = np.mean(dis)
            uniform_dis.append(coverage * dis_mean)

        uniform_dis = np.array(uniform_dis).astype(np.float32)
        uniform_measure[j, 0] = np.mean(uniform_dis)

    print('time cost for uniform :', time() - start_time)
    return uniform_measure


fieldnames = ["CD", "hausdorff", "p2f avg", "p2f std"]

fieldnames += ["uniform_%d" % d for d in range(precentages.shape[0])]
for D in [PRED_DIR]:
    avg_md_forward_value = 0
    avg_md_backward_value = 0
    avg_hd_value = 0
    counter = 0
    pred_paths = glob(os.path.join(D, "*.xyz"))

    gt_pred_pairs = []
    for p in pred_paths:
        name, ext = os.path.splitext(os.path.basename(p))
        assert (ext in (".ply", ".xyz"))
        try:
            gt = gt_paths[gt_names.index(name)]
        except ValueError:
            pass
        else:
            gt_pred_pairs.append((gt, p))

    tag = re.search("/(\w+)/result", os.path.dirname(gt_pred_pairs[0][1]))
    if tag:
        tag = tag.groups()[0]
    else:
        tag = D

    global_p2f = []
    global_density = []
    global_uniform = []

    with open(os.path.join(os.path.dirname(gt_pred_pairs[0][1]), "evaluation.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
        writer.writeheader()
        for gt_path, pred_path in gt_pred_pairs:
            gt = load(gt_path)[:, :3]
            gt = gt[np.newaxis, ...]
            pred = load(pred_path)
            pred = pred[:, :3]

            pred = pred[np.newaxis, ...]

            cd_forward_value, cd_backward_value = [cd_forward, cd_backward]

            md_value = np.mean(cd_forward_value) + np.mean(cd_backward_value)
            hd_value = np.max(np.amax(cd_forward_value, axis=0) + np.amax(cd_backward_value, axis=0))
            cd_forward_value = np.mean(cd_forward_value)
            cd_backward_value = np.mean(cd_backward_value)
            avg_md_forward_value += cd_forward_value
            avg_md_backward_value += cd_backward_value
            avg_hd_value += hd_value

            if os.path.isfile(pred_path[:-4] + "_point2mesh_distance.txt"):
                point2mesh_distance = load(pred_path[:-4] + "_point2mesh_distance.txt")
                print(point2mesh_distance)
                if point2mesh_distance.size == 0:
                    continue
                point2mesh_distance = point2mesh_distance[:, 3]
                global_p2f.append(point2mesh_distance)

            if os.path.isfile(pred_path[:-4] + "_disk_idx.txt"):
                idx_file = pred_path[:-4] + "_disk_idx.txt"
                radius_file = pred_path[:-4] + '_radius.txt'
                map_points_file = pred_path[:-4] + '_point2mesh_distance.txt'

                disk_measure = analyze_uniform(idx_file, radius_file, map_points_file)
                global_uniform.append(disk_measure)

            counter += 1

        row = OrderedDict()

        avg_md_forward_value /= counter
        avg_md_backward_value /= counter
        avg_hd_value /= counter
        avg_cd_value = avg_md_forward_value + avg_md_backward_value
        row["CD"] = avg_cd_value
        row["hausdorff"] = avg_hd_value
        if global_p2f:
            global_p2f = np.concatenate(global_p2f, axis=0)
            mean_p2f = np.nanmean(global_p2f)
            row["p2f avg"] = mean_p2f

        if global_uniform:
            global_uniform = np.array(global_uniform)
            uniform_mean = np.mean(global_uniform, axis=0)
            for i in range(precentages.shape[0]):
                row["uniform_%d" % i] = uniform_mean[i, 0]

        writer.writerow(row)
