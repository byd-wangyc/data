#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import argparse
import csv
from typing import List, Tuple

RANDOM_SEED = 42

# 预设若干汽车零部件同义词/等价实体组（不含品牌）
# 每个子列表表示同一实体的多种表达，正样本从同一组内采样，负样本跨组采样
PART_SYNONYM_GROUPS = [
    ["发动机", "引擎", "发动机总成"],
    ["气缸盖", "缸盖", "汽缸盖"],
    ["气缸体", "缸体", "汽缸体"],
    ["曲轴", "曲拐轴"],
    ["凸轮轴", "配气凸轮轴"],
    ["活塞", "发动机活塞"],
    ["活塞环", "活塞圈"],
    ["连杆", "连杆组件"],
    ["喷油嘴", "燃油喷射嘴", "喷油器"],
    ["节气门", "节流阀"],
    ["涡轮增压器", "涡轮", "增压器"],
    ["散热器", "水箱", "散热水箱"],
    ["中冷器", "中间冷却器"],
    ["机油滤清器", "机滤", "机油滤芯"],
    ["空气滤清器", "空滤", "空气滤芯"],
    ["燃油滤清器", "汽滤", "燃油滤芯"],
    ["空调压缩机", "A/C压缩机", "空调泵"],
    ["火花塞", "点火塞"],
    ["点火线圈", "点火包"],
    ["离合器压盘", "压盘"],
    ["离合器片", "摩擦片", "离合片"],
    ["变速箱", "变速器", "变速箱总成"],
    ["差速器", "主减速器"],
    ["万向节", "十字轴"],
    ["半轴", "驱动半轴"],
    ["制动盘", "刹车盘"],
    ["制动鼓", "刹车鼓"],
    ["刹车片", "制动片"],
    ["刹车钳", "制动钳"],
    ["转向机", "转向器"],
    ["转向拉杆", "内外拉杆"],
    ["减震器", "避震器"],
    ["弹簧", "悬架弹簧"],
    ["下摆臂", "控制臂", "下控制臂"],
    ["稳定杆", "平衡杆"],
    ["球头", "转向球头"],
    ["轮毂轴承", "轴承单元"],
    ["轮速传感器", "ABS传感器"],
    ["氧传感器", "氧传感"],
    ["节温器", "温控阀"],
    ["水泵", "发动机水泵"],
    ["机油泵", "润滑油泵"],
    ["正时皮带", "正时带"],
    ["正时链条", "正时链"],
    ["张紧轮", "涨紧轮"],
    ["惰轮", "导向轮"],
    ["发电机", "交流发电机"],
    ["起动机", "启动机", "马达"],
    ["电瓶", "蓄电池"],
    ["雨刮器", "雨刷"],
    ["雨刮电机", "雨刷电机"],
    ["大灯总成", "前大灯", "前照灯"],
    ["尾灯总成", "后尾灯"],
    ["雾灯", "前雾灯"],
    ["保险杠", "前保险杠"],
    ["后保险杠", "后杠"],
    ["翼子板", "叶子板"],
    ["引擎盖", "发动机盖"],
    ["后备箱盖", "行李箱盖"],
    ["车门内饰板", "门内饰板"],
    ["座椅滑轨", "座椅导轨"],
    ["安全气囊", "气囊"],
    ["安全带", "安全束带"],
    ["中控屏", "多媒体主机", "中控显示屏"],
    ["行车电脑", "仪表电脑"],
    ["倒车影像摄像头", "后视摄像头"],
    ["雷达传感器", "泊车雷达"],
]


def set_seed():
    random.seed(RANDOM_SEED)


def sample_positive_pairs(groups: List[List[str]], num_pairs: int) -> List[Tuple[str, str, int]]:
    pairs: List[Tuple[str, str, int]] = []
    candidates = [g for g in groups if len(g) >= 2]
    while len(pairs) < num_pairs and candidates:
        group = random.choice(candidates)
        left, right = random.sample(group, 2)
        pairs.append((left, right, 1))
    return pairs


def sample_negative_pairs(groups: List[List[str]], num_pairs: int) -> List[Tuple[str, str, int]]:
    pairs: List[Tuple[str, str, int]] = []
    while len(pairs) < num_pairs:
        g1, g2 = random.sample(groups, 2)
        left = random.choice(g1)
        right = random.choice(g2)
        pairs.append((left, right, 0))
    return pairs


def write_csv(path: str, rows: List[Tuple[str, str, int]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["entity1", "entity2", "label"])  # 保留表头
        for a, b, y in rows:
            writer.writerow([a, b, y])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=8000, help='训练样本数量（含正负各半）')
    parser.add_argument('--test_size', type=int, default=2000, help='测试样本数量（含正负各半）')
    parser.add_argument('--output_dir', type=str, default=os.path.join('data'), help='输出目录')
    args = parser.parse_args()

    set_seed()

    train_pos = sample_positive_pairs(PART_SYNONYM_GROUPS, args.train_size // 2)
    train_neg = sample_negative_pairs(PART_SYNONYM_GROUPS, args.train_size // 2)
    test_pos = sample_positive_pairs(PART_SYNONYM_GROUPS, args.test_size // 2)
    test_neg = sample_negative_pairs(PART_SYNONYM_GROUPS, args.test_size // 2)

    train_rows = train_pos + train_neg
    test_rows = test_pos + test_neg
    random.shuffle(train_rows)
    random.shuffle(test_rows)

    write_csv(os.path.join(args.output_dir, 'train.csv'), train_rows)
    write_csv(os.path.join(args.output_dir, 'test.csv'), test_rows)
    print(f"Wrote {len(train_rows)} train and {len(test_rows)} test samples to '{args.output_dir}'.")


if __name__ == '__main__':
    main()

