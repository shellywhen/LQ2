#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate scores for images with empirical measures.
[1] O’Donovan, P., Agarwala, A., & Hertzmann, A. (2014). Learning layouts for single-pagegraphic designs. IEEE Transactions on Visualization and Computer Graphics, 20(8), 1200-1213. http://www.dgp.toronto.edu/~donovan/layout/
"""
import json
import math
import pandas as pd
import numpy as np
import pickle as pkl
from PIL import Image, ImageDraw, ImageOps
import multiprocessing
from datetime import datetime
import os

def sigmoid(x, alpha=1.0):
    return math.atan(alpha*x) / math.atan(alpha)

def white_space_area(mask, alpha=2.0):
    w, h = mask.shape
    ratio =  (1-sum(sum(mask))) / (w*h)
    return -sigmoid(ratio, alpha)

def _get_distance_map(rect, size):
    """For a given rectangle, calculate the distance map for each pixel
    Note that the distance is the Euclidean distance between the pixel and the center point
    """
    dis = np.zeros((size[1], size[0]))
    center = np.array([(rect[0][0]+rect[2][0])/2, (rect[0][1]+rect[2][1])/2])
    for i in range(size[1]):
        for j in range(size[0]):
            p = np.array([i,j])
            dis[i][j] = np.linalg.norm(p-center)
    img = Image.new('L', size, 'white')
    draw = ImageDraw.Draw(img)
    polygon = np.array(rect).flatten().astype(int).tolist()
    draw.polygon(polygon, fill='black')
    mask = (np.array(img) / 255).astype(int)
    return mask * dis

def spread(maps, size, alpha=600.0):
    s = 0
    for i in range(size[1]):
        for j in range(size[0]):
            d = min([m[i][j] for m in maps])
            s += d**3
    return sigmoid(s/(size[0]*size[1]), alpha)

def distance(maps, alpha=50.0):
    n = len(maps)
    mindis = float(maps[0].shape[0] + maps[0].shape[1])
    res_sum = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = maps[i]
            q = maps[j]
            tmp = np.sqrt((p**2+q**2)/2)
            res = np.min(np.min(tmp))
            mindis = min(mindis, res)
            res_sum += 1 - sigmoid(mindis, alpha)
    return res_sum / n

def _get_graphic_size(graphics, shape):
    W, H = shape
    sizes = []
    for bar in graphics:
        [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] = bar
        w = max(abs(x0 - x1), abs(x0 - x2))
        h = max(abs(y0 - y2), abs(y0 - y1))
        sizes.append(w*h/(W*H))
    return sizes

def graphic_size(sizes, alpha=2.0):
    return sum([sigmoid(x, alpha) for x in sizes]) / len(sizes)

def graphic_size_var(sizes, alpha=200.0):
    return sigmoid(np.std(np.array(sizes)), alpha)

def graphic_size_min(sizes, tau=0.04):
    return sum([max(tau-x,0) for x in sizes])

def group_size_var(graphicsSizeVar):
    # adapt to our case
    return graphicsSizeVar/3

def _get_box_center_distance(a, b):
    """Compute the distance for center points
    """
    ma = np.array([(a[0][0] + a[2][0])/2, (a[0][1] + a[2][1])/2])
    mb = np.array([(b[0][0] + b[2][0])/2, (b[0][1] + b[2][1])/2])
    d = np.linalg.norm(ma-mb)
    return d

def _get_box_distance(a, b, flag=False):
    """Compute the minimum distance for boundaries
    :a/b: a list in the order of [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    if not flag:  # regular rect
        a, b = np.array(a).flatten(), np.array(b).flatten()
        ax, bx = a[[0,2,4,6]], a[[0,2,4,6]]
        ay, by = a[[1,3,5,7]], b[[1,3,5,7]]
        #print(ay, by, ax, bx)
        minx = min([min([abs(t1-t2) for t1 in ax]) for t2 in bx ])
        miny = min([min([abs(t1-t2) for t1 in ay]) for t2 in by ])
        #print(minx, miny)
        return max(minx, miny)
    else:         # rect with rotation
        lines  = np.array([[b[0],b[1]],[b[1],b[2]],[b[2],b[3]],[b[3], b[1]]])
        dis = [[min(np.cross(p[0]-t,p[1]-t)/np.linalg.norm(p[0]-p[1])) for p in lines] for t in a]
        return min(dis)

def group_distance_min(groups, flaglist=[False, False, False], alpha=200.0):
    res = 0
    for gi, group in enumerate(groups):
        group_n = len(group)
        flag = flaglist[gi]
        distance_sum = 0
        for i in range(group_n-1):
            for j in range(i+1, group_n-1):
                a = group[i]
                b = group[j]
                dis = _get_box_center_distance(a, b)
                #  dis = _get_box_distance(a, b, flag)
                distance_sum += dis
        res += distance_sum  / group_n
    return res/len(groups)

def _draw_local(group, size):
    img = Image.new('L', size, 'black')
    draw = ImageDraw.Draw(img)
    for rect in group:
        polygon = np.array(rect).flatten().astype(int).tolist()
        draw.polygon(polygon, fill='white')
    return img

def symmetry(groups, size, axis, alpha=1.0):
    final = 0
    for group in groups:
        img = _draw_local(group, size)
        flip = ImageOps.mirror(img) if axis == 'x' else ImageOps.flip(img)
        o = (np.array(img) / 255).astype(int)
        n = (np.array(flip) / 255).astype(int)
        total = sum(sum(o))
        divider = sum(sum(abs(o-n)*o))
        result = divider/total - 1
        final += result
        del img
    return sigmoid(final, alpha)

def asymmetry(groups, size, axis, alpha=1.0):
    final = 0
    for group in groups:
        img = _draw_local(group, size)
        flip = ImageOps.mirror(img) if axis == 'x' else ImageOps.flip(img)
        o = (np.array(img) / 255).astype(int)
        n = (np.array(flip) / 255).astype(int)
        total = sum(sum(o))
        divider = sum(sum(abs(o-n)*o))
        result = divider/total - 1
        final += result
        del img
    return sigmoid(final-1, alpha)

def _draw_local(group, size):
    img = Image.new('L', size, 'black')
    draw = ImageDraw.Draw(img)
    for rect in group:
        polygon = np.array(rect).flatten().astype(int).tolist()
        draw.polygon(polygon, fill='white')
    return img

def symmetry(groups, size, axis, alpha=1.0):
    final = 0
    for group in groups:
        img = _draw_local(group, size)
        flip = ImageOps.mirror(img) if axis == 'x' else ImageOps.flip(img)
        o = (np.array(img) / 255).astype(int)
        n = (np.array(flip) / 255).astype(int)
        total = sum(sum(o))
        divider = sum(sum(abs(o-n)*o))
        result = divider/total - 1
        final += result
        del img
    return sigmoid(final, alpha)

def asymmetry(groups, size, axis, alpha=1.0):
    final = 0
    for group in groups:
        img = _draw_local(group, size)
        flip = ImageOps.mirror(img) if axis == 'x' else ImageOps.flip(img)
        o = (np.array(img) / 255).astype(int)
        n = (np.array(flip) / 255).astype(int)
        total = sum(sum(o))
        divider = sum(sum(abs(o-n)*o))
        result = divider/total - 1
        final += result
        del img
    return sigmoid(final-1, alpha)

def _resize(data):
    """Rescale the image to a fixed height of 128 to accelerate computation.
    """
    newd = {}
    newh = int(128)
    h = int(data['height'])
    w = int(data['width'])
    neww = int(newh*w/h)
    newd['width'] = neww
    newd['height'] = newh
    for title in ['x-bbox', 'y-bbox', 'bar-bbox']:
        newd[title] = []
        for rect in data[title]:
            newpos = [[x*neww/w, y*newh/h]for x, y in rect]
            newd[title].append(newpos)
    newd['name'] = data['name']
    return newd

metricName = ['name','whiteSpaceArea', 'graphicSpread', 'graphicDistance','graphicSize', 'graphicSizeVar', 'graphicSizeMin','groupSizeVar', 'groupDistanceMin', 'graphicXSymmetry', 'graphicXAsymmetry', 'graphicYSymmetry', 'graphicYAsymmetry', 'textXsymmetry', 'textXAsymmetry', 'textYsymmetry','textYAsymmetry']

def step(d):
    """Pipeline to generate scores from the graphical features.
    """
    data = _resize(d)
    name = data['name']
    size =  (int(data['width']), int(data['height']))
    img = Image.new('L', size, 'black')
    polygons = data['x-bbox'] + data['y-bbox'] + data['bar-bbox']
    draw = ImageDraw.Draw(img)
    for rect in polygons:
        polygon = np.array(rect).flatten().astype(int).tolist()
        draw.polygon(polygon, fill='white')
    mask = (np.array(img) / 255).astype(int)
    xflag = True if data['x-bbox'][0][0][1] != data['x-bbox'][0][1][1] else False

    whiteSpaceArea = white_space_area(mask)
    _graphicDistanceMap = [_get_distance_map(rect, size) for rect in data['bar-bbox']]
    graphicSpread = spread(_graphicDistanceMap, size)
    graphicDistance = distance(_graphicDistanceMap)

    graphic_sizes= _get_graphic_size(data['bar-bbox'], mask.shape)
    graphicSize= graphic_size(graphic_sizes)
    graphicSizeVar = graphic_size_var(graphic_sizes)
    graphicSizeMin = graphic_size_min(graphic_sizes)

    groupSizeVar = group_size_var(graphicSize)
    groupDistanceMin = group_distance_min([data['bar-bbox'], data['x-bbox'], data['y-bbox']], [False, xflag, False])

    graphicXSymmetry = symmetry([data['bar-bbox']], size, 'x')
    graphicXAsymmetry = asymmetry([data['bar-bbox']], size, 'x')
    graphicYSymmetry = symmetry([data['bar-bbox']], size, 'y')
    graphicYAsymmetry = asymmetry([data['bar-bbox']], size, 'y')
    textXsymmetry = symmetry([data['x-bbox'], data['y-bbox']], size, 'x')
    textXAsymmetry = asymmetry([data['x-bbox'], data['y-bbox']], size, 'x')
    textYsymmetry = symmetry([data['x-bbox'], data['y-bbox']], size, 'y')
    textYAsymmetry = asymmetry([data['x-bbox'], data['y-bbox']], size, 'y')

    return [name, whiteSpaceArea, graphicSpread, graphicDistance,graphicSize, graphicSizeVar, graphicSizeMin, groupSizeVar, groupDistanceMin, graphicXSymmetry, graphicXAsymmetry, graphicYSymmetry, graphicYAsymmetry, textXsymmetry, textXAsymmetry, textYsymmetry, textYAsymmetry]

def calculate_measures(filename, savepath):
    with open(filename, 'rb') as f:
        dataset = pkl.load(f)
        totalcharts = len(dataset)
    params = [[x] for x in dataset]
    n_worker = max(int(multiprocessing.cpu_count()/2), 1)
    with multiprocessing.Pool(n_worker) as p:
        measureArray = p.starmap(step, params)
    df = pd.DataFrame(measureArray)
    df.columns= metricName
    df.to_csv(savepath, index=False)
    return df

if __name__ == '__main__':
    calculate_measures('../dataset/exp1/graphical_features', '../dataset/exp1/metrics.csv')
    calculate_measures('../dataset/exp2/graphical_features', '../dataset/exp2/metrics.csv')
