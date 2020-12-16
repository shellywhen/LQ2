#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parse graphical elements in the chart for empirical approach to score them.
"""
import json
import math
import altair as alt
from altair_saver import save
import bs4
import os

global char_px
char_px = 6

def get_translate(s):
    x, y = s.replace('translate(', '').replace(')', '').split(',')
    return float(x), float(y)

def get_rotate(s):
    angle = s.replace('rotate(', '').replace(')', '')
    return float(angle)

def get_parent_pos(ele):
    """Find the overall offset (x,y) from the element parents.
    :ele: a beautifulsoup-based svg element
    """
    sx, sy = 0.0, 0.0
    for p in ele.parents:
        if p.has_attr('transform'):
            x, y = get_translate(p['transform'])
            sx += x
            sy += y
    return sx, sy

def get_text_bbox_from_anchor(x, y, length, anchor):
    """Parse the bounding box of a horizontal text.
    """
    width = length * char_px
    offset = 0 if anchor == 'end' else (width/2 if anchor == 'middle' else width)
    x0 = x - width + offset
    y0 = y - char_px
    x1 = x + offset
    y1 = y - char_px
    x2 = x + offset
    y2 = y
    x3 = x - width + offset
    y3 = y
    return [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]

def reverse_rotated_text(sx, sy, length, angle, ex, ey):
    """Parse the bounding box of a rotated text.
    This function only apply to a limited range of texts with the angle of -45 and -90.
    """
    width = length * char_px
    [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] = get_text_bbox_from_anchor(sx, sy, length, 'end')
    if angle == 315:               # horrible code
        width /= math.sqrt(2)
        ex /= math.sqrt(2)
        ey /= math.sqrt(2)
        char = char_px / math.sqrt(2)
        x1 = sx - char + ex + ey
        y1 = sy - char + ex + ey
        x3 = sx - width + ex + ey
        y3 = sy + width + ex + ey
        x0 = sx - width - char + ex + ey
        y0 = sy + width - char + ex + ey
        x2 = sx + ex + ey
        y2 = sy + ex + ey
    elif angle == 270:
        x3 = sx + ey
        y3 = sy + char_px + ex
        x1 = sx - char_px + ey
        y1 = sy + ex + ex
        x0 = sx - char_px + ey
        y0 = sy + width + ex
        y2 += ex
        x2 += ey
    else:
        print('Error in parsing', text, angle, type(angle))
        return None
    return [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]

def deal_texts(texts):
    """Batch process of a group of texts, get their bboxes.
    """
    px, py = get_parent_pos(texts[0])
    positions = []
    for text in texts:
        content = text.text
        length = len(content)
        width = length * 6        # heuristic: a character ≈ 6px
        transforms = text['transform'].split(' ')
        if len(transforms) is 1:  # safe like y & normal x
            ox, oy = get_translate(text['transform'])
            x, y = ox+px, oy+py
            pos = get_text_bbox_from_anchor(x, y, length, text['text-anchor'])
        else:                     # rotate pattern
            ox, oy = get_translate(transforms[0])
            x, y = ox+px, oy+py
            a = int(get_rotate(transforms[1]))
            if a != 315 and a != 270:
                print('Error', text)
                return [], None
            ex, ey = get_translate(transforms[2])
            pos = reverse_rotated_text(x, y, length, a, ex, ey)
        positions.append(pos)
    return positions, '#000000'

def deal_bars(bars):
    """Batch process for rectangles. Get their bboxes.
    """
    px, py = get_parent_pos(bars[0])
    color = bars[0]['fill']
    positions = []
    for bar in bars:
        pathObj = parse_path(bar['d'])
        x0 =  px + pathObj[0].start.real
        y0 = py + pathObj[0].start.imag
        x1 = px + pathObj[1].end.real
        y1 = py + pathObj[2].end.imag
        positions.append([[x0,y0], [x1,y0], [x1,y1], [x0,y1]])
    return positions, color

def parse_svg(soup):
    svg = soup.find("svg")
    bars = soup.findAll('path', attrs={'role': 'graphics-symbol'})
    axes = soup.findAll('g', attrs={'class': 'role-axis-label'})
    xtexts = axes[0].findAll('text')
    ytexts = axes[1].findAll('text')
    bar_eles, bar_color = deal_bars(bars)
    xtext_eles, xtext_color = deal_texts(xtexts)
    ytext_eles, ytext_color = deal_texts(ytexts)
    regular = bar_eles + ytext_eles
    irregular = []
    if '(315)' in xtexts[0]['transform']:
        irregular = xtext_eles
    else:
        regular += xtext_eles
    result = {
        'width': svg['width'],
        'height': svg['height'],
        'color': list(set([bar_color, xtext_color, ytext_color])),
        'regular_bbox': regular,
        'irregular_bbox': irregular
    }
    return result

def get_elements_from_img(img_name, folder):
    """The pipeline to process the vega-lite spec and find out its shapes.
    """
    with open(f'{folder}{img_name}.json', 'r') as f:
        data = json.load(f)
    chart = alt.Chart.from_dict(data)
    chart.save('chart.svg')
    with open('chart.svg', 'r') as f:
        svgstr = f.read()
    soup = bs4.BeautifulSoup(svgstr, 'html.parser')
    result = parse_svg(soup)
    result['name'] = img_name
    return result, chart

def launch(folder, output_path):
    results = []
    names = list(set([x.split('.')[0] for x in os.listdir(folder)]))
    total = len(names)
    bar = 50
    for i, name in enumerate(names):
        res, chart = get_elements_from_img(name, folder)
        results.append(res)
        del chart
        progress = int(i / total * bar) +1
        print('\r['+ '>'*progress + ' '*(bar-progress)+']'+ f'  {i}', end='')
    with open(output_path, 'wb') as f:
        pkl.dump(results, f)
if __name__ == '__main__':
    launch('../dataset/exp1/vega_spec/', '../dataset/exp1/graphical_features.pkl')
    launch('../dataset/exp2/vega_spec/', '../dataset/exp2/graphical_features.pkl')
