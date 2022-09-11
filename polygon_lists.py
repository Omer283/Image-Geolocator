import torch
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib import cm


def which_continent(loc, polygon_list, cont_name_list):
    '''
    Returns the continent which a point belongs to
    :param loc: [lat, long] coords
    :param polygon_list: List of polygons
    :param cont_name_list: List of continent names
    :return: The continent name
    '''
    pt = Point(loc[1], loc[0])
    for i in range(len(polygon_list)):
        if polygon_list[i].contains(pt):
            return cont_name_list[i]
    return None


def conv_locs(locs, polygon_list):
    '''
    Converts locations to one hot vectors
    :param locs: The locations in [lat, long] form
    :param polygon_list: The polygon list
    :return: An array of one hot vectors
    '''
    continents = len(polygon_list)
    batch = len(locs)
    preds = [continents - 1 for _ in range(batch)]
    pt_list = [Point(loc[1], loc[0]) for loc in locs]
    for i in range(continents):
        for j in range(batch):
            if polygon_list[i].contains(pt_list[j]):
                preds[j] = i
    return torch.nn.functional.one_hot(torch.tensor(preds), num_classes=continents)


def display_polygon_split(filename, behind=True):
    '''
    Displays the polygon split specified by a file
    :param filename: The file specifying the polygon split
    :param behind: Will the polygons appear behind the map
    :return: Nothing
    '''
    import geopandas as gpd
    a, b, _, __ = get_polygons_dict(filename)
    color = cm.rainbow(np.linspace(0, 1, len(a)))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    plt.figure()
    axes = plt.gca()
    for name, c in zip(a, color):
        axes.add_patch(matplotlib.patches.Polygon(a[name], closed=True, facecolor=c))
    if behind:
        zord = 2
    else:
        zord = 0
    world.plot(ax=axes, zorder=zord)
    plt.show()


def get_polygons_dict(filename):
    '''
    Returns the polygon information
    :param filename: file name specifying the polygons
    :return: cont_poly, cont_idx, cont_poly_list, cont_name_list
    '''
    cont_poly = dict()
    cont_idx = dict()
    cont_poly_list = []
    cont_name_list = []
    cnt = 0
    with open(filename, 'r') as f:
        while True:
            name = f.readline().strip()
            if len(name) == 0:
                return cont_poly, cont_idx, cont_poly_list, cont_name_list
            pt_list = []
            while True:
                ln = f.readline().strip()
                if ln == '---':
                    break
                pt = ln.split(',')
                pt_list.append((float(pt[0]), float(pt[1])))
            cont_idx[name] = cnt
            cont_poly[name] = pt_list
            cont_poly_list.append(pt_list)
            cont_name_list.append(name)
            cnt += 1
    return cont_poly, cont_idx, cont_poly_list, cont_name_list