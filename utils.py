import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import shapely.geometry.polygon
import os
import polygon_lists
from single_image_dataset import SingleImageDataset
import torchvision
from PIL import Image
import random


def topk_categories(probabilities, categories, k):
    """
    Given probabilities vector and categories meaning, returns top-k scores and categories
    :param probabilities: Probabilities vector calculated in forward pass
    :param categories: Categories vector
    :param k: top k
    :return: Top k scores and categories
    """
    assert k <= len(categories)
    topk_prob, topk_catid = torch.topk(probabilities, k)
    return np.array([(categories[topk_catid[i]], topk_prob[i].item()) for i in range(k)])


def get_categories_from_file(filename):
    """
    Given category file, return categories
    :param filename: The file name
    :return: Categories list
    """
    with open(filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories


def write_points_to_csv(points, filename):
    """
    Writes given (lat, long) points to csv
    :param points: List of points
    :param filename: Filename to write to
    """
    with open(filename, "w") as f:
        f.write("Latitude,Longitude\n")
        for p in points:
            f.write(f'{p[0]},{p[1]}\n')


def great_circle_dist(lat1, lon1, lat2, lon2):
    '''
    Calculates the great circle distance between pairs of points
    :param lat1: vector of latitudes
    :param lon1: vector of longitudes
    :param lat2: vector of latitudes
    :param lon2: vector of longitudes
    :return: vector of the distances between the pairs of points
    '''
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_long = lon2 - lon1
    sin_lat = torch.sin(delta_lat / 2.0)
    sin_long = torch.sin(delta_long / 2.0)
    a_xy = sin_lat ** 2 + torch.cos(lat1) * torch.cos(lat2) * (sin_long ** 2)
    return 2 * 6371 * torch.asin(torch.sqrt(a_xy))


def custom_loss(outputs, labels):
    '''
    Calculates the mean great circle distance
    :param outputs: outputs of the model
    :param labels: correct labels of the model
    :return: The mean great circle distance
    '''
    return torch.mean(great_circle_dist(outputs[:,0], outputs[:,1], labels[:,0], labels[:,1]))


def plot_csv_points(filename):
    '''
    Plots the points in the CSV file on a map
    :param filename: filename to be plotted
    :return: Nothing
    '''
    import geopandas as gpd
    from geopandas import GeoDataFrame
    df = pd.read_csv(filename, delimiter=',', skiprows=0, low_memory=False)
    geometry = []
    for xy in zip(df['long'], df['lat']):
        geometry.append(Point(xy))
    gdf = GeoDataFrame(df, geometry=geometry)
    # this is a simple map that goes with geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=1)
    plt.show()


def imshow(img):
    '''
    Displays a specified image
    :param img: the image
    :return: Nothing
    '''
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_k_images(region, k=5,
                 dir_name ='100k_images_dir', polygon_file='continents_polygon2', image_file='20k_images_file.csv'):
    '''
    Finds and displays K images from a certain region
    :param region: The wanted region
    :param k: The amount of images
    :param dir_name: Dir to look for images
    :param polygon_file: File describing the polygons
    :param image_file: File describing the images
    :return: Nothing
    '''
    cont_poly, cont_idx, _, __ = polygon_lists.get_polygons_dict(polygon_file)
    counter = 0
    polygon = shapely.geometry.polygon.Polygon(cont_poly[region])
    BATCH_SIZE = 16
    t = transforms.compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    imds = SingleImageDataset(image_file, dir_name, transform=t)
    imdl = torch.utils.data.DataLoader(imds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    cur_tensor = None
    loc_list = []
    for data in imdl:
        inputs, labels = data
        for j in range(BATCH_SIZE):
            loc_j = labels[j].numpy()
            pt = Point(loc_j[1], loc_j[0]) #becuz i'm stupid
            if polygon.contains(pt):
                cur_img = inputs[j].unsqueeze(0)
                loc_list.append(loc_j)
                if cur_tensor is None:
                    cur_tensor = cur_img
                else:
                    cur_tensor = torch.cat((cur_tensor, cur_img))
                counter += 1
                if counter >= k:
                    break
        if counter >= k:
            break
    imshow(torchvision.utils.make_grid(cur_tensor))
    for j in range(k):
        print(loc_list[j])


def convert_file(filename, new_filename, polygon_file_name):
    '''
    Convert file to an updated file using different polygon splits
    :param filename: old file
    :param new_filename: new file
    :param polygon_file_name: polygon split
    :return: Nothing
    '''
    assert filename != new_filename
    a, b, c, d = polygon_lists.get_polygons_dict(polygon_file_name)
    c = [shapely.geometry.polygon.Polygon(x) for x in c]
    f = open(filename, 'r')
    g = open(new_filename, 'w+')
    f.readline()
    g.write('name,lat,long,cont\n')
    while True:
        x = f.readline()
        if len(x) == 0:
            break
        tokens = x.strip().split(',')
        pt = [float(tokens[1]), float(tokens[2])]
        name = polygon_lists.which_continent(pt, c, d)
        num = b[name]
        g.write(f'{tokens[0]},{tokens[1]},{tokens[2]},{num}\n')


def rename_dir(dir_name, digits=5):
    '''
    Renames directory according to more tidy naming scheme
    :param dir_name: Directory to be renamed
    :param digits: Number of digits in ID
    :return: Nothing
    '''
    for file_name in os.listdir(dir_name):
        path_name = os.path.join(dir_name, file_name)
        tokens = file_name.split('.')
        idx = tokens[0][5:]
        while len(idx) < digits:
            idx = '0' + idx
        new_pathname = os.path.join(dir_name, 'image' + idx + '.jpg')
        os.rename(path_name, new_pathname)


def resize_dir(dir_name, new_dir, up_size=1024, down_size=224):
    '''
    Resize all images in directory
    :param dir_name: Target directory
    :param new_dir: New directory
    :param up_size: Maximum dimension
    :param down_size: Minimum dimension
    :return: Nothing
    '''
    cnt = 0
    for name in os.listdir(dir_name):
        img = Image.open(os.path.join(dir_name, name))
        if img.size[0] > up_size or img.size[1] > up_size:
            r1 = img.size[0] / up_size
            r2 = img.size[1] / up_size
            r = max(r1, r2)
            x = max(down_size, int(img.size[0] / r))
            y = max(down_size, int(img.size[1] / r))
            img = img.resize((x, y), Image.ANTIALIAS)
        img.save(os.path.join(new_dir, name))
        cnt += 1


def downsample(old_file, new_file, upper_bound=15000):
    '''
    Downsamples the input according to some upper bound
    :param old_file: old src file
    :param new_file: new dest file
    :param upper_bound: The upper bound on the number of images in a certain region
    :return: Nothing
    '''
    f = open(old_file, 'r')
    g = open(new_file, 'w+')
    g.write(f.readline())
    cnts = [0 for _ in range(25)]
    while True:
        x = f.readline().strip()
        if len(x) == 0:
            break
        tokens = x.split(',')
        region = int(tokens[-1])
        cnts[region] += 1
        if cnts[region] <= upper_bound:
            g.write(x + '\n')


def split_into_regions(original_file, num_classes=15):
    '''
    Split file into files with different regions
    :param original_file: original file
    :param num_classes: number of classes
    :return: Nothing
    '''
    file_name = original_file.split('.')[0]
    new_dir_name = file_name + '_split'
    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)
    new_files = [open(os.path.join(new_dir_name, 'region' + str(i) + '.csv'), 'w+') for i in range(num_classes)]
    for f in new_files:
        f.write('name,lat,long,cont\n')
    g = open(original_file, 'r')
    g.readline()
    while True:
        x = g.readline().strip()
        if len(x) == 0:
            break
        region = int(x.split(',')[-1])
        new_files[region].write(x + '\n')


def less_photos(filename, train, test, remove=5000):
    f = open(filename, 'r')
    f.readline()
    lns = 0
    while True:
        x = f.readline().strip()
        if len(x) == 0:
            break
        lns += 1
    idx = list(range(lns))
    random.shuffle(idx)
    to_rem = set(idx[:remove])
    f = open(filename, 'r')
    g1 = open(train, 'w+')
    g2 = open(test, 'w+')
    g1.write('name,lat,long,cont\n')
    g2.write('name,lat,long,cont\n')
    f.readline()
    cur_idx = 0
    while True:
        x = f.readline().strip()
        if len(x) == 0:
            break
        if cur_idx in to_rem:
            g2.write(x + '\n')
        else:
            g1.write(x + '\n')
        cur_idx += 1