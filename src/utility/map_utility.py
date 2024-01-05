#get all the path of the map in ./map/f1tenth_racetracks
import os
import random
import numpy as np


wpt_delim= ';'
wpt_rowskip= 3
wpt_xind= 1
wpt_yind= 2
wpt_thind= 3
wpt_vind= 5

def get_map_path():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    map_path = "./map/f1tenth_racetracks"
    return map_path

def get_map_list():
    map_path = get_map_path()
    map_list = os.listdir(map_path)
    filtered_map_list = []

    for map_file in map_list:
        if os.path.isdir(os.path.join(map_path, map_file)):
            raceline_file = os.path.join(map_path, map_file, f"{map_file}_raceline.csv")
            if os.path.exists(raceline_file):
                filtered_map_list.append(map_file)

    return filtered_map_list

def get_map(map_name):
    map_path = get_map_path()
    map_file = map_path + "/" + map_name
    return map_file

def get_map_name(map_file):
    map_name = map_file.split("/")[-1]
    return map_name

def get_map_list_name():
    map_list = get_map_list()
    map_list_name = []
    for map_file in map_list:
        map_list_name.append(get_map_name(map_file))
    return map_list_name

def get_map_list_file():
    map_list = get_map_list()
    map_list_file = []
    for map_file in map_list:
        map_list_file.append(get_map(map_file))
    return map_list_file

def get_map_list_name_file():
    map_list = get_map_list()
    map_list_name_file = []
    for map_file in map_list:
        map_list_name_file.append(get_map_name(map_file))
        map_list_name_file.append(get_map(map_file))
    return map_list_name_file

def get_map_list_name_file_dict():
    map_list = get_map_list()
    map_list_name_file_dict = {}
    for map_file in map_list:
        map_list_name_file_dict[get_map_name(map_file)] = get_map(map_file)
    return map_list_name_file_dict



def get_one_random_map():
    map_list = get_map_list()
    map_file = random.choice(map_list)
    return get_map(map_file)


def get_raceline(map_folder):
    map_name = get_map_name(map_folder)
    raceline_file = map_folder + "/"+ map_name + "_raceline.csv"
    raceline = np.genfromtxt(raceline_file, delimiter=wpt_delim, skip_header=wpt_rowskip)
    return raceline

def get_x_y_theta_from_raceline(raceline):
    x = raceline[:, wpt_xind]
    y = raceline[:, wpt_yind]
    theta = raceline[:, wpt_thind]
    return x, y, theta

def get_start_pose(map_folder):
    raceline = get_raceline(map_folder)
    x, y, theta = get_x_y_theta_from_raceline(raceline)
    start_pose = [x[0], y[0], theta[0]]
    return start_pose




if __name__ == "__main__":
    print(get_one_random_map())
    print(get_start_pose(get_one_random_map()))
