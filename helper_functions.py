import numpy as numpy
import re
import glob
import shutil
import os

'''
    A couple of helper functions used in image_processor.py
'''

def find_uncompact_clusters(folder, threshold=0.5, record_cluster_nums=0):
    cluster_nums=[]
    with open(f"{folder}/cluster_descriptions.txt") as f:
        for line in f.readlines():
            matches = re.match("Cluster ([^ ]+) - ([^ ]+ ){15}([^\n]*)", line)
            if(matches):
                if(float(matches.group(3))<threshold):
                    cluster_nums.append(matches.group(1))
                    if record_cluster_nums:
                        with open(f"{folder}/uncompact_clusters.txt", "a") as f2:
                            f2.write(f"{matches.group(1)} {matches.group(3)}\n")
    return cluster_nums

def move_uncompact_plots(folder, file, new_folder):
    filenames=glob.glob(f"{folder}/*.png")
    cwd = os.getcwd()
    with open(f"{folder}/{file}", "r") as f:
        for line in f.readlines():
            cluster_number = re.match("([^ ]+)", line).group(1)
            print(cluster_number)
            for filename in filenames:
                if f"{cluster_number}." in filename:
                    shutil.move(f"{cwd}/{filename}", f"{cwd}/{folder}/{new_folder}/{cluster_number}.png")


#move_uncompact_plots("gcal-thresh40", "uncompact_clusters.txt", "uncompact")

