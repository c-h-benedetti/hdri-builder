import cv2 as ocv
import time_stamp as ts
import os
import re
import numpy as np
import tifffile as tiff

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     RECEIVES A LIST OF IMAGES AND WRITES IT AS A SEQUENCE                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def write_maps(maps, base_nom):
    ts.start_function("writer")
    written = []
    for index, map in enumerate(maps):
        nom = base_nom + "_" + str(index) + ".png"
        written.append(nom)
        maxi = map.max()
        if (maxi > 1.0):
            pass
        else:
            map = map * 255
        entiers = map.astype(int)
        ocv.imwrite(nom, entiers)
    ts.tour("maps written", "writer")


def write_maps_tiff(maps, base_nom):

    written = []

    for index, map in enumerate(maps):
        nom = base_nom + "__" + str(index) + "__" + ".tiff"
        written.append(nom)
        
        ocv.imwrite(nom, map.astype("float"))

    return written

def write_maps_png(maps, base_nom):

    written = []

    for index, map in enumerate(maps):
        nom = base_nom + "__" + str(index) + "__" + ".png"
        written.append(nom)
        
        ocv.imwrite(nom, map * 255)
        del map

    return written

def write_tiff(map, base_nom, index):
    written = base_nom + "__" + str(index) + "__" + ".tiff"
    tiff.imwrite(written, map.astype('float'))

    return written

def write_png(map, base_nom, index):
    written = base_nom + "__" + str(index) + "__" + ".png"
    ocv.imwrite(written, map * 255)

    return written

def rename_sequence(direc, seq_ini):
    entries = os.listdir(direc)
    detached = []

    for entry in entries: # We save only the extension, in lower case 
        temp = entry.split('.')
        temp[-1] = temp[-1].lower()
        detached.append(temp[-1])

    index = 0 # Future index of the files

    for ext, entry in zip(detached, entries):
        os.rename(direc+entry, direc+seq_ini+"-"+str(index)+"."+ext)
        index += 1

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def open_sequence(direc, raw=False):
    try:
        entries = os.listdir(direc)
        sequence = []
        entries.sort(key=natural_keys)
        f_format = entries[0].split('.')
        tf = (f_format[-1] == "tif") or (f_format[-1] == "tiff")

        for entry in entries:
            if tf:
                img = tiff.imread(direc+entry)
                img = ocv.cvtColor(img, ocv.COLOR_RGB2BGR)
                if raw:
                    sequence.append(img)
                else:
                    sequence.append(img / 255.0)
            else:
                if raw:
                    sequence.append(ocv.imread(direc+entry, -1))
                else:
                    sequence.append(np.float32(ocv.imread(direc+entry, -1)) / 255.0)
            print(direc+entry+" opened.")
        
        return sequence
    except FileNotFoundError:
        return []

def reduce_res():
    seq  = open_sequence("testing7/")
    shp = (int(seq[0].shape[1] / 3), int(seq[0].shape[0] / 3))
    print(shp)

    for idx, img in enumerate(seq):
        buffer = ocv.normalize(ocv.resize(src=img, dsize=shp, fx=0.33, fy=0.33, interpolation=ocv.INTER_LINEAR), None, 0, 255, ocv.NORM_MINMAX, ocv.CV_8UC3)
        ocv.imwrite("testing7/stitch_panel-"+str(idx)+".png", buffer)

def profile_through_mono(c, l, sequence, file):
    val = []
    for img in sequence:
        val.append(int(img[c][l] * 255))
    
    f = open(file + ".plt", "w")
    for index, entry in enumerate(val):
        f.write(str(index) + "  " + str(entry)+"\n")
    f.close()
