import cv2 as ocv
import numpy as np
import time_stamp as ts
import utilities as ut
import cupy as cp

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     CONVOLUTING AN IMAGE WITH A FILTER                                                                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def convolve(imIn, kernel, shift, pad):
    ts.start_function("convolve")
    imOut = ocv.filter2D(imIn, -1, kernel, borderType=ocv.BORDER_REPLICATE)
    ts.tour("Convolution done", "convolve")
    return imOut

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     SUBSAMPLE THE IMAGE IN PARAMETER                                                                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def sample(imIn, rate=2):
    ts.start_function("sample")
    (i_h, i_w) = imIn.shape
    (o_h, o_w) = (i_h // 2, i_w // 2)
    
    canvas = ocv.resize(imIn, (o_w, o_h), 1.0/rate, 1.0/rate, interpolation=ocv.INTER_NEAREST)
    ts.tour("Subsampling done", "sample")
    
    return canvas

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     BUILDS A GAUSSIAN PYRAMID OF A SINGLE CHANNEL IMAGE                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def gaussian_pyramid_mono(imIn, threshold=10, stop=5):
    ts.start_function("gaussian")
    # Acquisition des données de l'image de base
    (i_height, i_width) = imIn.shape
    nb_pixels = i_height * i_width
    maps = []
    maps.append(imIn)

    # Filtre flou médian
    filtre = np.ones((5, 5)) / (5 * 5)
    # Filtre flou gaussien
    #filtre = np.array((1, 2, 1, 2, 4, 2, 1, 2, 1)).reshape((3,3)) / 16.0
    count = 0

    while((nb_pixels > threshold) and (count < stop)): # Sécurité de boucle infinie
        produced = convolve(maps[-1], filtre, 1, 1)
        produced = sample(produced)
        maps.append(produced)
        (i_height, i_width) = produced.shape
        nb_pixels = i_height * i_width
        count += 1

    ts.tour("Gaussian mono done", "gaussian")

    return maps


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     BUILDS A GAUSSIAN PYRAMID FOR ANY IMAGE                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def gaussian_pyramid(imIn, complete=True):
    ts.start_function("gaussian pyramid")
    # Forme de l'image, profondeur comprise
    sp = imIn.shape
    # Canaux de l'image Gr || BGR
    canaux = []

    colored = (len(sp) >= 3) and (sp[2] > 1)

    if(not colored):
        # Image en niveaux de gris
        canaux.append(imIn)
    else:
        # 3 canaux de couleur
        b, g, r = ocv.split(imIn)
        canaux.append(b)
        canaux.append(g)
        canaux.append(r)

    # Ne contient qu'une liste de maps pour mono channel
    # Contient trois listes de maps pour tri channels
    list_maps = []
    
    for canal in canaux:
        factor = 1.0
        maps = gaussian_pyramid_mono(canal, 10, 5)
        new_maps = []
        for map in maps:
            new_maps.append(ocv.resize(map, (sp[1],sp[0]), fx=factor, fy=factor, interpolation=ocv.INTER_LINEAR)) #interpolation=ocv.INTER_NEAREST
            factor *= 2
        list_maps.append(new_maps)
        del canal
    del canaux
    
    final_maps = []
    canvas_size = list_maps[0][0].shape

    if(not complete):
        return (colored, list_maps)
    else:
        ts.tour("processing gaussian batch", "gaussian pyramid")
        if(colored):   
            for b,g,r in zip(list_maps[0], list_maps[1], list_maps[2]):
                canvas = ocv.merge((b,g,r))
                del b
                del g
                del r
                final_maps.append(canvas)
                ts.tour("batch gaussian done RGB", "gaussian pyramid")
            del list_maps[:]
        else:
            final_maps = list_maps[0]
            ts.tour("batch gaussian done B&W", "gaussian pyramid")
        return final_maps

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     BUILDS A LAPLACIAN PYRAMID FOR ANY IMAGE                                                                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def laplacian_pyramid(img, bw=False):
    maps = gaussian_pyramid(img)
    maps_lapla = []

    ts.start_function("laplacian")
    
    for index in range(len(maps) - 1):
        lapla = maps[index] - maps[index + 1] + 0.5
        canvas = lapla #ocv.normalize(lapla, None, 0.0, 1.0, ocv.NORM_MINMAX, ocv.CV_32FC3) # lapla #/!\
        maps_lapla.append(canvas)
    maps_lapla.append(maps[-1]) # Adding last gaussian map at the end of the list to avoid data loss when the pyramid is collapsed
    del maps[0:-1]
    ts.tour("Laplacian done", "laplacian")
    return maps_lapla

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            TESTING                                                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def testing_4():
    img = ocv.imread("data/basketball1.png", 0)
    img2 = ocv.imread("data/baboon.jpg", -1)

    ts.tour("~~~ START LAPLACIAN ~~~", "testing")
    maps0 = gaussian_pyramid(img)
    ts.tour("GAUSSIAN IMG B&W", "testing")
    maps3 = gaussian_pyramid(img2)
    ts.tour("GAUSSIAN IMG RGB", "testing")

    maps1 = laplacian_pyramid(img)
    ts.tour("LAPLACIAN IMG B&W", "testing")
    maps2 = laplacian_pyramid(img2)
    ts.tour("LAPLACIAN IMG RGB", "testing")

    ut.write_maps(maps0, "output/gaussian/g_map_bw")
    ut.write_maps(maps3, "output/gaussian/g_map_rgb")
    ut.write_maps(maps1, "output/laplacian/g_map_bw")
    ut.write_maps(maps2, "output/laplacian/g_map_rgb")


