import cv2 as ocv
import numpy as np
import time_stamp as ts


def convolve(imIn, kernel, shift, pad):
    ts.start_function("convolve")
    imOut = ocv.filter2D(imIn, -1, kernel, borderType=ocv.BORDER_REPLICATE)
    ts.tour("Convolution done", "convolve")
    return imOut


def sample(imIn, rate=2):
    ts.start_function("sample")
    (i_h, i_w) = imIn.shape
    (o_h, o_w) = (i_h // 2, i_w // 2)
    
    canvas = ocv.resize(imIn, (o_w, o_h), 1.0/rate, 1.0/rate, interpolation=ocv.INTER_NEAREST)
    ts.tour("Subsampling done", "sample")
    
    return canvas

def gaussian_pyramid_mono(imIn, threshold=10, stop=5):
    ts.start_function("gaussian")
    # Acquisition des données de l'image de base
    (i_height, i_width) = imIn.shape
    nb_pixels = i_height * i_width
    maps = []
    maps.append(imIn)

    # Filtre flou médian
    filtre = np.ones((5, 5)) / (5 * 5)
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

# Fonction qui écrit toutes les images qui lui sont passées sous la forme d'un tableau
def write_maps(maps, base_nom):
    ts.start_function("writer")
    for index, map in enumerate(maps):
        nom = base_nom + "_" + str(index) + ".png"
        entiers = map.astype(int)
        ocv.imwrite(nom, entiers)
    ts.tour("maps written", "writer")


def gaussian_pyramid(imIn, complete=True):
    ts.start_function("gaussian pyramid")
    # Forme de l'image, profondeur comprise
    sp = imIn.shape
    # Canaux de l'image Gr || BGR
    canaux = []

    colored = len(sp) >= 3

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
        maps = gaussian_pyramid_mono(canal, 20, 10)
        new_maps = []
        for map in maps:
            new_maps.append(ocv.resize(map, (sp[1],sp[0]), fx=factor, fy=factor, interpolation=ocv.INTER_LINEAR)) #interpolation=ocv.INTER_NEAREST
            factor *= 2
        list_maps.append(new_maps)
    
    final_maps = []
    canvas_size = list_maps[0][0].shape

    if(not complete):
        return (colored, list_maps)
    else:
        ts.tour("processing gaussian batch", "gaussian pyramid")
        if(colored):   
            for b,g,r in zip(list_maps[0], list_maps[1], list_maps[2]):
                canvas = ocv.merge((b,g,r))
                final_maps.append(canvas)
                ts.tour("batch gaussian done RGB", "gaussian pyramid")
            pass
        else:
            final_maps = list_maps[0]
            ts.tour("batch gaussian done B&W", "gaussian pyramid")
        return final_maps



def laplacian_pyramid(img, bw=False):
    gaussian = gaussian_pyramid(img, False)
    colored = gaussian[0]
    maps = gaussian[1]
    maps_lapla = []

    ts.start_function("laplacian")
    
    if(colored):
        for index in range(len(maps[0]) - 1):
            lapla_b = maps[0][index] - maps[0][index + 1] + 128
            lapla_g = maps[1][index] - maps[1][index + 1] + 128
            lapla_r = maps[2][index] - maps[2][index + 1] + 128
            canvas = None
            if(bw):
                canvas = lapla_b + lapla_g + lapla_r
            else:
                canvas = ocv.merge((lapla_b,lapla_g,lapla_r))
            
            canvas = ocv.normalize(canvas, None, 0, 255, ocv.NORM_MINMAX)
            maps_lapla.append(canvas)
    else:
        iv = 0
        for index in range(len(maps[0]) - 1):
            lapla = maps[0][index] - maps[0][index + 1] + 128
            canvas = ocv.normalize(lapla, None, 0, 255, ocv.NORM_MINMAX)
            maps_lapla.append(canvas)
    
    if(colored):
        ts.tour("Laplacian colored done", "laplacian")
    else:
        ts.tour("Laplacian b&w done", "laplacian")
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

    write_maps(maps0, "output/gaussian/g_map_bw")
    write_maps(maps3, "output/gaussian/g_map_rgb")
    write_maps(maps1, "output/laplacian/g_map_bw")
    write_maps(maps2, "output/laplacian/g_map_rgb")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            MAIN()                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
    testing_4()
    #liste_cos(7, 7, 5, 5)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            TRIGGER                                                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


ts.tour("--- Start ---", "main")
main()
ts.tour("--- End ---", "main")

#ts.sort_laps(ts.TimeSort.LONGER)
ts.write_laps("time_stamp_new_convo.log")

