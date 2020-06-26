import cv2 as ocv
import numpy as np
import time

start_time = 0.0
previous = 0.0
laps = []

# On suppose que l'image passée en argument a une taille qui marchera avec le filtre et le shift voulu

def convolve(imIn, kernel, shift, pad):
    tour("convolve start")
    # Isolation de la forme de l'image ET du filtre
    (i_height, i_width) = imIn.shape
    (k_height, k_width) = kernel.shape

    # Copie de travail de l'image avec les bords qui recopient les valeurs excentrées
    img_work = ocv.copyMakeBorder(
        imIn, 
        pad, 
        pad, 
        pad, 
        pad, 
        ocv.BORDER_REPLICATE
    )

    # Nouvelle taille d'image calculée selon le shift
    new_shape = ((np.array(imIn.shape) + (2 * pad) - k_width) / shift) + 1
    new_shape = new_shape.astype(int)

    # Canevas vide
    output = np.zeros(new_shape, dtype="float32")

    # Convolution
    l_out, c_out = 0, 0
    decal = int((k_width - 1) / 2)

    for l in range(pad, i_height+pad, shift):
        for c in range(pad, i_width+pad, shift):
            region = img_work[l - decal:l + decal + 1, c - decal:c + decal + 1].astype(float) # Region isolee dand l'image, de la taille du filtre
            mat = region * kernel
            output[l_out][c_out] = np.sum(mat)
            c_out += 1
        c_out = 0
        l_out += 1

    tour("convolve done")

    return output


def gaussian_pyramid_mono(imIn, threshold=10, stop=5):
    tour("gaussian mono start")
    # Acquisition des données de l'image de base
    (i_height, i_width) = imIn.shape
    nb_pixels = i_height * i_width
    maps = []
    maps.append(imIn)

    # Filtre flou médian
    filtre = np.ones((5, 5)) / (5 * 5)
    count = 0

    while((nb_pixels > threshold) and (count < stop)): # Sécurité de boucle infinie
        produced = convolve(maps[-1], filtre, 2, 2)
        maps.append(produced)
        (i_height, i_width) = produced.shape
        nb_pixels = i_height * i_width
        count += 1

    tour("gaussian mono done")

    return maps

# Fonction qui écrit toutes les images qui lui sont passées sous la forme d'un tableau
def write_maps(maps, base_nom):
    tour("starting writing maps")
    for index, map in enumerate(maps):
        nom = base_nom + "_" + str(index) + ".png"
        entiers = map.astype(int)
        ocv.imwrite(nom, entiers)
    tour("maps written")

# Fonction qui étend les maps gaussiennes à la taille de l'image génératrice
def expend(maps):
    tour("expending ...")
    target = maps[0].shape
    output = []
    factor = 1.0

    for map in maps:
        canvas = np.zeros(target)

        for l in range(0, target[0]):
            for c in range(0, target[1]):
                index_l = int(l * factor)
                index_c = int(c * factor)

                canvas[l][c] = map[index_l][index_c]

        output.append(canvas)
        factor /= 2.0
    tour("expending done")
    return output



def gaussian_pyramid_old(imIn):
    tour("starting batch gaussian pyramid")
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
        maps = gaussian_pyramid_mono(canal, 20, 10)
        maps = expend(maps)
        list_maps.append(maps)
    
    final_maps = []
    canvas_size = list_maps[0][0].shape

    if(colored):   
        for b,g,r in zip(list_maps[0], list_maps[1], list_maps[2]):
            canvas = ocv.merge((b,g,r))
            final_maps.append(canvas)
        pass
    else:
        final_maps = list_maps[0]
        pass
    tour("batch gaussian done")
    return final_maps


def gaussian_pyramid(imIn):
    tour("starting batch gaussian pyramid")
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
            new_maps.append(ocv.resize(map, (sp[1],sp[0]), fx=factor, fy=factor, interpolation=ocv.INTER_NEAREST))
            factor *= 2
        list_maps.append(new_maps)
    
    final_maps = []
    canvas_size = list_maps[0][0].shape

    if(colored):   
        for b,g,r in zip(list_maps[0], list_maps[1], list_maps[2]):
            canvas = ocv.merge((b,g,r))
            final_maps.append(canvas)
        pass
    else:
        final_maps = list_maps[0]
        pass
    tour("batch gaussian done")
    return final_maps


def testing():
    # Image test fake 
    img_sz = 7
    #img = np.arange(img_sz * img_sz).reshape(img_sz, img_sz)
    img = np.ones((img_sz, img_sz))
    img *= 2
    print(img)
    print("")

    # Filtre test fake
    filtre = np.ones((5, 5)) / (5 * 5)
    print(filtre)
    print("")

    # Settings
    shift = 2
    pad = 2

    output = convolve(img, filtre, shift, pad)
    print(output)


def testing_2():
    img = ocv.imread("data/basketball1.png", 0)
    filtre = np.ones((5, 5)) / (5 * 5)
    shift = 2
    pad = 2

    output = convolve(img, filtre, shift, pad)
    ocv.imwrite("basket_gauss.png", output)

def testing_3():
    img = ocv.imread("data/basketball1.png", 0)
    maps = gaussian_pyramid_mono(img, 20, 10)
    write_maps(maps, "g_map_raw")
    maps = expend(maps)
    write_maps(maps, "g_map_exp")

def testing_4():
    img = ocv.imread("data/basketball1.png", 0)
    img2 = ocv.imread("data/baboon.jpg", -1)
    maps1 = gaussian_pyramid(img)
    maps2 = gaussian_pyramid(img2)
    write_maps(maps1, "g_map_bw")
    write_maps(maps2, "g_map_rgb")
    pass

def disp_laps():
    for lap in laps:
        print(lap)

def write_laps(file):
    f = open(file, "w")
    for lap in laps:
        f.write(lap[0] + str(lap[1]) +'\n')
    f.close()

def tour(txt):
    global previous
    if(len(txt) > 32):
        print("Warning: Text too long")
        txt = txt[0:32]
    
    missing = 32 - len(txt)
    missing = " " * missing

    elapsed = time.time() - previous
    elapsed = round(elapsed, 4)
    laps.append((txt + missing + "|   ", elapsed))
    previous = time.time()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            MAIN()                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
    testing_4()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            TRIGGER                                                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


start_time = time.time()
previous = start_time
tour("start")
main()
tour("end")
laps = sorted(laps, key=lambda tup: tup[1], reverse=True)
write_laps("times_1.log")
print("--- %s seconds ---" % (time.time() - start_time))