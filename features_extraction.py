import cv2 as ocv
import numpy as np
import time_stamp as ts
import convolution as cvl
import utilities as ut
import Sequence as sq

g_index = 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            EXTRACTION OF CONTRAST MAP                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# The contrast map highlights the detailed areas 

def contrast_map(imIn):
    map = ocv.Laplacian(imIn, -1, ksize=3)
    return np.abs(map)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            EXTRACTION OF SATURATION MAP                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def saturation_map(imIn):
    b, g, r = ocv.split(imIn)
    avg = (b + g + r) / 3.0

    b_sq = np.power(b - avg, 2)
    g_sq = np.power(g - avg, 2)
    r_sq = np.power(r - avg, 2)

    del b
    del g
    del r

    sum_sqs = np.sqrt((b_sq + g_sq + r_sq) / 3.0)

    del b_sq
    del g_sq
    del r_sq

    return sum_sqs

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            EXTRACTION OF EXPOSURE MAP                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def well_exposedness(imIn, sigma):
    channels = [b, g, r] = ocv.split(imIn)
    output = []
    deno = pow(2 * sigma, 2)

    for channel in channels:
        c = channel
        c = -1.0 * (np.power((c - 0.5), 2) / deno)
        c = np.exp(c)
        output.append(c)
        del channel
    del channels

    render = output[0] * output[1] * output[2]

    return render

def expend(imIn, sigma):
    deno = pow(2 * sigma, 2)

    c = imIn
    c = -1.0 * (np.power((c - 0.5), 2) / deno)
    c = np.exp(c)
    return c

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            PROCESSING ALL THE FEATURES MAPS                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def combined_map(imIn, boost_contrast=1, boost_saturation=1, boost_exposure=1):
    global g_index
    imColor = imIn
    b,g,r = ocv.split(imIn)
    imGray = (b*0.11 + g*0.59 + r*0.3)
    del b
    del g
    del r

    map_contrast = np.power(contrast_map(imGray), boost_contrast)
    map_saturation = np.power(saturation_map(imColor), boost_saturation)
    map_exposure = np.power(well_exposedness(imColor, 0.2), boost_exposure)

    combined = (map_contrast, map_saturation, map_exposure)
    
    map_contrast = ocv.normalize(map_contrast, None, 0.0, 255.0, ocv.NORM_MINMAX)
    map_saturation = ocv.normalize(map_saturation, None, 0.0, 255.0, ocv.NORM_MINMAX)
    map_exposure = ocv.normalize(map_exposure, None, 0.0, 255.0, ocv.NORM_MINMAX)

    #combined = ocv.normalize(combined, None, 0.00, 1.0, ocv.NORM_MINMAX, ocv.CV_32FC2)

    ocv.imwrite("output/contrast-"+str(g_index)+".png", map_contrast)
    ocv.imwrite("output/saturation-"+str(g_index)+".png", map_saturation)
    ocv.imwrite("output/exposure-"+str(g_index)+".png", map_exposure)
    
    g_index += 1
    
    return combined

def gauss_spread(mat):
    fac = 1.0 / np.sqrt(2.0 * np.pi)
    mat = fac * np.exp(-0.5 * np.power((mat - 0.5), 2.0))
    return mat

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            PROCESSING OF FEATURES MAPS FOR A SEQUENCE OF IMAGE                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def weight_maps_sequence(sequence):

    # Containers of stacked maps (all maps are stored as a single image with a big depth)
    contrast = None
    saturation = None
    exposure = None

    initialization = True

    for img in sequence:
        cse = combined_map(img, 1.0, 1.0, 1.0)

        if(initialization):
            contrast = cse[0]
            saturation = cse[1]
            exposure = cse[2]
            initialization = False
        else:
            contrast = ocv.merge((contrast, cse[0]))
            saturation = ocv.merge((saturation, cse[1]))
            exposure = ocv.merge((exposure, cse[2]))
        
        del cse

    contrast =   ocv.normalize(contrast, None, 0.0001, 1.0, ocv.NORM_MINMAX, ocv.CV_32FC2)
    saturation = ocv.normalize(saturation, None, 0.0001, 1.0, ocv.NORM_MINMAX, ocv.CV_32FC2)
    exposure =   ocv.normalize(exposure, None, 0.0001, 1.0, ocv.NORM_MINMAX, ocv.CV_32FC2)
    maps = (np.power(contrast, 1.0) * np.power(saturation, 1.0) * np.power(exposure, 1.0))
    maps =   ocv.normalize(maps, None, 0.0001, 1.0, ocv.NORM_MINMAX, ocv.CV_32FC2)

    del contrast
    del saturation
    del exposure
    
    output = ocv.split(maps)
    
    del maps

    return output 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            BALANCED FEATURES MAPS (sum for each pixel == 1.0)                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def balance_maps(maps):
    global g_index
    buffer = np.zeros(maps[0].shape)
    for map in maps:
        buffer += map

    def_map = []
    for map in maps:
        def_map.append(map / buffer)
        del map
    del maps
    
    for map in def_map:
        ocv.imwrite("output/weights-"+str(g_index)+".png", map*255)
        g_index += 1

    return def_map

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            MERGING IMAGES SEQUENCE WITH WEIGHTS (deprecated)                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def assemble_layers(sequence, maps):
    buffer = np.zeros(sequence[0].shape)
    # L'image est un peu grise car tout le monde a un peu son mot à dire, il faudrait crank up l'écart entre les valeurs
    for img,map in zip(sequence, maps):
        factor = ocv.merge((map, map, map))
        buffer += (img * factor)
        del factor

    return buffer

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            CREATES NEW LAPLACIAN MAP FOR RENDER                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def blend_outputs(weights, sequence):
    laplacian = [] # Contains the Laplacian pyramid of each image from the bracketed set
    gaussian = [] # Contains the Gaussian pyramid of each weight map corresponding to images from the set
    blended_laplacian = [] # Will contain our new Laplacian pyramid, that carries blender for each level of detail

    # If there are N images in the bracketed set:
    #   - There are N pyramids in the 'gaussian' list
    #   - There are N-1 pyramids in the 'laplacian' list
    # So we will dump the last level of each of our gaussian pyramids

    laplacian = []
    ite_lapla = []
    gaussian = []
    ite_gauss = []

    for idx, img in enumerate(sequence):
        lapla = sq.Sequence("laplacian"+str(idx))
        lapla.write_seq(cvl.laplacian_pyramid(img))
        laplacian.append(lapla)
        ite_lapla.append(iter(lapla))
        
    for idx, weight in enumerate(weights):
        gauss = sq.Sequence("gaussian"+str(idx))
        gauss.write_seq(cvl.gaussian_pyramid(weight))
        gaussian.append(gauss)
        ite_gauss.append(iter(gauss))
 
    shape = sequence[0].shape # Shape of the desired canvas (== the shape of the biggest map)
    depth = laplacian[0].size() # Number of layers of the pyramid with the lowest number of floors
    

    for l in range(0, depth): # 'l' allows us to traverse the same layer of each pyramid at the same time.
        result = np.zeros(shape)
        for pyra_gauss, pyra_lapla in zip(ite_gauss, ite_lapla):
            gauss_work = next(pyra_gauss).reshape(shape[0], shape[1], 1) #ocv.merge((pyra_gauss[l], pyra_gauss[l], pyra_gauss[l]))
            lapla_pyr = next(pyra_lapla)
            result += (lapla_pyr * gauss_work)

        blended_laplacian.append(result)
    
    return blended_laplacian



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            COLLAPSING OF A LAPLACIAN MAP                                                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def collapse(blended_lap, coefs=None, brightness=1.0):
    result = np.zeros(blended_lap[0].shape)

    if(coefs == None):
        coefs = [1.0] * len(blended_lap)
        coefs[-1] = 1.5

    for coef, map in zip(coefs, blended_lap):
        result += (map * coef)
        del map
    del blended_lap

    result = ocv.normalize(result, None, 0.0, 1.0, ocv.NORM_MINMAX)
    result *= brightness
    thresh, result = ocv.threshold(result, 1.0, 1.0, ocv.THRESH_TRUNC)
    
    result = ocv.normalize(result, None, 0.0, 255.0, ocv.NORM_MINMAX, ocv.CV_8UC3)
    return result



#[(1.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.2, 0.8, 0.1), (0.1, 0.2, 0.2)]