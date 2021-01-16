import os
import numpy as np
import pickle
import header
import sys


import head_orientation_lib
import saldat_head_orientation
import saldat_saliency

if __name__ == "__main__":
    #specify dataset & video name to extract
    TOPIC = sys.argv[2]#for 6, modify 2 places, for loop vlength, and output file with _part
    DELTA = 0.06
    
    dataset = int(sys.argv[1])#saldat_head_orientation.HeadOrientation._DATASET2
    topic = TOPIC#dataset 1: paris, roller, venise,diving,timelapse, 
                   #dataset 2: '0', '1', '2', '3', '4', '5', '6', '7', '8'
                   #dataset 3: ['coaster2_', 'coaster_', 'diving', 'drive', 'game', 'landscape', 'pacman', 'panel', 'ride', 'sport']
    #specify output address to store the saliency maps
    
    #initialize head_oren
    print ("generating saliency maps for ds={}, topic={}".format(dataset, TOPIC))
    dirpath1 = header.dirpath1#u'./data/head-orientation/dataset1'
    dirpath2 = header.dirpath2#u'./data/head-orientation/dataset2/Experiment_1'
    dirpath3 = header.dirpath3#u'./data/head-orientation/dataset3/sensory/orientation'
    ext1 = header.ext1
    ext2 = header.ext2
    ext3 = header.ext3
    headoren = saldat_head_orientation.HeadOrientation(dirpath1, dirpath2, dirpath3, ext1, ext2, ext3)
    #initialize 
    var = 20
    salsal = saldat_saliency.Fixation(var)
    
    dirpath, filename_list, f_parse, f_extract_direction = headoren.load_filename_list(dataset, topic)
    series_ds = headoren.load_series_ds(filename_list, f_parse)
    vector_ds = headoren.headpos_to_headvec(series_ds, f_extract_direction)
    print(len(vector_ds[47]))
    vector_ds = headoren.cutoff_vel_acc(vector_ds, dataset=dataset)

    print(len(vector_ds))
    print(len(vector_ds[0]))
    print(len(vector_ds[47]))

    _, vlength, _, _ = head_orientation_lib.topic_info_dict[topic]
    # saliency_ds = []
    pixel_ds = []
    for t in np.arange(1, vlength, DELTA):#0.06
    #for t in np.arange( vlength/2, vlength, DELTA):
        try:
            pixel_list, fixation_list = headoren.get_fixation(vector_ds, t)
            # v_list = [item[1] for item in fixation_list]

            print (t, len(fixation_list))
            pixel_ds.append(pixel_list)
            # fmap0 = headoren.create_fixation_map(fixation_list, dataset)
            # heat_map0 = salsal.create_saliency(fixation_list, dataset)
            # saliency_ds.append([t, v_list, heat_map0])
        
        except:
            continue
    # pickle.dump(saliency_ds, open('./data/saliency_ds{}_topic{}'.format(dataset, topic), 'wb'))
    pickle.dump(pixel_ds, open('./data/pixel_list_ds{}_topic{}'.format(dataset, topic), 'wb'))
