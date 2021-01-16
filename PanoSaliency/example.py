import pickle

if __name__ == "__main__":
    #TODO: a simple example to load a saliency dataset file
    #RETURN: the values of the first record of the file. The record includes: timestamp, fixation_list, and saliency_map
    #note: this script assumes the dataset has been download from the LINK provided in the ./data folder
    
    #load the dataset file named `saliency_ds1_topicparis` (ds=1, video=paris). Assuming the dataset file is in ./data folder
    data = pickle.load(open('./data/saliency_ds1_topicparis'))
    
    #access the first record
    timestamp, fixation_list, saliency_map = data[0]
    
    #print out the values of fields in the first record
    print timestamp
    print fixation_list
    print saliency_map
    
    
