import pickle
import numpy as np
from sklearn import metrics
import head_orientation_lib

class SalEvaluation:
    topic_dict ={head_orientation_lib.DATASET1:['paris', 'roller', 'diving', 'timelapse', 'venise'],\
        head_orientation_lib.DATASET2:['0', '1', '2', '3', '4', '5_part1', '6_part1', '7', '8'],\
        head_orientation_lib.DATASET3:['coaster2_', 'coaster_', 'diving', 'drive', 'game', 'landscape', 'pacman', 'panel', 'ride', 'sport']}
    template = './data/saliency_ds{}_topic{}'
    dat_filepath = './data/saliency_evaldat'
   
    dat = {}
    
    bl_center_salmap = None
    #baseline_equator = None
    
    headoren = None
    salsal = None
    
    def __init__(self, headoren, salsal, verbose=False):
        #must supplement salat_headoren & saldat salsal objects when initialize
        
        self.headoren = headoren
        self.salsal = salsal
        
        print 'Read/Initialize samples for evaluation'
        try:
            dat_dict = pickle.load(open(self.dat_filepath))
        except:
            dat1 = self.create_sample(head_orientation_lib.DATASET1, verbose=verbose)
            dat2 = self.create_sample(head_orientation_lib.DATASET2, verbose=verbose)
            dat3 = self.create_sample(head_orientation_lib.DATASET3, verbose=verbose)
            
            np.random.shuffle(dat1)
            np.random.shuffle(dat2)
            np.random.shuffle(dat3)

            dat_dict = {head_orientation_lib.DATASET1: dat1[:1000], \
                        head_orientation_lib.DATASET2: dat2[:1000], \
                        head_orientation_lib.DATASET3: dat3[:1000]}
            pickle.dump(dat_dict, open(self.dat_filepath, 'wb'))
        self.dat = dat_dict
        
        print 'Initialize baselines (center & equator)'
        self.bl_center_salmap = salsal.create_saliency([[0.0, [-1.0, 0, 0], 0, 0]], 1)
        self.bl_center_salmap = self.bl_center_salmap*1.0 / self.bl_center_salmap.max()
        
        return
    
    def create_sample(self, dataset, verbose=False):
        #TODO: for each topic, 
        #open the ds file, 
        #create random idx then append the list of sample to result

        result = []

        

        topic_list = self.topic_dict[dataset]
        for topic in topic_list:
            if verbose==True:
                print 'reading {} - {}'.format(dataset, topic)
            
            filepath = self.template.format(dataset, topic)
            dat = pickle.load(open(filepath))      #load the saliency dataset file
            idx_randlist = np.arange(len(dat))
            np.random.shuffle(idx_randlist)
            idx_randlist = idx_randlist[:300]

            for idx in idx_randlist:
                timestamp, fix_list, sal_map = dat[idx]
                result.append([topic, timestamp, fix_list, sal_map])
        return result
    
    def get_negative_sample(self, dataset, topic, TOP=3):
        #return five sample same dataset, different topic
        temp = []
        for top, t, fixlist, salmap in self.dat[dataset]:
            if topic != top:
                temp.append([top, t, fixlist, salmap])
        np.random.shuffle(temp)
        return temp[:TOP]
    
    def get_negative_fixations(self, dataset, topic):
        neg_list = self.get_negative_sample(dataset, topic)
        neg_vlist = []
        for top, t, fixlist, salmap in neg_list:
            neg_vlist += fixlist
        return neg_vlist
    
    def model_blequator(self, pixel_list):
        #the equator assign saliency chance to list of pixel coordination (human fixation)
        return [1.0 - np.abs(head_orientation_lib.geoy_to_phi(hi, head_orientation_lib.H))/90.0 for (hi, wi) in pixel_list]
    
    def model_blcircle(self, pixel_list):
        return [self.bl_center_salmap[hi, wi] for (hi, wi) in pixel_list]
    
    def sauc(self, dataset, topic, vpos_list):
        vneg_list = self.get_negative_fixations(dataset, topic)

        #now I have positive, negative fixation, need to create saliency list
        fixpos_list = [(0, v, 0, 0) for v in vpos_list]
        fixneg_list = [(0, v, 0, 0) for v in vneg_list]
        fposmap = self.headoren.create_fixation_map(fixpos_list, dataset)
        fnegmap = self.headoren.create_fixation_map(fixneg_list, dataset)

        np.random.shuffle(fixpos_list)
        npos= len(fixpos_list)/3
        fixposeval_list = fixpos_list[:npos]
        fixpostrain_list = fixpos_list[npos:]
        val_salmap = self.salsal.create_saliency(fixposeval_list, dataset)
        train_fmap = self.headoren.create_fixation_map(fixpostrain_list, dataset)
        geoxy_train = zip(*np.where(train_fmap==1))
        geoxy_neg = zip(*np.where(fnegmap==1))

        #val_salmap must be evaluated against geoxy_train + geoxy_neg

        y_pred = [val_salmap[hi, wi] for (hi, wi) in geoxy_train]
        y_pred += [val_salmap[hi, wi] for (hi, wi) in geoxy_neg]

        y_true = [1 for item in geoxy_train]
        y_true += [0 for item in geoxy_neg]

        return metrics.roc_auc_score(y_true, y_pred)

        

    def cc(self, salmap, fixmap):
        return
    
    def nss(self, salmap, fixmap):
        prediction = salmap - np.mean(salmap)
        prediction = prediction / np.std(prediction)
        return np.mean(prediction[fixmap==1])