import os
from tqdm import tqdm
import numpy as np

def load_config(path):
    import json
    f = open(path)
    config = json.load(f)
    return(config)
    


def get_eval_data(path_test):
    all_pic_test = [path for path in os.listdir(path_test)]
    batch = []
    for pic_test in all_pic_test:
        glass = pic
        path_pic = os.path.join(path_test,pic_test)
        pic = img.load_img(pic_test,target_size=(256,256))
        pic = img.img_to_array(pic)
        batch.append(pic)
    if len(batch) != batch_size:
        for k in range(batch_size-len(batch)):
            batch.append(np.zeros((256,256)))
    pred_test = sess.run([output_net],feed_dict={"input_tensor:0":batch})
    return(pred_test)
    
def load_from_path(path):
    list_elmts = os.listdir(path)
    elmts = []
    print("Loading elements from ",path)
    for path_elmt in tqdm(list_elmts):
        path_util = os.path.join(path,path_elmt)
        elmt = np.load(path_util)
        elmts.append(elmt)
    return(elmts)


def save_in_path(path,elmts):
    if not os.path.isdir(path):
        os.makedirs(path)
    print("Saving in path ",path)
    for index,elmt in tqdm(enumerate(elmts)):
        path_save = os.path.join(path,str(index)+".npy")
        np.save(path_save,elmt)
        
    