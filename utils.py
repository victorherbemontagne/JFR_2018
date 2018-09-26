from tensorflow.python.keras.preprocessing import image as img

def load_config(path="config_for_outside.json"):
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