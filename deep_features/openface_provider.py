# OpenFace provider
# If vgg: provides with the VGG Net output; Specify on which layer is needed
# If image: provides with the image paths
# the return is a dictionary

import os.path as path
import glob
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.resnet50 import preprocess_input as res_preprocess
from keras.models import Model, load_model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.layers import Flatten
from tqdm import tqdm 
import os

openface_faces_dir = '/newdisk/AVEC2018/downloaded_dataset/openface_faces'
videos =  os.listdir(openface_faces_dir)
class Extractor():
    def __init__(self, layer = 'fc6', model = 'vgg16'):
        self.model = model
        self.layer = layer
        # Get model with pretrained weights. model: vgg16, resnet50
        vgg_model = VGGFace(model = model)
        if self.model == 'vgg16':
            # We'll extract features at the fc6 layer
            if layer=='fc6' or 'fc7':
                self.model = Model(
                        inputs=vgg_model.input,
                        outputs=vgg_model.get_layer(layer).output
                        )
            else:
                # convolution layer output needs to be flattened
                vgg_out = vgg_model.get_layer(layer).output
                out = Flatten(name='flatten')(vgg_out)
                self.model = Model(
                        inputs=vgg_model.input,
                        outputs=out
                        )
        elif self.model == 'resnet50':
            # layer=avg_pool
            resent_out = vgg_model.get_layer(layer).output
            out = Flatten(name='flatten')(resent_out)
            self.model = Model(
                    inputs=vgg_model.input,
                    outputs=out
                    )
        
    def extract(self, image_path):
        
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) 
        if self.model == 'vgg16':
            x = vgg_preprocess(x)
        elif self.model == 'resnet50':
            x = res_preprocess(x)
        # Get the prediction.
        features = self.model.predict(x)
        features = features[0]
        return features

def openface_provider(model, layer, is_image):
    """
    model: the model choose to extract features, one of 'vgg16' or 'resent50'
    layer: which layer is chosen to be feature output
    is_image : if true, the output will only be image paths
    """
    output_dictionary = {}
    if is_image:
        print('Return the image set ...')
        for video_name in tqdm(videos):
            pass
            if path.isdir(path.join(openface_faces_dir, video_name)):
                #check whether it is directory
                image_dir = path.join(openface_faces_dir, video_name, video_name+'_aligned')
                image_paths = sorted(glob.glob(path.join(image_dir, '*.bmp')))
                output_dictionary[video_name] = image_paths
    else:
        print('Feature extracting ...')
        extractor = Extractor(layer, model)
        for video_name in tqdm(videos):
            pass
            if path.isdir(path.join(openface_faces_dir, video_name)):
                #check whether it is directory
                image_dir = path.join(openface_faces_dir, video_name, video_name+'_aligned')
                image_paths = sorted(glob.glob(path.join(image_dir, '*.bmp')))
                paths_to_features = []
                for img in image_paths:
                    frame_num = img.split('/')[-1].split('.')[0].split('_')[-1]
                    output_file_name = model+'_'+layer+'_'+frame_num+'.npy'
                    output_file_path = path.join(image_dir, output_file_name)
                    if not path.exists(output_file_path):
                        feat = extractor.extract(img)
                        np.save(output_file_path, feat)
                    paths_to_features.append(output_file_path)
                output_dictionary[video_name] = paths_to_features
        return output_dictionary
#import pdb
#pdb.set_trace()
#openface_provider(None, None, is_image=True)
#openface_provider('vgg16','fc6', False)
#openface_provider('vgg16','pool5', False)

