import sys
import argparse
import cv2
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from torch.autograd import Function
sys.path.append('/data/bigbone/amorales/Deep_Learning/pytorch')
import vision2
from vision2.torchvision import models
from os.path import join

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if type(module) == type(torch.nn.Dropout()):
                x = module(x)
                for name2, module2 in self.model[1]._modules.items():
                    x = module2(x)
                    if name2 in self.target_layers:
                        x.register_hook(self.save_gradient)
                        outputs += [x]
                        return outputs, x                
            else:
                x = module(x)
                if name in self.target_layers:
                    x.register_hook(self.save_gradient)
                    outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                if len(module._modules) == 2:
                    if module == self.feature_module:
                        target_activations, x = self.feature_extractor(x)
                    else:
                        x = module(x)
                else:
                    if module == self.feature_module:
                        target_activations, x = self.feature_extractor(x)
                    elif "avgpool" in name.lower():
                        x = module(x)
                        x = x.view(x.size(0),-1)
                    else:
                        x = module(x)
            else:
                    if module == self.feature_module:
                        target_activations, x = self.feature_extractor(x)
                    elif "avgpool" in name.lower():
                        x = module(x)
                        x = x.view(x.size(0),-1)
                    else:
                        x = module(x)
        
        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))
    cv2.imwrite("mask.jpg", np.uint8(255 * mask))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


# class GuidedBackpropReLU(Function):

#     @staticmethod
#     def forward(self, input):
#         positive_mask = (input > 0).type_as(input)
#         output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
#         self.save_for_backward(input, output)
#         return output

#     @staticmethod
#     def backward(self, grad_output):
#         input, output = self.saved_tensors
#         grad_input = None

#         positive_mask_1 = (input > 0).type_as(grad_output)
#         positive_mask_2 = (grad_output > 0).type_as(grad_output)
#         grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
#                                    torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
#                                                  positive_mask_1), positive_mask_2)

#         return grad_input


# class GuidedBackpropReLUModel:
#     def __init__(self, model, use_cuda):
#         self.model = model
#         self.model.eval()
#         self.cuda = use_cuda
#         if self.cuda:
#             self.model = model.cuda()

#         def recursive_relu_apply(module_top):
#             for idx, module in module_top._modules.items():
#                 recursive_relu_apply(module)
#                 if module.__class__.__name__ == 'ReLU':
#                     module_top._modules[idx] = GuidedBackpropReLU.apply
                
#         # replace ReLU with GuidedBackpropReLU
#         recursive_relu_apply(self.model)

#     def forward(self, input):
#         return self.model(input)

#     def __call__(self, input, index=None):
#         if self.cuda:
#             output = self.forward(input.cuda())
#         else:
#             output = self.forward(input)

#         if index == None:
#             index = np.argmax(output.cpu().data.numpy())
#             # print(index)

#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0][index] = 1
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         if self.cuda:
#             one_hot = torch.sum(one_hot.cuda() * output)
#         else:
#             one_hot = torch.sum(one_hot * output)

#         # self.model.features.zero_grad()
#         # self.model.classifier.zero_grad()
#         one_hot.backward(retain_graph=True)

#         output = input.grad.cpu().data.numpy()
#         output = output[0, :, :, :]

#         return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--sampling', type=str, default='NewSphere',
                        help='Input image sampling')
    parser.add_argument('--norm', type=str, default='GroupNorm',
                        help='Input image normalization')
    parser.add_argument('--task', type=str, default='Pain',
                        help='Input image task')
    parser.add_argument('--split', type=str, default='Hold',
                        help='Input image split')
    parser.add_argument('--bone', type=str, default='Femur',
                        help='Input image bone')
    parser.add_argument('--biomarker', type=str, default='Thickness',
                        help='Input image biomarker')
    parser.add_argument('--predtype', type=str, default='truepositives',
                        help='Input image prediction type')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    
    data_dir = '/data/knee_mri5/Alejandro/BiomarkerFusionAll/NewSphere/GroupNorm/'
    save_path = '../data/'
    pd_path = '/data/knee_mri5/Alejandro/BiomarkerFusion/data'
    sampling = args.sampling
    norm = args.norm
    task = args.task
    split = args.split
    bone = args.bone
    biomarker = args.biomarker
    predtype = args.predtype
    pd_path = join(pd_path, sampling, norm, biomarker, bone, task)
    log_dir = pd_path.replace('on/data', 'on/logs')
    pdpath = join(log_dir.split(sampling)[0], task.lower() + 'performance.csv')
    checkpointpd = pd.read_csv(pdpath, index_col=['Sampling', 'Normalization', 'Bone', 'Incidence', 'Fusion'])
    best_checkpoint_path = checkpointpd.loc[(sampling, norm, bone, task, biomarker), 'Checkpoint']
    
    load_path = '/data/knee_mri3/Alejandro/biomarker_fusion_VBR/testagebmisex.mat'
    preddict = sio.loadmat(load_path)

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    num_classes = 2

    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    best_checkpoint_load = torch.load(best_checkpoint_path, map_location='cpu')
    model.load_state_dict(best_checkpoint_load['model_state_dict'])
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)

    target_index = None
    patientnames = [item[1:] for item in list(preddict.keys())[3:]]
    agesvalues = np.array([preddict[item]['Age'].item().item() for item in list(preddict.keys())[3:]])
    bmivalues = np.array([preddict[item]['BMI'].item().item() for item in list(preddict.keys())[3:]])
    sexvalues = np.array([preddict[item]['Sex'].item().item() for item in list(preddict.keys())[3:]])
    painvalues = np.array([preddict[item]['Pain_Status'].item().item() for item in list(preddict.keys())[3:]])
    klvalues = np.array([preddict[item]['KL'].item().item() for item in list(preddict.keys())[3:]])
    oavalues = (klvalues > 1).astype(int)

    cam_array = np.zeros((224,224,4,len(patientnames)), dtype=np.uint8())
    for num,image_name in enumerate(patientnames):
        image_path = join(data_dir, biomarker, bone, image_name + '.png')
        img = cv2.imread(image_path, 1)
        cam_array[..., :3, num] = img
        img = np.float32(cv2.resize(img, (224, 224)))/255
        input = preprocess_image(img)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        mask = grad_cam(input,target_index)
        cam_array[..., -1, num] = np.uint8(255*mask)
    
    cam_dict = {}
    cam_dict['gradcams'] = cam_array
    cam_dict['names'] = patientnames
    cam_dict['age'] = agesvalues
    cam_dict['bmi'] = bmivalues
    cam_dict['sex'] = sexvalues
    cam_dict['pain'] = painvalues
    cam_dict['kl'] = klvalues
    cam_dict['oa_status'] = oavalues

    mat_name = join(save_path, ''.join([bone,biomarker,task,'.mat']))
    sio.savemat(mat_name, cam_dict)