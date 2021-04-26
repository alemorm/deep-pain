# Finetuning Torchvision Models
# =============================
# **Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import click
import pickle
sys.path.append('/data/bigbone/amorales/Deep_Learning/pytorch')
import vision2
import torch.optim as optim
from os import listdir
from os.path import join, isfile, isdir, dirname
from torch.optim import lr_scheduler
from vision2.torchvision import datasets, models, transforms
from tqdm import tqdm
from decimal import Decimal
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--config_file', '-c',
    type=click.File('r'),
    help='Config file with the training parameters such as GPU ID, data directory, batch size and model type.'
         'If not provided, the default settings except the data directory will be used.',
)
@click.option(
    '--data_dir',
    '-d',
    default='/data/knee_mri5/Alejandro/BiomarkerFusion/data/',
    help='Directory with the image data according to the ImageFolder dataset format for classification. If not provided, an error will be raised.',
)
@click.option(
    '--knee_type',
    '-k',
    default='Femur',
    type=click.Choice(['Femur', 'Tibia', 'Patella']),
    help='Knee type for the model, options are "combined", "Femur", "Tibia", and "Patella".'
         'The default is "Femur" if no knee type is specified.',
)
@click.option(
    '--biomarker',
    '-x',
    default='Bone',
    type=click.Choice(['Bone', 'Thickness', 'T2Fusion', 'AllFusion', 'BoneCartFusion', 'CartFusion']),
    help='Biomarker type for the model, options are "Bone", "Thickness", "T2Fusion", "AllFusion", "BoneCartFusion" and "CartFusion".'
         'The default is "Bone" if no biomarker type is specified.',
)
@click.option(
    '--progression',
    '-i',
    default='DiagnosisNorm',
    type=click.Choice(['4year', '4year_worms', '4year_worms_norm','Diagnosis', 'DiagnosisNorm', 'DiagnosisAll', 'Pain']),
    help='The specific OA progression model to be used.'
         'The default model is "4year" progression if no model is specified.',
)
@click.option(
    '--sampling',
    default='NewSphere',
    type=click.Choice(['OldSphere', 'NewSphere', 'Cylinder']),
    help='The specific sampling used.'
         'The default is spherical sampling with half the articular surface.',
)
@click.option(
    '--norm',
    default='GroupNorm',
    type=click.Choice(['GroupNorm', 'PatientNorm']),
    help='The specific normalization used.'
         'The default is group normalized where each biomarker is out of the max from the entire distribution.',
)
@click.option(
    '--feature_extract',
    '-f',
    default='0',
    type=click.Choice(['0', '1', '2', '3']),
    help='Specify as True when you want to use feature extraction as a transfer learning strategy. If not specified, finetuning will be used.',
)
@click.option(
    '--gpu',
    '-g',
    default='0',
    type=click.Choice(['0', '1', '2', '3']),
    help='The GPU ID for training. Ranges from "0" to "3". Make sure to check with nvidia-smi before choosing a free GPU.',
)
@click.option(
    '--model_name',
    '-m',
    default='ResNet',
    # type=click.Choice(['DenseNet', 'ResNet', 'SqueezeNet', 'AlexNet', 'VGG', 'Inception'], case_sensitive=False),
    help='The type of classifier model, from one of the following: "DenseNet", "ResNet", "SqueezeNet", "AlexNet", "VGG", "Inception".',
)
@click.option(
    '--pretraining',
    default='ImageNet',
    type=click.Choice(['Random', 'ImageNet', 'Diagnosis', 'Pain']),
    help='Whether to use the ImageNet pretraining (True) or random weight initialization (False).',
)
@click.option(
    '--overwrite_flag',
    '-o',
    is_flag=True,
    help='Whether to overwrite existing checkpoints for the specified model. Not specifying this flag when previous checkpoints are found will raise an exception and stop the code execution.',
)
@click.option(
    '--learning_rate',
    '-l',
    default=1e-4,
    help='The learning rate for the model. Default is 1e-4.'
)
@click.option(
    '--drop_rate',
    '-r',
    default=0.5,
    help='The dropout rate for the model. Default is 0.5.'
)
@click.option(
    '--num_epochs',
    '-n',
    default=200,
    help='The max number of epochs for the training. Default is 50 epochs.',
)
@click.option(
    '--batch_size',
    '-b',
    default=300,
    help='The batch size for the dataloader used in training. Default is 80 samples during every forward pass but this depends on the model and the total GPU memory available.' 
    'Here are some guidelines to choose an appropriate batch size:'
    'batch_size = 600 is the optimal batch size to fill up 31 GiB for SqueezeNet' 
    'batch_size = 900 is the optimal batch size to fill up 31 GiB for ResNet' 
    'batch_size = 240 is the optimal batch size to fill up 32 GiB for DenseNet',
)
@click.option(
    '--num_classes',
    '-c',
    default=2,
    help='The number of classes for the classifier. Default is 2 for a binary model.',
)
@click.option(
    '--patience',
    default=50,
    help='Patience for preventing training overfitting. If the difference between the model training accuracy and the model validation accuracy is lesser than 1e-4 value for'
    'The specified patience value in epochs, the learning rate will be decreased by a factor of 0.1',
)
@click.option(
    '--prev_checkpoint_path',
    '-p',
    help='Previous best performing checkpoint path for early stopping. For now, just type the existing model name, tranfer learning type, learning rate and the epoch number: ',
)
@click.option(
    '--model_evaluate',
    '-e',
    type=click.Choice(['Train', 'Val', 'Hold', 'All', 'TrainVal', 'ValHold'], case_sensitive=False),
    help='Whether the model will be evaluated at the specified checkpoint. The evaluation is performed on each dataset and a dictionary with the logits, labels, file names and the class predictions is saved in the checkpoint directory as "model_perf.pickle." The option represent which data split will be used to measure performance with "all" being performed on all data splits and "train_val" being just the training and validation splits.',
)

def train_classifier(config_file, data_dir, knee_type, feature_extract, gpu, model_name, pretraining, overwrite_flag, learning_rate, drop_rate, num_epochs, batch_size, num_classes, patience, prev_checkpoint_path, model_evaluate, progression, biomarker, sampling, norm):
    '''This script trains a classifier CNN on a specified dataset. It can also infer on a previous checkpoint. The model availables are: "densenet121", "resnet18", "squeezenet", "alexnet", "Vgg11", "inceptionv3".'''
    
    feature_extract = int(feature_extract)
    
    if pretraining == 'Diagnosis':
        if '50' not in model_name:
            return
    elif pretraining == 'Random':
        if feature_extract > 0:
            return

    print(f'PyTorch Version: {torch.__version__}')
    print(f'Torchvision Version: {torchvision.__version__}')
    
    if progression in ['DiagnosisAll', 'Pain']:
        data_dir = join(data_dir, sampling, norm, biomarker, knee_type, progression)
    else:
        data_dir = join(data_dir, biomarker, knee_type, progression)
    log_dir = data_dir.replace('on/data', 'on/logs')


    # GPU ID to use
    torch.cuda.set_device(int(gpu))
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)

    # Flag for feature extracting. When False, we finetune the whole model, 
    # when True we only update the reshaped layer params
    
    transfer_learning_type = ['Finetuned', 'FeatureExtract', 'FirstLayerExtract', 'FirstTwoLayersExtract']
    transfer_learning_dict = {}
    transfer_learning_dict[1] = 'Linear'
    transfer_learning_dict[2] = 'layer2'
    transfer_learning_dict[3] = 'layer3'
    
    # Learning rate string for naming the checkpoint directory
    lr_exp = 'e'.join('{:.0E}'.format(Decimal(learning_rate)).split('E-'))
    
    # Unique training name
    if pretraining == 'Diagnosis' or model_evaluate:
        if progression in ['DiagnosisAll', 'Pain']:
            if pretraining == 'Diagnosis':
                diagnosispdpath = join(log_dir.split(sampling)[0], 'diagnosisallperformance.csv')
                checkpointpd = pd.read_csv(diagnosispdpath, index_col=['Sampling', 'Normalization', 'Bone', 'Incidence', 'Fusion'])
                best_checkpoint_path = checkpointpd.loc[(sampling, norm, knee_type, 'DiagnosisAll', biomarker), 'Checkpoint']
            elif pretraining == 'ImageNet':
                pass
            else:
                pdpath = join(log_dir.split(sampling)[0], progression.lower() + 'performance.csv')
                checkpointpd = pd.read_csv(pdpath, index_col=['Sampling', 'Normalization', 'Bone', 'Incidence', 'Fusion'])
                best_checkpoint_path = checkpointpd.loc[(sampling, norm, knee_type, progression, biomarker), 'Checkpoint']
        else:
            pdpath = join(log_dir.split(biomarker)[0], progression.lower() + 'performance.csv')
            checkpointpd = pd.read_csv(pdpath, index_col=['Bone', 'Incidence', 'Fusion'])
            best_checkpoint_path = checkpointpd.loc[(knee_type, progression, biomarker), 'Checkpoint']
        
        best_checkpoint_dir = dirname(best_checkpoint_path)
        prev_training_name = best_checkpoint_dir.split('/')[-1]
        
        print('Previous training name =', prev_training_name)
        print('Previous training directory =', best_checkpoint_dir)

        prev_feature_extract = transfer_learning_type.index([item for item in transfer_learning_type if item in prev_training_name][0])
        if learning_rate == 1e-4:
            learning_rate = float(prev_training_name[-3:].replace('e', 'e-'))/10
        elif learning_rate == 1e-5:
            learning_rate = float(prev_training_name[-3:].replace('e', 'e-'))/100
        elif learning_rate == 1e-6:
            learning_rate = float(prev_training_name[-3:].replace('e', 'e-'))/1000
        elif learning_rate == 1e-7:
            learning_rate = float(prev_training_name[-3:].replace('e', 'e-'))/10000
        lr_exp = 'e'.join('{:.0E}'.format(Decimal(learning_rate)).split('E-'))
        if pretraining == 'Diagnosis':
            training_name = 'Diagnosis' + model_name.capitalize() + transfer_learning_type[feature_extract] + lr_exp
        elif 'prev' in prev_checkpoint_path.lower() and model_evaluate:
            feature_extract = prev_feature_extract
            training_name = prev_training_name
        else:
            training_name = 'Prev' + model_name.capitalize() + transfer_learning_type[feature_extract] + lr_exp
    elif pretraining == 'ImageNet':
        training_name = 'ImageNet' + model_name.capitalize() + transfer_learning_type[feature_extract] + lr_exp
    elif pretraining == 'Random':
        training_name = 'Random' + model_name.capitalize() + transfer_learning_type[feature_extract] + lr_exp
    
    # Current Checkpoint path
    current_checkpoint_path = join(log_dir, training_name)

    print(f'Checkpoint Path: {current_checkpoint_path}')
    print(f'Transfer Learning: {transfer_learning_type[feature_extract]}')

    if not isdir(current_checkpoint_path):
        os.makedirs(current_checkpoint_path)
    elif os.listdir(current_checkpoint_path) and not model_evaluate:
        if not overwrite_flag:
            raise Exception('Previous checkpoints found and no overwrite flag specified.')

    if not model_evaluate:
        checkpoint_log = join(current_checkpoint_path, 'log_file.txt')
        checkpoint_header = '*Model Name*: ' + model_name.capitalize() + '_' + knee_type.capitalize() + '_' + progression.capitalize() + '\t*Transfer Learning Type*: ' + transfer_learning_type[feature_extract] + '\t*Learning Rate*: ' + lr_exp + '\n\n'
        write_type = 'w' # Write mode if file does not exist
        with open(checkpoint_log, write_type) as f_checkpoint:
            if pretraining == 'Diagnosis':
                f_checkpoint.write(checkpoint_header.replace(transfer_learning_type[feature_extract], transfer_learning_type[feature_extract] + '_Diagnosis'))
            elif pretraining == 'ImageNet':
                f_checkpoint.write(checkpoint_header.replace(transfer_learning_type[feature_extract], transfer_learning_type[feature_extract] + '_ImageNet'))
            elif pretraining == 'Random':
                f_checkpoint.write(checkpoint_header.replace(transfer_learning_type[feature_extract], transfer_learning_type[feature_extract] + '_Random'))
        write_type = 'a' # Append mode after file is created

    def train_model(model, dataloaders, criterion, optimizer, current_checkpoint_path, num_epochs=25, prev_model=0, is_inception=False):
        total_time = time.time()
        output_dict = {}
        output_dict['best_val_MCC'] = -1
        output_dict['best_val_TPN'] = 0
        output_dict['best_epoch'] = prev_model
        for phase in ['Train', 'Val']:
            output_dict[phase] = {}
            output_dict[phase]['auc'] = np.zeros(num_epochs)
            output_dict[phase]['acc'] = np.zeros(num_epochs)
            output_dict[phase]['mcc'] = np.zeros(num_epochs)
            output_dict[phase]['tpn'] = np.zeros(num_epochs)
            output_dict[phase]['loss'] = 100*np.ones(num_epochs)
            output_dict[phase]['epoch'] = np.zeros(num_epochs, dtype=int)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_MCC = -1
        best_TPN = 0
        best_epoch = 0

        # for epoch in tqdm(range(num_epochs)):
        for epoch in range(num_epochs):
            since = time.time()
            # print(f'\nEpoch {epoch}/{(num_epochs - 1)}')
            # print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['Train', 'Val']:
                if phase == 'Train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                running_corrects_positive = 0
                running_corrects_negative = 0
                running_positives = 0
                running_negatives = 0
                cnt = 0
                num_hold = len(dataloaders[phase].dataset)
                output_dict[phase]['labels'] = np.zeros(num_hold) - 1
                output_dict[phase]['softmax'] = np.zeros([num_hold,2])
                
                # Iterate over data.
                for inputs, labels, paths in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'Train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'Train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(torch.softmax(outputs, dim=1),1)

                        # backward + optimize only if in training phase
                        if phase == 'Train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_corrects_positive += np.logical_and(preds.cpu(), labels.data.cpu()).sum()
                    running_corrects_negative += len(labels.data) - np.logical_or(preds.cpu(), labels.data.cpu()).sum()
                    running_positives += torch.sum(labels.data)
                    running_negatives += len(labels.data) - torch.sum(labels.data).int()
                    output_dict[phase]['labels'][(cnt*batch_size):((cnt + 1)*batch_size)] = labels.cpu().numpy()
                    output_dict[phase]['softmax'][(cnt*batch_size):((cnt + 1)*batch_size),:] = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                    cnt += 1 
#                     print('True Positives: {} All Positives: {}'.format(running_corrects_positive, running_positives))
#                     print('True Negatives: {} All Negatives: {}'.format(running_corrects_negative, running_negatives))torch.sigmoid(outputs) 

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                epoch_tpr = running_corrects_positive / running_positives.double()
                epoch_tnr = running_corrects_negative / running_negatives.double()
                output_dict[phase]['auc'][epoch] = roc_auc_score(output_dict[phase]['labels'], output_dict[phase]['softmax'][:,1])
                output_dict[phase]['mcc'][epoch] = matthews_corrcoef(output_dict[phase]['labels'], output_dict[phase]['softmax'][:,1] >= 0.5)
                output_dict[phase]['tpn'][epoch] = epoch_tpr + epoch_tnr + output_dict[phase]['auc'][epoch]
                # print(f'TPR: {epoch_tpr.numpy():5.3f}')
                # print(f'TNR: {epoch_tnr.numpy():5.3f}')
                # print(f'AUC: {output_dict[phase]["auc"][epoch]:5.3f}')
                # print(f'MCC: {output_dict[phase]["mcc"][epoch]:6.3f}')
                # print(f'TPN: {output_dict[phase]["tpn"][epoch]:6.3f}')
                    
                # if phase == 'Val':
                #     time_elapsed = time.time() - since
                #     print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} \nTime since last epoch: {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
                # else:
                #     print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'Val' and (output_dict[phase]['tpn'][epoch] > best_TPN) and (epoch_tpr >= 0.5) and (epoch_tnr >= 0.5):
                    best_MCC = output_dict[phase]['mcc'][epoch]
                    best_TPN = output_dict[phase]['tpn'][epoch]
                    output_dict['best_val_MCC'] = best_MCC
                    output_dict['best_val_TPN'] = best_TPN
                    output_dict['best_epoch'] = prev_model + epoch
                    chkpath = join(current_checkpoint_path, 'best_epoch')
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'output_dict': output_dict}, chkpath)
                    print('Saved model in ' + chkpath)
                    # best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'Val':
                    output_dict['Val']['acc'][epoch] = epoch_acc
                    output_dict['Val']['loss'][epoch] = epoch_loss
                    output_dict['Val']['epoch'][epoch] = prev_model + epoch
#                     scheduler.step(1 - output_dict['Val']['auc'][epoch])
                if phase == 'Val' and (np.argmin(output_dict['Val']['loss']) + patience < epoch):
                    print(f'Early stopping due to validation loss not improving for {patience} epochs')
                    quit()
                if phase == 'Train':
                    output_dict['Train']['acc'][epoch] = epoch_acc
                    output_dict['Train']['loss'][epoch] = epoch_loss
                    output_dict['Train']['epoch'][epoch] = prev_model + epoch

            if epoch % 1 == 0:
                current_train_acc = output_dict['Train']['acc'][epoch]
                current_val_acc = output_dict['Val']['acc'][epoch]
                current_train_loss = output_dict['Train']['loss'][epoch]
                current_val_loss = output_dict['Val']['loss'][epoch]
                best_val_MCC = output_dict['best_val_MCC']
                best_val_TPN = output_dict['best_val_TPN']
                with open(checkpoint_log, write_type) as f_checkpoint:
                    f_checkpoint.write(f'*Epoch*: {output_dict["Train"]["epoch"][epoch]:3}  *Train Loss*: {current_train_loss:5.3f}' + \
                    f'  *Val Loss*: {current_val_loss:5.3f}  *TPR*: {epoch_tpr.numpy():5.3f}' + \
                    f'  *TNR*: {epoch_tnr.numpy():5.3f}  *AUC*: {output_dict["Val"]["auc"][epoch]:5.3f}  *MCC*: {output_dict["Val"]["mcc"][epoch]:6.3f}' + \
                    f'  *TPN*: {output_dict["Val"]["tpn"][epoch]:5.3f}  *Best Val TPN*: {best_val_TPN:6.3f}  *Best Epoch*: {output_dict["best_epoch"]:3}\n')

            # print()
            
        total_time_elapsed = time.time() - total_time
        print(f'Training complete in {(total_time_elapsed // 3600):.0f}h {(total_time_elapsed // 60):.0f}m {(total_time_elapsed % 60):.0f}s')
        print(f'Best Val MCC: {best_MCC:6.3f}')

        # load best model weights    
        model.load_state_dict(best_model_wts)
        return model

    # Set Model Parameters’ .requires_grad attribute
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # This helper function sets the .requires_grad attribute of the
    # parameters in the model to False when we are feature extracting. By
    # default, when we load a pretrained model all of the parameters have
    # .requires_grad=True, which is fine if we are training from scratch
    # or finetuning. However, if we are feature extracting and only want to
    # compute gradients for the newly initialized layer then we want all of
    # the other parameters to not require gradients. This will make more sense
    # later.
    
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting > 0:
            bypass_condition = 0
            for param_name, param in model.named_parameters():
                if transfer_learning_dict[feature_extracting] in param_name or bypass_condition:
                    param.requires_grad = True
                    bypass_condition = 1
                else:
                    param.requires_grad = False

    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if 'resnet' in model_name.lower():
            ''' Resnet50
            '''
            if '18' in model_name.lower():
                model_ft = models.resnet18(pretrained=use_pretrained)
            elif '34' in model_name.lower():
                model_ft = models.resnet34(pretrained=use_pretrained)
            elif '50' in model_name.lower():
                model_ft = models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            print(f'Fully Connected Layer Features = {num_ftrs}')
            # model_ft.fc.register_forward_hook(lambda m, inp, out: nn.functional.dropout(out, p=drop_rate, training=m.training))
            # if not feature_extract:
            if pretraining == 'Diagnosis':
                model_ft.layer1 = nn.Sequential(
                                            nn.Dropout(drop_rate),
                                            model_ft.layer1
                                            )
                model_ft.layer2 = nn.Sequential(
                                            nn.Dropout(drop_rate),
                                            model_ft.layer2
                                            )
                model_ft.layer3 = nn.Sequential(
                                            nn.Dropout(drop_rate),
                                            model_ft.layer3
                                            )
                model_ft.layer4 = nn.Sequential(
                                            nn.Dropout(drop_rate),
                                            model_ft.layer4
                                            )

                model_ft.fc = nn.Sequential(
                                            nn.Dropout(drop_rate),
                                            nn.Linear(num_ftrs, num_classes)
                                            )
            else:
                model_ft.fc = nn.Linear(num_ftrs, num_classes)
            # print(f'Layers = {model_ft.children}')
            input_size = 224

        elif model_name.lower() == 'alexnet':
            ''' Alexnet
            '''
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name.lower() == 'vgg':
            ''' VGG11_bn
            '''
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name.lower() == 'squeezenet':
            ''' Squeezenet
            '''
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name.lower() == 'densenet':
            ''' Densenet
            '''
            model_ft = models.densenet121(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
            input_size = 224

        elif model_name.lower() == 'inception':
            ''' Inception v3 
            Be careful, expects (299,299) sized images and has auxiliary output
            '''
            model_ft = models.inception_v3(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs,num_classes)
            input_size = 299

        else:
            print('Invalid model name, exiting...')
            exit()

        return model_ft, input_size

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=(pretraining == 'ImageNet'))

    # Load Data
    # ---------
    # Now that we know what the input size must be, we can initialize the data
    # transforms, image datasets, and the dataloaders. Notice, the models were
    # pretrained with the hard-coded normalization values, as described
    # https://pytorch.org/docs/master/torchvision/models.html

    # Data augmentation and normalization for training
    # Just normalization for validation
    
    data_transforms = {
        'Train': transforms.Compose([
            # No augmentation necessary for spherical maps since that is done before the transformation
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.RandomErasing(p=0.25)
        ]),
        'Val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Hold': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print(f'Transforms: {data_transforms}')

    # print('Initializing Datasets and Dataloaders...')

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(join(data_dir, x), data_transforms[x]) for x in ['Train', 'Val', 'Hold']}

    for split in ['Train', 'Val', 'Hold']:
        if 'year' in progression:
            image_datasets[split].class_to_idx = {'Healthy': 0, 'Incidence': 1}
        elif 'pain' in progression.lower():
                image_datasets[split].class_to_idx = {'Healthy': 0, 'Pain': 1}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8) for x in ['Train', 'Val', 'Hold']}

    # Detect if we have a GPU available3
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are 
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    if feature_extract > 0:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        num_params = len(params_to_update)
    else:
        num_params = len(list(model_ft.parameters()))

    # Observe that all parameters are being optimized
    print('Number of training parameters:', num_params)
    optimizer_ft = optim.Adam(params_to_update, lr=learning_rate, weight_decay=0.1)

    # Run Training and Validation Step
    # --------------------------------
    # Finally, the last step is to setup the loss for the model, then run the
    # training and validation function for the set number of epochs. Notice,
    # depending on the number of epochs this step may take a while on a CPU.
    # Also, the default learning rate is not optimal for all of the models, so
    # to achieve maximum accuracy it would be necessary to tune for each model
    # separately.
    
    pain_weights = [0.86589497, 1.18325617]
    # Setup the loss fxn
    class_weights = torch.FloatTensor(pain_weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f'Loss Class Weights = {pain_weights}')

    info = nvmlDeviceGetMemoryInfo(h)

    if feature_extract == 1:
        if '18' in model_name:
            batch_size = 905
        elif '34' in model_name:
            batch_size = 905
        else:
            batch_size = 705
        batch_step = 60
    elif feature_extract == 3:
        if '18' in model_name:
            batch_size = 855
        elif '34' in model_name:
            batch_size = 855
        else:
            batch_size = 305
        batch_step = 60
    elif feature_extract == 2:
        if '18' in model_name:
            batch_size = 955
        elif '34' in model_name:
            batch_size = 705
        else:
            batch_size = 155
        batch_step = 30
    elif feature_extract == 0:
        if '18' in model_name:
            batch_size = 425
        elif '34' in model_name:
            batch_size = 305
        else:
            batch_size = 105
        batch_step = 10

    if info.total > 1.5e10:
        batch_size = round(2.8*batch_size)
        batch_step = round(2*batch_step)
        
    n_channels = 3
    batch_adapt = 1
    print('Batch Size:', batch_size)
    print('Batch Step:', batch_step)
    model_ft.train()
    while batch_adapt:
        input_shape = (batch_size, n_channels, input_size, input_size)
        try:
            inputs = torch.randn(*input_shape, dtype=torch.float32).cuda()
            labels = torch.ones(batch_size, dtype=torch.int64).cuda()
            # zero the parameter gradients
            optimizer_ft.zero_grad()
            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()
            print('Allocated:', torch.cuda.max_memory_allocated())
            print('Cached:', torch.cuda.max_memory_cached())
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
            if (info.total - torch.cuda.max_memory_cached()) < 2e9:
                batch_adapt = 0
                print('Final Batch Size:', batch_size)
            else:
                batch_size += batch_step
        except RuntimeError as error:
            print(error)
            batch_size -= round(1.5*batch_step)
            batch_adapt = 0
            print('Final Batch Size:', batch_size)

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8) for x in ['Train', 'Val', 'Hold']}    

    if pretraining == 'Diagnosis':
        best_checkpoint_load = torch.load(best_checkpoint_path, map_location='cuda:' + gpu)
        model_ft.load_state_dict(best_checkpoint_load['model_state_dict'])
        print(f'Loading previous model: {best_checkpoint_path}')

    if model_evaluate:
        best_checkpoint_load = torch.load(best_checkpoint_path, map_location='cuda:' + gpu)
        model_ft.load_state_dict(best_checkpoint_load['model_state_dict'])
        print(f'Loading best model: {best_checkpoint_path}')

        # We're evaluating the model_load here:
        def model_eval(model_load, dataloaders, phase_range=['Val']):

            model_load.eval()   # Set model_load to evaluate mode
            results_dict = {}

            for phase in phase_range:
                cnt = 0
                num_hold = len(dataloaders[phase].dataset)
                results_dict[phase] = {}
                results_dict[phase]['file_names'] = []
                results_dict[phase]['labels'] = np.zeros(num_hold) - 1
                results_dict[phase]['class_predict'] = np.zeros(num_hold) - 1
                results_dict[phase]['logits'] = np.zeros([num_hold,2])
                results_dict[phase]['softmax'] = np.zeros([num_hold,2])
#                     results_dict[phase]['features'] = np.zeros([num_hold,2048])

                # Iterate over data.
                for inputs, labels, paths in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    results_dict[phase]['file_names'].extend(list(paths))
                    results_dict[phase]['labels'][(cnt*batch_size):((cnt + 1)*batch_size)] = labels.cpu().numpy()

                    with torch.set_grad_enabled(False): 
                        outputs = model_load(inputs)
                        results_dict[phase]['softmax'][(cnt*batch_size):((cnt + 1)*batch_size),:] = torch.softmax(outputs, dim=1).cpu().numpy()
                        results_dict[phase]['logits'][(cnt*batch_size):((cnt + 1)*batch_size),:] = outputs.cpu().numpy()
#                             results_dict[phase]['features'][(cnt*batch_size):((cnt + 1)*batch_size),:] = outputs.cpu().numpy()
                        # forward
                        _, preds = torch.max(torch.softmax(outputs, dim=1),1)
                        results_dict[phase]['class_predict'][(cnt*batch_size):((cnt + 1)*batch_size)] = preds.cpu().numpy()
                        # statistics
                        pred_comp = (labels.cpu().numpy() == preds.cpu().numpy())
                        cnt += 1  

            return results_dict
        
        def perf_measure(y_actual, y_hat):
            TP = 0
            FP = 0
            TN = 0
            FN = 0

            for i in range(len(y_hat)): 
                if y_actual[i]==y_hat[i]==1:
                    TP += 1
                if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                    FP += 1
                if y_actual[i]==y_hat[i]==0:
                    TN += 1
                if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                    FN += 1

            return TP, FP, TN, FN
        
        if model_evaluate.lower() == 'all':
            phase_range = ['Train','Val', 'Hold']
            eval_dict = model_eval(model_ft, dataloaders_dict, phase_range)
        elif model_evaluate.lower() == 'trainval':
            phase_range = ['Train','Val']
            eval_dict = model_eval(model_ft, dataloaders_dict, phase_range)
        elif model_evaluate.lower() == 'valhold':
            phase_range = ['Val', 'Hold']
            eval_dict = model_eval(model_ft, dataloaders_dict, phase_range)
        else:
            phase_range = [model_evaluate.lower().capitalize()]
            eval_dict = model_eval(model_ft, dataloaders_dict, phase_range)
        
        for i in phase_range:
            TP, FP, TN, FN = perf_measure(eval_dict[i]['labels'],eval_dict[i]['class_predict'])
            # print(TP, FP, TN, FN)
            sensitivity = TP/(TP + FN)
            specificity = TN/(TN + FP)
            eval_dict[i]['sensitivity'] = sensitivity 
            eval_dict[i]['specificity'] = specificity
            eval_dict[i]['auc'] = roc_auc_score(eval_dict[i]['labels'], eval_dict[i]['softmax'][:,1])
            if i == 'Hold':
                continue
            else:
                print("*{}* \nSensitivity = {} \nSpecificity = {} \nAUC = {}".format(i.capitalize(),sensitivity, specificity, eval_dict[i]['auc']))
        
        with open(join(best_checkpoint_dir, 'model_perf.pickle'), 'wb') as f:
            eval_dict['checkpoint'] = [best_checkpoint_path]
            pickle.dump(eval_dict,f)
        
        quit()

    else:
        if pretraining == 'Diagnosis':
            model_ft.layer1 = model_ft.layer1[1]
            model_ft.layer2 = model_ft.layer2[1]
            model_ft.layer3 = model_ft.layer3[1]
            model_ft.layer4 = model_ft.layer4[1]
            model_ft.fc = model_ft.fc[1]
        model_ft = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, current_checkpoint_path, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    
if __name__ == '__main__':
    train_classifier()
