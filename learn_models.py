from resnet_train import ResnetTrain
from vgg_train import VGGTrain
from inception_train import InceptionTrain
import torch
from torchvision import models
import sys
import os


def get_model(model_name, file_name, dir_name, val_percent, regression, num_workers):
    '''
    Returns the model given the model name
    :param model_name: Name of the model
    :param file_name: Training file name
    :param dir_name: Training dir name
    :param val_percent: Validation percentage
    :param regression: Regression or classification
    :param num_workers: Number of workers
    :return: The specified model
    '''
    if model_name == 'resnet':
        return ResnetTrain(model=models.resnet50(pretrained=True), #TODO not 50
                                 file_name_train=file_name, dir_name_train=dir_name, val_percent=val_percent, regression=regression,
                           num_workers=num_workers)
    elif model_name == 'resnext':
        ResnetTrain(model=models.resnext50_32x4d(pretrained=True),
                    file_name_train=file_name, dir_name_train=dir_name, val_percent=val_percent, regression=regression,
                    num_workers=num_workers)
    elif model_name == 'wideresnet':
        return ResnetTrain(model=models.wide_resnet50_2(pretrained=True),
                           file_name_train=file_name, dir_name_train=dir_name, val_percent=val_percent,
                           regression=regression, num_workers=num_workers)
    elif model_name == 'vgg':
        return VGGTrain(model=models.vgg19(pretrained=True),
                        file_name_train=file_name, dir_name_train=dir_name, val_percent=val_percent,
                        regression=regression, num_workers=num_workers)
    elif model_name == 'vggbn':
        return VGGTrain(models.vgg19_bn(pretrained=True),
                        file_name_train=file_name, dir_name_train=dir_name, val_percent=val_percent,
                        regression=regression, num_workers=num_workers)
    elif model_name == 'inception':
        return InceptionTrain(model=models.inception_v3(pretrained=True),
                        file_name_train=file_name, dir_name_train=dir_name, val_percent=val_percent,
                        regression=regression, num_workers=num_workers)


def main(model_name, out_dir_name, top_k=3, regression=False,
         file_name='100k_ds10k_train', dir_name='100k_images_dir_new', region=None):
    '''
    Trains the model
    :param model_name: Name of the model
    :param out_dir_name: Where to output the best model
    :param top_k: Top what
    :return: Nothing
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    val_percent = 7 #TODO also test?
    best_params = []
    num_workers = 8
    print(model_name)
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)
    if not regression:
        save_model_to = os.path.join(out_dir_name, 'best_model_' + model_name + '_top' + str(top_k))
    else:
        save_model_to = os.path.join(out_dir_name, 'best_model_' + model_name + '_region' + str(region))
        file_name += str(region) + '.csv'
    # TODO play with batch size? model size?
    best_model = None
    best_acc = 0
    for lr in [0.00001, 0.0001, 0.0005, 0.00005]:
        ft_lr = 1e-6
        print([lr, ft_lr])
        rn = get_model(model_name, file_name, dir_name, val_percent, regression, num_workers)
        rn.main_model.to(device)
        m1, acc = rn.feat_extract(lr=lr, epochs=30, top_k=top_k)
        if acc > best_acc:
            best_acc = acc
            best_model = m1
            best_params = [lr, ft_lr]
        m1, acc = rn.fine_tune(lr=ft_lr, epochs=10, top_k=top_k)
        if acc > best_acc:
            best_acc = acc
            best_model = m1
            best_params = [lr, ft_lr]
    torch.save(best_model, save_model_to)
    print(best_params)


if __name__ == '__main__':
    dir_name = sys.argv[1]
    region = int(sys.argv[2])
    main('resnet', dir_name, regression=True, file_name=os.path.join('../Geoguessr_workshop/100k_ds10k_train_split', 'region'), region=region)