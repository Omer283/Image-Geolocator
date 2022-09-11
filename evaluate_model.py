from PIL import Image
from torchvision import transforms
import torch
import utils
from polygon_lists import get_polygons_dict
from single_image_dataset import SingleImageDataset
from torch import nn


def eval_image(model, file_name, is_inception=False, regression=True):
    '''
    Infers an image with a given model
    :param model: The CNN to infer the image with
    :param file_name: Image file name
    :param is_inception: A flag specifying whether the given
    :param regression: A flag specifying whether the model regresses or classifies
    :return: The inference value
    '''
    if is_inception:
        resize, crop = 299, 299
    else:
        resize, crop = 256, 224
    image = Image.open(file_name)
    t = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop((crop, crop)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = t(image)
    model.eval()
    input = image.unsqueeze(0)
    preds = model(input)
    preds = preds.squeeze(0)
    if not regression:
        _, _, _, cont_list = get_polygons_dict('continents_polygon2')
        preds = 100 * torch.nn.functional.softmax(preds, dim=0)
        return utils.topk_categories(preds, cont_list, 15)
    else:
        lat = preds[0]
        long = preds[1]
        lat = (lat + 90) % 180 - 90
        long = (long + 180) % 360 - 180
        return torch.tensor([lat, long])


def eval_dataset(model, file_name, dir_name, regression=True, is_inception=False, top_k=1):
    '''
    Evaluates the accuracy of a specified dataset on a given model.
    :param model: The model
    :param file_name: File name of the dataset
    :param dir_name: Directory of the dataset
    :param is_inception: A flag specifying whether the given
    :param regression: A flag specifying whether the model regresses or classifies
    :param top_k: Eval according to top_k accuracy
    :return: (loss, accuracy)
    '''
    if is_inception:
        resize, crop = 299, 299
    else:
        resize, crop = 256, 224
    t = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop((crop, crop)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = SingleImageDataset(file_name, dir_name, transform=t, regression=regression)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    running_loss = 0.0
    running_corrects = 0.0
    if not regression:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = utils.custom_loss
    for inputs, labels in dl:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if not regression:
                _, preds = outputs.topk(top_k, 1, True, True)
                preds = preds.t()

        running_loss += loss.item() * inputs.size(0)
        if not regression:
            running_corrects += torch.sum(preds.eq(labels.view(1, -1).expand_as(preds)))
    epoch_loss = running_loss / len(ds)
    if regression:
        return epoch_loss
    else:
        epoch_acc = running_corrects.double() / len(ds)
        return epoch_loss, float(epoch_acc)
