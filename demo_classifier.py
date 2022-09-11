import sys
import argparse
import evaluate_model
import torch
import os


def infer_class(image_name, k=5, model_name=None):
    if model_name:
        is_inception = False
        if model_name == 'inception':
            is_inception = True
        path_name = os.path.join(str(k) + 'k_best', 'best_model_' + model_name + '_top' + str(k))
        model = torch.load(path_name, map_location=torch.device('cpu'))
        model.eval()
        image_val = evaluate_model.eval_image(model, image_name, regression=False, is_inception=is_inception)
    else:
        path_name = os.path.join('../Geoguessr_workshop/100k_best', 'best_model_' + 'resnet' + '_top' + str(k))
        model = torch.load(path_name, map_location=torch.device('cpu'))
        model.eval()
        image_val = evaluate_model.eval_image(model, image_name, regression=False, is_inception=False)
    return image_val


def main():
    if len(sys.argv) < 3:
        print('Invalid usage')
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('-k', default='5')
    parser.add_argument('-type', default=None)
    args = parser.parse_args()
    image_name = args.image_path
    k = int(args.k)
    model_name = args.type
    print(infer_class(image_name, k, model_name))


if __name__ == '__main__':
    main()