import sys
import argparse
import evaluate_model
import torch
import os
import polygon_lists


def infer_location(image_name, k=5, model_name=None, bound=15, model_reg_list=None):
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
    image_val = list(image_val)
    for i in range(len(image_val)):
        image_val[i] = list(image_val[i])
    _, mem, _, _ = polygon_lists.get_polygons_dict('continents_polygon2')
    for i in range(bound):
        continent, _ = image_val[i]
        idx = mem[continent]
        reg_path = os.path.join('../Geoguessr_workshop/reg_best', 'best_model_' + 'resnet' + '_region' + str(idx))
        if model_reg_list is None:
            model_reg = torch.load(reg_path, map_location=torch.device('cpu'))
            model_reg.eval()
        else:
            model_reg = model_reg_list[idx]
        image_reg = evaluate_model.eval_image(model_reg, image_name, regression=True, is_inception=False)
        image_val[i].append(image_reg)
    return image_val


def main():
    if len(sys.argv) < 3:
        print('Invalid usage')
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('-k', default='5')
    parser.add_argument('-type', default=None)
    parser.add_argument('-restrict', default='15')
    args = parser.parse_args()
    image_name = args.image_path
    k = int(args.k)
    model_name = args.type
    bound = int(args.restrict)
    print(infer_location(image_name, k, model_name, bound))


if __name__ == '__main__':
    main()