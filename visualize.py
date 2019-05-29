import argparse
import logging
import math
import nltk
import pickle
import string
import pandas as pd

import os
import random
import torch
import torch.nn as nn

from PIL import Image
from termcolor import colored
from tqdm import tqdm

from torchvision import transforms
from torchvision.models import vgg19
from torch.serialization import default_restore_location

from captioner import models, utils
from captioner.data.dataset import CaptionDataset, BatchSampler
from captioner.data.dictionary import Dictionary
from captioner.generator import SequenceGenerator


def get_args():
    parser = argparse.ArgumentParser('Caption Generation')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--dataset-path', required=True, help='path to datasets')
    parser.add_argument('--test-caption', default='annotations/reports_test_supra.csv', help='reference captions')
    parser.add_argument('--test-image', default='images/imgs', help='path to test images')
    parser.add_argument('--caption-ids', default=None, type=int, nargs='+', help='caption ids')
    parser.add_argument('--image-size', type=int, default=256, help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--checkpoint-path', default='checkpoints/checkpoint_best.pt', help='path to the model file')

    # Add generation arguments
    parser.add_argument('--beam-size', default=5, type=int, help='beam size')
    parser.add_argument('--max-len', default=200, type=int, help='maximum length of generated sequence')
    parser.add_argument('--stop-early', default='True', help='stop generation immediately after finalizing hypotheses')
    parser.add_argument('--normalize_scores', default='True', help='normalize scores by the length of the output')
    parser.add_argument('--len-penalty', default=1, type=float, help='length penalty: > 1.0 favors longer sentences')
    parser.add_argument('--unk-penalty', default=0, type=float, help='unknown word penalty: >0 produces fewer unks')
    return parser.parse_args()


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load arguments from checkpoint (no need to load pretrained embeddings or write to log file)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args = argparse.Namespace(**{**vars(state_dict['args']), **vars(args), 'embed_path': None, 'log_file': None})
    utils.init_logging(args)

    # Load dictionary
    dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
    logging.info('Loaded a dictionary of {} words'.format(len(dictionary)))

    # Load dataset
    test = pd.read_csv(os.path.join(args.dataset_path, args.test_caption))
    if args.caption_ids is None:
        args.caption_ids = random.sample(test['uId'].tolist(), 10)
    image_ids = [test.loc[test['uId'] == id]['imgId'].tolist()[0] for id in args.caption_ids]
    reference_captions = [test.loc[test['uId'] == id]['report'].tolist()[0] for id in args.caption_ids]
    image_names = [os.path.join(args.dataset_path, args.test_image, id + '.png') for id in image_ids]

    # Transform image
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
    ])
    images = [transform(Image.open(filename)) for filename in image_names]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    sample = torch.stack([transform(image.convert('RGB')) for image in images], dim=0)

    # Extract image features
    vgg = vgg19(pretrained=True).eval().cuda()
    model = nn.Sequential(*list(vgg.features.children())[:-2])
    image_features = model(utils.move_to_cuda(sample))
    image_features = image_features.view(*image_features.size()[:-2], -1)
    # B x C x (H x W) -> B x (H x W) x C
    image_features = image_features.transpose(1, 2)

    # Load model and build generator
    model = models.build_model(args, dictionary).cuda()
    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {}'.format(args.checkpoint_path))
    generator = SequenceGenerator(
        model, dictionary, beam_size=args.beam_size, maxlen=args.max_len, stop_early=eval(args.stop_early),
        normalize_scores=eval(args.normalize_scores), len_penalty=args.len_penalty, unk_penalty=args.unk_penalty,
    )

    # Generate captions
    with torch.no_grad():
        hypos = generator.generate(image_features)
    for i, (id, image, reference_caption) in enumerate(zip(args.caption_ids, images, reference_captions)):
        output_image = os.path.join('images', '{}.png'.format(id))
        attention = hypos[i][0]['attention'].view(14, 14, -1).cpu().numpy()
        system_tokens = [dictionary.words[tok] for tok in hypos[i][0]['tokens'] if tok != dictionary.eos_idx]
        ''' dumpeo dict para entender mejor las predicciones y referencias'''
        prediction = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in system_tokens]).strip()
        dic = {'prediction' : prediction, 'reference': reference_caption}
        with open('images/{}.p'.format(id), 'wb') as f:
            pickle.dump(dic, f)
        utils.plot_image_caption(image, output_image, system_tokens, reference_caption, attention)



if __name__ == '__main__':
    args = get_args()
    main(args)
