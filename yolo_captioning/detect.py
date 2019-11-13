#coding=utf-8
from __future__ import division
import time
from util import *
import argparse

from  model import EncoderCNN, AttnDecoderRNN
from data_loader import get_loader
from prepro import Vocabulary
from torchvision import transforms
from utils import *
from torch.nn.utils.rnn import pack_padded_sequence
from LoggerSu import Logger
import pickle
from nltk.translate.bleu_score import corpus_bleu

best_bleu4=0.0
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    base_path=r'/data/antonio/images_data/'
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images',default = "imgs", help ="Image / Directory containing images to perform detection upon", type = str)
    parser.add_argument("--det", dest = 'det', default = "det",help ="Image / Directory to store detections to", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 3)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = "Config file", default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile", default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', default = "224", type = str,help ="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed")

    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",default = "1,2,3", type = str)

    parser.add_argument('--embed_dim', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder rnn')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--fine_tune_encoder', type=bool, default='False', help='fine-tune encoder')


    parser.add_argument('--vec_filename', type=str, default='word/doc_vecs.txt', help='path for saving trained models')
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default=base_path+'vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default=base_path+'esized2014', help='directory for resized images')
    parser.add_argument('--image_dir_val', type=str, default=base_path+'val2014_resized', help='directory for resized images')
    parser.add_argument('--caption_path', type=str,default=base_path+'annotations/captions_train2014.json',help='path for train annotation json file')
    parser.add_argument('--caption_path_val', type=str,default=base_path+'annotations/captions_val2014.json',help='path for val annotation json file')

    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_lr', type=float, default=4e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--alpha_c', type=float, default=1.)
    parser.add_argument('--epochs_since_improvement', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default=None, help='path for checkpoints')
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    parser.add_argument('--data_name', type=str, default='coco_5_cap_per_img_5_min_word_freq')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    return parser.parse_args()

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch,stdout):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens,allcaps) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        prediction, feature13, feature26, feature52= encoder(imgs)

        detection_box = non_max_suppression(prediction, 80, args.conf_thres, args.nms_thres)
        box_feature=getresult(detection_box, feature13, feature26, feature52)

        # alphas=(32*23*196)
        scores, caps_sorted, decode_lengths, alphas = decoder(feature13,box_feature, caps, caplens,detection_box)
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)

        targets = caps_sorted[:, 1:]
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = criterion(scores, targets)
        loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        train_time = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
        # Print status
        if i % args.log_step == 0:
            s='Epoch: [{0}][{1}/{2}]\t Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\tData Load Time {data_time.val:.3f} ({data_time.avg:.3f})' \
              '\tLoss {loss.val:.4f} ({loss.avg:.4f})\t Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f}) excute time:{time}'\
              .format(epoch, i, len(train_loader),batch_time=batch_time, data_time=data_time, loss=losses,top5=top5accs,time=train_time)
            stdout.write(s)


def validate(val_loader, encoder, decoder, criterion,word_map,stdout):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)

        # Forward prop.
        #if encoder is not None:
        prediction, feature13, feature26, feature52 = encoder(imgs)
        detection_box = non_max_suppression(prediction, 80, args.conf_thres, args.nms_thres)
        box_feature=getresult(detection_box, feature13, feature26, feature52)

        # alphas=(32*23*196)
        scores, caps_sorted, decode_lengths, alphas = decoder(feature13,box_feature, caps, caplens)
        #scores, caps_sorted, decode_lengths, alphas = decoder(feature13, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        val_time = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
        if i % args.log_step == 0:
           s='Validation: [{0}/{1}]\tBatch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})' \
             '\tTop-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\texcute time:{time}'\
             .format(i, len(val_loader), batch_time=batch_time, loss=losses, top5=top5accs,time=val_time)
           stdout.write(s)

        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        # References
        # allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)

        assert len(references) == len(hypotheses)


    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    s='LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}'.format(loss=losses,top5=top5accs,bleu=bleu4)
    stdout.write(s)
    return bleu4

def getresult(detection_box,feature13, feature26, feature52):
    """
    If the image has many box,take the max area of the box
    If the image has no box,take this (x1, x2, y1, y2 = 4, 8, 4, 8) feature
    :param detection_box: batch_size*num_box*7
    :param feature13: batch_size*depth*N*N
    :param feature26:
    :param feature52:
    :return:
    """
    #feature13 = feature13.permute(0, 2, 3, 1)
    #features include all the image feature,and take the max feature of image
    features=[]
    result=0
    max=0
    #If the box is none,the cordinates is as follow
    x1, x2, y1, y2 = 4, 8, 4, 8
    # Iterate through the batch, detection's meaning is each image detection box
    for img_i, detection in enumerate(detection_box):
        # If the box is none,the cordinates( x1, x2, y1, y2 = 4, 8, 4, 8) will be taken
        if  detection is None:
            feature_max_box = feature13[img_i, :, x1:x2, y1:y2]
            features,max=feature_cal(feature_max_box, features, feature13, max)
            continue
        #record the area of box
        area_box=detection.new(detection.size(0))
        # every box area, box's dim is [7]
        for num_box,box in enumerate(detection):
            area_box[num_box]=(box[2]-box[0])*(box[3]-box[1])
        #get the max box, and the index is 0
        _,index=torch.sort(area_box,descending=False)
        max_box = detection[index[0]]
        #scale to the 13*13 feature
        max_box=(max_box//32).int()
        #get the cordinates
        x1, x2,y1,y2=max_box[0],max_box[2],max_box[1],max_box[3]
        #avoid the tensor overfitting
        if max_box[0]-max_box[2]==0:
            if max_box[0]==12:
                x1,x2=11,12
            else:
                x1=max_box[0]
                x2=max_box[2]+1
        if max_box[1] - max_box[3] == 0 :
            if max_box[1] == 12:
                y1,y2=11,12
            else:
                y1=max_box[1]
                y2=max_box[3]+1


        feature_max_box=feature13[img_i,:,x1:x2,y1:y2]
        features,max=feature_cal(feature_max_box, features, feature13, max)
    #unify the scale of each feature
    for i,feature in enumerate(features):
        #if feature less than max ,padding 0
        if max-feature.size(1)>0:
            #torch.zeros have to initial to gpu,
            t = torch.zeros(feature.size(0), max - feature.size(1)).to(device)
            feature=torch.cat((feature,t),dim=1).unsqueeze(0)
        else:
            feature=feature.unsqueeze(0)
        if i==0:
            result=feature
        else:
            result=torch.cat((result,feature),dim=0)
    return result

def feature_cal(feature_max_box,features,feature13,max):
    feature_max_box = feature_max_box.contiguous()
    depth = feature_max_box.size(0)
    #depth*nums_feature_pot
    feature_max_box = feature_max_box.view(depth, -1)

    features.append(feature_max_box)
    if feature_max_box.size(1) > max:
        max = feature_max_box.size(1)
    return features,max


def main(args):
    global best_bleu4
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    encoder = EncoderCNN(reso=args.reso)
    decoder = AttnDecoderRNN(attention_dim=args.attention_dim,
                             embed_dim=args.embed_dim,
                             decoder_dim=args.decoder_dim,
                             filename=args.vec_filename,
                             vocab=vocab,
                             dropout=args.dropout)
    encoder.to(device)
    decoder.to(device)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=args.decoder_lr)

    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=args.encoder_lr) if args.fine_tune_encoder else None

    criterion = nn.CrossEntropyLoss().to(device)
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Build data loader
    train_loader = get_loader(args.image_dir, args.caption_path, vocab,
                              transform, args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = get_loader(args.image_dir_val, args.caption_path_val, vocab,
                            transform, args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    TrainStdout = Logger('train.txt')
    ValStdout = Logger('val.txt')
    for epoch in range(args.start_epoch, args.epochs):
        if args.epochs_since_improvement == 20:
            break
        if args.epochs_since_improvement > 0 and args.epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if args.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              stdout=TrainStdout)
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                word_map=vocab.word2idx,
                                stdout=ValStdout)

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            args.epochs_since_improvement +=1
            print ("\nEpoch since last improvement: %d\n" %(args.epochs_since_improvement,))
        else:
            args.epochs_since_improvement = 0

        save_checkpoint(args.data_name, epoch, args.epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                        recent_bleu4, is_best)

if __name__ ==  '__main__':
    args = arg_parse()
    main(args)