
import os

import nltk
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO


class DataLoader(data.Dataset):
    def __init__(self, root, json, vocab, transform=None):

        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
    #在具体加载数据的时候，执行以下函数,index对应的就是图片的关键值，通过index可以得到图片得信息
    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
	    #获取图片路径和对应的caption
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # caption = []
        #
        # caption.append(vocab('<start>'))
        # caption.extend([vocab(token) for token in tokens])
        # caption.append(vocab('<end>'))
        # target = torch.Tensor(caption)

        tokens = []
        allcaps = []
        # 将caption中的大写字符转化为小写字符，而后分词(按照空格、逗号、句号等)
        for i in range(5):
            allcap = coco.imgToAnns[img_id][i]['caption']
            tokens.append(nltk.tokenize.word_tokenize(str(allcap).lower()))
        tokens.sort(key=lambda x: len(x), reverse=True)
        # 在main函数中得到单词和数值对应的vocab，通过如下语句将单词转化为数值
        for i in range(5):
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens[i]])
            caption.append(vocab('<end>'))
            allcaps.append(caption)
        target = torch.Tensor(allcaps[0])
        return image, target,allcaps

    def __len__(self):
        return len(self.ids)
#一批数据执行完成__getitem__之后，会执行如下方法，批量处理数据，
#这里的参数data是__getitem__方法返回的image，target
#data是list，内部是tuple,分别是image和caption
def collate_fn(data):
    #将data中的数据按照caption中的单词数长度逆序排列，
    data.sort(key=lambda  x: len(x[1]), reverse=True)
    #通过zip方法将图像和caption分别存储到images,captions；image 是一个list，list大小为32，每个list中的元素是3*224*224
    images, captions ,allcap= zip(*data)
    Batch = len(allcap)
    Num_sentences = 5
	#此时images变为32*3*224*224
    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]


    allcaps=torch.zeros(Batch, Num_sentences,max(lengths)).long()
    for i,cap in enumerate(allcap):
        for j, c in enumerate(cap):
            end = len(c)
            c = torch.Tensor(c)
            allcaps[i,j,:end]=c[:end]

    return images, targets, lengths, allcaps

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    coco = DataLoader(root=root, json=json, vocab=vocab, transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
