import copy
import nltk
import random
import re
from nltk.tokenize import TreebankWordTokenizer

from torch.utils.data import Sampler
from torchdata.datapipes.iter import IterableWrapper, FileOpener
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

def get_pipe(trn_file_path, tst_file_path=None):
    trn_datapipe = IterableWrapper([trn_file_path])
    trn_datapipe = FileOpener(trn_datapipe, mode='b')
    trn_datapipe = trn_datapipe.parse_csv()
    tst_datapipe = None
    
    if tst_file_path is not None:
        tst_datapipe = IterableWrapper([tst_file_path])
        tst_datapipe = FileOpener(tst_datapipe, mode='b')
        tst_datapipe = tst_datapipe.parse_csv()
        
    return trn_datapipe, tst_datapipe


def preprocess_english(text):
    my_text = copy.copy(text)
    my_text = my_text.replace('\n', '')
    sents = nltk.sent_tokenize(my_text)
    tokenizer = TreebankWordTokenizer()
    stopwords = nltk.corpus.stopwords.words('english')
    
    p = re.compile('[^A-Za-z]')
    result = []
    for sent in sents:
        sent = sent.lower() # 소문자로 변환
        sent = p.sub(' ', sent) # 각 문장에서 특수문자 제거
        word_tokens = tokenizer.tokenize(sent) # word tokenization
        for token in word_tokens:
            if token not in stopwords:
                result.append(token) # stopwords removal
    return result


def yield_tokens(data_iter, tokenizer, data_type = 'description'):
    assert data_type in ['class','title','description']
    for label, title, text in data_iter:
        if data_type == 'description':
            yield tokenizer(text)
        elif data_type == 'title':
            yield tokenizer(title)
        elif data_type == 'class':
            yield [label]

            
def get_vocab(train_datapipe, tokenizer, data_type = 'description', specials = ["<UNK>", "<PAD>"], vocab_size=None):
    assert data_type in ['class','title','description']
    vocab = build_vocab_from_iterator(yield_tokens(train_datapipe, tokenizer, data_type),
                                      min_freq=3,
                                      specials=specials,
                                      max_tokens=vocab_size)
    vocab.set_default_index(vocab["<UNK>"])
    return vocab


class BatchSamplerSimilarLength(Sampler):
    def __init__(self, dataset, batch_size, tokenizer, indices=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # get the indices and length
        self.indices = [(i, len(tokenizer(s[1]))) for i, s in enumerate(dataset)]
        # if indices are passed, then use only the ones passed (for ddp)
        if indices is not None:
            self.indices = torch.tensor(self.indices)[indices].tolist()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(self.indices), self.batch_size * 10):
            pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 10], key=lambda x: x[1]))
        self.pooled_indices = [x[0] for x in pooled_indices]

        # Comment in for validation
        #self.pooled_lengths = [x[1] for x in pooled_indices]
        #print(self.pooled_lengths)
        #print(self.pooled_indices)

        # yield indices for current batch
        batches = [self.pooled_indices[i:i + self.batch_size] for i in
                   range(0, len(self.pooled_indices), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.pooled_indices) // self.batch_size