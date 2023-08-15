import torch
from torch.utils.data import Dataset, DataLoader
import config
from transformers import AutoTokenizer


class FiNEdataset(Dataset):
    def __init__(self, dataset_type, cfg=config):
        self.label_list = None
        self.label_dict = None
        self.data_path = cfg.DataInfo.data_cache_path[dataset_type]
        self.data = None
        self.size = cfg.DataInfo.num_sample[dataset_type]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.plm)
        self.label2id = cfg.LabelInfo.label2id
        self.id2label = cfg.LabelInfo.id2label
        self.max_len = cfg.Exp.max_length if dataset_type == 'train' else 512
        self.flatL2id = {}
        self.flatid2L = {}
        tag_list = ['START', 'END', 'O']
        for cls in cfg.LabelInfo.categories:
            tag_list.append('B-%s' % cls)
            tag_list.append('I-%s' % cls)
            tag_list.append('E-%s' % cls)
            tag_list.append('S-%s' % cls)
        for idx, cate in enumerate(tag_list):
            self.flatL2id[cate] = idx
            self.flatid2L[idx] = cate

    def __len__(self):
        return self.size

    def load_data(self):
        self.data = torch.load(self.data_path)

    def transform(self, cate_seq, len_seq):
        Most = [[self.flatL2id['O']] * len(item) for item in cate_seq]
        for i in range(len(Most)):
            B_cate_item = [self.id2label[int(item)] for item in cate_seq[i]]
            B_seg_item = len_seq[i]
            for j in range(len(Most[i])):
                if j == 0:
                    Most[i][j] = self.flatL2id['START']
                elif j == len(Most[i]) - 1:
                    Most[i][j] = self.flatL2id['END']
                else:
                    if B_cate_item[j] != 'O':
                        cate = B_cate_item[j]
                        L = int(B_seg_item[j])
                        for k in range(L):
                            if k == 0:
                                seg = 'B' if L > 1 else 'S'
                            elif k == L - 1:
                                seg = 'E'
                            else:
                                seg = 'I'
                            Most[i][j + k] = self.flatL2id['-'.join([seg, cate])]

        return Most

    def collate_fcn(self, batch):
        id_list = [item['id'] for item in batch]
        text = [item['text_ori'] for item in batch]
        gt = [item['gt'] for item in batch]
        # TODO: 用Bmax 构建outermost，Bmin 构建 innermost
        Bmin_cate = [item['Bmin_cate'] for item in batch]
        Bmax_cate = [item['Bmax_cate'] for item in batch]
        Bmin_seg = [item['Bmin_seg'] for item in batch]
        Bmax_seg = [item['Bmax_seg'] for item in batch]

        innerMost = self.transform(Bmin_cate, Bmin_seg)
        outerMost = self.transform(Bmax_cate, Bmax_seg)

        innerMost_processed = []
        outerMost_processed = []
        for i in range(len(innerMost)):
            innerMost_processed += innerMost[i]
        for i in range(len(outerMost)):
            outerMost_processed += outerMost[i]
        innerMost_processed = torch.tensor(innerMost_processed).long()
        outerMost_processed = torch.tensor(outerMost_processed).long()

        # prepare tokenized texts
        text_split_list = [item['text'] for item in batch]
        tokenized = self.tokenizer(text_split_list, padding=True, truncation=True, max_length=self.max_len,
                                   is_split_into_words=True, return_tensors="pt")

        samples = {
            'id_list': id_list,
            'text_list': text,
            'innerMost': innerMost,
            'outerMost': outerMost,
            'gt': gt,
            'innerMost_processed': innerMost_processed,
            'outerMost_processed': outerMost_processed,
            'tokenized': tokenized,
        }

        return samples

    def __getitem__(self, idx):
        if self.data is None:
            self.load_data()
        return self.data[idx]


def get_dataloader(dataset_type, cfg=config):
    dataset = FiNEdataset(dataset_type, cfg)
    if cfg.Exp.num_workers <= 0:
        return DataLoader(dataset, batch_size=cfg.Exp.batch_size,
                          shuffle=cfg.Exp.shuffle if dataset_type == 'train' else False,
                          sampler=None,
                          batch_sampler=None, num_workers=cfg.Exp.num_workers, collate_fn=dataset.collate_fcn,
                          pin_memory=True, drop_last=False, timeout=0,
                          worker_init_fn=None)
    else:
        return DataLoader(dataset, batch_size=cfg.Exp.batch_size,
                          shuffle=cfg.Exp.shuffle if dataset_type == 'train' else False,
                          sampler=None,
                          batch_sampler=None, num_workers=cfg.Exp.num_workers, collate_fn=dataset.collate_fcn,
                          pin_memory=True, drop_last=False, timeout=0,
                          worker_init_fn=None, prefetch_factor=cfg.Exp.prefetch_factor,
                          persistent_workers=True)
