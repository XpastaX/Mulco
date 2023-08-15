import torch
from torch.utils.data import Dataset, DataLoader
import config
from transformers import AutoTokenizer
import os


class NNERData(Dataset):
    def __init__(self, dataset_type, cfg=config):
        self.label_list = None
        self.label_dict = None
        self.data_path = cfg.DataInfo.data_cache_path[dataset_type]
        self.data = None
        self.size = cfg.DataInfo.num_sample[dataset_type]
        self.max_len = cfg.Exp.max_length if dataset_type == 'train' else 512

    def __len__(self):
        return self.size

    def load_data(self):
        self.data = torch.load(self.data_path)

    def __getitem__(self, idx):
        if self.data is None:
            self.load_data()
        return self.data[idx]


class MyCollator(object):
    def __init__(self, tokenizer, cfg=config):
        self.max_length = cfg.Exp.max_length
        self.tokenizer = tokenizer
        self.use_sampler = cfg.Exp.use_sampler
        self.O = cfg.LabelInfo.label2id['O']
        self.language = cfg.DataInfo.language
        self.use_2level_cate = cfg.Exp.use_2level_cate
        self.use_2level_seg = cfg.Exp.use_2level_seg
        self.use_ratio = cfg.Exp.use_ratio

    def __call__(self, batch):
        id_list = [item['id'] for item in batch]
        gt = [item['gt'] for item in batch]

        if self.use_2level_cate:
            B2_cate = [item['B2_cate'] for item in batch]
            E2_cate = [item['E2_cate'] for item in batch]
            C2_cate = [item['C2_cate'] for item in batch]
            B2_cate_processed = torch.cat(B2_cate, dim=-1)
            E2_cate_processed = torch.cat(E2_cate, dim=-1)
            C2_cate_processed = torch.cat(C2_cate, dim=-1)
            label_cate = [B2_cate_processed, None, E2_cate_processed, None, C2_cate_processed, None]
            label_cate_for_mask = label_cate
        if not self.use_2level_cate or not self.use_2level_seg:
            Bmin_cate = [item['Bmin_cate'] for item in batch]
            Bmax_cate = [item['Bmax_cate'] for item in batch]
            Emin_cate = [item['Emin_cate'] for item in batch]
            Emax_cate = [item['Emax_cate'] for item in batch]
            Cmin_cate = [item['Cmin_cate'] for item in batch]
            Cmax_cate = [item['Cmax_cate'] for item in batch]

            Bmin_cate_processed = torch.cat(Bmin_cate, dim=-1)
            Bmax_cate_processed = torch.cat(Bmax_cate, dim=-1)
            Emin_cate_processed = torch.cat(Emin_cate, dim=-1)
            Emax_cate_processed = torch.cat(Emax_cate, dim=-1)
            Cmin_cate_processed = torch.cat(Cmin_cate, dim=-1)
            Cmax_cate_processed = torch.cat(Cmax_cate, dim=-1)
            if not self.use_2level_seg:
                label_cate_for_mask = [Bmin_cate_processed, Bmax_cate_processed,
                              Emin_cate_processed, Emax_cate_processed,
                              Cmin_cate_processed, Cmax_cate_processed]
            if not self.use_2level_cate:
                label_cate = [Bmin_cate_processed, Bmax_cate_processed,
                              Emin_cate_processed, Emax_cate_processed,
                              Cmin_cate_processed, Cmax_cate_processed]
                label_cate_for_mask = label_cate

        if self.use_2level_seg:
            B2_seg = [item['B2_seg'] for item in batch]
            E2_seg = [item['E2_seg'] for item in batch]
            C2_seg = [item['C2_seg'] for item in batch]
            B2_seg_processed = torch.cat(B2_seg, dim=-1)
            E2_seg_processed = torch.cat(E2_seg, dim=-1)
            C2_seg_processed = torch.cat(C2_seg, dim=-1)
            label_seg = [B2_seg_processed, None, E2_seg_processed, None, C2_seg_processed, None]
        else:
            if self.use_ratio:
                Bmin_seg = [item['Bmin_seg_ratio'] for item in batch]
                Bmax_seg = [item['Bmax_seg_ratio'] for item in batch]
                Emin_seg = [item['Emin_seg_ratio'] for item in batch]
                Emax_seg = [item['Emax_seg_ratio'] for item in batch]
                Cmin_seg = [item['Cmin_seg_ratio'] for item in batch]
                Cmax_seg = [item['Cmax_seg_ratio'] for item in batch]
            else:
                Bmin_seg = [item['Bmin_seg'] for item in batch]
                Bmax_seg = [item['Bmax_seg'] for item in batch]
                Emin_seg = [item['Emin_seg'] for item in batch]
                Emax_seg = [item['Emax_seg'] for item in batch]
                Cmin_seg = [item['Cmin_seg'] for item in batch]
                Cmax_seg = [item['Cmax_seg'] for item in batch]

            Bmin_seg_processed = torch.cat(Bmin_seg, dim=-1)
            Bmax_seg_processed = torch.cat(Bmax_seg, dim=-1)
            Emin_seg_processed = torch.cat(Emin_seg, dim=-1)
            Emax_seg_processed = torch.cat(Emax_seg, dim=-1)
            Cmin_seg_processed = torch.cat(Cmin_seg, dim=-1)
            Cmax_seg_processed = torch.cat(Cmax_seg, dim=-1)

            label_seg = [Bmin_seg_processed, Bmax_seg_processed,
                         Emin_seg_processed, Emax_seg_processed,
                         Cmin_seg_processed, Cmax_seg_processed]

        text_split_list = [item['text'] for item in batch]
        tokenized = self.tokenizer(text_split_list, padding=True, truncation=True, max_length=self.max_length,
                                   is_split_into_words=True, return_tensors="pt")

        mask = tokenized['attention_mask'].contiguous().eq(1)

        # construct mask for C scope
        L = len(tokenized['attention_mask'][0])
        mask_C = []
        for item in batch:
            length = 2 * (item['length'] + 2) - 1
            pad = 2 * (L - item['length'] - 2)
            mask_C.append([True] * length + [False] * pad)
        try:
            mask_C = torch.tensor(mask_C).bool().contiguous()
        except:
            pass
        # generate token mask for sampler

        active = []
        for idx, item in enumerate(label_cate_for_mask):
            if item is None:
                active.append(torch.tensor([False]*len([label_cate[idx-1]])))
            else:
                if self.use_sampler:
                    __active = item > self.O
                    tmp = item > self.O
                    tmp[:-1] = tmp[:-1] + __active[1:]
                    tmp[1:] = tmp[1:] + __active[:-1]
                    active.append(tmp)
                else:
                    active.append(item > self.O)

        samples = {
            'id_list': id_list,
            'text_ori': [item['text_ori'] for item in batch],
            'gt': gt,
            'processed_cate': label_cate,
            'processed_seg': label_seg,
            'tokenized': tokenized,
            'mask': mask,
            'Cmask': mask_C,
            'entity_mask': active,
            'length': [item['length'] for item in batch]
        }
        return samples


def get_dataloader(dataset_type, cfg=config):
    dataset = NNERData(dataset_type, cfg)
    tokenizer = AutoTokenizer.from_pretrained(config.plm)
    collate_fcn = MyCollator(tokenizer, cfg)
    if cfg.Exp.num_workers <= 0:
        return DataLoader(dataset, batch_size=cfg.Exp.batch_size,
                          shuffle=cfg.Exp.shuffle if dataset_type == 'train' else False,
                          sampler=None,
                          batch_sampler=None, num_workers=cfg.Exp.num_workers, collate_fn=collate_fcn,
                          pin_memory=True, drop_last=False, timeout=0,
                          worker_init_fn=None)
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        return DataLoader(dataset, batch_size=cfg.Exp.batch_size,
                          shuffle=cfg.Exp.shuffle if dataset_type == 'train' else False,
                          sampler=None,
                          batch_sampler=None, num_workers=cfg.Exp.num_workers, collate_fn=collate_fcn,
                          pin_memory=True, drop_last=False, timeout=0,
                          worker_init_fn=None, prefetch_factor=cfg.Exp.prefetch_factor,
                          persistent_workers=True)


if __name__ == '__main__':
    pass
