import json
import torch
import config
from operator import itemgetter
from util.util import read_json, join_cate
import numpy as np
from util.metric import decode, decode_CE_test
import torch
from tqdm import tqdm


def preprocess(dataset_type, cfg=config):
    print(f'preprocessing {dataset_type}')
    data = __text2label(read_json(cfg.DataInfo.data_path[dataset_type]), cfg)
    torch.save(data, cfg.DataInfo.data_cache_path[dataset_type])
    print(f'saved at {cfg.DataInfo.data_cache_path[dataset_type]}')


def __text2label(data, cfg=config):
    preprocessed = []
    L_max = cfg.Exp.max_length - 2
    label2id = cfg.LabelInfo.label2id
    label2id_2 = cfg.LabelInfo.label2id_2
    seg2id_2 = cfg.LabelInfo.seg2id_2
    if cfg.DataInfo.language == 'en':
        label_key = 'label_ori' if cfg.Exp.use_indice else 'label'
    else:
        label_key = 'label'
    for idx, item in enumerate(tqdm(data)):
        if cfg.DataInfo.language == 'cn':
            L = len(item['text'])
            text_split = [char.replace(' ', '[UNK]') for char in list(item['text'])]
        else:
            L = item['length']
            text_split = item['text'].split()
        B_cate = [[] for i in range(L)]
        E_cate = [[] for i in range(L)]
        C_cate = [[] for i in range(2 * L - 1)]
        B_seg = [[] for i in range(L)]
        E_seg = [[] for i in range(L)]
        C_seg = [[] for i in range(2 * L - 1)]
        B_seg_ratio = [[] for i in range(L)]
        E_seg_ratio = [[] for i in range(L)]
        C_seg_ratio = [[] for i in range(2 * L - 1)]

        for cate in item[label_key].keys():
            for entity in item[label_key][cate].keys():
                for start_idx, end_idx in item[label_key][cate][entity]:
                    # length_ratio = (end_idx - start_idx + 1) / len(text_split)
                    length = min(end_idx - start_idx + 1, cfg.Exp.max_entity_length)
                    length_ratio = length / min(cfg.Exp.max_length - 2, len(text_split))
                    B_cate, B_seg, B_seg_ratio = __replace(B_cate, B_seg, B_seg_ratio, start_idx, cate, length,
                                                           length_ratio)
                    E_cate, E_seg, E_seg_ratio = __replace(E_cate, E_seg, E_seg_ratio, end_idx, cate, length,
                                                           length_ratio)
                    C_cate, C_seg, C_seg_ratio = __replace(C_cate, C_seg, C_seg_ratio, end_idx + start_idx, cate,
                                                           length, length_ratio)

        for i in range(L):
            if len(B_cate[i]) == 0:
                B_cate[i] = ['O']
                B_seg[i] = [0]
                B_seg_ratio[i] = [0]
            if len(E_cate[i]) == 0:
                E_cate[i] = ['O']
                E_seg[i] = [0]
                E_seg_ratio[i] = [0]
        for i in range(2 * L - 1):
            if len(C_cate[i]) == 0:
                C_cate[i] = ['O']
                C_seg[i] = [0]
                C_seg_ratio[i] = [0]

        gt = []
        gt_char = []
        for label in item[label_key]:
            for entity in item[label_key][label]:
                for pl, pr in item[label_key][label][entity]:
                    gt_char.append([label, pl, pr])
                    gt.append([cfg.LabelInfo.label2id[label], pl, pr])
        gt.sort()
        gt_char.sort()

        Bmin_cate = ['START'] + [B_cate[i][0] for i in range(L)][:L_max] + ['END']
        Bmax_cate = ['START'] + [B_cate[i][-1] for i in range(L)][:L_max] + ['END']
        Emin_cate = ['START'] + [E_cate[i][0] for i in range(L)][:L_max] + ['END']
        Emax_cate = ['START'] + [E_cate[i][-1] for i in range(L)][:L_max] + ['END']
        Cmin_cate = ['START'] * 2 + [C_cate[i][0] for i in range(2 * L - 1)][:L_max * 2 - 1] + ['END'] * 2
        Cmax_cate = ['START'] * 2 + [C_cate[i][-1] for i in range(2 * L - 1)][:L_max * 2 - 1] + ['END'] * 2

        Bmin_seg = [0] + [B_seg[i][0] for i in range(L)][:L_max] + [0]
        Bmax_seg = [0] + [B_seg[i][-1] for i in range(L)][:L_max] +[0]
        Emin_seg = [0] + [E_seg[i][0] for i in range(L)][:L_max] + [0]
        Emax_seg = [0] + [E_seg[i][-1] for i in range(L)][:L_max] + [0]
        Cmin_seg = [0] * 2 + [C_seg[i][0] for i in range(2 * L - 1)][:L_max * 2 - 1] + [0] * 2
        Cmax_seg = [0] * 2 + [C_seg[i][-1] for i in range(2 * L - 1)][:L_max * 2 - 1] + [0] * 2

        sample = {'id': idx,
                  'text_ori': item['text'],
                  'text': text_split[:L_max],
                  'gt': gt,
                  'gt_char': gt_char,
                  'length': L,
                  'Bmin_cate': torch.tensor([label2id[item] for item in Bmin_cate]),
                  'Bmin_seg': torch.tensor(Bmin_seg).long(),
                  'Bmin_seg_ratio': torch.tensor([0.] + [B_seg_ratio[i][0] for i in range(L)][:L_max] + [0.]),

                  'Bmax_cate': torch.tensor([label2id[item] for item in Bmax_cate]),
                  'Bmax_seg': torch.tensor(Bmax_seg).long(),
                  'Bmax_seg_ratio': torch.tensor([0.] + [B_seg_ratio[i][-1] for i in range(L)][:L_max] + [0.]),

                  'Emin_cate': torch.tensor([label2id[item] for item in Emin_cate]),
                  'Emin_seg': torch.tensor(Emin_seg).long(),
                  'Emin_seg_ratio': torch.tensor([0.] + [E_seg_ratio[i][0] for i in range(L)][:L_max] + [0.]),

                  'Emax_cate': torch.tensor([label2id[item] for item in Emax_cate]),
                  'Emax_seg': torch.tensor(Emax_seg).long(),
                  'Emax_seg_ratio': torch.tensor([0.] + [E_seg_ratio[i][-1] for i in range(L)][:L_max] + [0.]),

                  'Cmin_cate': torch.tensor([label2id[item] for item in Cmin_cate]),
                  'Cmin_seg': torch.tensor(Cmin_seg).long(),
                  'Cmin_seg_ratio': torch.tensor(
                      [0.] * 2 + [C_seg_ratio[i][0] for i in range(2 * L - 1)][:L_max * 2 - 1] + [0.] * 2),

                  'Cmax_cate': torch.tensor([label2id[item] for item in Cmax_cate]),
                  'Cmax_seg': torch.tensor(Cmax_seg).long(),
                  'Cmax_seg_ratio': torch.tensor(
                      [0.] * 2 + [C_seg_ratio[i][-1] for i in range(2 * L - 1)][:L_max * 2 - 1] + [0.] * 2),
                  }

        # add 2-level labels
        B2_cate = []
        E2_cate = []
        C2_cate = []
        B2_seg = []
        E2_seg = []
        C2_seg = []
        for i in range(len(sample['Bmin_cate'])):
            if i == 0:
                Bcate = 'START'
                Ecate = 'START'
            elif i == len(sample['Bmin_cate'])-1:
                Bcate = 'END'
                Ecate = 'END'
            else:
                Bcate1 = Bmin_cate[i]
                Bcate2 = Bmax_cate[i]
                Ecate1 = Emin_cate[i]
                Ecate2 = Emax_cate[i]
                if Bcate1 == 'O':
                    assert Bcate1 == Bcate2
                elif Bcate2 == Bcate1:
                    if sample['Bmin_seg'][i] == sample['Bmax_seg'][i]:
                        Bcate2 = 'O'
                        Bmax_seg[i] = 0
                if Ecate1 == 'O':
                    assert Ecate1 == Ecate2
                elif Ecate2 == Ecate1:
                    if sample['Emin_seg'][i] == sample['Emax_seg'][i]:
                        Ecate2 = 'O'
                        Emax_seg[i] = 0
                Bcate = join_cate(Bcate1, Bcate2)
                Ecate = join_cate(Ecate1, Ecate2)
            B2_cate.append(label2id_2[Bcate])
            E2_cate.append(label2id_2[Ecate])
            B2_seg.append(seg2id_2[join_cate(str(Bmin_seg[i]), str(Bmax_seg[i]))])
            E2_seg.append(seg2id_2[join_cate(str(Emin_seg[i]), str(Emax_seg[i]))])

        for i in range(len(sample['Cmin_cate'])):
            if i <= 1:
                Ccate = 'START'
            elif i >= len(sample['Cmin_cate'])-2:
                Ccate = 'END'
            else:
                Ccate1 = Cmin_cate[i]
                Ccate2 = Cmax_cate[i]
                if Ccate1 == 'O':
                    assert Ccate1 == Ccate2
                elif Ccate2 == Ccate1:
                    if sample['Cmin_seg'][i] == sample['Cmax_seg'][i]:
                        Ccate2 = 'O'
                        Cmax_seg[i] = 0
                Ccate = join_cate(Ccate1, Ccate2)
            C2_cate.append(label2id_2[Ccate])
            C2_seg.append(seg2id_2[join_cate(str(Cmin_seg[i]), str(Cmax_seg[i]))])
        sample['B2_cate'] = torch.tensor(B2_cate)
        sample['E2_cate'] = torch.tensor(E2_cate)
        sample['C2_cate'] = torch.tensor(C2_cate)
        sample['B2_seg'] = torch.tensor(B2_seg).long()
        sample['E2_seg'] = torch.tensor(E2_seg).long()
        sample['C2_seg'] = torch.tensor(C2_seg).long()
        preprocessed.append(sample)

        # debug
        # chunk = decode_CE_test(
        #     torch.stack((sample['Bmin_cate'], sample['Bmin_seg']), dim=-1),
        #     torch.stack((sample['Bmax_cate'], sample['Bmax_seg']), dim=-1),
        #     torch.stack((sample['Emin_cate'], sample['Emin_seg']), dim=-1),
        #     torch.stack((sample['Emax_cate'], sample['Emax_seg']), dim=-1),
        #     torch.stack((sample['Cmin_cate'], sample['Cmin_seg']), dim=-1),
        #     torch.stack((sample['Cmax_cate'], sample['Cmax_seg']), dim=-1),
        # )
        # wrong = False
        # for en in chunk:
        #     if en not in gt:
        #         wrong = True
        # for en in gt:
        #     if en not in chunk:
        #         wrong = True
        # if wrong:
        #     chunk.sort()
        #     gt.sort()
        #     print('--------------')
        #     print(item['text'])
        #     print([[cfg.LabelInfo.id2label[ent[0]], item['text'][ent[1]: ent[2]+1]] for ent in gt])
        #     print([[cfg.LabelInfo.id2label[ent[0]], item['text'][ent[1]: ent[2]+1]] for ent in chunk])
    return preprocessed


def __replace(cate, seg, seg_ratio, idx, rep_cate, rep_seg, rep_seg_ratio):
    cate[idx].append(rep_cate)
    seg[idx].append(rep_seg)
    seg_ratio[idx].append(rep_seg_ratio)
    indice = np.argsort(seg[idx])
    seg[idx] = [seg[idx][i] for i in indice]
    cate[idx] = [cate[idx][i] for i in indice]
    seg_ratio[idx] = [seg_ratio[idx][i] for i in indice]
    return cate, seg, seg_ratio
