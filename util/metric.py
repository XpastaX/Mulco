# def decode(seg,center):
#     """
#     :param seg: [B,I,I,E,O,O,...]
#     :param center: [I,I,I,C1-xxx,I,I,I,O,O,O,...] length=2*len(seg)-1
#     :return: [['PER', 0,1], ['LOC', 3, 3]]
#     """
#     chunck = []     # list of entities and their corresponding indices
#     L = len(seg)    # length of the input sentence
#     for cen_idx, cen_tag in enumerate(center):
#         if cen_tag[0] == 'C':   # find center point
#             if cen_idx%2==1:    # entity has even number of characters
#                 pl=int(cen_idx/2)   # left pointer
#                 pr=int(cen_idx/2)+1 # right pointer
#             else:   # entity has odd number of characters
#                 pl=int(cen_idx/2)
#                 pr=int(cen_idx/2)
#             # extract the number of entities and their categories
#             num,cate=cen_tag.split('-')
#             num=int(num[1:])
#             cate=cate.split('/')
#
#             j=0     # j-th entity
#             while pl>=0 and pr <=L:
#                 found=False
#                 if pl==pr: # odd
#                     if seg[pl]=='S':    # if single char entity
#                         found=True
#                     # elif seg[pl] == 'I':
#                     #     continue
#                 elif seg[pl] == 'B' and seg[pr] == 'E':
#                     found=True
#
#                 elif seg[pl] == 'S' and seg[pr] =='E':
#                     found = True
#
#                 elif seg[pl] == 'B' and seg[pr] == 'S':
#                     found = True
#                 else:   # BI,IE,II,BO,OE,OI,IO
#                     pass
#                 if found:
#                     chunck.append([cate[j], pl, pr])
#                     j += 1
#                     num -= 1
#                 pl-=1
#                 pr+=1
#                 if num <=0: break
#     return chunck
from collections import Counter
import config
import torch



class decode(object):
    def __init__(self, cfg=config):
        self.O = cfg.LabelInfo.label2id['O']
        self.entity = {}
        self.confidence = {}
        self.scope = {}
        self.cfg = cfg

    def init(self):
        self.entity = {}
        self.confidence = {}
        self.scope = {}

    def __get_entity_length(self, idx, l_ent, scope):
        if scope[0] == 'B':
            start_idx = idx
            end_idx = idx + l_ent - 1
        elif scope[0] == 'E':
            end_idx = idx
            start_idx = idx - l_ent + 1
        elif scope[0] == 'C':
            if idx % 2 == 1:  # length is even
                start_idx = (idx - 1) / 2 - l_ent / 2 + 1
            else:
                start_idx = idx / 2 - (l_ent - 1) / 2
            end_idx = start_idx + l_ent - 1
        else:
            raise NotImplementedError
        return int(start_idx), int(end_idx)

    def __decode_scope(self, L, seq, scope):
        L_seq = L if scope[0] != 'C' else 2 * L - 1
        for idx in range(L_seq):  # for each token in a sentence
            item_seq = seq[idx]  # [cate,seg_length,confidence]
            if item_seq[0] <= self.O or item_seq[1] == 0: continue
            L_ent = item_seq[1] if not self.cfg.Exp.use_ratio else torch.round(L * item_seq[1])
            start_idx, end_idx = self.__get_entity_length(idx, L_ent, scope)
            self.__add(start_idx, end_idx, L, item_seq[0], item_seq[-1], scope=scope)

    def __call__(self, Bmin, Bmax, Emin, Emax, Cmin=None, Cmax=None, return_scope=False):
        self.init()
        chunk = []
        scope_list = []
        Bmin = Bmin[1:-1]
        Bmax = Bmax[1:-1]
        Emin = Emin[1:-1]
        Emax = Emax[1:-1]

        L = len(Bmin)  # length of the seq
        self.__decode_scope(L, Bmin, 'Bmin')
        self.__decode_scope(L, Bmax, 'Bmax')
        self.__decode_scope(L, Emin, 'Emin')
        self.__decode_scope(L, Emax, 'Emax')
        if Cmin is not None:
            Cmin = Cmin[2:-2]
            Cmax = Cmax[2:-2]
            self.__decode_scope(L, Cmin, 'Cmin')
            self.__decode_scope(L, Cmax, 'Cmax')

        for key in self.entity:
            start_idx, end_idx = key.split('-')
            chunk.append([int(self.entity[key]), int(start_idx), int(end_idx)])
            scope_list.append(self.scope[key])
        if return_scope:
            return chunk, scope_list
        else:
            return chunk

    def __add(self, start_idx, end_idx, L, cate, con_score=None, scope=None):
        if start_idx < 0: return
        if end_idx < start_idx: return
        if end_idx >= L: return
        key = f"{start_idx}-{end_idx}"
        if key not in self.entity:
            self.entity[key] = cate
            if self.confidence is not None:
                self.confidence[key] = con_score
            if scope is not None:
                self.scope[key] = ['-'.join([scope, str(int(cate))])]
        else:
            if self.confidence is not None:
                if self.confidence[key] < con_score:
                    self.entity[key] = cate
                    self.confidence[key] = con_score
                    if scope is not None:
                        self.scope[key].append('-'.join([scope, str(int(cate))]))


def __add(entity, start_idx, end_idx, L, cate, confidence=None, score=None, scope=None):
    start_idx = int(min(max(0, start_idx), L))
    end_idx = int(min(max(start_idx, end_idx), L))
    key = f"{start_idx}-{end_idx}"
    if key not in entity:
        entity[key] = cate
        if confidence is not None:
            confidence[key] = score
    else:
        if confidence is not None:
            if confidence[key] < score:
                entity[key] = cate
                confidence[key] = score


def decode_CE_test(Bmin, Bmax, Emin, Emax, Cmin=None, Cmax=None, O='O'):
    """
    :param Bmin: [[cate,length,confidence]...]
    :param Bmax: [[cate,length,confidence]...]
    :param Emin: [[cate,length,confidence]...]
    :param Emax: [[cate,length,confidence]...]
    :return: chuncks  [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunk = []
    Bmin = Bmin[1:-1]
    Bmax = Bmax[1:-1]
    Emin = Emin[1:-1]
    Emax = Emax[1:-1]
    if Cmin is not None:
        Cmin = Cmin[2:-2]
        Cmax = Cmax[2:-2]
    L = len(Bmin)
    entity = {}
    O = 2
    # if type(Bmin[0][1]) == torch.Tensor:
    round = torch.round
    for idx in range(L):
        item_Bmin = Bmin[idx]
        item_Bmax = Bmax[idx]
        item_Emin = Emin[idx]
        item_Emax = Emax[idx]
        if item_Bmin[0] != O:
            start_idx = idx
            end_idx = idx + item_Bmin[1] - 1
            cate = item_Bmin[0]
            __add(entity, start_idx, end_idx, L, cate, )
        if item_Bmax[0] != O:
            start_idx = idx
            end_idx = idx + item_Bmax[1] - 1
            cate = item_Bmax[0]
            __add(entity, start_idx, end_idx, L, cate, )
        if item_Emin[0] != O:
            end_idx = idx
            start_idx = idx - item_Emin[1] + 1
            cate = item_Emin[0]
            __add(entity, start_idx, end_idx, L, cate, )
        if item_Emax[0] != O:
            end_idx = idx
            start_idx = idx - item_Emax[1] + 1
            cate = item_Emax[0]
            __add(entity, start_idx, end_idx, L, cate, )
    # for idx in range(2 * L - 1):
    #     item_Cmin = Cmin[idx]
    #     item_Cmax = Cmax[idx]
    #     if item_Cmin[0] != O:
    #         length = item_Cmin[1]
    #         if idx % 2 == 1:  # length is even
    #             start_idx = (idx - 1) / 2 - length / 2 + 1
    #         else:
    #             start_idx = idx / 2 - (length - 1) / 2
    #         end_idx = start_idx + length - 1
    #         cate = item_Cmin[0]
    #         __add(entity, start_idx, end_idx, L, cate, )
        # if item_Cmax[0] != O:
        #     length = item_Cmax[1]
        #     if idx % 2 == 1:  # length is even
        #         start_idx = (idx - 1) / 2 - length / 2 + 1
        #     else:
        #         start_idx = idx / 2 - (length - 1) / 2
        #     end_idx = start_idx + length - 1
        #     cate = item_Cmax[0]
        #     __add(entity, start_idx, end_idx, L, cate, )
    for key in entity:
        start_idx, end_idx = key.split('-')
        chunk.append([int(entity[key]), int(start_idx), int(end_idx)])
    return chunk


def generate_result(text, pred, gt, scope, id2label, is_en=False, tokenizer=None):
    pred_ent = []
    gt_ent = []
    for i, [cate, start, end] in enumerate(pred):
        if not is_en:
            entity = text[start:end + 1]
        else:
            entity = tokenizer.decoder(text[start:end + 1])
        pred_ent.append(['-'.join([entity, id2label[cate], str(start), str(end)]), scope[i]])
    for cate, start, end in gt:
        if not is_en:
            entity = text[start:end + 1]
        else:
            entity = tokenizer.decoder(text[start:end + 1])
        gt_ent.append('-'.join([entity, id2label[cate],  str(start), str(end)]))
    pred_ent.sort()
    gt_ent.sort()
    return pred_ent, gt_ent


class SeqEntityScore(object):
    def __init__(self):
        self.origins = []
        self.founds = []
        self.rights = []
        self.right_cut = []
        self.ori_cut = []
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []
        self.right_cut = []
        self.ori_cut = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])

        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        right_cut = len(self.right_cut)
        recall, precision, f1 = self.compute(origin, found, right)
        cut_recall = 0 if origin == 0 else right_cut / origin
        cut_acc = 0 if found == 0 else right_cut / found
        cut_f1 = 0. if cut_recall + cut_acc == 0 else (2 * cut_acc * cut_recall) / (cut_acc + cut_recall)
        pred_acc = 0 if right_cut == 0 else right / right_cut
        return {'acc': precision, 'recall': recall, 'cut_acc': cut_acc, 'cut_recall': cut_recall, 'pred_acc': pred_acc,
                'cut_f1': cut_f1, 'f1': f1, }, class_info

    def update(self, gt, pred):
        for label_entities, pre_entities in zip(gt, pred):
            ori_cut = [l_e[1:] for l_e in label_entities]
            self.origins.extend(label_entities)
            self.ori_cut.extend(ori_cut)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])
            self.right_cut.extend([pre_entity for pre_entity in pre_entities if pre_entity[1:] in ori_cut])
