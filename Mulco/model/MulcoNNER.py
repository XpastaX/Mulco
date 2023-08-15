import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import config
from model.classifier.Mulco_cls_lstm import BaseClassifier
from util.metric import decode
from datetime import datetime


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss


class Mulco(nn.Module):
    def __init__(self, cfg=config):
        super(Mulco, self).__init__()
        self.cfg = cfg
        self.device = None
        self.encoder = AutoModel.from_pretrained(cfg.plm)
        self.classifier = BaseClassifier(cfg)
        self.decoder = decode(cfg)
        self.CE = torch.nn.CrossEntropyLoss() if not cfg.Exp.use_Focal else FocalLoss()
        if cfg.Exp.use_ratio:
            self.seg_cri = nn.L1Loss()
        else:
            self.seg_cri = torch.nn.CrossEntropyLoss() if not cfg.Exp.use_Focal else FocalLoss()
        self.cate_out = cfg.Exp.cate_out_dim
        self.seg_out = cfg.Exp.seg_out_dim
        self.O = cfg.LabelInfo.label2id['O']
        self.scope = cfg.Exp.scope
        self.hasC = 'C' in self.scope
        self.llid2lid = self.cfg.LabelInfo.llid2lid
        self.ssid2sid = self.cfg.LabelInfo.ssid2sid

    def forward(self, text, l_cate, l_seg, active, seq_mask, seq_mask2, indice=None):
        if self.device is None:
            self.device = next(self.parameters()).device
        token = {key: text[key].to(self.device) for key in text.keys()}
        for i in range(len(active)):
            active[i] = active[i].to(self.device)
        seq_mask = seq_mask.to(self.device).reshape(-1)
        if self.hasC:
            seq_mask2 = seq_mask2.to(self.device).reshape(-1)
        fea = self.encoder(**token)['last_hidden_state']
        L = token['attention_mask'].sum(dim=-1)
        x = self.classifier(fea, L)

        # cal loss
        L_Bmax_cate = 0
        L_Emax_cate = 0
        L_Bmax_seg = 0
        L_Emax_seg = 0
        L_Cmax_seg = 0
        L_Cmax_cate = 0
        L_Cmin_seg = 0
        L_Cmin_cate = 0

        L_Bmin_cate = self.CE(x[0].contiguous().view(-1, self.cate_out)[seq_mask], l_cate[0].to(self.device))
        L_Emin_cate = self.CE(x[2].contiguous().view(-1, self.cate_out)[seq_mask], l_cate[2].to(self.device))
        L_Bmin_seg = self.seg_cri(x[6].contiguous().view(-1, self.seg_out)[seq_mask].squeeze()[active[0]],
                                  l_seg[0].to(self.device)[active[0]])
        L_Emin_seg = self.seg_cri(x[8].contiguous().view(-1, self.seg_out)[seq_mask].squeeze()[active[2]],
                                  l_seg[2].to(self.device)[active[2]])
        if self.hasC:
            L_Cmin_cate = self.CE(x[4].contiguous().view(-1, self.cate_out)[seq_mask2], l_cate[4].to(self.device))
            L_Cmin_seg = self.seg_cri(x[10].contiguous().view(-1, self.seg_out)[seq_mask2].squeeze()[active[4]],
                                      l_seg[4].to(self.device)[active[4]])

        if not self.cfg.Exp.use_2level_cate:
            L_Bmax_cate = self.CE(x[1].contiguous().view(-1, self.cate_out)[seq_mask], l_cate[1].to(self.device))
            L_Emax_cate = self.CE(x[3].contiguous().view(-1, self.cate_out)[seq_mask], l_cate[3].to(self.device))
            if self.hasC:
                L_Cmax_cate = self.CE(x[5].contiguous().view(-1, self.cate_out)[seq_mask2], l_cate[5].to(self.device))

        if not self.cfg.Exp.use_2level_seg:
            L_Bmax_seg = self.seg_cri(x[7].contiguous().view(-1, self.seg_out)[seq_mask].squeeze()[active[1]],
                                      l_seg[1].to(self.device)[active[1]])
            L_Emax_seg = self.seg_cri(x[9].contiguous().view(-1, self.seg_out)[seq_mask].squeeze()[active[3]],
                                      l_seg[3].to(self.device)[active[3]])
            if self.hasC:
                L_Cmax_seg = self.seg_cri(x[11].contiguous().view(-1, self.seg_out)[seq_mask2].squeeze()[active[5]],
                                          l_seg[5].to(self.device)[active[5]])

        L_cate = L_Bmin_cate + L_Bmax_cate + L_Emin_cate + L_Emax_cate + L_Cmin_cate + L_Cmax_cate
        L_seg = L_Bmin_seg + L_Bmax_seg + L_Emin_seg + L_Emax_seg + L_Cmin_seg + L_Cmax_seg
        L = L_cate + L_seg
        return L, L_cate, L_seg

    def inference(self, text, mask, Cmask):
        # start = datetime.now()
        if self.device is None:
            self.device = next(self.parameters()).device

        token = {key: text[key].to(self.device) for key in text.keys()}
        # extract fea
        fea = self.encoder(**token)['last_hidden_state']
        # actual length of each sentence in the batch (+2 for CLS and SEP)
        L = token['attention_mask'].sum(dim=-1)
        # cls for result
        # returned [B_cate, Bmax_cate, E_cate, Emax_cate, C_cate, Cmax_cate,
        #           B_seg, Bmax_seg, E_seg, Emax_seg, C_seg, Cmax_seg]
        x = self.classifier(fea, L)

        if not self.cfg.Exp.use_ratio:
            for i in range(6, 12):
                if x[i] is not None:
                    x[i] = torch.argmax(x[i], dim=-1).to(torch.device('cpu'))

        # pre-decode if use 2level components
        Bmin_con, Bmin_sp = torch.max(x[0].to(torch.device('cpu')), dim=-1)
        Emin_con, Emin_sp = torch.max(x[2].to(torch.device('cpu')), dim=-1)
        if self.cfg.Exp.use_2level_cate:
            min_tmp = []
            max_tmp = []
            for item in Bmin_sp:
                tmp1=[]
                tmp2=[]
                for token in item:
                    a, b = self.llid2lid[int(token)]
                    tmp1.append(a)
                    tmp2.append(b)
                min_tmp.append(tmp1)
                max_tmp.append(tmp2)
            Bmin_sp = torch.tensor(min_tmp)
            Bmax_sp = torch.tensor(max_tmp)
            Bmax_con = Bmin_con = Bmin_con
            min_tmp = []
            max_tmp = []
            for item in Emin_sp:
                tmp1 = []
                tmp2 = []
                for token in item:
                    a, b = self.llid2lid[int(token)]
                    tmp1.append(a)
                    tmp2.append(b)
                min_tmp.append(tmp1)
                max_tmp.append(tmp2)
            Emin_sp = torch.tensor(min_tmp)
            Emax_sp = torch.tensor(max_tmp)
            Emax_con = Emin_con
        else:
            Bmax_con, Bmax_sp = torch.max(x[1].to(torch.device('cpu')), dim=-1)
            Emax_con, Emax_sp = torch.max(x[3].to(torch.device('cpu')), dim=-1)

        #--------------------------------------
        if self.cfg.Exp.use_2level_seg:
            min_tmp = []
            max_tmp = []
            for item in x[6]:
                tmp1 = []
                tmp2 = []
                for token in item:
                    a, b = self.ssid2sid[int(token)]
                    tmp1.append(a)
                    tmp2.append(b)
                min_tmp.append(tmp1)
                max_tmp.append(tmp2)
            x[6] = torch.tensor(min_tmp)
            x[7] = torch.tensor(max_tmp)
            min_tmp = []
            max_tmp = []
            for item in x[8]:
                tmp1 = []
                tmp2 = []
                for token in item:
                    a, b = self.ssid2sid[int(token)]
                    tmp1.append(a)
                    tmp2.append(b)
                min_tmp.append(tmp1)
                max_tmp.append(tmp2)
            x[8] = torch.tensor(min_tmp)
            x[9] = torch.tensor(max_tmp)

        Bmin = torch.stack([Bmin_sp, x[6].squeeze(), Bmin_con], dim=-1).to(torch.device('cpu'))
        Bmax = torch.stack([Bmax_sp, x[7].squeeze(), Bmax_con], dim=-1).to(torch.device('cpu'))
        Emin = torch.stack([Emin_sp, x[8].squeeze(), Emin_con], dim=-1).to(torch.device('cpu'))
        Emax = torch.stack([Emax_sp, x[9].squeeze(), Emax_con], dim=-1).to(torch.device('cpu'))

        pred = []
        scope = []
        Cmin = None
        Cmax = None
        if self.hasC:
            Cmin_con, Cmin_sp = torch.max(x[4].to(torch.device('cpu')), dim=-1)
            if self.cfg.Exp.use_2level_cate:
                min_tmp = []
                max_tmp = []
                for item in Cmin_sp:
                    tmp1 = []
                    tmp2 = []
                    for token in item:
                        a, b = self.llid2lid[int(token)]
                        tmp1.append(a)
                        tmp2.append(b)
                    min_tmp.append(tmp1)
                    max_tmp.append(tmp2)
                Cmin_sp = torch.tensor(min_tmp)
                Cmax_sp = torch.tensor(max_tmp)
                Cmax_con = Cmin_con
            else:
                Cmax_con, Cmax_sp = torch.max(x[5].to(torch.device('cpu')), dim=-1)
            if self.cfg.Exp.use_2level_seg:
                min_tmp = []
                max_tmp = []
                for item in x[10]:
                    tmp1 = []
                    tmp2 = []
                    for token in item:
                        a, b = self.ssid2sid[int(token)]
                        tmp1.append(a)
                        tmp2.append(b)
                    min_tmp.append(tmp1)
                    max_tmp.append(tmp2)
                x[10] = torch.tensor(min_tmp)
                x[11] = torch.tensor(max_tmp)
            else:
                x[11] = x[11].to(torch.device('cpu'))

            Cmin = torch.stack([Cmin_sp, x[10].squeeze(), Cmin_con], dim=-1).to(torch.device('cpu'))
            Cmax = torch.stack([Cmax_sp, x[11].squeeze(), Cmax_con], dim=-1).to(torch.device('cpu'))
        # print(f"特征耗时：{datetime.now()-start}")
        # start = datetime.now()
        # decode
        for idx in range(len(text['input_ids'])):
            Bmin_input = Bmin[idx][mask[idx]]
            Bmax_input = Bmax[idx][mask[idx]]
            Emin_input = Emin[idx][mask[idx]]
            Emax_input = Emax[idx][mask[idx]]
            if self.hasC:
                Cmin_input = Cmin[idx][Cmask[idx]]
                Cmax_input = Cmax[idx][Cmask[idx]]
            else:
                Cmin_input = None
                Cmax_input = None
            _pred, _scope = self.decoder(Bmin_input, Bmax_input, Emin_input, Emax_input, Cmin_input, Cmax_input, True)
            pred.append(_pred)
            scope.append(_scope)
        # print(f"解码耗时：{datetime.now() - start}")
        return pred, scope

    def save(self, save_name, print_path=False):
        ckpt_path = self.cfg.dir_ckpt + save_name + '.ckpt'
        torch.save(self.state_dict(), ckpt_path)
        if print_path:
            print('model saved at:')
            print(ckpt_path)

    def load(self, load_name=None, path=None):
        if path is not None:
            ckpt_path = path
        else:
            ckpt_path = self.cfg.dir_ckpt + load_name + '.ckpt'
        self.load_state_dict(torch.load(ckpt_path, map_location=next(self.parameters()).device))
        print(f'model loaded from {ckpt_path}')

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True]
        for param in self.parameters():
            param.requires_grad = train_module[1]
        for param in self.encoder.parameters():
            param.requires_grad = train_module[0]
