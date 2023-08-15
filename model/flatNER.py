import torch
import torch.nn as nn
from transformers import AutoModel
import config
from model.classifier.flat_cls import BaseClassifier


class Plain(nn.Module):
    def __init__(self, cfg=config):
        super(Plain, self).__init__()
        self.name = cfg.name
        self.cfg = cfg
        self.O = 2
        self.para_device = None
        self.fea_size = cfg.Exp.token_fea_dim
        drop_out = cfg.Exp.drop_out
        self.out_size = len(cfg.LabelInfo.categories)*4+3
        # extractor
        self.encoder = AutoModel.from_pretrained(cfg.plm)
        # decoder, output dim is number of label types plus CLS and SEP
        self.decoder = BaseClassifier(self.fea_size, [], self.out_size, drop_out, self.name)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, text, ground_truth):
        if self.para_device is None:
            self.para_device = next(self.parameters()).device

        ground_truth = ground_truth.to(self.para_device)
        token = {key: text[key].to(self.para_device) for key in text.keys()}
        mask_index = token['attention_mask'].contiguous().reshape(-1).eq(1)
        fea = self.encoder(**token)['last_hidden_state']
        x = self.decoder(fea).contiguous().view(-1, self.out_size)[mask_index]  # batch,seq,labels
        loss1 = self.criterion(x, ground_truth)
        return loss1

    def predict(self, text):
        if self.para_device is None:
            self.para_device = next(self.parameters()).device
        token = {key: text[key].to(self.para_device) for key in text.keys()}
        mask = token["attention_mask"]
        output = self.encoder(**token)  # batch,seq,768
        x = output['last_hidden_state']
        x = self.decoder(x)  # batch,seq,labels
        pred = torch.argmax(x, dim=2)
        pred_list = [sample[mask[i] == 1][1:-1] for i, sample in enumerate(pred)]
        # sequence = [list(itemgetter(*sample.int().tolist())(self.id2target)) for i, sample in enumerate(pred_list)]

        return pred_list

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True]
        for param in self.encoder.parameters():
            param.requires_grad = train_module[0]
        for param in self.decoder.parameters():
            param.requires_grad = train_module[1]

    def save(self, save_name, print_path=False, save_mode=False):
        # save all modules
        if save_mode:
            encoder_ckpt_path = self.cfg.dir_ckpt + save_name + '.encoder'
            torch.save(self.encoder.state_dict(), encoder_ckpt_path)
            if print_path:
                print('encoder saved at:')
                print(encoder_ckpt_path)

            decoder_ckpt_path = self.cfg.dir_ckpt + save_name + '.decoder'
            torch.save(self.decoder.state_dict(), decoder_ckpt_path)
            if print_path:
                print('decoder saved at:')
                print(decoder_ckpt_path)

        ckpt_path = self.cfg.dir_ckpt + save_name + '.ckpt'
        torch.save(self.state_dict(), ckpt_path)

        if print_path:
            print('model saved at:')
            print(ckpt_path)

    def load(self, load_name, direct=False, load_mode=False):
        if load_mode:
            if direct:
                encoder_ckpt_path = load_name[0]
                decoder_ckpt_path = load_name[1]
            else:
                encoder_ckpt_path = self.cfg.dir_ckpt + load_name + '.encoder'
                decoder_ckpt_path = self.cfg.dir_ckpt + load_name + '.decoder'
            self.encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=next(self.parameters()).device))
            self.decoder.load_state_dict(torch.load(decoder_ckpt_path, map_location=next(self.parameters()).device))
            print(f'encoder loaded from {encoder_ckpt_path}')
            print(f'decoder loaded from {decoder_ckpt_path}')
        else:
            if direct:
                ckpt_path = load_name
            else:
                ckpt_path = self.cfg.dir_ckpt + load_name + '.ckpt'

            self.load_state_dict(torch.load(ckpt_path, map_location=next(self.parameters()).device))
            print(f'model loaded from {ckpt_path}')
