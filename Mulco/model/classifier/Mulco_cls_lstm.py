import torch.nn as nn
import config
import torch


def direct_re(x):
    return x


class BaseClassifier(nn.Module):
    def __init__(self, cfg=config):
        super(BaseClassifier, self).__init__()
        self.cfg = cfg
        self.l_emb = cfg.Exp.pos_emb_dim
        self.activate = nn.GELU()
        if cfg.Exp.use_ratio:
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.sigmoid = direct_re

        self.drop = torch.nn.Dropout(p=cfg.Exp.drop_out)

        if cfg.Exp.lstm_layer > 0:
            self.lstm = torch.nn.LSTM(cfg.Exp.token_fea_dim, cfg.Exp.token_fea_dim,
                                      num_layers=cfg.Exp.lstm_layer, batch_first=True,
                                      dropout=cfg.Exp.drop_out, bidirectional=True)
            input_dim = cfg.Exp.token_fea_dim * 2
        else:
            input_dim = cfg.Exp.token_fea_dim
            self.lstm = None

        if self.l_emb != 0:
            self.Emb = torch.nn.Embedding(cfg.Exp.max_length, self.l_emb)
        else:
            self.Emb = None
        self.fc_Bmin = nn.Linear(input_dim, cfg.Exp.latent_dim)
        self.fc_Emin = nn.Linear(input_dim, cfg.Exp.latent_dim)
        self.cate_Bmin = nn.Linear(cfg.Exp.latent_dim, cfg.Exp.cate_out_dim)
        self.cate_Emin = nn.Linear(cfg.Exp.latent_dim, cfg.Exp.cate_out_dim)
        self.seg_B = nn.Linear(cfg.Exp.latent_dim + self.l_emb, cfg.Exp.seg_out_dim)
        self.seg_E = nn.Linear(cfg.Exp.latent_dim + self.l_emb, cfg.Exp.seg_out_dim)
        if 'C' in cfg.Exp.scope:
            self.fc_C = nn.Linear(input_dim, cfg.Exp.latent_dim)
            self.fc_C2 = nn.Linear(input_dim * 2, cfg.Exp.latent_dim)
            self.cate_C = nn.Linear(cfg.Exp.latent_dim, cfg.Exp.cate_out_dim)
            self.seg_C = nn.Linear(cfg.Exp.latent_dim + self.l_emb, cfg.Exp.seg_out_dim)

        if not cfg.Exp.use_2level_cate or not cfg.Exp.use_2level_seg:
            self.fc_Bmax = nn.Linear(input_dim, cfg.Exp.latent_dim)
            self.fc_Emax = nn.Linear(input_dim, cfg.Exp.latent_dim)
            if 'C' in cfg.Exp.scope:
                self.fc_Cmax = nn.Linear(input_dim, cfg.Exp.latent_dim)
                self.fc_Cmax2 = nn.Linear(input_dim * 2, cfg.Exp.latent_dim)
        if not cfg.Exp.use_2level_cate:
            self.cate_Bmax = nn.Linear(cfg.Exp.latent_dim, cfg.Exp.cate_out_dim)
            self.cate_Emax = nn.Linear(cfg.Exp.latent_dim, cfg.Exp.cate_out_dim)
            if 'C' in cfg.Exp.scope:
                self.cate_Cmax = nn.Linear(cfg.Exp.latent_dim, cfg.Exp.cate_out_dim)
        if not cfg.Exp.use_2level_seg:
            self.seg_Bmax = nn.Linear(cfg.Exp.latent_dim + self.l_emb, cfg.Exp.seg_out_dim)
            self.seg_Emax = nn.Linear(cfg.Exp.latent_dim + self.l_emb, cfg.Exp.seg_out_dim)
            if 'C' in cfg.Exp.scope:
                self.seg_Cmax = nn.Linear(cfg.Exp.latent_dim + self.l_emb, cfg.Exp.seg_out_dim)
        self.hasC = 'C' in cfg.Exp.scope

    def forward(self, x, L):  # batch seq dim
        cCmin = None
        cCmax = None
        sCmin = None
        sCmax = None
        L = (L - 2)

        if self.cfg.Exp.lstm_layer > 0:
            x = self.lstm(x)[0]
        else:
            x = self.drop(x)
        x2 = torch.cat((x[:, :-1], x[:, 1:]), dim=-1)

        pBmin = self.activate(self.fc_Bmin(x))
        pEmin = self.activate(self.fc_Emin(x))
        pBmax = self.activate(self.fc_Bmax(x))
        pEmax = self.activate(self.fc_Emax(x))

        cBmin = self.cate_Bmin(pBmin)
        cEmin = self.cate_Emin(pEmin)
        cBmax = self.cate_Bmax(pBmax)
        cEmax = self.cate_Emax(pEmax)

        if self.hasC:
            pCmin = self.activate(self.fc_C(x))
            pCmin2 = self.activate(self.fc_C2(x2))
            pCmin2 = torch.cat((torch.stack((pCmin[:, :-1], pCmin2), dim=-2).view(pCmin2.shape[0], -1, pCmin2.shape[2]),
                                pCmin[:, -1:]), dim=-2)
            cCmin = self.cate_C(pCmin2)
            pCmax = self.activate(self.fc_Cmax(x))
            pCmax2 = self.activate(self.fc_Cmax2(x2))
            pCmax2 = torch.cat((torch.stack((pCmax[:, :-1], pCmax2), dim=-2).
                                view(pCmax2.shape[0], -1, pCmax2.shape[2]),
                                pCmax[:, -1:]), dim=-2)
            cCmax = self.cate_Cmax(pCmax2)

        if self.Emb is not None:
            Lemb = self.Emb(L).expand((len(cBmin[0]), len(L), self.l_emb)).transpose(0, 1)
            sBmin = self.seg_B(torch.cat((pBmin, Lemb), dim=-1))
            sEmin = self.seg_E(torch.cat((pEmin, Lemb), dim=-1))
            sBmax = self.seg_Bmax(torch.cat((pBmax, Lemb), dim=-1))
            sEmax = self.seg_Emax(torch.cat((pEmax, Lemb), dim=-1))
            if self.hasC:
                Lemb2 = self.Emb(L).expand((len(cCmin[0]), len(L), self.l_emb)).transpose(0, 1)
                sCmin = self.seg_C(torch.cat((pCmin2, Lemb2), dim=-1))
                sCmax = self.sigmoid(self.seg_Cmax(torch.cat((pCmax2, Lemb2), dim=-1)))
        else:
            sBmin = self.sigmoid(self.seg_B(pBmin))
            sEmin = self.sigmoid(self.seg_E(pEmin))
            sBmax = self.seg_Bmax(pBmax)
            sEmax = self.seg_Emax(pEmax)
            if self.hasC:
                sCmin = self.seg_C(pCmin2)
                sCmax = self.seg_Cmax(pCmax2)

        return [cBmin, cBmax, cEmin, cEmax, cCmin, cCmax, sBmin, sBmax, sEmin, sEmax, sCmin, sCmax]
