import torch
import config
from util.util import write_log, ProgressBar, check_dir
from util.metric import SeqEntityScore, generate_result
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import logging
from dataloader.loader import get_dataloader
import datetime
from model.MulcoNNER import Mulco

logging.set_verbosity_error()


class NNER_trainer(object):
    def __init__(self, cfg=config):
        self.cfg = cfg
        # dataloader
        self.train_loader = get_dataloader('train', cfg=cfg)
        # define all keys
        self.cate_key = 'processed_cate'
        self.seg_key = 'processed_seg'
        self.entity_mask = 'entity_mask'
        self.input_key = 'tokenized'
        self.mask_key = 'mask'
        self.Cmask_key = 'Cmask'
        # get model
        self.model = Mulco(cfg).to(cfg.device)
        # metric
        self.metric = SeqEntityScore()
        # setting learning rate and decay with exceptions
        no_decay = ["bias", "LayerNorm.weight"]
        name = []
        for n, p in self.model.encoder.named_parameters():
            name.append(n)
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in name)],
             "lr": cfg.Exp.lr, "weight_decay": self.cfg.Exp.decay},
            {"params": [p for n, p in self.model.named_parameters() if (any(nd in n for nd in name) and not any(nd in n for nd in no_decay))],
             "lr": cfg.Exp.bert_lr, "weight_decay": self.cfg.Exp.decay},
            {"params": [p for n, p in self.model.named_parameters() if (any(nd in n for nd in name) and any(nd in n for nd in no_decay))],
             "lr": cfg.Exp.bert_lr, "weight_decay": 0}]

        total_step = self.cfg.Exp.epoch * len(self.train_loader)
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=int(self.cfg.Exp.warm_up_ratio * total_step),
                                                         num_training_steps=total_step)

    # ========================================================================================================
    #                                              Trainer
    # ========================================================================================================
    def train(self):
        best_result = ''
        best_f1 = 0
        # self.eval()
        self.model.train()
        self.model.set_train([True, True])
        # for step, batch in enumerate(self.train_loader, start=1):
        #     pass
        for epoch in range(1, self.cfg.Exp.epoch + 1):
            pbar = ProgressBar(n_total=len(self.train_loader), desc='Training')
            for step, batch in enumerate(self.train_loader, start=1):
                self.optimizer.zero_grad()
                loss, l_cate, l_seg = self.model(batch[self.input_key], batch[self.cate_key], batch[self.seg_key],
                                                 batch[self.entity_mask], batch[self.mask_key], batch[self.Cmask_key],
                                                 )
                pbar(step, {'epoch': epoch, 'loss': loss.item(), 'l_cate': l_cate.item(), 'l_seg': l_seg.item()})

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # evaluate
                if epoch >= self.cfg.Exp.start_eval:
                    if step % self.cfg.Exp.steps_per_eval == 0 or step == len(self.train_loader):
                        f1_score, log = self.eval(epoch, step)
                        if best_f1 < f1_score:
                            best_f1 = f1_score
                            best_result = log
                            prt_txt = f"\nFound best f1 socre at {epoch}-{step}:{best_f1}\n"
                            print(log)
                            self.model.save(self.cfg.dataset + '_best', print_path=True)
                        else:
                            prt_txt = f"\nf1 socre at {epoch}-{step}:{f1_score}\n"
                        write_log(prt_txt, self.cfg.file_log)

            print('')  # to show the progress bar for each epoch
        # print best evaluation
        best_result = "********* NAME: %s BEST EVAL F1 is %s **********\n" % (self.cfg.name, best_f1,) + best_result
        write_log(best_result, self.cfg.file_log)
        # print test result

        f1_score, log = self.eval(test=True, save_result=True)
        log = "********* NAME: %s BEST TEST F1 is %s **********\n" % (self.cfg.name, f1_score,) + log
        write_log(log, self.cfg.file_log)

    # ========================================================================================================
    #                                       Evaluation and Testing
    # ========================================================================================================
    def eval(self, epoch=0, step=0, test=False, path=None, load=True, save_result=False, name=None):
        if name is None:
            name = self.cfg.name
        if test:
            if load:
                self.model.load(self.cfg.dataset + '_best', path)
            valid_loader = get_dataloader('test', cfg=self.cfg)
        else:
            valid_loader = get_dataloader('valid', cfg=self.cfg)
        # collect all results
        all_text = []
        all_pred = []
        all_gt = []
        all_id = []
        self.metric.reset()
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(valid_loader, start=1):
                pred, scope = self.model.inference(batch[self.input_key], batch[self.mask_key],
                                                   batch[self.Cmask_key])
                self.metric.update(batch['gt'], pred)

                if save_result:
                    for i in range(len(pred)):
                        if self.cfg.DataInfo.language == 'en':
                            pred_ent, gt_ent = generate_result(batch[self.input_key]['input_ids'][i][1:-1], pred[i],
                                                               batch['gt'][i],
                                                               scope[i], self.cfg.LabelInfo.id2label, is_en=True,
                                                               tokenizer=valid_loader.collate_fn.tokenizer)
                        else:
                            pred_ent, gt_ent = generate_result(batch['text_ori'][i], pred[i], batch['gt'][i],
                                                               scope[i], self.cfg.LabelInfo.id2label,
                                                               is_en=False)
                        all_text.append(batch['text_ori'][i])
                        all_pred.append(pred_ent)
                        all_gt.append(gt_ent)
                        all_id.append(batch['id_list'][i])

            eval_info, entity_info = self.metric.result()
            f1_score, log = self.print(eval_info, entity_info, epoch, step)
            self.model.train()

            if save_result:
                # for i in range(len(all_id)):
                # print(f'------------------ID:{all_id[i]}------------------')
                # print(all_text[i])
                # print(f'  gt:{all_gt[i]}')
                # print(f'pred:{all_pred[i]}')
                check_dir('pred_result', creat=True)
                torch.save([all_text, all_pred, all_gt, all_id], 'pred_result/' + name + '.torch')

            return f1_score, log,

    def print(self, eval_info, entity_info, epoch=0, step=0):
        results = {f'{key}': value for key, value in eval_info.items()}
        log = f"********* EVAL RESULTS E{epoch} S{step}  **********\n"
        log += "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        log += "\n************* ENTITY RESULTS  *************\n"
        for key in sorted(entity_info.keys()):
            log += "\t\t ---%s results--- \n" % self.cfg.LabelInfo.id2label[int(key)]
            log += "-".join([f' {_key}: {value:.4f} ' for _key, value in entity_info[key].items()])
            log += "\n"
        log += '*******************************************\n'
        # print(log)
        return results['f1'], log
