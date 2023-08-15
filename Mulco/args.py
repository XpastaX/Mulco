import argparse
import config
import torch
from util.util import check_dir, write_log, join_cate
import math

# parser used to read argument
parser = argparse.ArgumentParser(description='xxxxx')

# ==================================
#              BASIC
# ==================================
parser.add_argument('--seed', type=int, default=config.seed)
parser.add_argument('--device', type=str, default=config.device, help='device to run on')
parser.add_argument('--data', type=str, default='TNNER', help='dataset name, e.g., Weibo')
parser.add_argument('--force_process', action="store_true", help='force preprocess')
# ==================================
#      Model & Training  Settings
# ==================================
parser.add_argument('--use_ratio', action="store_true", help='whether to use ratio on seg')
parser.add_argument('--scope', type=str, default=None, help='what scopes to use')
parser.add_argument('--use_sampler', action="store_true", help='whether to use O labeled token in seg training')
parser.add_argument('--use_Focal', action="store_true", help='whether to use Focal Loss')
parser.add_argument('--use_2level_cate', action="store_true", help='whether to use 2level cate')
parser.add_argument('--use_2level_seg', action="store_true", help='whether to use 2level seg')
parser.add_argument('--loss_type', type=str, default='not_defined', help='loss type for feas')
parser.add_argument('--batch_size', type=int, default=0, help='batch size')
parser.add_argument('--max_length', type=int, default=0, help='max_length')
parser.add_argument('--epoch', type=int, default=0, help='num of epoch')
parser.add_argument('--start_train', type=int, default=0, help='start_train')
parser.add_argument('--start_eval', type=int, default=0, help='start_eval')
parser.add_argument('--lr', type=float, default=0, help='initial learning rate')
parser.add_argument('--decay', type=float, default=0, help='decay factor')
parser.add_argument('--drop_out', type=float, default=-1, help='decay factor')
parser.add_argument('--steps_per_eval', type=int, default=0, help='initial learning rate')


# ==================================
#         Initialization
# ==================================
def init_config(args, cfg):
    # update universal parameters
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.name = args.data + cfg.ID
    cfg.dataset = args.data
    # whether force processing data
    if args.force_process:
        cfg.force_process = True
    # update output path
    cfg.dir_output += cfg.name + r'/'
    cfg.dir_ckpt = cfg.dir_output + r'ckpt/'
    cfg.file_log = cfg.dir_output + r'log.txt'
    cfg.file_config = cfg.dir_output + r'config.torch'
    check_dir(cfg.dir_ckpt)
    # overwrite Dataset and Exp with corresponding data settings
    cfg = overwrite_config(cfg)
    # update exp settings in args
    cfg = update_Exp(cfg, args)
    # update label info
    cfg = update_LabelInfo(cfg)
    # save and print config
    save_config(cfg)
    return cfg


def overwrite_config(cfg):
    if cfg.dataset == "TNNER":
        import settings.TNNER as target
    elif cfg.dataset == "ACE2005":
        import settings.ACE2005 as target
    else:
        raise NotImplementedError(f'Data {cfg.dataset} is not Implemented')
    cfg.DataInfo = target.DataInfo
    cfg.Exp = target.Exp
    cfg.LabelInfo.categories = target.categories
    return cfg


def update_Exp(cfg, args):
    cfg.Exp.use_Focal = True if args.use_Focal else cfg.Exp.use_Focal
    cfg.Exp.use_ratio = True if args.use_ratio else cfg.Exp.use_ratio
    cfg.Exp.use_sampler = True if args.use_sampler else cfg.Exp.use_sampler
    cfg.Exp.use_2level_cate = True if args.use_2level_cate else cfg.Exp.use_2level_cate
    cfg.Exp.use_2level_seg = True if args.use_2level_seg else cfg.Exp.use_2level_seg
    cfg.Exp.scope = args.scope if args.scope is not None else cfg.Exp.scope

    cfg.Exp.batch_size = args.batch_size if args.batch_size != 0 else cfg.Exp.batch_size
    cfg.Exp.max_length = args.max_length if args.max_length != 0 else cfg.Exp.max_length
    cfg.Exp.epoch = args.epoch if args.epoch != 0 else cfg.Exp.epoch
    cfg.Exp.start_train = args.start_train if args.start_train != 0 else cfg.Exp.start_train
    cfg.Exp.start_eval = args.start_eval if args.start_eval != 0 else cfg.Exp.start_eval
    cfg.Exp.lr = args.lr if args.lr != 0 else cfg.Exp.lr
    cfg.Exp.decay = args.decay if args.decay != 0 else cfg.Exp.decay
    cfg.Exp.drop_out = args.drop_out if args.drop_out != -1 else cfg.Exp.drop_out
    cfg.Exp.steps_per_eval = args.steps_per_eval if args.steps_per_eval != 0 \
        else math.ceil(cfg.DataInfo.num_sample['train'] / cfg.Exp.batch_size / cfg.Exp.eval_per_step)
    return cfg


def save_config(cfg):
    log = '====================PARAMETER====================\n'
    log += f'           seed:  {cfg.seed}\n' \
           f'           name:  {cfg.name}\n' \
           f'   dataset_name:  {cfg.dataset}\n' \
           f"     train_path:  {cfg.DataInfo.data_path['train']}\n" \
           f"     valid_path:  {cfg.DataInfo.data_path['valid']}\n" \
           f"      test_path:  {cfg.DataInfo.data_path['test']}\n" \
           f"            plm:  {cfg.plm}\n" \
           f'  force_process:  {cfg.force_process}\n' \
           f'            MODEL PARA\n' \
           f'      use_ratio:  {cfg.Exp.use_ratio}\n' \
           f'    use_sampler:  {cfg.Exp.use_sampler}\n' \
           f'      use_Focal:  {cfg.Exp.use_Focal}\n' \
           f'use_2level_cate:  {cfg.Exp.use_2level_cate}\n' \
           f' use_2level_seg:  {cfg.Exp.use_2level_seg}\n' \
           f'        pos_dim:  {cfg.Exp.pos_emb_dim}\n'\
           f'     latent_dim:  {cfg.Exp.latent_dim}\n' \
           f'     lstm_layer:  {cfg.Exp.lstm_layer}\n' \
           f'   cate_out_dim:  {cfg.Exp.cate_out_dim}\n' \
           f'    seg_out_dim:  {cfg.Exp.seg_out_dim}\n' \
           f'          batch:  {cfg.Exp.batch_size}\n' \
           f'        max_len:  {cfg.Exp.max_length}\n' \
           f'              TRAINING\n' \
           f'          epoch:  {cfg.Exp.epoch}\n' \
           f'  warm_up_ratio:  {cfg.Exp.warm_up_ratio}\n' \
           f'    start_train:  {cfg.Exp.start_train}\n' \
           f'             lr:  {cfg.Exp.lr}\n' \
           f'        bert_lr:  {cfg.Exp.bert_lr}\n' \
           f'          decay:  {cfg.Exp.decay}\n' \
           f'       drop_out:  {cfg.Exp.drop_out}\n' \
           f' steps_per_eval:  {cfg.Exp.steps_per_eval}\n'
    log += '=======================END=======================\n'
    write_log(log, cfg.file_log)


def update_LabelInfo(cfg=config):
    all_cate = ['START', 'END', 'O'] + cfg.LabelInfo.categories
    cfg.LabelInfo.label2id = {cate: idx for idx, cate in enumerate(all_cate)}
    cfg.LabelInfo.id2label = {cfg.LabelInfo.label2id[cate]: cate for cate in cfg.LabelInfo.label2id}
    # generate 2-lv labels
    cfg.LabelInfo.label2id_2 = {cate: idx for idx, cate in enumerate(['START', 'END', 'O-O'])}
    # initialize llid2lid mapping
    for i in range(len(cfg.LabelInfo.label2id_2)):
        cfg.LabelInfo.llid2lid[i] = (i, i)
    cfg.LabelInfo.id2label_2 = {}
    # start to build label mapping of 2level categories
    counter = len(cfg.LabelInfo.label2id_2)
    for cate1 in cfg.LabelInfo.categories:
        for cate2 in cfg.LabelInfo.categories + ['O']:
            key = join_cate(cate1, cate2)
            if key not in cfg.LabelInfo.label2id_2:
                cfg.LabelInfo.label2id_2[key] = counter
                cfg.LabelInfo.id2label_2[counter] = key
                cfg.LabelInfo.llid2lid[counter] = (cfg.LabelInfo.label2id[cate1], cfg.LabelInfo.label2id[cate2])
                counter += 1

    entity_len = cfg.Exp.max_entity_length + 1
    counter = 0
    for seg1 in range(entity_len):
        for seg2 in range(entity_len):
            if seg1 == 0 and seg2 != 0: continue
            if seg2 <= seg1 and seg2 != 0: continue
            key = join_cate(str(seg1), str(seg2))
            if key not in cfg.LabelInfo.seg2id_2:
                cfg.LabelInfo.seg2id_2[key] = counter
                cfg.LabelInfo.id2seg_2[counter] = (seg1, seg2)
                counter += 1
    cfg.LabelInfo.ssid2sid = cfg.LabelInfo.id2seg_2

    # cfg.LabelInfo.label2id['IGNORE'] = -100
    # cfg.LabelInfo.id2label[-100] = 'IGNORE'
    # cfg.LabelInfo.label2id_2['IGNORE'] = -100
    # cfg.LabelInfo.id2label_2[-100] = 'IGNORE'
    # cfg.LabelInfo.seg2id_2['IGNORE'] = -100
    # cfg.LabelInfo.id2seg_2[-100] = 'IGNORE'

    # update corresponding dimensions
    cfg.Exp.cate_out_dim = len(cfg.LabelInfo.label2id_2) if cfg.Exp.use_2level_cate else len(cfg.LabelInfo.label2id)
    entity_len = len(cfg.LabelInfo.seg2id_2) if cfg.Exp.use_2level_seg else cfg.Exp.max_length - 2
    cfg.Exp.seg_out_dim = 1 if cfg.Exp.use_ratio else entity_len
    return cfg
