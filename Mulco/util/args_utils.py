import math


def overwrite_config(cfg):
    if cfg.dataset == "TTNER":
        import settings.TNNER as target
    else:
        raise NotImplementedError(f'Data {cfg.dataset} is not Implemented')
    cfg.DataInfo = target.DataInfo
    cfg.Exp = target.Exp
    cfg.LabelInfo.categories = target.categories
    return cfg


def update_Exp(cfg, args):
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
    log = '-----------------PARAMETERS--------------------\n'
    log += f'          seed:\t{cfg.seed}\n' \
           f'          name:\t{cfg.name}\n' \
           f'  dataset_name:\t{cfg.dataset}\n' \
           f"    train_path:\t{cfg.DataInfo.data_path['train']}\n" \
           f"    valid_path:\t{cfg.DataInfo.data_path['valid']}\n" \
           f"    test_path:\t{cfg.DataInfo.data_path['test']}\n" \
           f' force_process:\t{cfg.force_process}\n' \
           f'         batch:\t{cfg.Exp.batch_size}\n' \
           f'       max_len:\t{cfg.Exp.max_length}\n' \
           f'         epoch:\t{cfg.Exp.epoch}\n' \
           f' warm_up_ratio:\t{cfg.Exp.warm_up_ratio}\n' \
           f'   start_train:\t{cfg.Exp.start_train}\n' \
           f'            lr:\t{cfg.Exp.lr}\n' \
           f'         decay:\t{cfg.Exp.decay}\n' \
           f'      drop_out:\t{cfg.Exp.drop_out}\n' \
           f'steps_per_eval:\t{cfg.Exp.steps_per_eval}\n'
    write_log(log, cfg.file_log)