from util.util import check_dir
import math

categories = ['PER', 'ORG', 'GPE', 'LOC', 'FAC', 'WEA', 'VEH']


class DataInfo:
    language = 'cn'
    data_path_root = 'data/ACE2005/'
    data_path = {'train': data_path_root + 'train.json',
                 'valid': data_path_root + 'dev.json',
                 'test': data_path_root + 'test.json'}
    num_sample = {'train': 5999, 'valid': 727, 'test': 734}
    cache_path_root = data_path_root + 'cache/'
    check_dir(cache_path_root)
    data_cache_path = {'train': cache_path_root + 'train.pk', 'valid': cache_path_root + 'valid.pk',
                       'test': cache_path_root + 'test.pk'}


class Exp:
    scope = 'BE'
    use_ratio = False
    use_sampler = False
    use_Focal = False
    use_2level_cate = False
    use_2level_seg = False

    token_fea_dim = 768
    max_length = 512
    max_entity_length = 128
    latent_dim = 512
    lstm_layer = 4
    pos_emb_dim = 0
    cate_out_dim = 0    # will be overwritten in args.py
    seg_out_dim = 0   # will be overwritten in args.py

    batch_size = 16
    shuffle = True
    num_workers = 0
    prefetch_factor = 2
    persistent_workers = True

    epoch = 50
    warm_up_ratio = 0.01
    start_train = 1
    start_eval = 1
    lr = 1e-3
    bert_lr = 1e-5
    decay = 5e-2
    drop_out = 0.5
    eval_per_step = 2


    #
    # token_fea_dim = 768
    # max_length = 512
    # max_entity_length = 128
    # latent_dim = 512
    # pos_emb_dim = 0
    # cate_out_dim = 0    # will be overwritten in args.py
    # seg_out_dim = 0   # will be overwritten in args.py
    #
    # batch_size = 16
    # shuffle = True
    # num_workers = 0
    # prefetch_factor = 2
    # persistent_workers = True
    #
    # epoch = 50
    # warm_up_ratio = 0.01
    # start_train = 1
    # start_eval = 25
    # lr = 1e-3
    # bert_lr = 5e-6
    # decay = 5e-2
    # drop_out = 0.5
    # eval_per_step = 2
