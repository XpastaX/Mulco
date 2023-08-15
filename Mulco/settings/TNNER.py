from util.util import check_dir
import math

categories = ['Person', 'Organization', 'Location', 'Time', 'Event', 'Food', 'Creature', 'Work', 'Product', 'Medicine']


class DataInfo:
    language = 'cn'
    data_path_root = 'data/TNNER/'
    data_path = {'train': data_path_root + 'new_train.json',
                 'valid': data_path_root + 'new_eval.json',
                 'test': data_path_root + 'new_test.json'}
    num_sample = {'train': 17500, 'valid': 1000, 'test': 1500}
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
    max_entity_length = 60
    latent_dim = 768
    lstm_layer = 2
    pos_emb_dim = 0
    cate_out_dim = 0  # will be overwritten in args.py
    seg_out_dim = 0  # will be overwritten in args.py

    batch_size = 8
    shuffle = True
    num_workers = 0
    prefetch_factor = 2
    persistent_workers = True

    epoch = 50
    warm_up_ratio = 0.01
    start_train = 1
    start_eval = 25
    lr = 2e-5
    bert_lr = 2e-5
    decay = 5e-2
    drop_out = 0.3
    eval_per_step = 1

# class Exp: #79.75
#     use_CE = True
#     use_mask = False
#     use_Focal = False
#
#     token_fea_dim = 768
#     max_length = 512
#     latent_dim = 128
#     cate_out_dim = 13
#     seg_out_dim = 1 if not use_CE else max_length - 2
#
#     batch_size = 16
#     shuffle = True
#     num_workers = 0
#     prefetch_factor = 2
#     persistent_workers = True
#
#     epoch = 50
#     warm_up_ratio = 0.1
#     start_train = 1
#     start_eval = 5
#     lr = 2e-5
#     decay = 5e-2
#     drop_out = 0.5
#     eval_per_step = 1
