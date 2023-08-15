from util.util import check_dir
import random
import datetime
import torch

seed = 42
ID = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_' + str(random.randint(1000, 9999))
dataset = 'Default'
name = dataset + ID  # will be used to create separate ckpt dir to save parameters and logs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
force_process = False
# path
dir_output = 'output/'
dir_ckpt = dir_output + r'ckpt/'
file_log = dir_output + r'log.txt'
file_config = dir_output + r'config.torch'
dir_plm = r"data/plm/"
check_dir(dir_plm)
plm = dir_plm + 'bert-base-chinese/'
plm_en = dir_plm + 'bert-large-uncased/'


class LabelInfo:
    # ======================================
    # will be updated during parsing args
    # ======================================
    categories = []
    labels = []
    # statics
    num_categories = 0
    num_labels = 0
    # conversion between index and labels
    id2label = {}
    label2id = {}
    id2label_2 = {}
    label2id_2 = {}
    seg2id_2 = {}
    id2seg_2 = {}
    # map between 2 level and single level
    llid2lid = {}
    ssid2sid = {}


class DataInfo:
    language = 'cn'
    data_path_root = ''
    data_path = {'train': '', 'valid': '', 'test': ''}
    num_sample = {'train': 0, 'valid': 0, 'test': 0}
    cache_path_root = data_path_root + 'cache/'
    data_cache_path = {'train': '', 'valid': '', 'test': ''}


class Exp:
    scope = 'BE'
    use_ratio = False
    use_Focal = False
    use_sampler = True
    use_indice = False
    use_2level_cate = False
    use_2level_seg = False

    token_fea_dim = 768
    latent_dim = 100
    pos_emb_dim = 100
    cate_out_dim = 11
    seg_out_dim = 1

    max_length = 0
    max_entity_length = 0
    batch_size = 2
    shuffle = True
    num_workers = 0
    prefetch_factor = None
    persistent_workers = True

    epoch = 0
    warm_up_ratio = 0
    lr = 0
    bert_lr = 0
    decay = 0
    drop_out = 0

    start_eval = 1
    steps_per_eval = 0
    start_train = 0
