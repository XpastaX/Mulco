import config
import args as arguments
from util.util import set_seed
from trainer.NNER import NNER_trainer
import os


def run(args, cfg=config):
    cfg = arguments.init_config(args, cfg)
    if cfg.force_process or not os.path.isfile(cfg.DataInfo.data_cache_path['train']):
        # if cfg.DataInfo.language == 'cn':
        from dataloader.preprocess import preprocess
        # elif cfg.DataInfo.language == 'en':
        #     from dataloader.preprocess_en import preprocess
        # else:
        #     raise NotImplementedError
        for dataset in cfg.DataInfo.data_path:
            preprocess(dataset, config)

    set_seed(cfg.seed)

    trainer = NNER_trainer(cfg=cfg)

    trainer.train()


if __name__ == '__main__':
    run(arguments.parser.parse_args())
