from trainer.NNER import NNER_trainer
import config
import args as arguments


def run(args, cfg=config):
    cfg = arguments.init_config(args, cfg)
    trainer = NNER_trainer(cfg=cfg)
    name = 'TNNER20220808172230_6420'
    dataset = 'TNNER'
    model_path = f'output/{name}/ckpt/{dataset}_best.ckpt'
    trainer.model.load(path=model_path)

    f1_score, log = trainer.eval(test=True, load=False, save_result=False, name=dataset + '_result')
    log = "********* NAME: %s BEST TEST F1 is %s **********\n" % (dataset, f1_score,) + log
    print(log)


if __name__ == '__main__':
    run(arguments.parser.parse_args())
