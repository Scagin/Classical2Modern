import logging
import argparse
from utils import load_hparams, save_hparams


class Hparams:
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', help="run mode.[train, test, infer, version]")

    ## vocabulary
    parser.add_argument('--vocab_size', default=16121, type=int)
    parser.add_argument('--vocab', default='data/vocab_char.txt', help="vocabulary file path")

    # train
    ## files
    parser.add_argument('--train1', default='data/train_sample.src', help="classical Chinese training data file")
    parser.add_argument('--train2', default='data/train_sample.dst', help="modern Chinese training data file")
    parser.add_argument('--eval1', default='data/test_sample.src', help="classical Chinese evaluation data file")
    parser.add_argument('--eval2', default='data/test_sample.dst', help="modern Chinese evaluation data file")
    # parser.add_argument('--eval3', default='data/test_sample.dst', help="english evaluation unsegmented data")

    # training scheme
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/2", help="log directory")
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--evaldir', default="eval/2", help="evaluation dir")

    # model
    parser.add_argument('--d_model', default=256, type=int, help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=512, type=int, help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=3, type=int, help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int, help="number of attention heads")
    parser.add_argument('--maxlen1', default=50, type=int, help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=50, type=int, help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float, help="label smoothing rate")

    # test
    parser.add_argument('--test1', default='data/test_sample.src', help="classical Chinese test data file")
    parser.add_argument('--test2', default='data/test_sample.dst', help="modern Chinese test data file")
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="test/v1.0.0", help="test result dir")

    def get_params(self):
        hparams = Hparams()
        parser = hparams.parser
        hp = parser.parse_args()
        logging.info("Hyper Params :")
        logging.info("\n" + "\n".join(["{} = {}".format(key.rjust(20, " "), val) for key, val in hp._get_kwargs()]))
        return hp
