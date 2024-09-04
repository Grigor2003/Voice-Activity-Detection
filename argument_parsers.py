import argparse
from models import MODELS_COUNT, NAMES


def get_range(fr=0, to=1, tp=float):
    def is_in_range(value):
        try:
            fvalue = tp(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not a valid {tp}")
        if fr <= fvalue <= to:
            return fvalue
        raise argparse.ArgumentTypeError(f"{value} is out of range ({fr} to {to})")

    return is_in_range


def get_range_parser(first, second, order=True):
    def parse_tpl(s):
        try:
            start, end = map(int, s.split(','))
            s = first(start)
            e = second(end)
            if order and s > e:
                raise argparse.ArgumentTypeError(f"{s} cannot be greater than {e}")
            return s, e
        except Exception as err:
            raise argparse.ArgumentTypeError(f"Range must be in the form start,end. Error: {str(err)}")

    return parse_tpl


def add_def_arg(long, default=None, desc="", tp=str):
    train_parser.add_argument(long, required=False, default=default, type=tp,
                              help=desc)


def add_def_arg_short(long, short, default=None, desc="", tp=str):
    train_parser.add_argument(short, long, required=False, default=default, type=tp,
                              help=desc)


train_parser = argparse.ArgumentParser(description="Training script for Open SLR dataset with noise",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

add_def_arg("--clean", default=r"data\train-clean-100", desc="Path to clean data")
add_def_arg("--labels", default=r"data\8000_30_50_100_50_max.csv", desc="Path to labels")
add_def_arg("--noise", default=r"data\noise-16k", desc="Path to noises data")

add_def_arg("--train_res", default=r"train_results", desc="The directory to save train results")
train_parser.add_argument("--no_plot", required=False, action="store_true",
                          help="If not exists the plots of current training run will be saved in results folder")

train_parser.add_argument("-u", "--use_last", required=False, action="store_true",
                          help="Indicates whether the last model from the training results will be used")
train_parser.add_argument("--model_path", required=False, type=str,
                          help="The path to the model which will be used")
add_def_arg_short("--model_name", short="-m", default=None,
                  desc=f"The name of the model from models.py one of {NAMES}")
add_def_arg_short("--model_id", short="-i", default=0, tp=get_range(0, MODELS_COUNT - 1, int),
                  desc=f"The id of the model from models.py from 0 to {MODELS_COUNT - 1}")
add_def_arg("--val_ratio", default=0.05, tp=get_range(), desc="The ratio of validation data taken from clean data")
add_def_arg_short("--epoch", short="-e", default=1, tp=get_range(0, 1000, int),
                  desc="The number of epochs to train the model in current session")
add_def_arg("--noise_pool", default=500, tp=get_range(0, 5000, int),
            desc="The size of the noise pool for audio augmentation")
add_def_arg("--noise_count", default=2, tp=get_range(0, 10, int),
            desc="The count of noises added on a single audio")
add_def_arg("--noise_duration", default=(5, 10), tp=get_range_parser(get_range(0, 60), get_range(0, 60)),
            desc="The duration of added noises in seconds seperated by f,t")
add_def_arg_short("--workers", short="-w", default=1, tp=get_range(0, 32, int),
                  desc="The number of workers for data loading during training")
add_def_arg("--val_workers", default=1, tp=get_range(0, 32, int),
            desc="The number of workers for data loading during validation score calculation")
add_def_arg_short("--batch", short="-b", default=2 ** 6, tp=get_range(1, 2 ** 15, int),
                  desc="The size of the batch for simultaneous usage of samples during training")
add_def_arg("--val_batch", default=2 ** 6, tp=get_range(1, 2 ** 15, int),
            desc="The size of the batch for simultaneous usage of samples during validation score calculation")
add_def_arg("--threshold", default=0.7, tp=get_range(),
            desc="The threshold for labeling the prediction 0 or 1")
add_def_arg("--snr", default=3, tp=get_range(-20, 20),
            desc="The size of the batch for simultaneous usage of samples during validation score calculation")
add_def_arg("--val_every", default=1, tp=get_range(0, 1000, int),
            desc="The count of epoches to pass before calculating the validation score")
add_def_arg("--verbose", default=2, tp=get_range(0, 2, int),
            desc="The level of detail of printing")
add_def_arg("--lr", default=-3, tp=get_range(-2000, 2000),
            desc="The learning rate given by power of 10")
add_def_arg("--saves_count", default=1, tp=get_range(1, 100, int),
            desc="The count of saves (one save is always done in the end of run)")
