import os
import config
from datetime import datetime


def gen_prefix(name, date):
    file_path = os.path.join(config.ROOT_DIR, 'saved_model', '_'.join((date, name)))
    return file_path


def logging2file(file_path, type, info, file_name=None):
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    if type == 'message':
        with open(os.path.join(file_path, 'message.txt'), 'a+') as f:
            f.write(info)
            f.flush()
    elif type == 'log':
        with open(os.path.join(file_path, 'log.txt'), 'a+') as f:
            f.write(info)
            f.flush()
    elif type == 'code':
        with open(os.path.join(file_path, 'code.pys'), 'a+') as f, open(file_name) as it:
            f.write(it.read())
            f.flush()

if __name__ == '__main__':
    date_now = datetime.now().strftime("%m-%d-%H:%M:%S")
    log_file_path = gen_prefix('conv_model', date_now)

    logging2file(log_file_path, 'message', 'something.')
    logging2file(log_file_path, 'code', 'something.')
    logging2file(log_file_path, 'log', 'something.')