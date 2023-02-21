# coding=utf-8
import os.path as osp
import os
import argparse
import zipfile
import yaml

from utils.logger import Logger


def get_format_time(timezone_str = 'Asia/Seoul', time_format = '%Y%m%d-%H%M%S'):
    from pytz import timezone
    from datetime import datetime
    KST = timezone(timezone_str)
    return datetime.now(tz = KST).strftime(time_format)

def hyperparams2yaml(path, context, backup_file = True):

    params = vars(context)
    os.makedirs(osp.join(path, 'yaml'), exist_ok=True)
    f_time = get_format_time()
    with open(osp.join(path, 'yaml', '%s.yaml'%f_time), 'w',
              encoding="UTF-8") as f:
        yaml.dump(params, f, sort_keys=False, allow_unicode = True, indent =4)

    if backup_file:
        backup(osp.join(path, 'yaml', '%s.zip'%f_time))
    Logger.info('Backup:', osp.join(path, 'yaml', '%s'%f_time), '.'*50)
    return osp.join(path, 'yaml', '%s'%f_time)

def yaml2hyperparams(args):
    if osp.exists(args.config):
        opt = vars(args)
        args = yaml.load(open(args.config), Loader = yaml.FullLoader)
        opt.update(args)
        args = argparse.Namespace(**opt)

    return args

def backup(path, base_path = None, suffixs = ['py', 'yaml'], exclude_folder_names = ['output', '.idea']):
    base_path = osp.join(osp.dirname(osp.abspath(__file__)), '../') if base_path is None else base_path

    zipf = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED)
    for root, _, filenames in os.walk(base_path):
        r = osp.abspath(root)
        if sum([e.lower() in r.lower() for e in exclude_folder_names]) == 0:
            for filename in filenames:
                if filename.split('.')[-1] in suffixs:
                    zipf.write(os.path.join(root, filename),
                               os.path.relpath(os.path.join(root, filename),
                                               os.path.join(base_path, '..')))
    zipf.close()