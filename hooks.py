import os
import shutil

def move_well_known(config, **kwargs):
    site_dir = config['site_dir']
    docs_dir = config['docs_dir']
    shutil.copytree(os.path.join(docs_dir, '.well-known/'), os.path.join(site_dir, '.well-known'))
    # print(docs_dir)