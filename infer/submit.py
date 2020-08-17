import json
import argparse
import tarfile
try:
    import xmlrpclib
except ImportError:
    import xmlrpc.client as xmlrpclib
import hashlib

FILE_LIMITS = 100*1024*1024
SERVER_IP = 'http://43.254.11.51:8009/'
PROXY = xmlrpclib.ServerProxy(SERVER_IP)

class Submitter(object):
    def __init__(self, user_name, user_key):
        self.user_info = {
            'user_name': user_name,
            'user_key': user_key
        }

    def get_jobs_info(self):
        print('Getting running jobs ...')
        jobs = PROXY.get_jobs_info(self.user_info)
        print(jobs)

    def get_gpus_info(self, node):
        gpus = PROXY.get_gpus_info(self.user_info, node)
        print(gpus)

    def get_available_gpus(self):
        print('Getting available gpus ...')
        nodes_gpus_str = PROXY.get_free_gpus(self.user_info)
        print(nodes_gpus_str)

    def submit_job(self, job_config):
        def exclude_function(tarinfo):
            exclude_files = ['.git', '.idea', '.pyc', '__pycache__', 'checkpoint', 'work_dirs',
                             'tmp', '.jpg', '.webp', 'pfm']
            name = tarinfo.name

            for i in exclude_files:
                if name.find(i) > 0:
                    return None
            else:
                return tarinfo

        tar_file = tarfile.open('tmp.tar', 'w')
        tar_file.add(job_config['code_dir'], arcname=job_config['job_name'], filter=exclude_function)
        tar_file.close()
        with open('tmp.tar', 'rb') as file_handle:
            code_str = xmlrpclib.Binary(file_handle.read())
            print(hashlib.md5(code_str.data).hexdigest())
            # os.remove('tmp.tar')
            if len(code_str.data) > FILE_LIMITS:
                print('Too large, please remove unused files in the repo.')
            else:
                msg = PROXY.collect(self.user_info, json.dumps(job_config), code_str)
                print(msg)

    def cancel_job(self, job_config):
        msg = PROXY.remote_cancel(self.user_info, json.dumps(job_config))
        print(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action',
                        default='g', type=str,
                        dest='action', help='[get_available_gpus|'
                                            'get_gpus_info|'
                                            'get_jobs_info|'
                                            'submit_job|'
                                            'cancel_job]')
    args = parser.parse_args()

    user_name = 'changshu'
    submitter = Submitter(user_name=user_name, user_key='28c201374ad533f15032e7cac34a3fde')

    if args.action == 'get_available_gpus' or args.action == 'a':
        submitter.get_available_gpus()
    if args.action == 'get_gpus_info' or args.action == 'g':
        str = 'GPU-node12'
        print(str)
        submitter.get_gpus_info(str)
    if args.action == 'get_jobs_info' or args.action == 'j':
        submitter.get_jobs_info()

    if args.action in ['submit_job', 'cancel_job', 's', 'c']:
        job_config = {
            'code_dir': '../job-thick-200211', # local directory contains your code
            'dataset': '',
            'interpreter': 'Anaconda3/envs/py37t11/bin/python',
            'node': 'GPU-node12',
            'gpus': [0,1,2,3,4,5,6,7],

            'job_name': 'fm_odo',
            'entry': '-m torch.distributed.launch --master_port=9900 --nproc_per_node=8 train.py',
            'args': './config/cfg_odom_fm.py --launcher pytorch',

            # 'job_name': 'info',
            # 'entry': './scripts/node_test.py',
            # 'args': '',
        }

    if args.action == 'submit_job' or  args.action == 's':
        submitter.submit_job(job_config)
    if args.action == 'cancel_job' or args.action == 'c':
        submitter.cancel_job(job_config)
