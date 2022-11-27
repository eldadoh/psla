import numpy as np
import json
import os
from pathlib import Path
from loguru import logger


def resample_raw_FSD50K(fsd_path: str) -> None:
    """Resample the audio dataset files service"""
    # convert all samples to 16kHZ
    logger.info('Now converting all FSD50K audio to 16kHz, this may take dozens of minutes.')

    def get_immediate_files(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

    resample_cnt = 0
    set_list = ['dev', 'eval']
    for set in set_list:
        basepath = Path(fsd_path + '/FSD50K.' + set + '_audio/')
        targetpath = Path(fsd_path + '/FSD50K.' + set + '_audio_16k/')
        if os.path.exists(targetpath) == False:
            os.makedirs(targetpath, exist_ok=True)
        files = get_immediate_files(basepath)
        for audiofile in files:
            os.system(
                'sox ' + './' + str(basepath) + '/' + audiofile + ' -r 16000 ' + './' + str(
                    targetpath) + '/' + audiofile + ' > /dev/null 2>&1')
            resample_cnt += 1
            if resample_cnt % 1000 == 0:
                print('Resampled {:d} samples.'.format(resample_cnt))
    print('Resampling finished.')
    print('--------------------------------------------')

    return


def prepare_fsd_dataset(resample: bool = False) -> None:
    # dataset downloaded from https://zenodo.org/record/4060432#.YXXR0tnMLfs
    # please change it to your FSD50K dataset path
    # the data organization might change with versioning, the code is tested early 2021
    fsd_path = 'data/waveforms'
    if resample:
        resample_raw_FSD50K(fsd_path)

    # create json datafiles for training, validation, and evaluation set

    # training set and validation set are from the official 'dev' set, we use the official training and validation set split.
    fsdeval = '/home/ubuntu/psla/data/FSD50K.ground_truth/dev.csv'
    fsdeval = np.loadtxt(fsdeval, skiprows=1, dtype=str)

    tr_cnt, val_cnt = 0, 0

    # only apply to the vocal sound data
    fsd_tr_data = []
    fsd_val_data = []
    for i in range(len(fsdeval)):
        try:
            fileid = fsdeval[i].split(',"')[0]
            labels = fsdeval[i].split(',"')[2][0:-1]
            set_info = labels.split('",')[1]
        except:
            fileid = fsdeval[i].split(',')[0]
            labels = fsdeval[i].split(',')[2]
            set_info = fsdeval[i].split(',')[3][0:-1]

        labels = labels.split('",')[0]
        label_list = labels.split(',')
        new_label_list = []
        for label in label_list:
            new_label_list.append(label)
        new_label_list = ','.join(new_label_list)
        # note, all recording we use are 16kHZ.
        cur_dict = {"wav": fsd_path + '/FSD50K.dev_audio_16k/' + fileid + '.wav', "labels": new_label_list}

        if set_info == 'trai':
            fsd_tr_data.append(cur_dict)
            tr_cnt += 1
        elif set_info == 'va':
            fsd_val_data.append(cur_dict)
            val_cnt += 1
        else:
            raise ValueError('unrecognized set')

    if os.path.exists('data/datafiles') == False:
        os.mkdir('data/datafiles')

    with open('./data/datafiles/fsd50k_tr_full.json', 'w') as f:
        json.dump({'data': fsd_tr_data}, f, indent=1)
    print('Processed {:d} samples for the FSD50K training set.'.format(tr_cnt))

    with open('./data/datafiles/fsd50k_val_full.json', 'w') as f:
        json.dump({'data': fsd_val_data}, f, indent=1)
    print('Processed {:d} samples for the FSD50K validation set.'.format(val_cnt))

    ## process the evaluation set
    fsdeval = '/home/ubuntu/psla/data/FSD50K.ground_truth/eval.csv'
    fsdeval = np.loadtxt(fsdeval, skiprows=1, dtype=str)

    cnt = 0

    # only apply to the vocal sound data
    vc_data = []
    for i in range(len(fsdeval)):
        try:
            fileid = fsdeval[i].split(',"')[0]
            labels = fsdeval[i].split(',"')[2][0:-1]
        except:
            fileid = fsdeval[i].split(',')[0]
            labels = fsdeval[i].split(',')[2]

        label_list = labels.split(',')
        new_label_list = []
        for label in label_list:
            new_label_list.append(label)

        if len(new_label_list) != 0:
            new_label_list = ','.join(new_label_list)
            cur_dict = {"wav": fsd_path + '/FSD50K.eval_audio_16k/' + fileid + '.wav', "labels": new_label_list}
            vc_data.append(cur_dict)
            cnt += 1

    with open('./data/datafiles/fsd50k_eval_full.json', 'w') as f:
        json.dump({'data': vc_data}, f, indent=1)
    print('Processed {:d} samples for the FSD50K evaluation set.'.format(cnt))

    # generate balanced sampling weight file
    os.system(
        'python3 ./src/gen_weight_file.py --dataset fsd50k --label_indices_path {:s} --datafile_path {:s}'.format(
            '/home/ubuntu/psla/egs/fsd50k/class_labels_indices.csv', '.data/datafiles/fsd50k_tr_full.json'))

    # (optional) create label enhanced set.
    # Go to /src/label_enhancement/


if __name__ == '__main__':
    prepare_fsd_dataset(resample=True)
