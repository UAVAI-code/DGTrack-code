import os
import sys
import gdown
import re
import shutil
import argparse
import tempfile
import _init_paths

from lib.test.evaluation.environment import env_settings

pytracking_results_link_dict = {
    "dimp": {
        "prdimp50_003.zip": "1p13j3iwcOCubBi3ms0hLwqnP6-x0J8Mc",
        "prdimp50_002.zip": "1PPKgrAepbuyM2kjfzYAozQKTL6AjcQOz",
        "prdimp50_001.zip": "17NFBObEDeK6mW4Mk2vN5Ekk1SGbFvxRS",
        "prdimp50_000.zip": "1r3Efq7AumML2yGQ_KV4zmf4ATKVE1bo6",
        "prdimp18_004.zip": "1DF4ZJQAa4CwvN_OiT4te33AV0kpsO7JM",
        "prdimp18_003.zip": "1RgwJAN4TxnzgVgsfvrHIg1OUXD1EBZkO",
        "prdimp18_002.zip": "17lMllYhygCqgE81DoHX4BZar3xc3auzM",
        "prdimp18_001.zip": "1Yg7DmGYOnn2k0MYtSjjKlGyzO1Uimj4G",
        "prdimp18_000.zip": "1DuZJSBJ-23WJBQTOWSAaoPYSbGAJJN2Z",
        "prdimp50_004.zip": "1f9bx9-dtx3B5_IvIJhjjJyp-cnXciqLO",
        "dimp50_004.zip": "1Lj3p8mYCoIqxzdQXZkWFTw-MA8c6eeLa",
        "dimp50_000.zip": "1LCgf5sg453Z4bY37A_W5mbXeG68U1fET",
        "dimp18_000.zip": "17M7dJZ1oKrIY4-O5lL_mlQPEubUn034g",
        "dimp18_001.zip": "1AsiliVgISyDTouYOQYVOXA0srj3YskhJ",
        "dimp18_002.zip": "1I0GrBaPnySOyPWSvItHhXH8182tFCi_Y",
        "dimp50_001.zip": "1XfPvwAcymW88J1rq7RlhyKmqsawJDK-K",
        "dimp18_004.zip": "1EztF6bpROFwZ1PSJWgMB7bQ4G_Z08YIg",
        "dimp18_003.zip": "1iuiFLv04WE7GfBjm8UkZXFq4gheG2Ru8",
        "dimp50_003.zip": "1rLsgeQXyKpD6ryl9BjlIVdO3vd27ekwy",
        "dimp50_002.zip": "1wj2jUwlpHgsP1hAcuxXAVriUPuEspsu4",
    },
    "atom": {
        "default_004.zip": "1BapnQh_8iRM44DXj862eOZV4q8zQLdmT",
        "default_003.zip": "1YpfOBLBEUQQiX0fWMPA5pnW3dm0NG3E5",
        "default_000.zip": "1x6fKGZk3V839mX99Gl_pw7JUaiMaTxc5",
        "default_002.zip": "1QIlQFv3p6MBTwsYdIMYmzUDBDQGxGsUC",
        "default_001.zip": "1-K2--GNCURDKEgUuiEF18K4DcCLvDEVt",
    },
    "kys": {
        "default_004.zip": "1QdfkA3d4MzKwdDiBOM1ZhDJWk9NmALxD",
        "default_000.zip": "1SCs79_ePTc8zxPDzRAgAmbbRlnmE89SN",
        "default_003.zip": "1TCzq38QW4YiMrgU5VR6NAEefJ85gwzfT",
        "default_002.zip": "1_9u1ybCFxHu0yJmW5ZzDR4-isJMEUsDf",
        "default_001.zip": "1utJhdosNj6vlI75dfzUxGM3Vy8OjWslT",
    },
}


def _download_file(file_id, path):
    link = 'https://drive.google.com/uc?id=' + file_id
    gdown.download(link, path, quiet=True)


def download_results(download_path, trackers='pytracking'):
    """
    Script to automatically download tracker results for PyTracking.
    args:
        download_path - Directory where the zipped results are downloaded
        trackers - Tracker results which are to be downloaded.
                   If set to 'pytracking', results for all pytracking based trackers will be downloaded.
                   If set to 'external', results for available external trackers will be downloaded.
                   If set to 'all', all available results are downloaded.
                   If set to a name of a tracker (e.g. atom), all results for that tracker are downloaded.
                   Otherwise, it can be set to a dict, where the keys are the names of the trackers for which results are
                   downloaded. The value can be set to either 'all', in which case all available results for the
                    tracker are downloaded. Else the value should be a list of parameter file names.
    """
    print('Using download path ''{}'''.format(download_path))

    os.makedirs(download_path, exist_ok=True)

    if isinstance(trackers, str):
        if trackers == 'all':
            all_trackers = list(pytracking_results_link_dict.keys()) + list(external_results_link_dict.keys())
            trackers = {k: 'all' for k in all_trackers}
        elif trackers == 'pytracking':
            trackers = {k: 'all' for k in pytracking_results_link_dict.keys()}
        elif trackers == 'external':
            trackers = {k: 'all' for k in external_results_link_dict.keys()}
        elif trackers in pytracking_results_link_dict or trackers in external_results_link_dict:
            trackers = {trackers: 'all'}
        else:
            raise Exception('tracker_list must be set to ''all'', a tracker name, or be a dict')
    elif isinstance(trackers, dict):
        pass
    else:
        raise Exception('tracker_list must be set to ''all'', or be a dict')

    common_link_dict = pytracking_results_link_dict

    for trk, runfiles in trackers.items():
        trk_path = os.path.join(download_path, trk)
        if not os.path.exists(trk_path):
            os.makedirs(trk_path)

        if runfiles == 'all':
            for params, fileid in common_link_dict[trk].items():
                print('Downloading: {}/{}'.format(trk, params))
                _download_file(fileid, os.path.join(trk_path, params))
        elif isinstance(runfiles, (list, tuple)):
            for p in runfiles:
                for params, fileid in common_link_dict[trk].items():
                    if re.match(r'{}(|_(\d\d\d)).zip'.format(p), params) is not None:
                        print('Downloading: {}/{}'.format(trk, params))
                        _download_file(fileid, os.path.join(trk_path, params))

        else:
            raise Exception('tracker_list values must either be set to ''all'', or be a list of param names')



def unpack_tracking_results(download_path, output_path=None):
    """
    Unpacks zipped benchmark results. The directory 'download_path' should have the following structure
    - root
        - tracker1
            - param1.zip
            - param2.zip
            .
            .
        - tracker2
            - param1.zip
            - param2.zip
        .
        .
    args:
        download_path - Path to the directory where the zipped results are stored
        output_path - Path to the directory where the results will be unpacked. Set to env_settings().results_path
                      by default
    """

    if output_path is None:
        output_path = env_settings().results_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    trackers = os.listdir(download_path)

    for t in trackers:
        runfiles = os.listdir(os.path.join(download_path, t))

        for r in runfiles:
            save_path = os.path.join(output_path, t)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            shutil.unpack_archive(os.path.join(download_path, t, r), os.path.join(save_path, r[:-4]), 'zip')


def main():
    parser = argparse.ArgumentParser(description='Download and unpack zipped results')
    parser.add_argument('--tracker', type=str, default='pytracking',
                        help='Name of tracker results to download, or "pytracking" (downloads results for PyTracking'
                             ' based trackers, or "external" (downloads results for external trackers) or "all"')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to the directory where the results will be unpacked.')
    parser.add_argument('--temp_download_path', type=str, default=None,
                        help='Temporary path used for downloading the Zip files.')
    parser.add_argument('--download', type=bool, default=True,
                        help='Whether to download results or unpack existing downloaded files.')
    args = parser.parse_args()

    download_path = args.temp_download_path
    if download_path is None:
        download_path = '{}/pytracking_results/'.format(tempfile.gettempdir())

    if args.download:
        download_results(download_path, args.tracker)

    unpack_tracking_results(download_path, args.output_path)


if __name__ == '__main__':
    main()