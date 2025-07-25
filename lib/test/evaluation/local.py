from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()


    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/lzw/LEF/Github_load/DGTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/lzw/LEF/Github_load/DGTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/lzw/LEF/Github_load/DGTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/home/lzw/LEF/Github_load/DGTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/lzw/LEF/Github_load/DGTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/lzw/LEF/Github_load/DGTrack/data/lasot'
    settings.network_path = '/home/lzw/LEF/Github_load/DGTrack/output/test/networks'
    settings.nfs_path = '/home/lzw/LEF/Github_load/DGTrack/data/nfs'
    settings.otb_path = '/home/lzw/LEF/Github_load/DGTrack/data/otb'
    settings.prj_dir = '/home/lzw/LEF/Github_load/DGTrack'
    settings.result_plot_path = '/home/lzw/LEF/Github_load/DGTrack/output/test/result_plots'
    settings.results_path = '/home/lzw/LEF/Github_load/DGTrack/output/test/tracking_results'
    settings.save_dir = '/home/lzw/LEF/Github_load/DGTrack/output'
    settings.segmentation_path = '/home/lzw/LEF/Github_load/DGTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/lzw/LEF/Github_load/DGTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/lzw/LEF/Github_load/DGTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/lzw/LEF/Github_load/DGTrack/data/trackingnet'
    settings.uav_path = '/home/lzw/LEF/Github_load/DGTrack/data/uav'
    settings.vot18_path = '/home/lzw/LEF/Github_load/DGTrack/data/vot2018'
    settings.vot22_path = '/home/lzw/LEF/Github_load/DGTrack/data/vot2022'
    settings.vot_path = '/home/lzw/LEF/Github_load/DGTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    settings.uavdt_path = '/media/lzw/LEF-SSD/compare_data/train_data_uavdt/uavdt/'

    return settings

