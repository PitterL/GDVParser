from parse import DebugViewLog
import pandas as pd
import numpy as np
import os
import sys
import argparse
from typing import List, Tuple, Dict, Any
from lib.dotdict import DotDict

FILE_TYPE_SUPPORT = DotDict(
    REF_MU = 'refs_mu',
    REF_SC = 'refs_sc',
    REF_KEY = 'refs_key'
)

FILE_EXTENSIONS = DotDict(
    EXTENSTION = ".csv"
)

class IDS(object):
    SCT_NAME = 'sct'
    MU_NAME = 'mu'
    KEY_NAME = 'key'
    DUALX_NAME = 'dualx'
    AA_NAME = 'aa'

    DF_COLUMN_LEVEL = ('major', 'minor')
    DF_COLUMN_TITLE = (('project',''), ('device', ''), ('catagory', ''), ('file_id', ''), ('data_id', ''))
    
    DF_COLUMN_AA_WO_DUALX = ((AA_NAME, 'max'), (AA_NAME, 'min'), (AA_NAME, 'range'))
    DF_COLUMN_KEY = ((KEY_NAME, 'max'), (KEY_NAME, 'min'), (KEY_NAME, 'range'))
    
    DF_COLUMN_DUALX_NAME = (AA_NAME, DUALX_NAME)
    DF_COLUMN_DUALX = (DF_COLUMN_DUALX_NAME, )
    DF_COLUMN_MERGE_KEY = DF_COLUMN_TITLE[:3]

    PAT_SPLIT_FILE_NAME = (SCT_NAME, MU_NAME, KEY_NAME, DUALX_NAME)

if __name__ == '__main__':
    def _parse_file(filename: str, tool:DebugViewLog.DV_DATA_TOOL, size:Tuple[int, int], mode:str):
        #limit = DebugViewLog.parse_range(args.range)
        if not DebugViewLog.supported_tool(tool):
            return
        
        if not DebugViewLog.supported_mode(mode):
            return

        size = DebugViewLog.parse_size(size)

        # Creat Log object
        return DebugViewLog(tool, size).parse(filename, mode)

    def _walk_and_parse(dir, args):                
        logcons: Dict[str: List[DebugViewLog]] = {}

        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if not filename.lower().endswith('.csv'):
                    continue

                mode = DebugViewLog.supported_mode(args.mode)
                if not mode:
                    if FILE_TYPE_SUPPORT.REF_MU in filename: 
                        mode = DebugViewLog.DV_DATA_MODE.MU
                    elif FILE_TYPE_SUPPORT.REF_SC in filename:
                        mode = DebugViewLog.DV_DATA_MODE.SC
                    elif FILE_TYPE_SUPPORT.REF_KEY in filename:
                        mode = DebugViewLog.DV_DATA_MODE.KEY
                    else:
                        # mode is not clear in filename
                        print ("Unknown file mode: ", filename)
                
                if mode:
                    log: DebugViewLog = _parse_file(os.path.join(dirpath, filename), args.tool, args.size, mode)
                    if log and len(log.frames):
                        if not logcons.get(mode):
                            logcons[mode] = []
                        logcons[mode].append(log)
            else:
                if len(filenames):
                    print("finish parse dir: %s", dirpath)
              
        return logcons

    def _folder_name(path):
        path = path.rstrip(os.sep)
        name = os.path.basename(path)
        return name

    def name_info(filename):
        name = os.path.basename(filename)
        infos = name.split('_')
        if (len(infos) < 3):
            raise ValueError("File name {} is not matched [type_mode_device] format".format(name))

        pos = 0
        for i, name in enumerate(reversed(infos)):
            if name in ('key', 'mu', 'sct', 'dualx'):
                pos = i
                break

        if pos <= 0:
            raise ValueError("File name {} device information is not found".format(name))

        pos = len(infos) - i
        _catagory = infos[0]
        _modes = infos[1: pos]
        _devices = infos[pos:]

        return _catagory, _modes, _devices

    def _build_series(ids, dats):
        raw_dict = dict(zip(ids, dats))
        return  pd.Series(raw_dict)
    
    def build_title(project, device, catagory, file_id, data_id):
        return _build_series(IDS.DF_COLUMN_TITLE, (project, device, catagory, file_id, data_id))
    
    def build_aa(v_max, v_min, v_range, dualx):
        return _build_series(IDS.DF_COLUMN_AA_WO_DUALX + IDS.DF_COLUMN_DUALX, (v_max, v_min, v_range, dualx))

    def build_key(v_max, v_min, v_range):
        return _build_series(IDS.DF_COLUMN_KEY, (v_max, v_min, v_range))

    def do_statics(df: pd.DataFrame, func):
        ids = IDS.DF_COLUMN_AA_WO_DUALX + IDS.DF_COLUMN_KEY
        raw_dict = dict(zip(ids, (func(df.loc[:, id]) for id in ids)))
        raw_dict[IDS.DF_COLUMN_DUALX_NAME] = df.iloc[-1][IDS.DF_COLUMN_DUALX_NAME]
        new_row = pd.Series(raw_dict)

        return new_row

    def log_to_df(project:str, logcons:Dict[str, List[DebugViewLog]]):
        
        col_name = IDS.DF_COLUMN_TITLE + IDS.DF_COLUMN_AA_WO_DUALX + IDS.DF_COLUMN_DUALX + IDS.DF_COLUMN_KEY
        columns = pd.MultiIndex.from_tuples(
            col_name,
            names=IDS.DF_COLUMN_LEVEL
        )

        dfcons:Dict[str, pd.DataFrame] = {}
        for mode in DebugViewLog.DV_DATA_MODE.values():
            dfcons[mode] = pd.DataFrame(columns=columns)

        for mode in logcons.keys():
            logs: List[DebugViewLog] = logcons[mode]
            for i, log in enumerate(logs):

                try:
                    _catagory, _modes, _devices = name_info(log.filename)
                    device = '_'.join(_devices)
                    catagory = _catagory
                except:
                    print("Project {}, String {}, Can't break ,skip the data".format(project, log.filename))
                    continue

                for j, d in enumerate(log.frames):
                    df = dfcons[mode]
                    new_row = build_title(project, device, catagory, i, j)
                    if mode == DebugViewLog.DV_DATA_MODE.MU:
                        new_append = build_aa(d.max, d.min, d.range, d.dualx)        
                    elif mode == DebugViewLog.DV_DATA_MODE.KEY:
                        new_append = build_key(d.max, d.min, d.range)   
                        
                    new_row = pd.concat([new_row, new_append])
                    df.loc[len(df)] = new_row

        # Merge MU and key
        df1 = dfcons[DebugViewLog.DV_DATA_MODE.MU].drop(columns=[(IDS.KEY_NAME,)])
        df2 = dfcons[DebugViewLog.DV_DATA_MODE.KEY].drop(columns=[(IDS.AA_NAME,)])
        if len(df1) or len(df2):
            merged_df = pd.merge(df1, df2, on=IDS.DF_COLUMN_MERGE_KEY, how='inner', 
                                 suffixes=("({})".format(IDS.KEY_NAME), "({})".format(IDS.AA_NAME)))
            dfcons['Summary'] = merged_df
            print(df1, df2, merged_df)

            # dualx df
            condition = ((merged_df[IDS.DF_COLUMN_DUALX_NAME] == True))
            dualx_df = merged_df[condition]
            
            # normal df
            condition = ((merged_df[IDS.DF_COLUMN_DUALX_NAME] == False))
            normal_df = merged_df[condition]

            dflist = (normal_df, dualx_df)
            # statics of `mean` and `std`
            proclist = {
                'mean': pd.DataFrame.mean,
                'std': pd.DataFrame.std
            }
            for df in dflist:
                for k, v in proclist.items():
                    # add to df tail
                    catagory = "{} ({})".format(_catagory, k)
                    title = build_title(project, device, catagory, None, None)
                    new_append = do_statics(df, v)
                    new_row = pd.concat([title, new_append])
                    merged_df.loc[len(merged_df)] = new_row

            # statics of 3*std
            # catagory = "{} ({})".format(_catagory, "3sigma")
            # title = build_title(project, device, catagory, None, None)

        return dfcons

    def save_to_file(dfcons: Dict[str, pd.DataFrame], output=None):

        writer = pd.ExcelWriter(output)
        for mode in dfcons.keys():
            df = dfcons[mode]

            if len(df):
                print("Write ", mode, len(df))
                df.to_excel(writer, sheet_name=str(mode))

        if writer.close:
            writer.close()
        else:
            writer.save()

        print("Save to:", output)


    def runstat(args=None):
        parser = parse_args(args)
        aargs = args if args is not None else sys.argv[1:]
        args = parser.parse_args(aargs)
        print(args)

        if not args.filename and not args.tool:
            parser.print_help()
            return

        if os.path.isfile(args.filename):
            dir = os.path.dirname(args.filename)
        elif os.path.isdir(args.filename):
            dir = args.filename
        else:
            return

        project = _folder_name(dir)
        print("Project name: ", project)

        ## _walk_and_parse_recusive(dir, 0, args)
        logcons: Dict[str: List[DebugViewLog]] = _walk_and_parse(dir, args)
        dfcons: Dict[str: pd.DataFrame] = log_to_df(project, logcons)
        save_to_file(dfcons, os.path.join(dir, project + ".xlsx"))

    def parse_args(args=None):

        parser = argparse.ArgumentParser(
            prog='xparse',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Tools for parsing debug log to xlxs file')

        parser.add_argument('--version',
                            action='version', version='%(prog)s v1.0.1',
                            help='show version')

        parser.add_argument('-f', '--filename', required=True,
                            nargs='?',
                            default='',
                            metavar='LOG_FILE',
                            help='where the \'XCFG|TXT\' file will be load')

        parser.add_argument('-t', '--tool', required=True,
                            nargs='?',
                            default='',
                            const='.',
                            metavar='hawkeye|mxtapp|studio',
                            help='format of of file data content')

        parser.add_argument('-m', '--mode', required=False,
                            nargs='?',
                            default='',
                            const='.',
                            metavar='sc|mc|key',
                            help='sensing mode of of file data content')

        parser.add_argument('-r', '--range', required=False,
                            nargs='?',
                            default='',
                            const='.',
                            metavar='^(low, hight)',
                            help='value range to save result, store to (low, high), ^ mean not in range')

        parser.add_argument('-s', '--size', required=False,
                            nargs='?',
                            default='',
                            const='.',
                            metavar='^(XSIZE, YSIZE)',
                            help='value to told XSIZE/YSIZE')

        return parser


    cmd = [
        '-t',
        'mxtapp',
        '-f',
        r'D:\trunk\customers3\Desay\Desay_Toyota_23MM_429D_1296M1_18581_Goworld\log\20240613 production log\502D_C SAMPLE_SW VER20240419'
    ]
    #cmd = None
    
    runstat(cmd)