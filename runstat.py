from parse import DebugViewLog
import pandas as pd
import numpy as np
import os
import sys
import argparse
from typing import List, Tuple, Dict, Any
from lib.dotdict import DotDict

FILE_MODE_NAMES = DotDict(
    REF_MU = "refs_mu",
    REF_SC = "refs_sc",
    REF_KEY = "refs_key"
)

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

    def _walk_and_parse_recusive(dir, depth, args):
        logs = [[] for _ in range(DebugViewLog.DV_DATA_MODE.NUM_DATA_MODES.value)]

        for dirpath, dirnames, filenames in os.walk(dir):
            for dirname in dirnames:
                if depth == 0:
                    device = _folder_name(dirname)
                
                path = os.path.join(dirpath, dirname)
                sublogs = _walk_and_parse_recusive(path, depth + 1, args)
                logs = [row1 + row2 for row1, row2 in zip(logs, sublogs)]

                if depth == 0:
                    print("Parsed device: ", device, logs)
            else:
                for filename in filenames:
                    if not filename.lower().endswith('.csv'):
                        continue

                    mode = DebugViewLog.supported_mode(args.mode)
                    if not mode:
                        if FILE_MODE_NAMES.REF_MU in filename: 
                            mode = DebugViewLog.DV_DATA_MODE.MU
                        elif FILE_MODE_NAMES.SC in filename:
                            mode = DebugViewLog.DV_DATA_MODE.SC
                        elif FILE_MODE_NAMES.KEY in filename:
                            mode = DebugViewLog.DV_DATA_MODE.KEY
                        else:
                            # mode is not clear in filename
                            print ("Unknown file mode: ", filename)
                    
                    if mode:
                        log = _parse_file(os.path.join(dirpath, filename), args.tool, args.size, mode)
                        if log and len(log.frames):
                            logs[mode].append(log)
                else:
                    if len(filenames):
                        print("finish parse dir: %s", dirpath)
            
            # don't enter sub folder
            dirnames[:] = []
             
        return logs

    def _walk_and_parse(dir, args):                
        logcons: Dict[str: List[DebugViewLog]] = {}

        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if not filename.lower().endswith('.csv'):
                    continue

                mode = DebugViewLog.supported_mode(args.mode)
                if not mode:
                    if filename.startswith("refs_mu"): 
                        mode = DebugViewLog.DV_DATA_MODE.MU
                    elif filename.startswith("refs_sct"):
                        mode = DebugViewLog.DV_DATA_MODE.SC
                    elif filename.startswith("refs_key"):
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

    IDS_NAMES = ('major', 'minor')
    IDS_TITLE = (('project',''), ('device', ''), ('catagory', ''), ('file_id', ''), ('data_id', ''))
    IDS_AA_WO_DUALX = (('aa', 'max'), ('aa', 'min'), ('aa', 'range'))
    ID_DUALX = ('aa', 'dualx')
    IDS_DUALX = (ID_DUALX, )
    IDS_KEY = (('key', 'max'), ('key', 'min'), ('key', 'range'))

    def _build_series(ids, dats):
        raw_dict = dict(zip(ids, dats))
        return  pd.Series(raw_dict)
    
    def build_title(project, device, catagory, file_id, data_id):
        return _build_series(IDS_TITLE, (project, device, catagory, file_id, data_id))
    
    def build_aa(v_max, v_min, v_range, dualx):
        return _build_series(IDS_AA_WO_DUALX + IDS_DUALX, (v_max, v_min, v_range, dualx))

    def build_key(v_max, v_min, v_range):
        return _build_series(IDS_KEY, (v_max, v_min, v_range))

    def do_statics(df: pd.DataFrame, func):
        ids = IDS_AA_WO_DUALX + IDS_KEY
        raw_dict = dict(zip(ids, (func(df.loc[:, id]) for id in ids)))
        raw_dict[ID_DUALX] = df.iloc[-1][ID_DUALX]
        new_row = pd.Series(raw_dict)

        return new_row

    def log_to_df(project:str, logcons:Dict[str, List[DebugViewLog]]):
        
        col_name = IDS_TITLE + IDS_AA_WO_DUALX + IDS_DUALX + IDS_KEY
        columns = pd.MultiIndex.from_tuples(
            col_name,
            names=IDS_NAMES
        )

        dfcons:Dict[str, pd.DataFrame] = {}
        for mode in DebugViewLog.DV_DATA_MODE.values():
            dfcons[mode] = pd.DataFrame(columns=columns)

        for mode in logcons.keys():
            logs: List[DebugViewLog] = logcons[mode]
            for i, log in enumerate(logs):

                try:
                    _catagory, _modes, _devices = name_info(log.filename)
                except:
                    print("Project {}, String {}, Can't break ,skip the data".format(project, log.filename))
                    continue

                for j, d in enumerate(log.frames):
                    df = dfcons[mode]
                    new_row = build_title(project, '_'.join(_devices), _catagory, i, j)
                    if mode == DebugViewLog.DV_DATA_MODE.MU:
                        new_append = build_aa(d.max, d.min, d.range, d.dualx)        
                    elif mode == DebugViewLog.DV_DATA_MODE.KEY:
                        new_append = build_key(d.max, d.min, d.range)   
                        
                    new_row = pd.concat([new_row, new_append])
                    df.loc[len(df)] = new_row

        # Merge MU and key
        df1 = dfcons[DebugViewLog.DV_DATA_MODE.MU].drop(columns=[('key',)])
        df2 = dfcons[DebugViewLog.DV_DATA_MODE.KEY].drop(columns=[('aa',)])
        if len(df1) or len(df2):
            merged_df = pd.merge(df1, df2, on=IDS_TITLE[:3], how='inner', suffixes=('key', 'aa'))
            dfcons['summary'] = merged_df
            print(df1, df2, merged_df)

            # dualx df
            condition = ((merged_df[ID_DUALX] == True))
            dualx_df = merged_df[condition]
            
            # normal df
            condition = ((merged_df[ID_DUALX] == False))
            normal_df = merged_df[condition]

            dflist = (normal_df, dualx_df)
            # statics
            proclist = {
                'mean': pd.DataFrame.mean,
                'std': pd.DataFrame.std
            }

            for df in dflist:
                for k, v in proclist.items():
                    # add to df tail
                    title = build_title(project, '_'.join(_devices), "{} ({})".format(_catagory, k), None, None)
                    new_append = do_statics(df, v)
                    new_row = pd.concat([title, new_append])
                    merged_df.loc[len(merged_df)] = new_row

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
    
    runstat(cmd)