import pandas as pd
import numpy as np
import os
import sys
import argparse
import copy
from typing import List, Tuple, Dict, Any
from lib.dotdict import DotDict
from parse.gdv import DebugViewLog

FILE_EXTENSIONS = DotDict(
    EXTENSTION = ".csv"
)

class IDS(object):
    SC_NAME = DebugViewLog.DV_DATA_MODE.SC
    MU_NAME = DebugViewLog.DV_DATA_MODE.MU
    KEY_NAME = DebugViewLog.DV_DATA_MODE.KEY
    MU_DUALX_SUBNAME = DebugViewLog.DV_DATA_SUBMODE.DUALX
    SC_SCT_SUBNAME = DebugViewLog.DV_DATA_SUBMODE.SCT
   
    DF_COLUMN_FIRST_LEVEL = 'major'
    DF_COLUMN_SECOND_LEVEL = 'minor'
    DF_COLUMN_LEVELS = (DF_COLUMN_FIRST_LEVEL, DF_COLUMN_SECOND_LEVEL)

    DF_COLUMN_CATAGORY = ('catagory', '')

    DF_COLUMN_2ND_MAX = 'max'
    DF_COLUMN_2ND_MIN = 'min'
    DF_COLUMN_2ND_RANGE = 'range'
    DF_COLUMN_2ND_MARK = 'mark'

    DF_COLUMN_TITLE = (('project',''), ('device', ''), DF_COLUMN_CATAGORY, ('file_id', ''), ('data_id', ''))
    
    DF_COLUMN_AA_WO_DUALX = ((MU_NAME, DF_COLUMN_2ND_MAX), (MU_NAME, DF_COLUMN_2ND_MIN), (MU_NAME, DF_COLUMN_2ND_RANGE))
    DF_COLUMN_KEY = ((KEY_NAME, DF_COLUMN_2ND_MAX), (KEY_NAME, DF_COLUMN_2ND_MIN), (KEY_NAME, DF_COLUMN_2ND_RANGE))
    DF_COLUMN_SC = ((SC_NAME, DF_COLUMN_2ND_MAX), (SC_NAME, DF_COLUMN_2ND_MIN), (SC_NAME, DF_COLUMN_2ND_RANGE))
    DF_COLUMN_STAT = (DF_COLUMN_2ND_MAX, DF_COLUMN_2ND_MIN, DF_COLUMN_2ND_RANGE)

    DF_COLUMN_DUALX_NAME = (MU_NAME, MU_DUALX_SUBNAME)
    DF_COLUMN_DUALX = (DF_COLUMN_DUALX_NAME, )
    DF_COLUMN_SC_MARK = ((SC_NAME, DF_COLUMN_2ND_MARK), )

    DF_COLUMN_ALL = DF_COLUMN_TITLE + DF_COLUMN_SC + DF_COLUMN_SC_MARK + DF_COLUMN_AA_WO_DUALX + DF_COLUMN_DUALX + DF_COLUMN_KEY
    DF_COLUMN_MERGE_KEY = DF_COLUMN_TITLE[:3]

    FILENAME_SPLIT = (MU_NAME, KEY_NAME, SC_SCT_SUBNAME, MU_DUALX_SUBNAME)

    STAT_NAME = 'STATS'
    STAT_MEAN = 'mean'
    STAT_STD = 'std'
    STAT_3SIGMA = '3sigma'
    STAT_MARGIN = 'margin'
    STAT_LIMIT = 'limit'
    STAT_RESULT = 'Sigmal Limit'

    SUMMARY_NAME = 'Summary'

if __name__ == '__main__':

    def _walk_and_parse(dir, args):
        """
            @dir: the log directory
            @args: parsing args
                # cate: the category of the data <signal/ref,delta>
                # mode: the data mode <sc/mu/key>
                # tool: logging tool <studio/mxtapp/haweye>
                # size: the matrix size of the log

            @return the container of DebugViewLog with the dict format of `(cate, mode) = List[DebugViewLog]`
                we can get the prased Dataframe|Series from DebugViewLog.frames[n].data
        """
        logcons: Dict[Tuple[DebugViewLog.ENUM_DV_DATA_CATE, DebugViewLog.ENUM_DV_DATA_MODE], List[DebugViewLog]] = {}

        # check logging tool
        tool = DebugViewLog.supported_tool(args.tool)
        if not tool:
            return logcons

        # check size paramenter with Tuple[int, int], None for auto
        size = DebugViewLog.parse_size(args.size)

        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if not filename.lower().endswith('.csv'):
                    continue

                cate = DebugViewLog.supported_cate(args.cate if args.cate else filename)
                mode = DebugViewLog.supported_mode(args.mode if args.mode else filename)               
                if mode:
                    log: DebugViewLog = DebugViewLog(tool, size).parse(os.path.join(dirpath, filename), cate, mode)
                    if log and len(log.frames):
                        k = (cate, mode)
                        if not logcons.get(k):
                            logcons[k] = []
                        logcons[k].append(log)
                else:
                     # mode is not clear in filename
                        print ("Unknown file mode: ", filename)
            # else:
            #    if len(filenames):
            #        print("finish parse dir: ", dirpath)
              
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
            if name in IDS.FILENAME_SPLIT:
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
        """
            @ids: the 2 level columns's name
            @dats: the data array
            @return pd.Series
        """
        # added 2nd level index
        index = pd.MultiIndex.from_arrays(zip(*ids), names=IDS.DF_COLUMN_LEVELS)

        # build series with dict
        raw_dict = dict(zip(ids, dats))

        return  pd.Series(raw_dict, index=index)
    
    def build_title(project, device, catagory, file_id, data_id):
        return _build_series(IDS.DF_COLUMN_TITLE, (project, device, catagory, file_id, data_id))
    
    def build_sc(v_max, v_min, v_range, mark):
        return _build_series(IDS.DF_COLUMN_SC + IDS.DF_COLUMN_SC_MARK, (v_max, v_min, v_range, mark))

    def build_aa(v_max, v_min, v_range, dualx):
        return _build_series(IDS.DF_COLUMN_AA_WO_DUALX + IDS.DF_COLUMN_DUALX, (v_max, v_min, v_range, dualx))

    def build_key(v_max, v_min, v_range):
        return _build_series(IDS.DF_COLUMN_KEY, (v_max, v_min, v_range))

    def multiply_if_number(x, A):
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return x * A
        else:
            return x

    def do_mean_sigma(dfcons: Dict[str, pd.DataFrame], percent: float):
        """
            @dfcons: dataframe contatiner with the format of `([IDS.SUMMARY_NAME]_[signal|ref|delta]) = DataFrame`
            @percent: the margin percent to calculate the signal limit range
            @return: dfcons with signal limit data filled
        """
        # statistic with delta/ref/signal individually
        for shname, df in dfcons.items():

            # check sheet name with Summary only
            if not shname.startswith(IDS.SUMMARY_NAME):
                continue

            if IDS.DF_COLUMN_DUALX_NAME in df.columns:
                # normal df
                condition = ((df[IDS.DF_COLUMN_DUALX_NAME] == False))
                df_normal = df[condition]

                # dualx df
                condition = ((df[IDS.DF_COLUMN_DUALX_NAME] == True))
                df_dualx = df[condition]
                dflist = (df_normal, df_dualx)
            else:
                dflist = (df, )
            
            # statistic with dual on/off individually
            for i, d in enumerate(dflist):
                
                # <1> Calculate mean, std
                title = build_title('', IDS.STAT_NAME, IDS.STAT_STD, None, None)
                # select 2nd level index is not Null
                filtered_columns = d.loc[:, d.columns.get_level_values(IDS.DF_COLUMN_SECOND_LEVEL).isin(IDS.DF_COLUMN_STAT)]
                
                mean_values = filtered_columns.mean()
                new_row = pd.concat([title, mean_values])
                new_row[IDS.DF_COLUMN_DUALX_NAME] = True if i == 1 else False

                df.loc[len(df)] = new_row # added to tail
                mean_row = new_row

                title = build_title('', IDS.STAT_NAME, IDS.STAT_MEAN, None, None)
                # select 2nd level index is not Null
                # filtered_columns = d.loc[:, d.columns.get_level_values(IDS.DF_COLUMN_SECOND_LEVEL).isin(IDS.DF_COLUMN_STAT)]
                std_values = filtered_columns.std()
                new_row = pd.concat([title, std_values])
                new_row[IDS.DF_COLUMN_DUALX_NAME] = True if i == 1 else False

                df.loc[len(df)] = new_row # added to tail
                std_row = new_row

                # <2> Calculate 3-sigma, margin, limit range
                if len(std_row) and len(mean_row):
                    # <2.1> 3-sigma
                    sigma_row = std_row.apply(multiply_if_number, args=(3,))
                    
                    sigma_row.loc[IDS.DF_COLUMN_CATAGORY] = IDS.STAT_3SIGMA
                    df.loc[len(df)] = sigma_row # added to tail

                    # <2.2> margin
                    margin_row = mean_row.apply(multiply_if_number, args=(percent,))
                    
                    margin_row.loc[IDS.DF_COLUMN_CATAGORY] = f'{IDS.STAT_MARGIN}-({percent*100: .0f})%'
                    df.loc[len(df)] = margin_row # added to tail

                    # <2.3> limit range (normal and dualx individually)
                    # calculate the limit as (avg +/- (margin + 3-sigma)), so we get the average row and filter the max/min/range
                    limit_row = copy.deepcopy(mean_row)
                    max_cond = (limit_row.index.get_level_values(IDS.DF_COLUMN_SECOND_LEVEL) == IDS.DF_COLUMN_2ND_MAX)
                    min_cond = (limit_row.index.get_level_values(IDS.DF_COLUMN_SECOND_LEVEL) == IDS.DF_COLUMN_2ND_MIN)
                    range_cond = (limit_row.index.get_level_values(IDS.DF_COLUMN_SECOND_LEVEL) == IDS.DF_COLUMN_2ND_RANGE)

                    # high_limit/range: max + margin + 3sigma, low limit: min - margin - 3sigma
                    limit_row[max_cond] = limit_row[max_cond] * (1 + percent) + sigma_row[max_cond]
                    limit_row[min_cond] = limit_row[min_cond] * (1 - percent) - sigma_row[min_cond]
                    limit_row[range_cond] = limit_row[range_cond] * (1 + percent) + sigma_row[range_cond]
                    
                    limit_row.loc[IDS.DF_COLUMN_CATAGORY] = IDS.STAT_LIMIT
                    df.loc[len(df)] = limit_row # added to tail

            # <3> combine the high_limit between dualx and normal 
            df_limit_rows = df.loc[df[IDS.DF_COLUMN_CATAGORY] == IDS.STAT_LIMIT]
            if len(df_limit_rows) == 2:
                max_values = df_limit_rows.max()

                max_values[IDS.DF_COLUMN_DUALX_NAME] = '-'
                max_values[IDS.DF_COLUMN_CATAGORY] = IDS.STAT_RESULT
                df.loc[len(df)] = max_values # added to tail

        return dfcons

    def organize_data(project: str, logcons: Dict[Tuple[DebugViewLog.ENUM_DV_DATA_CATE, DebugViewLog.ENUM_DV_DATA_MODE], List[DebugViewLog]]):
        """
            @project: project name
            @logcons: the container of DebugViewLog with the dict format of `(cate, DebugViewLog.DV_DATA_MODE) = List[DebugViewLog]`

            @return:
              dfsum: the summary of each data `([IDS.SUMMARY_NAME]_[signal|ref|delta]) = DataFrame`
              dfeach: the details of each data `(name) = frame`
        """

        ### create dataframe columns' name
        col_name = IDS.DF_COLUMN_ALL
        columns = pd.MultiIndex.from_tuples(
            col_name,
            names=IDS.DF_COLUMN_LEVELS
        )

        ### initialize the Null data container, which will storage each frame's statistic data(max, min, range...)
        dfcons: Dict[DebugViewLog.ENUM_DV_DATA_CATE,
                     Dict[DebugViewLog.ENUM_DV_DATA_MODE, pd.DataFrame]] = {}

        ### the summary of ouput content
        dfsum: Dict[str, pd.DataFrame] = {}
        ### each detail of the ouput data
        dfeach: Dict[str, pd.DataFrame] = {}

        for (cate, mode), logs in logcons.items():
            # loop the logs
            for i, log in enumerate(logs):
                try:
                    _catagory, _modes, _devices = name_info(log.filename)
                    device = '_'.join(_devices)
                    catagory = _catagory
                except:
                    print("Project {}, String {}, Can't break ,skip the data".format(project, log.filename))
                    continue

                for j, d in enumerate(log.frames):
                    # create the dataframe fo current frame data
                    if not cate in dfcons.keys():
                        dfcons[cate] = { mode: pd.DataFrame(columns=columns) }
                        cates = dfcons[cate]
                    else:
                        cates = dfcons[cate]
                        if mode not in cates.keys():
                            cates[mode] = pd.DataFrame(columns=columns)

                    # current frame statistic table 
                    tb = cates[mode]
                        
                    # add the data into the data frame
                    new_row = build_title(project, device, catagory, i, j)
                    if mode == DebugViewLog.DV_DATA_MODE.MU:
                        new_append = build_aa(d.max, d.min, d.range, d.dualx)
                    elif mode == DebugViewLog.DV_DATA_MODE.KEY:
                        new_append = build_key(d.max, d.min, d.range)
                    elif mode == DebugViewLog.DV_DATA_MODE.SC:
                        new_append = build_sc(d.max(), d.min(), d.range(), d.submode)
                    else:
                        print("Unsupport mode in logcons:", mode, d)
                        continue

                    new_row = pd.concat([new_row, new_append])
                    # insert to last line in the dataframe data
                    tb.loc[len(tb)] = new_row

                    # store each data detal
                    dfeach[f'{cate}_{mode}_{i}_{j}'] = d.data


        for cate, cons in dfcons.items():
            # remove null frames
            keys = cons.keys()
            for mode in keys:
                df = cons[mode]
                if not len(df):
                    del cons[mode]

            # drop unused columns
            for mode, df in cons.items():
                if mode == DebugViewLog.DV_DATA_MODE.SC:
                    df.drop(columns=[IDS.KEY_NAME, IDS.MU_NAME], level='major', inplace=True)
                elif mode == DebugViewLog.DV_DATA_MODE.MU:
                    df.drop(columns=[IDS.KEY_NAME, IDS.SC_NAME], level='major', inplace=True)
                elif mode == DebugViewLog.DV_DATA_MODE.KEY:
                    df.drop(columns=[IDS.MU_NAME, IDS.SC_NAME], level='major', inplace=True)
                else:
                    print("Unknow mode of the data:", mode, df)


            modes = list(cons.keys())
            merged_mode = modes[0]
            merged_df = cons[merged_mode]
            for mode in modes[1: ]:
                df = cons[mode]
                if len(df):
                    merged_df = pd.merge(merged_df, df, on=IDS.DF_COLUMN_MERGE_KEY, how='inner', 
                                        suffixes=(f"({merged_mode})", f"({mode})"))
                    merged_mode = mode

            #print(merged_df)

            dfsum[f'{IDS.SUMMARY_NAME}_{cate}'] = merged_df

        return dfsum, dfeach

    def save_to_file(dfcons: Dict[str, pd.DataFrame], trans=False, output=None):

        handled = False
        writer = pd.ExcelWriter(output)
        for mode, df in dfcons.items():
            if trans:
                if isinstance(df, pd.Series):
                    df = pd.DataFrame(df)
                df = df.T

            if len(df):
                print("Write ", mode, len(df))
                df.to_excel(writer, sheet_name=str(mode))
                handled = True
            else:
                print("Null table", mode)

        if handled:
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

        # Log to Log container, the container will split the log into different mode (DV_DATA_CATE, DV_DATA_MODE) as the key.
        #   the parsed data is stored in  DebugViewLog.frames[].data with format Dataframe|Series
        logcons: Dict[Tuple[DebugViewLog.ENUM_DV_DATA_CATE, DebugViewLog.ENUM_DV_DATA_MODE], List[DebugViewLog]] = _walk_and_parse(dir, args)

        # Log data to Summary
        # dfcons: Dict[str, pd.DataFrame]
        # dfeach: Dict[str, pd.DataFrame|pd.Series]
        (dfsum, dfeach) = organize_data(project, logcons)

        try:
            percent = (int(args.percent) / 100)
        except:
            percent = 0

        do_mean_sigma(dfsum, percent)

        # Data frame output
        save_to_file(dfsum, False,  os.path.join(dir, project + "_summary.xlsx"))
        save_to_file(dfeach, True, os.path.join(dir, project + "_details.xlsx"))

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
                            help='the tool of the data captured')

        parser.add_argument('-m', '--mode', required=False,
                            nargs='?',
                            default='',
                            const='.',
                            metavar='sc|mu|key',
                            help='the sensing mode of data: sc/mc/key')
        
        parser.add_argument('-c', '--cate', required=False,
                            nargs='?',
                            default='',
                            const='.',
                            metavar='ref|delta|signal',
                            help='the categroy of data: signal/ref/delta')
        
        parser.add_argument('-r', '--range', required=False,
                            nargs='?',
                            default='',
                            const='.',
                            metavar='^(low, hight)',
                            help='value range to save result, store to (low, high), ^ mean not in range')

        parser.add_argument('-p', '--percent', required=False,
                            nargs='?',
                            default='0',
                            const='.',
                            metavar='%',
                            help='percent of mean value to set signal limt')
        
        parser.add_argument('-s', '--size', required=False,
                            nargs='?',
                            default='',
                            const='.',
                            metavar='^(XSIZE, YSIZE)',
                            help='value to told XSIZE/YSIZE')

        return parser


    cmd = '-t mxtapp -p 5 -c ref'.split() + \
        ['-f', r'D:\trunk\customers3\Desay\Desay_Toyota_23MM_429D_1296M1_18581_Goworld\log\20240613 production log\502D_C SAMPLE_SW VER20240419']
    cmd = None

    runstat(cmd)