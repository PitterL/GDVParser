# Parser/gdv.py

import re
import os
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Union
from enum import Enum
from lib.dotdict import DotDict

SENSOR = DotDict(
    X = 0,
    Y = 1,
    ANY = 3
)
ENUM_SENSOR= Enum("ENUM_SENSOR", SENSOR)

# Check whether we can found any element in a given string, return the element if found 
def _check_contains(s, tuple_of_strings):
    for item in tuple_of_strings:
        if item in s:
            return item
    return None

class loader(object):
    """
        base object to load target file
        outupt:
            .f: the file handle
            .fileanme: the file name
    """
    def __init__(self, filename=None):
        self.f = None
        self.load_file(filename)

    def load_file(self, filename):
        if filename:
            if self.f:
                self.f.close()
            try:
                self.f = open(filename, 'r')
                self.filename = filename
                #print("Loader get file: %s", filename)
            except:
                print("unable to open file {}".format(filename))

    def close_file(self):
        self.f.close()
        self.f = None

class RsdDataSc(object):
    """
        @np_data_x ndarray of x channel
        @np_data_y: ndarray of y channel
        @submode: DV_DATA_SUBMODE
        @cate: DV_DATA_CATE

        @return:
            .maxtrix: the x, y channels configure
            .np_array: the ndarray of x, y channels's data
            .data: Series of data
    """
    def __init__(self, np_data_x, np_data_y, submode, cate):

        self.submode = submode
        self.cate = cate

        mx = len(np_data_x)
        my = len(np_data_y)
        self.maxtrix = (mx, my)
        self.np_array = (np_data_x, np_data_y)

        key_name_x = ["{}_X{}".format(submode, i) for i in range(mx)]
        key_name_y = ["{}_Y{}".format(submode, i) for i in range(my)]
        name = key_name_x + key_name_y
        d = np.concatenate(self.np_array, axis=0)
        self.data = pd.Series(data=d, 
                              index= name)

    def max(self, stp: ENUM_SENSOR = SENSOR.ANY):
        if stp < SENSOR.ANY:
            return self.np_array[stp].max()
        else:
            return max(self.np_array[SENSOR.X].max(), self.np_array[SENSOR.Y].max())

    def min(self, stp: ENUM_SENSOR = SENSOR.ANY):
        if stp < SENSOR.ANY:
            return self.np_array[stp].min()
        else:
            return min(self.np_array[SENSOR.X].min(), self.np_array[SENSOR.Y].min())

    def range(self, stp: ENUM_SENSOR = SENSOR.ANY):
        return self.max(stp) - self.min(stp)

    def mean(self, stp: ENUM_SENSOR = SENSOR.ANY):
        if stp < SENSOR.ANY:
            return np.nanmean(self.np_array[stp])
        else:
            d = np.concatenate(self.np_array, axis=0)
            return np.nanmean(d)

class RsdDataMu(object):
    """
        @dt: data time
        @n: the data number
        @dualx: indicate whether it's dualx data
        @np_data: re-shaped ndarray of x, y channels's data 

        @return:
            .dualx: dualx mode
            .maxtrix: the x, y channels configure
            .np_data: the raw ndarray of x, y channels's data after re-shaped
            .data: Dataframe of data
    """
    def __init__(self, dt, n, dualx, np_data):
        self.dt = dt
        self.n = n
        self.dualx = dualx
        (mx, my) = (len(np_data), len(np_data[0]))
        if dualx:
            mx = mx - 1
            np_data = np_data[:-1, :]

        self.maxtrix = (mx, my)
        self.np_data = np_data
        self.size = None

        row_name = ["X{}".format(i) for i in range(mx)]
        col_name = ["Y{}".format(i) for i in range(my)]
        self.data = pd.DataFrame(np_data, index=row_name, columns=col_name)

    def set_size(self, size):
        if isinstance(size, (tuple, list)):
            self.size = size

    @property
    def max(self):
        if isinstance(self.size, (tuple, list)):
            xsize, ysize = self.size
            if self.dualx:
                xsize = xsize - 1

            d = self.data.iloc[:xsize, :ysize]
        else:
            d = self.data
        
        return d.max().max()

    @property
    def min(self):
        if isinstance(self.size, (tuple, list)):
            xsize, ysize = self.size
            if self.dualx:
                xsize = xsize - 1
            d = self.data.iloc[:xsize, :ysize]
        else:
            d = self.data

        return d.min().min()

    @property
    def range(self):
        return self.max - self.min
    
    @property
    def mean(self):
        if isinstance(self.size, (tuple, list)):
            xsize, ysize = self.size
            if self.dualx:
                xsize = xsize - 1
            d = self.data.iloc[:xsize, :ysize]
        else:
            d = self.data

            return np.nanmean(d)

class RsdDataKey(object):
    """
        @np_data: ndarray of key channels's data 

        @return:
            .maxtrix: the key channels configure
            .np_data: the raw ndarray of key channels's data
            .data: Dataframe of data
    """
    def __init__(self, np_data):
        mx = len(np_data)
        self.maxtrix = (mx, 1)
        self.np_data = np_data

        key_name = ["Key{}".format(i) for i in range(mx)]
        self.data = pd.Series(data=np_data, index= key_name)

    @property
    def max(self):
        return self.np_data.max()

    @property
    def min(self):
        return self.np_data.min()

    @property
    def range(self):
        return self.max - self.min
    
    @property
    def mean(self):
        d = self.np_data
        return np.nanmean(d)

class DebugViewLog(loader):

    DV_DATA_TOOL = DotDict(MXTAPP='mxtapp', HAWKEYE='hawkeye', STUDIO='studio')
    DV_DATA_CATE = DotDict(SIGNAL='signal', REF='ref', DELTA='delta')
    DV_DATA_MODE = DotDict(SC='sc', MU='mu', KEY='key')

    DV_DATA_SUBMODE = DotDict(DUALX='dualx', SCT='sct', SCP='scp')

    ENUM_DV_DATA_CATE = Enum("ENUM_DV_DATA_CATE", DV_DATA_CATE)
    ENUM_DV_DATA_MODE = Enum("ENUM_DV_DATA_MODE", DV_DATA_MODE)

    MU_AA_FILE_FORMAT = DotDict({
        DV_DATA_TOOL.MXTAPP: {
            'sep':',',
            'title': { 'pat': r'[xX](\d+)[yY](\d+)_([Rr]eference16|[Dd]elta16)', 'st': 2, 'end': None},
            'row': {'name': '%H:%M:%S.%f', 'st': 2, 'end': None } 
        },
        DV_DATA_TOOL.HAWKEYE: {
            'sep':',',
            'title': { 'pat': r'[xX](\d+)[yY](\d+)', 'st': 0, 'end': None},
            'row': {'name': '%H:%M:%S %f', 'st': 1, 'end': None } 
        },
        DV_DATA_TOOL.STUDIO: {
            'sep':',',
            'title': { 'pat': r'[xX](\d+)[yY](\d+)', 'st': 0, 'end': None},
            'row': {'name': '%H:%M:%S %f', 'st': 1, 'end': None } 
        }
    })

    SC_AA_FILE_FORMAT = DotDict({
        DV_DATA_TOOL.MXTAPP: {
            'sep':',',
            'title': { 'pat': r'([xXyY])(\d+)', 'st': 2, 'end': None},
            'row': {'name': '%H:%M:%S.%f', 'st': 2, 'end': None } 
        },
        DV_DATA_TOOL.HAWKEYE: {
            'sep':',',
            'title': { 'pat': r'[Key](\d+)', 'st': 1, 'end': None},
            'row': {'name': None, 'st': 1, 'end': None } 
        },
        DV_DATA_TOOL.STUDIO: {
            'sep':',',
            'title': { 'pat': r'[Key](\d+)', 'st': 1, 'end': None},
            'row': {'name': None, 'st': 1, 'end': None } 
        }
    })

    KEY_FILE_FORMAT =  DotDict({
        DV_DATA_TOOL.MXTAPP: {
            'sep':',',
            'title': { 'pat': r'Key(\d)+', 'st': 1, 'end': None},
            'row': {'name': None, 'st': 0, 'end': None } 
        },
        DV_DATA_TOOL.HAWKEYE: {
            'sep':',',
            'title': { 'pat': r'[Key](\d+)', 'st': 1, 'end': None},
            'row': {'name': None, 'st': 1, 'end': None } 
        },
        DV_DATA_TOOL.STUDIO: {
            'sep':',',
            'title': { 'pat': r'[Key](\d+)', 'st': 1, 'end': None},
            'row': {'name': None, 'st': 1, 'end': None } 
        }
    })

    def __init__(self, tool, size: Tuple[int, int]):
        super().__init__()

        self.title = None
        self.channel_size = size
        self.matrix_size = None

        self.FILE_PARSING_FORMAT: Dict[str: dict] = {
            self.DV_DATA_MODE.MU: self.MU_AA_FILE_FORMAT[tool],
            self.DV_DATA_MODE.SC: self.SC_AA_FILE_FORMAT[tool],
            self.DV_DATA_MODE.KEY: self.KEY_FILE_FORMAT[tool]
        }

        self.frames: List[Union[RsdDataMu, RsdDataKey]] = []

    # look the key from given value, return `None` if not found
    @classmethod
    def supported_tool(cls, tool: str):
        return _check_contains(tool, cls.DV_DATA_TOOL.values())

    @classmethod
    def supported_cate(cls, name: str):
        return _check_contains(name, cls.DV_DATA_CATE.values())

    @classmethod
    def supported_mode(cls, name: str):
        return _check_contains(name, cls.DV_DATA_MODE.values())

    def set_maxtrix_size(self, size):
        self.matrix_size = size
        if not self.channel_size:
            self.channel_size = size    #default value for channel size
    
    @property
    def dualx(self):
        if self.filename:
            return self.DV_DATA_SUBMODE.DUALX in os.path.basename(self.filename)

    def _parse_sc(self, cate, mode=DV_DATA_MODE.SC):
        """
            parese each line of rawdata to RsdDataSc structure in frames[]
                the RsdDataSc.data is Series
        """
        patterns = self.FILE_PARSING_FORMAT[self.DV_DATA_MODE.SC]

        for i, line in enumerate(list(self.f)):
            if line.isspace():
                continue

            content = line.split(patterns.sep)
            if content[-1].isspace():
                content.pop()   #remove null tail

            if not self.title:
                channel_list = content[patterns.title.st: patterns.title.end]
                tag = channel_list[0].lower()
                if not (cate in tag and mode in tag):
                    print(f"file content not match {cate} and {mode}: {tag}", os.path.basename(self.filename))
                    return
                
                pat = re.compile(patterns.title.pat)

                mx, my = (0, 0)
                for name in channel_list:
                    channel, submode, cate = name.split('_')
                    t = pat.match(channel)

                    if t and t.lastindex == 2:
                        try:
                            val = int(t[2])
                        except:
                            raise ValueError("Parse string: {} with pat {} return {}".format(name, patterns.title.pat, t))

                        if t[1] in 'xX':
                            if mx < val:
                                mx = val
                        elif t[1] in 'yY':
                            if my < val:
                                my = val
                        else:
                            print("Unknown sefcap name:", name)
                    else:
                        print(f"Channel list parse failed {name}", os.path.basename(self.filename))
                        return

                size = (mx + 1, my + 1)
                self.set_maxtrix_size(size)
                self.title = line
            else:
                try:
                    st = patterns.row.st
                    end = patterns.row.end

                    # Add blocksize check
                    blocksize = self.matrix_size[SENSOR.X] + self.matrix_size[SENSOR.Y]
                    extendsize = blocksize - (len(content) - st)
                    if extendsize > 0:
                        print("Block size:", blocksize, self.matrix_size, " extend ", extendsize, os.path.basename(self.filename))
                        if extendsize > 0:
                            content.extend([-1] * extendsize)
                        else:
                            end = blocksize + st
                    elif extendsize < 0:
                        print("Drop data:", extendsize, os.path.basename(self.filename))

                    # FIXME: Here we only store the sct data, scp data is dropped
                    v = np.array(list(map(int, content[st: end])))
                    d = RsdDataSc(v[:self.matrix_size[SENSOR.X]], v[self.matrix_size[SENSOR.X]: self.matrix_size[SENSOR.Y]] , submode, cate)
                    self.frames.append(d)
                except:
                    print("Invalid data value, maxtrix", self.matrix_size, "content size", len(content))    
    
    def _parse_mu(self, cate, mode=DV_DATA_MODE.MU):
        """
            parese each line of rawdata to RsdDataMu structure in frames[]
                the RsdDataMu.data is Dataframe
        """
        patterns = self.FILE_PARSING_FORMAT[self.DV_DATA_MODE.MU]

        for i, line in enumerate(list(self.f)):
            if line.isspace():
                continue

            content = line.split(patterns.sep)
            if content[-1].isspace():
                content.pop()   #remove null tail

            if not self.title:
                channel_list = content[patterns.title.st: patterns.title.end]
                tag = channel_list[0].lower()
                if not (cate in tag):
                    print(f"file content not match {cate}: {tag}", os.path.basename(self.filename))
                    return

                pat = re.compile(patterns.title.pat)

                mx, my = (0, 0)
                for name in channel_list:
                    t = pat.match(name)
                    if t and t.lastindex == 3:
                        x, y = map(int, t.group(1, 2))
                        if mx < x:
                            mx = x

                        if my < y:
                            my = y
                    else:
                        print(f"Channel list parse failed {name}", os.path.basename(self.filename))
                        return

                size = (mx + 1, my + 1)
                self.set_maxtrix_size(size)
                self.title = line
            else:
                try:
                    dt = datetime.strptime(content[0], patterns.row.name)
                    st = patterns.row.st
                    end = patterns.row.end
                    if st > 1:
                        n = int(content[1])
                    else:
                        n = i

                    # Add blocksize check
                    blocksize = self.matrix_size[SENSOR.X] * self.matrix_size[SENSOR.Y]
                    extendsize = blocksize - (len(content) - st)
                    if extendsize > 0:
                        print("Block size:", blocksize, self.matrix_size, " extend ", extendsize, os.path.basename(self.filename))
                        if extendsize > 0:
                            content.extend([-1] * extendsize)
                        else:
                            end = blocksize + st
                    elif extendsize < 0:
                        print("Drop data:", extendsize, os.path.basename(self.filename))

                    v = np.array(list(map(int, content[st: end]))).reshape(self.matrix_size)
                    d = RsdDataMu(dt, n, self.dualx ,v)
                    self.frames.append(d)
                except:
                    print("Invalid data value, maxtrix", self.matrix_size, "content size", len(content))


    def _parse_key(self, cate, mode=DV_DATA_MODE.KEY):
        """
            parese each line of rawdata to RsdDataMu structure in frames[]
                the RsdDataKey.data is Series
        """
        patterns = self.FILE_PARSING_FORMAT[self.DV_DATA_MODE.KEY]

        for i, line in enumerate(list(self.f)):
            if line.isspace():
                continue

            content = line.split(patterns.sep)
            if content[-1].isspace():
                content.pop()   #remove null tail

            if not self.title:
                channel_list = content[patterns.title.st: patterns.title.end]
                tag = channel_list[0].lower()
                if not (cate in tag and mode in tag):
                    print(f"file content not match {cate} and {mode}: {tag}", os.path.basename(self.filename))
                    return
                
                pat = re.compile(patterns.title.pat)

                max_key = 0
                for name in channel_list:
                    t = pat.match(name)
                    if t and t.lastindex == 1:
                        try:
                            val = int(t[1])
                        except:
                            raise ValueError("Parse string: {} with pat {} return {}".format(name, patterns.title.pat, t))

                        if max_key < val:
                            max_key = val
                    else:
                        print(f"Channel list parse failed {name}", os.path.basename(self.filename))
                        return
                    
                size = ((max_key + 1), 1)
                self.set_maxtrix_size(size)
                self.title = line
            else:
                try:
                    st = patterns.row.st
                    end = patterns.row.end

                    # Add blocksize check
                    blocksize = self.matrix_size[SENSOR.X] * self.matrix_size[SENSOR.Y]
                    extendsize = blocksize - (len(content) - st)
                    if extendsize:
                        print("Block size:", blocksize, self.matrix_size, " extend ", extendsize)
                        if extendsize > 0:
                            content.extend([-1] * extendsize)
                        else:
                            end = blocksize + st
                    elif extendsize < 0:
                        print("Drop data:", extendsize, os.path.basename(self.filename))

                    v = np.array(list(map(int, content[st: end])))
                    d = RsdDataKey(v)
                    self.frames.append(d)
                except:
                    print("Invalid data value, maxtrix", self.matrix_size, "content size", len(content))


    def parse(self, filename: str, cate: ENUM_DV_DATA_CATE, mode: ENUM_DV_DATA_MODE):
        """
            @filename: the taget filename with path
            @cate: the 'delta/ref/signal' types of DV_DATA_CATE
            @mode: the 'sc/mu/key' types of DV_DATA_MODE
            @return: the parsed result (with format "RsdDataSc|RsdDataMu|RsdDataKey") stored in self.frames List
                the RsdDataxx stored information:
                    .maxtrix: the x/y channel configure
                    .np_array/np_data: the raw ndarray data
                    .data: DataFrame|Series
        """
        if filename:
            self.load_file(filename)

        if not self.f:
            return

        # remove all old data
        self.frames = []

        if mode == self.DV_DATA_MODE.SC:
            self._parse_sc(cate)
        elif mode == self.DV_DATA_MODE.MU:
            self._parse_mu(cate)
        elif mode == self.DV_DATA_MODE.KEY:
            self._parse_key(cate)
        else:
            print("Unsupport mode", mode, filename)

        self.close_file()

        return self

    def save_to_file(self, limit, output=None):
        if not len(self.frames):
            return

        if not output:
            output = self.filename + '.xlsx'

        writer = pd.ExcelWriter(output)

        for frame in self.frames:
            if limit:
                inside, low, hight = limit
                v_max = frame.max(self.channel_size)
                v_min = frame.min(self.channel_size)
                if v_max <= hight and v_min >= low:
                    v_inside = True
                else:
                    v_inside = False

                if v_inside != inside:
                    continue

            print("Save page", frame.n)
            frame.data.to_excel(writer, sheet_name=str(frame.n))

        if writer.close:
            writer.close()
        else:
            writer.save()

        print("Save to:", output)

    @staticmethod
    def parse_range(limit_txt):
        if not limit_txt:
            return

        limit_txt = limit_txt.strip()
        pat = re.compile(r'\^?\((-?\d+)[ \t]*,[ \t]*(-?\d+)\)')
        if limit_txt:
            result = pat.match(limit_txt)
            if result:
                try:
                    low, high = result.groups()
                    if limit_txt[0] == '^':
                        inside = False
                    else:
                        inside = True
                    limit = (inside, int(low), int(high))
                    print("Set limit:", limit)
                    return limit
                except:
                    print("Unsupport range parameters: ", limit_txt)
                    return None

    @staticmethod
    def parse_size(size_txt):
        if not size_txt:
            return size_txt

        size_txt = size_txt.strip()
        pat = re.compile(r'\((-?\d+)[ \t]*,[ \t]*(-?\d+)\)')
        if size_txt:
            result = pat.match(size_txt)
            if result:
                try:
                    low, high = result.groups()
                    size = (int(low), int(high))
                    print("Set size:", size)
                    return size
                except:
                    print("Unsupport size parameters: ", size_txt)
                    return None

    def runstat(self, path, mode: str):
        if os.path.exists(path):
            #log_parser = DebugViewLog(format, size)
            self.parse(path, mode)
            #self.save_to_file(limit)
        else:
            print('Un-exist file name \'{:s}\''.format(path))

if __name__ == '__main__':
    # log = DebugViewLog('HAWKEYE')
    # #filename = r"D:\trunk\customers2\BYD\12.8_Qin100_1664s\log\ref2.log"
    # filename = r"D:\trunk\tools\maXStudio control files\Hawkeye_20180125_190246.csv"
    # log.parse(filename)
    # log.save_to_file()

    import os
    import sys
    import argparse

    def runstat(args=None):
        parser = parse_args(args)
        aargs = args if args is not None else sys.argv[1:]
        args = parser.parse_args(aargs)
        print(args)

        if not args.filename and not args.tool:
            parser.print_help()
            return

        tool = args.tool
        if not DebugViewLog.supported_tool(tool):
            print("Unsupported tool", tool)
            return

        size = args.size
        limit = DebugViewLog.parse_range(args.range)
        mode = DebugViewLog.supported_mode(args.mode)

        path = args.filename
        if path:
            if os.path.exists(path):
                log_parser = DebugViewLog(tool, size)
                log_parser.parse(path, mode)
                log_parser.save_to_file(limit)
            else:
                print('Un-exist file name \'{:s}\''.format(path))

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
                            default='mc',
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


    cmd = None
    #cmd = r'-t mxtapp -r ^(19000,28000) -s (30,52) -f D:\trunk\customers2\BYD\BYD_SUV_VP146_1665T_T146A02_Token\log\146_tp_ref.log'.split()
    #cmd = r"-t studio -r ^(-100,100) -f  D:\trunk\customers2\Desay\Desay_DFLZM_SX7_10.1Inch_641T_14804_Goworld\log\Graphical_Debug_Viewer_Log_19_三月_11_22_48.csv".split()
    #cmd = r'-t studio -f D:\log\ref.csv'.split()
    cmd = [
        '-t',
        'mxtapp',
        '-f',
        r'D:\trunk\customers3\Desay\Desay_Toyota_23MM_429D_1296M1_18581_Goworld\log\20240613 production log\429D\NV2414_32_82_S001\tmp\refs_mu_NV2414_32_82_S001.csv'
    ]
    
    runstat(cmd)