import configparser
import enum
import os
import random
import re
import shutil
import sys


def main():
    '''Sort replays by version'''
    manager = ReplayManager()
    manager.sort_roas_into_subdatasets()


class Game:
    FRAMES_PER_SECOND = 60.0
    SUBDATASET_PATTERN = re.compile('[0-9]{2}_[0-9]{2}_[0-9]{2}')

    class State(enum.Enum):
        SPLASH_SCREEN = enum.auto()
        MAIN_MENU = enum.auto()
        REPLAY_MENU = enum.auto()
        REPLAY_PLAYBACK = enum.auto()
        CHARACTER_SELECTION = enum.auto()
        STAGE_SELECTION = enum.auto()
        GAMEPLAY = enum.auto()

    class Stage(enum.Enum):
        TREETOP_LODGE = 1
        FIRE_CAPITOL = 2
        AIR_ARMADA = 3
        ROCK_WALL = 4
        MERCHANT_PORT = 5
        BLAZING_HIDEOUT = 7
        TOWER_OF_HEAVEN = 8

    class StageMode(enum.Enum):
        BASIC = 0
        AETHER = 1

    class Character(enum.Enum):
        ZETTERBURN = 2
        ORCANE = 3
        WRASTOR = 4
        KRAGG = 5
        FORSBURN = 6
        MAYPUL = 7
        ABSA = 8
        ETALUS = 9
        ORI = 10
        RANNO = 11
        CLAIREN = 12

    class Sequence:
        splash_to_main = [1.5, 'Z', 'X', 'Z', 'Z', 'Z', 'Z']
        main_to_replay = [0.5, 'DOWN', 'DOWN', 'DOWN', 'Z', 1, 'Z']
        start_replay_1 = [1, 'Z', 'Z']
        back_and_forth = [1, 'X', 'Z']


class ReplayManager:
    def __init__(self):
        # Open the configuration file
        self.config = configparser.ConfigParser()
        config.read('roa.ini')
        # Establish path to replays folder
        self.replays_apath = config['RivalsofAether']['PathToReplays']
        print('-Set path to replays folder:', self.replays_apath)
        # Initialize subdataset values to 'None'
        self.__flush_subdataset()

    def sort_roas_into_subdatasets(self):
        '''Sort .roa files by version into subdatasets'''
        print('-Sorting replays by game version')
        for dirent in os.listdir(self.replays_apath):
            if not dirent.endswith('.roa'):
                continue
            print('--Found replay file:', dirent)
            dirent_apath = os.path.join(self.replays_apath, roa_fname)
            # Open the .roa file
            with open(dirent_apath) as fin:
                # Get the version string
                ln = fin.readline()
                version = '{}_{}_{}'.format(str(ln[1:3]),
                                            str(ln[3:5]),
                                            str(ln[5:7]))
                print('---Version string:', version)
                # Ensure subdataset folder exists
                subdataset_apath = os.path.join(self.replays_apath, version)
                if not os.path.exists(subdataset_apath):
                    os.mkdir(subdataset_apath)
                    print('---Created new folder for subdataset')
                # Move the replay to the new directory
                new_dirent_apath = os.path.join(subdataset_apath, dirent)
                os.rename(dirent_apath, new_dirent_apath)
                print('---Moved replay file')

    def load_subdataset(self, subdataset_dname=None):
        '''Load the subdataset for a particular game version.'''
        # If no folder provided, choose the one for version specified in config
        if not subdataset_dname:
            subdataset_dname = self.get_current_version(as_dname=True)
        self.subdataset_apath = os.path.join(self.replays_apath,
                                               subdataset_dname)
        # Load subdataset from folder
        self.subdataset = [
            dirent for dirent in os.listdir(self.subdataset_apath)
            if dirent.endswith('.roa')
            ]
        print('-Loaded subdataset into memory:', subdataset_dname)
        print('--Subdataset size:', len(self.subdataset))
        self.subdataset_unvisited = list(self.subdataset) # Ensure it's a copy
        self.subdataset_visited = []

    def get_current_version(self, as_dname=False):
        '''Get the current game version as specified in the config file.'''
        # Get in this format: x.y.z
        version = self.config['RivalsofAether']['GameVersion']
        if as_dname:
            # Convert to the following format: xx_yy_zz (with leading 0s)
            version = '_'.join([
                x if len(x) > 1 else '0' + x
                for x in version.split('.') if x.isdigit()
                ])
        return version

    def get_existing_subdatasets(self):
        '''Get a list of subdatasets, where each represents a game version.'''
        p = SUBDATASET_PATTERN
        return [ x for x in os.listdir(self.replays_apath) if p.match(x) ]

    def next_roa(self, apath=False):
        '''Get a new .roa file.'''
        print('-Retrieving replay file from unvisited subdataset')
        # Get the next .roa from the unvisited list
        if not self.dataset_unvisited:
            return None
        roa_fname = self.subdataset_unvisited[0]

        # Replace current .roas with the next .roa and mark latter as visited
        self.__flush_replays()
        self.__transfer_roa(roa_fname)
        self.__visit(roa_fname)

        if apath:
            return os.path.join(self.subdataset_apath, roa_fname)
        print('-Retrieved replay file:', roa_fname)
        return roa_fname

    def __flush_subdataset(self):
        '''Reset subdataset to starting values.'''
        self.subdataset_apath = None
        self.subdataset = None
        self.subdataset_unvisited = None
        self.subdataset_visited = None
        print('--Flushed subdataset')

    def __flush_replays(self):
        ''' Remove all .roa files from the replays folder.'''
        # Get contents of replays folder
        for dirent in os.listdir(self.replays_apath):
            # Check for .roa extension
            if dirent.endswith('.roa'):
                # Delete the .roa file
                dirent_apath = os.path.join(self.replays_apath, dirent)
                os.remove(dirent_apath)
        print('--Flushed replays folder')

    def __transfer_roa(self, roa_fname):
        '''Copy specified .roa file from subdataset folder to replays folder.'''
        # Copy the specified replay file to the game's replays folder
        roa_apath = os.path.join(self.subdataset_apath, roa_fname)
        shutil.copy(roa_apath, self.replays_apath)
        print('--Transferred,' roa_fname, 'to replays folder')

    def __visit_roa(self, roa_fname):
        '''Mark specified .roa file as visited.'''
        self.dataset_visited.append(roa_fname)
        self.dataset_unvisited.remove(roa_fname)
        print('--Marked', roa_fname, 'as visited.'


if __name__ == '__main__':
    main()
