import configparser
import enum
import os
import random
import re
import shutil
import sys
import time


import numpy as np


def main():
    '''Sort replays by version'''
    manager = ReplayManager()
    manager.sort_roas_into_subdatasets()

def version_to_dname(string):
    '''Convert x.x.x to xx_xx_xx'''
    return '_'.join([
        x if len(x) > 1 else '0' + x
        for x in string.split('.') if x.isdigit()
        ])

def dname_to_version(string):
    '''Convert xx_xx_xx to x.x.x'''
    return '.'.join([
        str(int(x))
        for x in string.split('_') if x.isdigit()
        ])


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

    class Sequence:
        '''The first element is the default delay between key presses, and all
        subsequent elements are either key names or timed delays.'''
        splash_to_main = [1.5, 'Z', 'X', 'Z', 'Z', 'Z', 'Z']
        main_to_replay = [0.5, 'DOWN', 'DOWN', 'DOWN', 'Z', 1, 'Z']
        start_replay_1 = [1, 'Z', 'Z', 0]
        back_and_forth = [1, 'X', 'Z']
        end_postreplay = [1, 'Z', 'Z']


class ReplayManager:
    def __init__(self):
        '''Sets up a new replay manager'''
        # Open the configuration file
        self.config = configparser.ConfigParser()
        config_dname = os.path.abspath(os.path.dirname(__file__))
        config_apath = os.path.join(config_dname, 'roa.ini')
        self.config.read(config_apath)
        # Establish path to replays folder
        self.replays_apath = self.config['RivalsofAether']['PathToReplays']
        # Initialize subdataset values to 'None'
        self.__flush_subdataset()
        # Ensure frames folder exists
        self.frames_apath = os.path.join(self.replays_apath, 'frames')
        if not os.path.isdir(self.frames_apath):
            os.mkdir(self.frames_apath)

    def sort_roas_into_subdatasets(self):
        '''Sort .roa files by version into subdatasets'''
        print('Establishing subdatasets:')
        for dirent in os.listdir(self.replays_apath):
            if not dirent.endswith('.roa'):
                continue
            dirent_apath = os.path.join(self.replays_apath, dirent)
            # Open the .roa file
            with open(dirent_apath) as fin:
                # Get the version string
                ln = fin.readline()
                version = '{}_{}_{}'.format(str(ln[1:3]),
                                            str(ln[3:5]),
                                            str(ln[5:7]))
                # Ensure subdataset folder exists
                subdataset_apath = os.path.join(self.replays_apath, version)
                if not os.path.exists(subdataset_apath):
                    os.mkdir(subdataset_apath)
                # Move the replay to the new directory
                new_dirent_apath = os.path.join(subdataset_apath, dirent)
                os.rename(dirent_apath, new_dirent_apath)
                print('Sorted "{}" into "{}"'.format(dirent, version))

    def load_subdataset(self, subdataset_dname=None):
        '''Load the subdataset for a particular game version'''
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
        self.subdataset_unvisited = list(self.subdataset) # Ensure it's a copy
        self.subdataset_visited = []
        print('Loaded subdataset for version "{}"'.format(subdataset_dname))
        print('Subdataset size:', len(self.subdataset))

    def get_current_version(self, as_dname=False):
        '''Get the current game version as specified in the config file'''
        # Get in this format: x.y.z
        version = self.config['RivalsofAether']['GameVersion']
        if as_dname:
            # Convert to the following format: xx_yy_zz (with leading 0s)
            version = version_to_dname(version)
        return version

    def get_existing_subdatasets(self):
        '''Get a list of subdatasets, where each represents a game version'''
        p = SUBDATASET_PATTERN
        return [ x for x in os.listdir(self.replays_apath) if p.match(x) ]

    def next_roa(self, apath=False):
        '''Get a new roa file.'''
        # Get the next .roa from the unvisited list
        if not self.subdataset_unvisited:
            return None
        roa_fname = self.subdataset_unvisited[0]

        # Replace current .roas with the next .roa and mark latter as visited
        self.__flush_replays()
        self.__transfer_roa(roa_fname)
        self.__visit_roa(roa_fname)

        print('Fetching replay file "{}"'.format(roa_fname))
        if apath:
            return os.path.join(self.replays_apath, roa_fname)
        return roa_fname

    def save_frame(self, frame, frame_offset, return_apath=False):
        # Get the name of the current replay file
        roa_fname = self.__detect_roa()
        if not roa_fname:
            return

        # Ensure existence of folder for this replay file's frames
        roa_frames_dname = os.path.splitext(roa_fname)[0]
        roa_frames_apath = os.path.join(self.frames_apath, roa_frames_dname)
        if not os.path.isdir(roa_frames_apath):
            os.mkdir(roa_frames_apath)

        # Write the numpy array to a file in that folder
        fout_fname = str(frame_offset) + '.np'
        fout_apath = os.path.join(roa_frames_apath, fout_fname)
        result = ''
        with open(fout_apath, 'wb') as fout:
            np.save(fout, frame)
            fout_rpath = os.path.join(roa_frames_dname, fout_fname)
            result = fout_rpath
        return result

    def __detect_roa(self):
        return [
            x for x in os.listdir(self.replays_apath) if x.endswith('.roa')
            ][0]

    def __flush_subdataset(self):
        '''Reset subdataset to starting values'''
        self.subdataset_apath = None
        self.subdataset = None
        self.subdataset_unvisited = None
        self.subdataset_visited = None

    def __flush_replays(self):
        ''' Remove all roa files from the replays folder'''
        # Get contents of replays folder
        for dirent in os.listdir(self.replays_apath):
            # Check for .roa extension
            if dirent.endswith('.roa'):
                # Delete the .roa file
                dirent_apath = os.path.join(self.replays_apath, dirent)
                os.remove(dirent_apath)

    def __transfer_roa(self, roa_fname):
        '''Copy specified roa file from subdataset folder to replays folder'''
        # Copy the specified replay file to the game's replays folder
        roa_apath = os.path.join(self.subdataset_apath, roa_fname)
        shutil.copy(roa_apath, self.replays_apath)

    def __visit_roa(self, roa_fname):
        '''Mark specified roa file as visited'''
        self.subdataset_visited.append(roa_fname)
        self.subdataset_unvisited.remove(roa_fname)


class PlaybackTimer:
    def start(self, duration):
        '''Hit the clock'''
        self.start_time = time.time()
        self.duration = duration
        self.end_time = self.start_time +  duration

    def is_playing(self):
        '''Returns true if the replay is still playing'''
        return time.time() < self.end_time

    def seconds_elapsed(self):
        '''Returns the number of seconds elapsed now'''
        return self.seconds_elapsed_since(time.time())

    def seconds_elapsed_since(self, timestamp):
        '''Returns the number of seconds elapsed for a particular time'''
        return timestamp - self.start_time

    def seconds_remaining(self):
        '''Returns the number of seconds remaining now'''
        return self.seconds_remaining_after(time.time())

    def seconds_remaining_after(self, timestamp):
        '''Returns the number of seconds remaining for a particular time'''
        return self.end_time - timestamp


if __name__ == '__main__':
    main()
