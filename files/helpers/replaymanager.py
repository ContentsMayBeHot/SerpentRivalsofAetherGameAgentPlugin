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
        start_replay_1 = [1, 'Z', 'Z']
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
        print('-Set path to replays folder:', self.replays_apath)
        # Initialize subdataset values to 'None'
        self.__flush_subdataset()
        # Ensure frames folder exists
        self.frames_apath = os.path.join(self.replays_apath, 'frames')
        if not os.path.isdir(self.frames_apath):
            os.mkdir(self.frames_apath)

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
        print('-Loaded subdataset into memory:', subdataset_dname)
        print('--Subdataset size:', len(self.subdataset))
        self.subdataset_unvisited = list(self.subdataset) # Ensure it's a copy
        self.subdataset_visited = []

    def get_current_version(self, as_dname=False):
        '''Get the current game version as specified in the config file'''
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
        '''Get a list of subdatasets, where each represents a game version'''
        p = SUBDATASET_PATTERN
        return [ x for x in os.listdir(self.replays_apath) if p.match(x) ]

    def next_roa(self, apath=False):
        '''Get a new roa file.'''
        print('-Retrieving replay file from unvisited subdataset')
        # Get the next .roa from the unvisited list
        if not self.subdataset_unvisited:
            return None
        roa_fname = self.subdataset_unvisited[0]

        # Replace current .roas with the next .roa and mark latter as visited
        self.__flush_replays()
        self.__transfer_roa(roa_fname)
        self.__visit_roa(roa_fname)

        print('-Retrieved replay file:', roa_fname)
        if apath:
            return os.path.join(self.replays_apath, roa_fname)
        return roa_fname

    def save_frame(self, frame, identifier):
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
        fout_fname = str(int(identifier)) + '.np'
        fout_apath = os.path.join(roa_frames_apath, fout_fname)
        print('--Saving frame:', fout_apath)
        with open(fout_apath, 'wb') as fout:
            np.save(fout, frame)

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
        print('--Flushed subdataset')

    def __flush_replays(self):
        ''' Remove all roa files from the replays folder'''
        # Get contents of replays folder
        for dirent in os.listdir(self.replays_apath):
            # Check for .roa extension
            if dirent.endswith('.roa'):
                # Delete the .roa file
                dirent_apath = os.path.join(self.replays_apath, dirent)
                os.remove(dirent_apath)
        print('--Flushed replays folder')

    def __transfer_roa(self, roa_fname):
        '''Copy specified roa file from subdataset folder to replays folder'''
        # Copy the specified replay file to the game's replays folder
        roa_apath = os.path.join(self.subdataset_apath, roa_fname)
        shutil.copy(roa_apath, self.replays_apath)
        print('--Transferred', roa_fname, 'to replays folder')

    def __visit_roa(self, roa_fname):
        '''Mark specified roa file as visited'''
        self.subdataset_visited.append(roa_fname)
        self.subdataset_unvisited.remove(roa_fname)
        print('--Marked', roa_fname, 'as visited.')


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
        '''Returns the number of seconds elapsed since playback began'''
        return self.seconds_elapsed_since(time.time())

    def seconds_elapsed_since(self, timestamp):
        return timestamp - self.start_time

    def seconds_remaining(self):
        '''Returns the number of seconds remaining until the end of playback'''
        return self.seconds_remaining_after(time.time())

    def seconds_remaining_after(self, timestamp):
        return self.end_time - timestamp


if __name__ == '__main__':
    main()
