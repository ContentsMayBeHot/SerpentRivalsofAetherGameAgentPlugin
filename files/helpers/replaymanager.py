import configparser
import enum
import os
import random
import re
import shutil
import sys
import time


import skimage.exposure
import numpy as np


def main():
    '''Sort replays by version'''
    manager = ReplayManager()
    manager.sort_roas_into_subdatasets()

def version_to_dname(string):
    '''Convert x.x.x to xx_xx_xx'''
    return '_'.join([
        char if len(char) > 1 else '0' + char
        for char in string.split('.') if char.isdigit()
        ])

def dname_to_version(string):
    '''Convert xx_xx_xx to x.x.x'''
    return '.'.join([
        str(int(char))
        for char in string.split('_') if char.isdigit()
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
        end_postreplay = [1, 5, 'Z', 'Z', 2]


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
        # Ensure frames folder exists
        self.frames_apath = os.path.join(self.replays_apath, 'frames')
        if not os.path.isdir(self.frames_apath):
            os.mkdir(self.frames_apath)

    def sort_roas_into_subdatasets(self):
        '''Purpose: Sort .roa files by version into subdatasets
        Pre: None
        Post: Replays sorted into subdatasets
        '''
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

    def load_subdataset(self):
        '''Purpose: Load the subdataset for a particular game version
        Pre: Replays sorted into subdatasets
        Post: Subdataset loaded
        '''
        # Get the name of the subdataset from the configuration file
        version = self.config['RivalsofAether']['GameVersion']
        subdataset_dname = version_to_dname(version)
        self.subdataset_apath = os.path.join(self.replays_apath, subdataset_dname)
        # Load subdataset from its folder
        self.subdataset = [
            dirent for dirent in os.listdir(self.subdataset_apath)
            if dirent.endswith('.roa')
            ]
        # Initialize the unvisited set
        self.subdataset_unvisited = []
        self.subdataset_visited = []
        for roa_fname in self.subdataset:
            if self.__are_frames_collected(roa_fname):
                self.subdataset_visited.append(roa_fname)
            else:
                self.subdataset_unvisited.append(roa_fname)
        print('Loaded subdataset for version "{}"'.format(subdataset_dname))
        print('Subdataset size:', len(self.subdataset))
        print('Unvisited size:', len(self.subdataset_unvisited))

    def next_roa(self):
        '''Purpose: Get a new roa file
        Pre: Subdataset loaded
        Post: Replaces current replay with an unvisited one and marks as visited
        '''
        # Select the next replay file name from the unvisited subdataset
        if not self.subdataset_unvisited:
            return None
        roa_fname = self.subdataset_unvisited[0]
        # Delete any existing replay files from the replays folder
        for dirent in os.listdir(self.replays_apath):
            if dirent.endswith('.roa'):
                dirent_apath = os.path.join(self.replays_apath, dirent)
                os.remove(dirent_apath)
        # Copy the next replay file into the replays folder
        roa_apath = os.path.join(self.subdataset_apath, roa_fname)
        shutil.copy(roa_apath, self.replays_apath)
        # Mark the replay file as visited
        self.subdataset_visited.append(roa_fname)
        self.subdataset_unvisited.remove(roa_fname)
        # Ensure the existence of a frames folder for the replay
        roa_frames_dname = os.path.splitext(roa_fname)[0]
        roa_frames_apath = os.path.join(frames_apath, roa_frames_dname)
        if not os.path.isdir(roa_frames_apath):
            os.mkdir(roa_frames_apath)
        # Update state variables
        self.roa_fname = roa_fname
        self.roa_apath = roa_apath
        self.roa_frames_dname = roa_frames_dname
        self.roa_frames_apath = roa_frames_apath
        # Return absolute path to the replay file
        print('Fetching replay file "{}"'.format(roa_fname))
        return roa_apath

    def save_frame(self, frame, frame_offset):
        '''Purpose: Save a game frame
        Pre: Subdataset loaded
        Post: Saves game frame as NumPy pickle
        '''
        # Write the numpy array to a file in that folder
        fout_fname = str(frame_offset) + '.np'
        fout_apath = os.path.join(self.roa_frames_apath, fout_fname)
        fout_rpath = os.path.join(self.roa_frames_dname, fout_fname)
        result = ''
        with open(fout_apath, 'wb') as fout:
            np.save(fout, frame)
            result = fout_rpath
        return result

    def save_parsed_roa(self, roa_matrix):
        result = []
        for i,roa in enumerate(roa_matrix):
            fout_fname = 'roa_' + str(i) + '.np'
            fout_apath = os.path.join(self.roa_frames_apath, fout_fname)
            fout_rpath = os.path.join(self.roa_frames_dname, fout_fname)
            with open(fout_apath, 'wb') as fout:
                np.save(fout, roa)
                result.append(fout_rpath)
        return result


    def cull_low_contrast(self):
        '''Purpose: Delete all frames with low contrast
        Pre: Subdataset loaded
        Post: Frames with low contrast deleted
        '''
        i = 0
        for roa_fname in self.subdataset_visited:
            # Get a list of all of the frame dump files
            roa_frames = [
                dirent for dirent in os.listdir(self.roa_frames_apath)
                if dirent.endswith('.np')
                ]
            # Sort by frame index in descending order
            roa_frames = sorted(roa_frames,
                                key=lambda x: int(os.path.splitext(x)[0]),
                                reverse=True)
            # Cull until low contrast images are gone
            for frame_fname in roa_frames:
                frame_apath = os.path.join(self.roa_frames_apath, frame_fname)
                frame = np.load(frame_apath)
                if skimage.exposure.is_low_contrast(frame):
                    os.remove(frame_apath)
                    i += 1
                else:
                    break
        print('Deleted {} low contrast frames'.format(i))

    def __are_frames_collected(self):
        '''Check if a frames folder exists for the current roa'''
        # Check if a folder exists in frames
        if os.path.isdir(self.roa_frames_apath):
            if len(os.listdir(self.roa_frames_apath) > 0:
                return True
        return False


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
