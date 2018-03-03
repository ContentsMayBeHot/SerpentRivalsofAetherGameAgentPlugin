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


#TODO: Rename subdataset to version_subset
#TODO: Create subset class


def main():
    args = sys.argv
    if len(args) != 2:
        print_help()
        sys.exit()
    cmd = args[1].lower()

    manager = ReplayManager()
    if cmd == 'sort':
        manager.sort_roas_into_subdatasets()
    elif cmd == 'sample':
        manager.make_ml_sets()
    elif cmd == 'test-sample':
        manager.make_random_test_sample()
    else:
        print_help()

def print_help():
    print('Rivals of Aether Replay Manager:')
    print(' SORT        - Sort replay files by game version')
    print(' SAMPLE      - Create training and testing sets')
    print(' TEST-SAMPLE - Create random sample for validation')

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
    ROA_PATTERN = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{12}')

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
        # Establish data paths
        self.replays_apath = self.config['RivalsofAether']['PathToReplays']
        self.frames_apath = os.path.join(self.replays_apath, 'frames')
        self.sets_path = os.path.join(self.replays_apath, 'sets')
        self.labels_apath = os.path.join(self.replays_apath, 'labels')
        # Ensure data paths exist
        self.__ensure_directory_exists__(self.frames_apath)
        self.__ensure_directory_exists__(self.labels_apath)
        self.__ensure_directory_exists__(self.sets_path)

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

    def make_random_test_sample(self, sample_size=10):
        # Ensure directory for random sample
        random_set_apath = os.path.join(self.sets_path, 'random')
        self.__ensure_directory_exists__(random_set_apath)
        # Create random subset
        dataset = [
            dirent for dirent in os.listdir(self.frames_apath)
            if os.path.isdir(os.path.join(self.frames_apath, dirent))
            ]
        random_batch = np.random.choice(dataset, sample_size, replace=False)
        self.__copy_batch_into_set__(random_batch, random_set_apath)

    def make_ml_sets(self):
        # Ensure directories for training and testing sets
        training_set_apath = os.path.join(self.sets_path, 'training')
        testing_set_apath = os.path.join(self.sets_path, 'testing')
        self.__ensure_directory_exists__(training_set_apath)
        self.__ensure_directory_exists__(testing_set_apath)
        # Establish training and testing sets with probability distribution
        dataset = [
            dirent for dirent in os.listdir(self.frames_apath)
            if os.path.isdir(os.path.join(self.frames_apath, dirent))
            ]
        training_set = []
        testing_set = []
        random = np.random.choice([True, False], len(dataset), p=[0.80, 0.20])
        for x,r in zip(dataset, random):
            if r:
                training_set.append(r)
            else:
                testing_set.append(r)
        self.__copy_batch_into_set__(training_set, training_set_apath)
        self.__copy_batch_into_set__(testing_set, testing_set_apath)


    def __ensure_directory_exists__(self, apath):
        if not os.path.isdir(apath):
            os.mkdir(apath)

    def __copy_batch_into_set__(self, batch, dst_root):
        n = len(batch)
        for i,roa_dname in enumerate(batch):
            print('Copying', roa_dname, 'into set [{}/{}]'.format(i+1,n))
            # Establish paths
            frames_src = os.path.join(self.frames_apath, roa_dname)
            frames_dst = os.path.join(dst_root, 'frames', roa_dname)
            labels_src = os.path.join(self.labels_apath, roa_dname)
            labels_dst = os.path.join(dst_root, 'labels', roa_dname)
            if os.path.isdir(frames_dst) or os.path.isdir(labels_dst):
                continue
            # Copy
            shutil.copytree(frames_src, frames_dst, symlinks=False, ignore=None)
            shutil.copytree(labels_src, labels_dst, symlinks=False, ignore=None)

    def load_subdataset(self):
        '''Purpose: Load the subdataset for a particular game version
        Pre: Replays sorted into subdatasets
        Post: Subdataset loaded
        '''
        # Get the name of the subdataset from the configuration file
        version = self.config['RivalsofAether']['GameVersion']
        version_dname = version_to_dname(version)
        self.subdataset_apath = os.path.join(self.replays_apath, version_dname)
        # Load subdataset from its folder
        self.subdataset = [
            dirent for dirent in os.listdir(self.subdataset_apath)
            if dirent.endswith('.roa')
            ]
        # Initialize the unvisited set
        self.subdataset_unvisited = []
        self.subdataset_visited = []
        for roa_fname in self.subdataset:
            if self.__is_collected__(roa_fname):
                self.subdataset_visited.append(roa_fname)
            else:
                self.subdataset_unvisited.append(roa_fname)
        print('Loaded subdataset for version "{}"'.format(version_dname))
        print('Subdataset size:', len(self.subdataset))
        print('Unvisited size:', len(self.subdataset_unvisited))

    def next_roa(self):
        '''Purpose: Get a new roa file
        Pre: Subdataset loaded
        Post: Replaces current replay with an unvisited one
        '''
        # Select the next replay file name from the unvisited subdataset
        if not self.subdataset_unvisited:
            return None
        self.roa_fname = self.subdataset_unvisited[0]
        # Delete any existing replay files from the replays folder
        for dirent in os.listdir(self.replays_apath):
            if dirent.endswith('.roa'):
                dirent_apath = os.path.join(self.replays_apath, dirent)
                os.remove(dirent_apath)
        # Copy this replay file into the replays folder
        self.roa_apath = os.path.join(self.subdataset_apath, self.roa_fname)
        shutil.copy(self.roa_apath, self.replays_apath)
        # Mark this replay file as visited
        self.subdataset_visited.append(self.roa_fname)
        self.subdataset_unvisited.remove(self.roa_fname)
        # Ensure the existence of a frames and labels folders for this replay
        self.roa_dname = os.path.splitext(self.roa_fname)[0]
        self.roa_frames_apath = os.path.join(self.frames_apath, self.roa_dname)
        self.__ensure_directory_exists__(self.roa_frames_apath)
        self.roa_labels_apath = os.path.join(self.labels_apath, self.roa_dname)
        self.__ensure_directory_exists__(self.roa_labels_apath)
        # Return absolute path to this replay file
        print('Fetching replay file "{}"'.format(self.roa_fname))
        return self.roa_apath

    def save_frame(self, frame, frame_offset):
        '''Purpose: Save a game frame
        Pre: Subdataset loaded
        Post: Saves game frame as NumPy pickle
        '''
        # Write the numpy array to a file in that folder
        fout_fname = str(frame_offset) + '.np'
        fout_apath = os.path.join(self.roa_frames_apath, fout_fname)
        fout_rpath = os.path.join(self.roa_dname, fout_fname)
        result = ''
        with open(fout_apath, 'wb') as fout:
            np.save(fout, frame)
            result = fout_rpath
        return result

    def save_labels(self, roa_matrices):
        '''Purpose: Save a set of labels
        Pre: Subdataset loaded
        Post: Saves a label as a NumPy pickle
        '''
        result = []
        for i,roa_matrix in enumerate(roa_matrices):
            fout_fname = 'roa_' + str(i) + '.np'
            fout_apath = os.path.join(self.roa_labels_apath, fout_fname)
            fout_rpath = os.path.join(self.roa_dname, fout_fname)
            with open(fout_apath, 'wb') as fout:
                np.save(fout, np.array(roa_matrix, dtype=object))
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

    def __is_collected__(self, roa_fname):
        '''Check if frames and labels exists for the current roa'''
        roa_dname = os.path.splitext(roa_fname)[0]
        frames = False
        roa_frames_apath = os.path.join(self.frames_apath, roa_dname)
        if os.path.isdir(roa_frames_apath):
            if os.listdir(roa_frames_apath):
                frames = True
        labels = False
        roa_labels_apath = os.path.join(self.labels_apath, roa_dname)
        if os.path.isdir(roa_labels_apath):
            if os.listdir(roa_labels_apath):
                labels = True
        return frames and labels


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
