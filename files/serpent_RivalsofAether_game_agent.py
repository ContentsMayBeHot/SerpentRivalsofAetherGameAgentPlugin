from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey

from serpent.frame_grabber import FrameGrabber
from serpent.input_controller import KeyboardKey

from .helpers.manager.replaymanager import ReplayManager, PlaybackTimer
from .helpers.parser.roaparser import Replay

import enum
import datetime
import keras
import numpy as np
import os
import time
import sys


class SerpentRivalsofAetherGameAgent(GameAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handlers["COLLECT"] = self.handle_collect
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_setups["COLLECT"] = self.setup_collect
        self.analytics_client = None

    def setup_common(self):
        '''Perform setup for both play and frame collection'''
        self.input_mapping = {
            'Z': KeyboardKey.KEY_Z,
            'X': KeyboardKey.KEY_X,
            'C': KeyboardKey.KEY_C,
            'D': KeyboardKey.KEY_D,
            'S': KeyboardKey.KEY_S,
            'UP': KeyboardKey.KEY_UP,
            'DOWN': KeyboardKey.KEY_DOWN,
            'LEFT': KeyboardKey.KEY_LEFT,
            'RIGHT': KeyboardKey.KEY_RIGHT
        }
        self.key_mapping = {
            KeyboardKey.KEY_Z.name: 'JUMP',
            KeyboardKey.KEY_X.name: 'ATTACK',
            KeyboardKey.KEY_C.name: 'SPECIAL',
            KeyboardKey.KEY_D.name: 'STRONG',
            KeyboardKey.KEY_S.name: 'DODGE',
            KeyboardKey.KEY_UP.name: 'UP',
            KeyboardKey.KEY_DOWN.name: 'DOWN',
            KeyboardKey.KEY_LEFT.name: 'LEFT',
            KeyboardKey.KEY_RIGHT.name: 'RIGHT'
        }

    def setup_play(self):
        '''Perform setup for play'''
        self.setup_common()
        # Turn off CPU feature warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # Load ML model
        model_path = os.path.join('plugins',
                                  'SerpentRivalsofAetherGameAgentPlugin',
                                  'files', 'ml_models', 'rival.h5')
        self.model = keras.models.load_model(model_path)

    def setup_collect(self):
        '''Perform setup for frame collection'''
        self.setup_common()

        self.manager = ReplayManager()
        self.manager.load_subdataset()
        self.playback = PlaybackTimer()

        self.game_state = Game.State.REPLAY_MENU

    def handle_play(self, game_frame):
        '''Frame handler for play mode. To invoke, run:
        serpent play RivalsofAether SerpentRivalsofAetherGameAgent PLAY
        '''
        x = np.array([game_frame.quarter_resolution_frame])
        print(x.shape)
        prediction = self.model.predict(x)
        print(prediction)

    def handle_collect(self, game_frame):
        '''Frame handler for frame collection mode. To invoke, run:
        serpent play RivalsofAether SerpentRivalsofAetherGameAgent COLLECT
        '''
        # State 1: Replay menu
        if self.game_state is Game.State.REPLAY_MENU:
            roa_apath = self.manager.next_roa()
            # Case 1-exception: No replays left
            if not roa_apath:
                total = len(self.manager.subdataset)
                print('^w^ ~ Done collecting for {} replays'.format(total))
                sys.exit()
            self.roa = Replay(roa_apath)
            duration = self.roa.get_duration()
            self.tap_sequence(Game.Sequence.back_and_forth)
            self.tap_sequence(Game.Sequence.start_replay_1)
            self.game_state = Game.State.REPLAY_PLAYBACK
            time.sleep(1)
            self.playback.start(duration)
        # State 2: Playback
        elif self.game_state is Game.State.REPLAY_PLAYBACK:
            # State 2-A: Playback in progress
            if self.playback.is_playing():
                printout = 'owo ~ Watching'
                # Get information about time left
                elapsed = self.playback.seconds_elapsed()
                remaining = self.playback.seconds_remaining()
                printout += '\t{0:.2f}'.format(elapsed)
                printout += ':{0:.2f}'.format(remaining)
                # Get frame data and frame offset for saving
                #timestamp = game_frame.timestamp
                #time_offset = self.playback.seconds_elapsed_since(timestamp)
                frame_offset = int(elapsed * 60)
                frame = game_frame.quarter_resolution_frame
                frame_rpath = self.manager.save_frame(frame, frame_offset)
                printout += '\tWrote: {}'.format(frame_rpath)
                print(printout)
            # State 2-B: Playback end
            else:
                printout = 'uwu ~ Finished watching'
                visited = len(self.manager.subdataset_visited)
                unvisited = len(self.manager.subdataset_unvisited)
                printout += '\t{}:{}'.format(visited, unvisited)
                print(printout)
                roa_matrices = [
                    p.collapse_actions() for p in self.roa.players
                    ]
                rpaths = self.manager.save_labels(roa_matrices)
                for rp in rpaths:
                    print('Wrote: {}'.format(rp))
                self.tap_sequence(Game.Sequence.end_postreplay)
                self.game_state = Game.State.REPLAY_MENU

    def tap_sequence(self, sequence, delay_override=None):
        '''Pass input sequence to the input controller'''
        # Must contain delay value and one input/wait token
        if len(sequence) < 2:
            return

        # Retrieve delay value
        delay = sequence[0]
        if delay_override:
            delay = delay_override

        # Read the rest of the sequence
        for token in sequence[1:]:
            # Check if token is a key name
            if isinstance(token, str):
                mapped_input = self.input_mapping[token.upper()]
                self.input_controller.tap_key(mapped_input)
                time.sleep(delay)
            # Check if token is a number
            elif isinstance(token, int):
                token = int(token)
                if token > 0:
                    time.sleep(token)


class Game:
    FRAMES_PER_SECOND = 60.0
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
        end_postreplay = [2, 6, 'Z', 'Z', 3]
