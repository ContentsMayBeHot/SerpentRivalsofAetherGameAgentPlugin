from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey

from serpent.frame_grabber import FrameGrabber
from serpent.input_controller import KeyboardKey

from .helpers.manager.replaymanager import ReplayManager, PlaybackTimer
from .helpers.parser.roaparser import Replay

import enum
import datetime
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import signal
import sys
import time
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, GlobalAveragePooling2D, AveragePooling3D, Input, LSTM, TimeDistributed  # noqa
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization


CLASSES = 9
IMG_U = 135
IMG_V = 240
IMG_C = 1
CLIP_LENGTH = 1
VISION_INPUT_SHAPE = (1, CLIP_LENGTH, IMG_U, IMG_V, IMG_C)
ACTIONS_INPUT_SHAPE = (1, CLIP_LENGTH, CLASSES)
OUTPUT_SHAPE = (1, CLIP_LENGTH, CLASSES)

FILTERS = 10
POOL_SIZE = (1, 135, 240)
KERNEL_SIZE = (3, 3)

LSTM_UNITS = 32

DEEP_UNITS = 64


def model_ConvLSTM2D():
    # Define ConvLSTM2D model
    model = Sequential()
    model.add(ConvLSTM2D(
            filters=FILTERS,
            kernel_size=KERNEL_SIZE,
            batch_input_shape=VISION_INPUT_SHAPE,
            data_format='channels_last',
            padding='same',
            return_sequences=True,
            stateful=True
    ))  # noqa
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(
            filters=FILTERS,
            kernel_size=KERNEL_SIZE,
            data_format='channels_last',
            padding='same',
            return_sequences=True,
            stateful=True
    ))  # noqa
    model.add(BatchNormalization())
    model.add(AveragePooling3D(POOL_SIZE))
    model.add(Reshape((-1, FILTERS)))
    model.add(Dense(CLASSES, activation='sigmoid'))
    model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy']
    )  # noqa
    return model


def model_functional():
    # Primary
    vision_input = Input(
            batch_shape=VISION_INPUT_SHAPE,
            name='vision_input')
    vision_x = ConvLSTM2D(
            filters=FILTERS,
            kernel_size=KERNEL_SIZE,
            batch_input_shape=VISION_INPUT_SHAPE,
            data_format='channels_last',
            padding='same',
            return_sequences=True,
            stateful=True)(vision_input)
    vision_x = BatchNormalization()(vision_x)
    vision_x = ConvLSTM2D(
            filters=FILTERS,
            kernel_size=KERNEL_SIZE,
            data_format='channels_last',
            padding='same',
            return_sequences=True,
            stateful=True)(vision_x)
    vision_x = BatchNormalization()(vision_x)
    vision_x = AveragePooling3D(pool_size=POOL_SIZE)(vision_x)
    vision_out = Reshape(target_shape=(-1, FILTERS))(vision_x)

    # Auxiliary
    actions_input = Input(
        batch_shape=ACTIONS_INPUT_SHAPE,
        name='actions_input'
    )
    actions_out = LSTM(
        units=LSTM_UNITS,
        return_sequences=True,
        stateful=True)(actions_input)

    # Combined primary and auxiliary
    x = keras.layers.concatenate([vision_out, actions_out])
    x = Dense(units=DEEP_UNITS, activation='relu')(x)
    x = Dense( units=DEEP_UNITS, activation='relu')(x)
    x = Dense( units=DEEP_UNITS, activation='relu')(x)
    main_output = Dense(
        units=CLASSES,
        activation='sigmoid',
        name='main_output')(x)

    # Finished model
    model = Model(
            inputs=[vision_input, actions_input],
            outputs=[main_output])
    model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy'])
    return model


class Classes(enum.Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    ATTACK = 4
    SPECIAL = 5
    JUMP = 6
    DODGE = 7
    STRONG = 8


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
            'JUMP': KeyboardKey.KEY_Z,
            'X': KeyboardKey.KEY_X,
            'ATTACK': KeyboardKey.KEY_X,
            'C': KeyboardKey.KEY_C,
            'SPECIAL': KeyboardKey.KEY_C,
            'D': KeyboardKey.KEY_D,
            'STRONG': KeyboardKey.KEY_D,
            'S': KeyboardKey.KEY_S,
            'DODGE': KeyboardKey.KEY_S,
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
        self.model = model_functional()
        self.model.summary()
        weights_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  'ml_models',
                                  'rival2-w.h5')
        print('Loading weights at', weights_path)
        self.model.load_weights(weights_path)
        self.predictions = []
        self.thresholds = {
            Classes.LEFT.value: 0.6,
            Classes.RIGHT.value: 0.6,
            Classes.UP.value: 0.5,
            Classes.DOWN.value: 0.4,
            Classes.ATTACK.value: 0.3,
            Classes.SPECIAL.value: 0.2,
            Classes.JUMP.value: 0.5,
            Classes.DODGE.value: 0.2,
            Classes.STRONG.value: 0.1
        }
        self.y1 = np.zeros(9,)
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signal, frame):
        print('CTRL+C detected')
        cols = [ 'L', 'R', 'U', 'D', 'ATK', 'SPC', 'JMP', 'DGD', 'STR' ]
        df = pd.DataFrame(data=self.predictions, columns=cols)
        results_path = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                'ml_models',
                'results.csv')
        df.to_csv(results_path)
        print('Saved predictions to', results_path)
        sys.exit(0)

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
        # https://stackoverflow.com/a/12201744
        x = np.dot(x[...,:3], [0.299, 0.587, 0.114]).reshape(1, 1, 135, 240, 1)

        # Make prediction
        y = self.model.predict([x, self.y1.reshape((1, 1, 9))])
        self.y1 = y
        y = y.tolist()[0][0]  # [BATCH [TIMESTEP [ACTIONS ...]]]
        self.predictions.append(y)
        # Display labels
        actnames = [ 'L', 'R', 'U', 'D', 'ATK', 'SPC', 'JMP', 'DGD', 'STR' ]
        printout = ''
        for a,v in zip(actnames, y):
            printout += a + ': {0:.3f}'.format(v) + '\t'
        print(printout)

        # Move left/right
        left = Classes.LEFT.value
        right = Classes.RIGHT.value
        if (y[left] >= self.thresholds[left]
        and y[left] > y[right]):
            self.input_controller.press_key(self.input_mapping['LEFT'])
            self.input_controller.release_key(self.input_mapping['RIGHT'])
        elif (y[right] >= self.thresholds[right]
        and y[right] > y[left]):
            self.input_controller.press_key(self.input_mapping['RIGHT'])
            self.input_controller.release_key(self.input_mapping['LEFT'])
        else:
            self.input_controller.release_key(self.input_mapping['LEFT'])
            self.input_controller.release_key(self.input_mapping['RIGHT'])

        # Move up/down
        up = Classes.UP.value
        down = Classes.DOWN.value
        if (y[up] >= self.thresholds[up]
            and y[up] > y[down]):
            self.input_controller.press_key(self.input_mapping['UP'])
            self.input_controller.release_key(self.input_mapping['DOWN'])
        elif (y[down] >= self.thresholds[down]
            and y[down] > y[up]):
            self.input_controller.press_key(self.input_mapping['DOWN'])
            self.input_controller.release_key(self.input_mapping['UP'])
        else:
            self.input_controller.release_key(self.input_mapping['UP'])
            self.input_controller.release_key(self.input_mapping['DOWN'])

        # Attack
        attack = Classes.ATTACK.value
        if y[attack] >= self.thresholds[attack]:
            self.input_controller.press_key(self.input_mapping['ATTACK'])
        else:
            self.input_controller.release_key(self.input_mapping['ATTACK'])

        # Special attack
        special = Classes.SPECIAL.value
        if y[special] >= self.thresholds[special]:
            self.input_controller.press_key(self.input_mapping['SPECIAL'])
        else:
            self.input_controller.release_key(self.input_mapping['SPECIAL'])

        # Jump
        jump = Classes.JUMP.value
        if y[jump] >= self.thresholds[jump]:
            self.input_controller.press_key(self.input_mapping['JUMP'])
        else:
            self.input_controller.release_key(self.input_mapping['JUMP'])

        # Dodge
        dodge = Classes.DODGE.value
        if y[dodge] >= self.thresholds[dodge]:
            self.input_controller.press_key(self.input_mapping['DODGE'])
        else:
            self.input_controller.release_key(self.input_mapping['DODGE'])

        # Strong attack
        strong = Classes.STRONG.value
        if y[strong] >= self.thresholds[strong]:
            self.input_controller.press_key(self.input_mapping['STRONG'])
        else:
            self.input_controller.release_key(self.input_mapping['STRONG'])

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
