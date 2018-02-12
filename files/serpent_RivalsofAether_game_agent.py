from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey

from serpent.frame_grabber import FrameGrabber
from serpent.input_controller import KeyboardKey

from .helpers.replaymanager import ReplayManager, PlaybackTimer, Game
from .helpers.parser.replayparser import Replay, Character, Action

import datetime
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
        # TODO Additional setup

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
        pass # TODO this

    def handle_collect(self, game_frame):
        '''Frame handler for frame collection mode. To invoke, run:
        serpent play RivalsofAether SerpentRivalsofAetherGameAgent COLLECT
        '''
        # State 1: Replay menu
        if self.game_state is Game.State.REPLAY_MENU:
            roa_apath = self.manager.next_roa(apath=True)
            self.roa = Replay(roa_apath)

            self.tap_sequence(Game.Sequence.back_and_forth)

            self.playback.start(self.roa.get_duration())
            
            self.tap_sequence(Game.Sequence.start_replay_1)

            self.game_state = Game.State.REPLAY_PLAYBACK
        # State 2: Playback
        elif self.game_state is Game.State.REPLAY_PLAYBACK:
            # State 2-A: Playback in progress
            if self.playback.is_playing():
                elapsed = self.playback.seconds_elapsed()
                remaining = self.playback.seconds_remaining()
                print('owo ~ Watching\t{}/{}'.format(elapsed, remaining))
                # TODO: Everything
                timestamp = game_frame.timestamp
                time_offset = self.playback.seconds_elapsed_since(timestamp)
                self.manager.save_frame(game_frame.frame, time_offset * 60)
            # State 2-B: Playback end
            else:
                print('uwu ~ Finished watching ')
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
            if isinstance(token, str):
                mapped_input = self.input_mapping[token.upper()]
                self.input_controller.tap_key(mapped_input)
                time.sleep(delay)
            elif isinstance(token, int):
                time.sleep(token)
