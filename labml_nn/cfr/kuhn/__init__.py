"""
---
title: CFR on Kuhn Poker
summary: >
  This is an annotated implementation/tutorial of CFR on Kuhn Poker
---

# [Counterfactual Regret Minimization (CFR)](../index.html) on Kuhn Poker

This applies [Counterfactual Regret Minimization (CFR)](../index.html) to Kuhn poker.

[Kuhn Poker](https://en.wikipedia.org/wiki/Kuhn_poker) is a two player 3-card betting game.
The players are dealt one card each out of Ace, King and Queen (no suits).
There are only three cards in the pack so one card is left out.
Ace beats King and Queen and King beats Queen - just like in normal ranking of cards.

Both players ante $1$ chip (blindly bet $1$ chip).
After looking at the cards, the first player can either pass or bet $1$ chip.
If first player passes, the the player with higher card wins the pot.
If first player bets, the second play can bet (i.e. call) $1$ chip or pass (i.e. fold).
If the second player bets and the player with the higher card wins the pot.
If the second player passes (i.e. folds) the first player gets the pot.
This game is played repeatedly and a good strategy will optimize for the long term utility (or winnings).

Here's some example games:

* `KAp` - Player 1 gets K. Player 2 gets A. Player 1 passes. Player 2 doesn't get a betting chance and Player 2 wins the pot of $2$ chips.
* `QKbp` - Player 1 gets Q. Player 2 gets K. Player 1 bets a chip. Player 2 passes (folds). Player 1 gets the pot of $4$ because Player 2 folded.
* `QAbb` - Player 1 gets Q. Player 2 gets A. Player 1 bets a chip. Player 2 also bets (calls). Player 2 wins the pot of $4$.

He we extend the `InfoSet` class and `History` class defined in [`__init__.py`](../index.html)
with Kuhn Poker specifics.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/cfr/kuhn/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/7c35d3fad29711eba588acde48001122)
"""

from typing import List, cast, Dict

import numpy as np

from labml import experiment
from labml.configs import option
from labml_nn.cfr import History as _History, InfoSet as _InfoSet, Action, Player, CFRConfigs
from labml_nn.cfr.infoset_saver import InfoSetSaver

# Kuhn poker actions are pass (`p`) or bet (`b`)
ACTIONS = cast(List[Action], ['p', 'b'])
# The three cards in play are Ace, King and Queen
CHANCES = cast(List[Action], ['A', 'K', 'Q'])
# There are two players
PLAYERS = cast(List[Player], [0, 1])


class InfoSet(_InfoSet):
    """
    ## [Information set](../index.html#InfoSet)
    """

    @staticmethod
    def from_dict(data: Dict[str, any]) -> 'InfoSet':
        """Does not support save/load"""
        pass

    def actions(self) -> List[Action]:
        """
        Return the list of actions. Terminal states are handled by `History` class.
        """
        return ACTIONS

    def __repr__(self):
        """
        Human readable string representation - it gives the betting probability
        """
        total = sum(self.cumulative_strategy.values())
        total = max(total, 1e-6)
        bet = self.cumulative_strategy[cast(Action, 'b')] / total
        return f'{bet * 100: .1f}%'


class History(_History):
    """
    ## [History](../index.html#History)

    This defines when a game ends, calculates the utility and sample chance events (dealing cards).

    The history is stored in a string:
    * First two characters are the cards dealt to player 1 and player 2
    * The third character is the action by the first player
    * Fourth character is the action by the second player
    """

    # History
    history: str

    def __init__(self, history: str = ''):
        """
        Initialize with a given history string
        """
        self.history = history

    def is_terminal(self):
        """
        Whether the history is terminal (game over).
        """
        # Players are yet to take actions
        if len(self.history) <= 2:
            return False
        # Last player to play passed (game over)
        elif self.history[-1] == 'p':
            return True
        # Both players called (bet) (game over)
        elif self.history[-2:] == 'bb':
            return True
        # Any other combination
        else:
            return False

    def _terminal_utility_p1(self) -> float:
        """
        Calculate the terminal utility for player $1$,  $u_1(z)$
        """
        # $+1$ if Player 1 has a better card and $-1$ otherwise
        winner = -1 + 2 * (self.history[0] < self.history[1])

        # Second player passed
        if self.history[-2:] == 'bp':
            return 1
        # Both players called, the player with better card wins $2$ chips
        elif self.history[-2:] == 'bb':
            return winner * 2
        # First player passed, the player with better card wins $1$ chip
        elif self.history[-1] == 'p':
            return winner
        # History is non-terminal
        else:
            raise RuntimeError()

    def terminal_utility(self, i: Player) -> float:
        """
        Get the terminal utility for player $i$
        """
        # If $i$ is Player 1
        if i == PLAYERS[0]:
            return self._terminal_utility_p1()
        # Otherwise, $u_2(z) = -u_1(z)$
        else:
            return -1 * self._terminal_utility_p1()

    def is_chance(self) -> bool:
        """
        The first two events are card dealing; i.e. chance events
        """
        return len(self.history) < 2

    def __add__(self, other: Action):
        """
        Add an action to the history and return a new history
        """
        return History(self.history + other)

    def player(self) -> Player:
        """
        Current player
        """
        return cast(Player, len(self.history) % 2)

    def sample_chance(self) -> Action:
        """
        Sample a chance action
        """
        while True:
            # Randomly pick a card
            r = np.random.randint(len(CHANCES))
            chance = CHANCES[r]
            # See if the card was dealt before
            for c in self.history:
                if c == chance:
                    chance = None
                    break

            # Return the card if it was not dealt before
            if chance is not None:
                return cast(Action, chance)

    def __repr__(self):
        """
        Human readable representation
        """
        return repr(self.history)

    def info_set_key(self) -> str:
        """
        Information set key for the current history.
        This is a string of actions only visible to the current player.
        """
        # Get current player
        i = self.player()
        # Current player sees her card and the betting actions
        return self.history[i] + self.history[2:]

    def new_info_set(self) -> InfoSet:
        # Create a new information set object
        return InfoSet(self.info_set_key())


def create_new_history():
    """A function to create an empty history object"""
    return History()


class Configs(CFRConfigs):
    """
    Configurations extends the CFR configurations class
    """
    pass


@option(Configs.create_new_history)
def _cnh():
    """
    Set the `create_new_history` method for Kuhn Poker
    """
    return create_new_history


def main():
    """
    ### Run the experiment
    """

    # Create an experiment, we only write tracking information to `sqlite` to speed things up.
    # Since the algorithm iterates fast and we track data on each iteration, writing to
    # other destinations such as Tensorboard can be relatively time consuming.
    # SQLite is enough for our analytics.
    experiment.create(name='kuhn_poker', writers={'sqlite'})
    # Initialize configuration
    conf = Configs()
    # Load configuration
    experiment.configs(conf)
    # Set models for saving
    experiment.add_model_savers({'info_sets': InfoSetSaver(conf.cfr.info_sets)})
    # Start the experiment
    with experiment.start():
        # Start iterating
        conf.cfr.iterate()


#
if __name__ == '__main__':
    main()
