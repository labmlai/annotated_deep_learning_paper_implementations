"""
---
title: Regret Minimization in Games with Incomplete Information
summary: >
  This is an annotated implementation/tutorial of Regret Minimization in Games with Incomplete Information
---

# Regret Minimization in Games with Incomplete Information

The paper
[Regret Minimization in Games with Incomplete Information](http://martin.zinkevich.org/publications/regretpoker.pdf)
introduces notion counterfactual regret and how minimizing counterfactual regret through self-play
can be used to reach Nash equilibrium.
The paper uses this technique to solve Texas Hold'em Poker.

We tried to keep our Python implementation easy-to-understand like a tutorial; therefore
it is not very efficient.
We run it on [a very simple imperfect information game called Kuhn poker](kuhn.html).

## Introduction

Counterfactual Regret minimization, in each iteration,
 explores the full game tree by trying all player actions.
It samples chance events only once per iteration.
Chance events are things like dealing cards; they are kept constant for each game tree exploration.
Then it calculates the *regret* of not taking each action, and following the current strategy.
Then it updates the strategy based on these regrets for the next iteration.
Finally it computes the average of the strategies throughout the iterations.
This becomes very close to the Nash equilibrium.

We will first introduce the mathematical notation and theory.
It was difficult to have the theory introduced in-line with code.

### Player

A player is denoted by $i \in N$, where $N$ is the set of players.

### [History](#History)

History $h \in H$ is a sequence of actions including chance events,
 and $H$ is the set of all histories.

$Z \subseteq H$ is the set of of terminal histories (game over).

### Action

Action $a$, $A(h) = \{a: (h, a) \in H}$ where $h \in H$ is a non-terminal [history](#History).

### [Information Set $I_i$](#InfoSet)

**Information set** $I_i \in \mathcal{I}_i$ for player $i$
is a similar to a history $h \in H$
but only contain the actions visible to player $i$.
That is, the history $h$ will contain actions/event such as cards dealt to the
opposing player while $I_i$ will not have them.

$\mathcal{I}_i$ is known as the **information partition** of player $i$.

<a id="Strategy"></a>
### Strategy

**Strategy of player** $i$, $\sigma_i \in \Sigma_i$ is a distribution over actions $A(I_i)$,
where $\Sigma_i$ is the set of all strategies for player $i$.
Strategy on $t$-th iteration is denoted by $\sigma^t_i$.

Strategy is defined as a probability for taking an action $a$ in for a given information set $I$,

$$\sigma_i(I)(a)$$

$\sigma$ is the **strategy profile** which consists of strategies of all players
 $\sigma_1, \sigma_2, \ldots$

$\sigma_{-i}$ is strategies of all players except $\sigma_i$

<a id="HistoryProbability"></a>
### Probability of History

$\pi^\sigma(h)$ is the probability of reaching the history $h$ with strategy profile $\sigma$.
$\pi^\sigma(h)_{-i}$ is the probability of reaching $h$ without player $i$'s contribution;
 i.e. player $i$ took the actions to follow $h$ with a probability of $1$.

$$\pi^\sigma(I) = \sum_{h \in I} \pi^\sigma(h)$$

for information set $I$, is the probability of reaching information set $I$ with
strategy profile $\sigma$.

### Utility (Pay off)

The [terminal utility](#terminal_utility) is the utility (or pay off)
 of a player $i$ for a terminal history $h$.

$$u_i(h)$$ where $h \in Z$

$u_i(\sigma)$ is the expected utility (payoff) for player $i$ with strategy profile $\sigma$.

$$u_i(\sigma) = \sum_{h \in Z} u_i(h) \pi^\sigma(h)$$

### Nash Equilibrium

Nash equilibrium is state where none of the players can increase their expected utility (or payoff)
by changing her strategy alone.

For two players, Nash equilibrium is a [strategy profile](#Strategy) where

\begin{align}
u_1(\sigma) &\ge \max_{\sigma'_1 \in \Sigma_1} u_1(\sigma'_1, \sigma_2) \\
u_2(\sigma) &\ge \max_{\sigma'_2 \in \Sigma_2} u_1(\sigma_1, \sigma'_2) \\
\end{align}

$\epsilon$-Nash equilibrium is,

\begin{align}
u_1(\sigma) + \epsilon &\ge \max_{\sigma'_1 \in \Sigma_1} u_1(\sigma'_1, \sigma_2) \\
u_2(\sigma)  + \epsilon &\ge \max_{\sigma'_2 \in \Sigma_2} u_1(\sigma_1, \sigma'_2) \\
\end{align}

### Regret Minimization

Regret is the utility (or pay off) that the player didn't get because
 she didn't follow the optimal strategy or took the best action.

Average overall regret for Player $i$ is, the average regret of not following the
optimal strategy in all $T$ rounds of game play.

$$R^T_i = \frac{1}{T} \max_{\sigma^*_i \in \Sigma_i} \sum_{t=1}^T
\Big( u_i(\sigma^*_i, \sigma^t_{-i}) - u_i(\sigma^t) \Big)$$

where $\sigma^t$ is the strategy profile of all players in round $t$,
and

$$(\sigma^*_i, \sigma^t_{-i})$$

is the strategy profile $\sigma^t$ with player $i$'s strategy
replaced with $\sigma^*_i$.

The average strategy is the average of strategies followed in each round,
 for all $I \in \mathcal{I}, a \in A(I)$

$$\bar{\sigma}^T_i(I)(a) = \frac{\sum_{t=1}^T \pi_i^{\sigma^t}(I)\sigma^t(I)(a)}{\sum_{t=1}^T \pi_i^{\sigma^t}(I)}$$

That is the mean regret of not playing with the optimal strategy.

If $R^T_i < \epsilon$ for all players then $\bar{\sigma}^T_i(I)(a)$ is a
$2\epsilon$-Nash equilibrium.

\begin{align}
R^T_i &< \epsilon \\
R^T_i &= \frac{1}{T} \max_{\sigma^*_i \in \Sigma_i} \sum_{t=1}^T
\Big( u_i(\sigma^*_i, \sigma^t_{-i}) - u_i(\sigma^t) \Big) \\
&= \frac{1}{T} \max_{\sigma^*_i \in \Sigma_i} \sum_{t=1}^T u_i(\sigma^*_i, \sigma^t_{-i})
- \frac{1}{T} \sum_{t=1}^T u_i(\sigma^t) < \epsilon
\end{align}

Since $u_1 = -u_2$ because it's a zero-sum game, we can add $R^T_1$ and $R^T_i$ and the
second term will cancel out.

\begin{align}
2\epsilon &>
\frac{1}{T} \max_{\sigma^*_1 \in \Sigma_1} \sum_{t=1}^T u_1(\sigma^*_1, \sigma^t_{-1}) +
\frac{1}{T} \max_{\sigma^*_2 \in \Sigma_2} \sum_{t=1}^T u_2(\sigma^*_2, \sigma^t_{-2})
\end{align}

The average of utilities over a set of strategies is equal to the utility of the average strategy.

$$\frac{1}{T} \sum_{t=1}^T u_i(\sigma^t) = u_i(\bar{\sigma}^T)$$

Therefore,
\begin{align}
2\epsilon &>
\max_{\sigma^*_1 \in \Sigma_1} u_1(\sigma^*_1, \bar{\sigma}^T_{-1}) +
\max_{\sigma^*_2 \in \Sigma_2} u_2(\sigma^*_2, \bar{\sigma}^T_{-2})
\end{align}

From definition of $\max$,
$$\max_{\sigma^*_2 \in \Sigma_2} u_2(\sigma^*_2, \bar{\sigma}^T_{-2}) \ge u_2(\bar{\sigma}^T)
 = -u_1(\bar{\sigma}^T)$$

Then,
\begin{align}
2\epsilon &>
\max_{\sigma^*_1 \in \Sigma_1} u_1(\sigma^*_1, \bar{\sigma}^T_{-1}) +
-u_1(\bar{\sigma}^T) \\
u_1(\bar{\sigma}^T) + 2\epsilon &> \max_{\sigma^*_1 \in \Sigma_1} u_1(\sigma^*_1, \bar{\sigma}^T_{-1})
\end{align}

That is, $2\epsilon$-Nash equilibrium. You can similarly prove for games with more than 2 players.

So we need to minimize $R^T_i$ to get close to a Nash equilibrium.
"""
from typing import NewType, Dict, List, Callable, cast, Optional

from labml import monit, tracker, logger
from labml.configs import BaseConfigs, option
from labml_helpers.training_loop import TrainingLoop

# A player $i \in N$ where $N$ is the set of players
Player = NewType('Player', int)
# Action $a$, $A(h) = \{a: (h, a) \in H}$ where $h \in H$ is a non-terminal [history](#History)
Action = NewType('Action', str)


class History:
    """
    <a id="History"></a>
    ## History

    History $h \in H$ is a sequence of actions including chance events,
     and $H$ is the set of all histories.
    """

    def is_terminal(self):
        """
        Whether it's a terminal history; i.e. game over.
        $h \in Z$
        """
        raise NotImplementedError()

    def terminal_utility(self, i: Player) -> float:
        """
        <a id="terminal_utility"></a>
        Utility of player $i$ for a terminal history.
        $u_i(h)$ where $h \in Z$
        """
        raise NotImplementedError()

    def player(self) -> Player:
        """
        Get current player, denoted by $P(h)$, where $P$ is known as **Player function**.

        If $P(h) = c$ it means that current event is a chance $c$ event.
        Something like dealing cards, or opening common cards in poker.
        """
        raise NotImplementedError()

    def is_chance(self) -> bool:
        """
        Whether the next step is a chance step; something like dealing a new card.
        $P(h) = c$
        """
        raise NotImplementedError()

    def sample_chance(self) -> Action:
        """
        Sample a chance when $P(h) = c$.
        """
        raise NotImplementedError()

    def __add__(self, action: Action):
        """
        Add an action to the history.
        """
        raise NotImplementedError()

    def info_set_key(self) -> str:
        """
        Get [information set](#InfoSet) for the current player
        """
        raise NotImplementedError

    def new_info_set(self) -> 'InfoSet':
        """
        Create a new [information set](#InfoSet) for the current player
        """
        raise NotImplementedError()

    def __repr__(self):
        """
        Human readable representation
        """
        raise NotImplementedError()

class InfoSet:
    """
    <a id="InfoSet"></a>
    ## Information Set $I_i$
    """

    # Unique key identifying the information set
    key: str
    # $\sigma_i$, the strategy of player $i$
    strategy: Dict[Action, float]
    # Current regret of not taking each action $A(I_i)$,
    #
    # $$\pi^{\sigma^t}_{-i} (I) \Big( u_i(\sigma_t |_{I \rightarrow a}, I) - u_i(\sigma^t, I) \Big)$$
    # where $\sigma_t |_{I \rightarrow a}$ is the same as strategy $\sigma_t$ but always taking action $a$
    # at $I$.
    current_regret: Dict[Action, float]
    # Total regret of not taking each action $A(I_i)$,
    #
    # $$R^T_i(I, a) = \frac{1}{T} \sum_{t=1}^T
    # \pi^{\sigma^t}_{-i} (I) \Big( u_i(\sigma_t |_{I \rightarrow a}, I) - u_i(\sigma^t, I) \Big)$$
    regret: Dict[Action, float]
    # Average strategy
    average_strategy: Dict[Action, float]

    def __init__(self, key: str):
        """
        Initialize
        """
        self.key = key
        self.regret = {a: 0 for a in self.actions()}
        self.current_regret = {a: 0 for a in self.actions()}
        self.average_strategy = {a: 0 for a in self.actions()}
        self.calculate_strategy()

    def actions(self) -> List[Action]:
        """
        Actions $A(I_i)$
        """
        raise NotImplementedError()

    @staticmethod
    def from_dict(data: Dict[str, any]) -> 'InfoSet':
        """
        Load information set from a saved dictionary
        """
        raise NotImplementedError()

    def to_dict(self):
        """
        Save the information set to a dictionary
        """
        return {
            'key': self.key,
            'regret': self.regret,
            'average_strategy': self.average_strategy,
        }

    def load_dict(self, data: Dict[str, any]):
        """
        Load data from a saved dictionary
        """
        self.regret = data['regret']
        self.average_strategy = data['average_strategy']
        self.calculate_strategy()

    def calculate_strategy(self):
        """
        ## Calculate strategy

        Strategy of time $T + 1$ is,

        \begin{align}
        \sigma_i^{T+1}(I)(a) =
        \begin{cases}
        \frac{R^{T,+}_i(I, a)}{\sum_{a'\in A(I)}R^{T,+}_i(I, a')},  & \text{if} \sum_{a'\in A(I)}R^{T,+}_i(I, a') \gt 0 \\
        \frac{1}{\lvert A(I) \rvert}, & \text{otherwise}
        \end{cases}
        \end{align}

        where $R^{T,+}_i(I, a) = \max \Big(R^T_i(I, a), 0 \Big)$
        """
        # $$R^{T,+}_i(I, a) = \max \Big(R^T_i(I, a), 0 \Big)$$
        regret = {a: max(r, 0) for a, r in self.regret.items()}
        # $$\sum_{a'\in A(I)}R^{T,+}_i(I, a')$$
        regret_sum = sum(regret.values())
        # if $\sum_{a'\in A(I)}R^{T,+}_i(I, a') \gt 0$
        if regret_sum > 0:
            # \frac{R^{T,+}_i(I, a)}{\sum_{a'\in A(I)}R^{T,+}_i(I, a')}
            self.strategy = {a: r / regret_sum for a, r in regret.items()}
        # Otherwise
        else:
            # $\lvert A(I) \rvert$
            count = len(list(a for a in self.regret))
            # $\frac{1}{\lvert A(I) \rvert}$
            self.strategy = {a: 1 / count for a, r in regret.items()}

    def get_average_strategy(self):
        """
        ## Get normalized average strategy
        """
        cum_strategy = {a: self.average_strategy.get(a, 0.) for a in self.actions()}
        strategy_sum = sum(cum_strategy.values())
        if strategy_sum > 0:
            return {a: s / strategy_sum for a, s in cum_strategy.items()}
        else:
            count = len(list(a for a in cum_strategy))
            return {a: 1 / count for a, r in cum_strategy.items()}

    def clear(self):
        """
        Clear current regrets
        """
        self.current_regret = {a: 0 for a in self.actions()}

    def update_regrets(self):
        """
        Update regrets with current regrets
        """
        for k, v in self.current_regret.items():
            self.regret[k] += v

    def __repr__(self):
        """
        Human readable representation
        """
        raise NotImplementedError()


class CFR:
    """
    $$r^t_i(a) = \max_{\sigma^*_i} u_i(\sigma^*_i, \sigma^t_{-i}) - u_i(\sigma^t)$$
    where,

    * $\sigma^t_i$ is the current strategy of player $i$,
    * $\sigma^t$ is the **strategy profile** which consists of strategies of all players
     $\sigma^t_1, \sigma^t_2, ...$
    * $\sigma^t_{-i}$ is strategies of all players except $\sigma^t_i$
    * $u_i(\sigma^t)$ is the overall utility of player $i$
    $$u_i(\sigma) = \sum_{h \in Z} u_i(h) \pi^\sigma(h)$$
    where $u_i(h)$ is the [terminal utility](#terminal_utility)
    and $\pi^\sigma(h)$ is the probability of $h$ occurring if players chose actions
    according to $\sigma$.
    """
    update_get_cfv: 'UpdateAndGetCounterFactualValue'
    track_frequency: int
    info_sets: Dict[str, InfoSet]
    is_online_update: bool
    n_players: int
    create_new_history: Callable[[], History]
    epochs: int

    def __init__(self, *,
                 create_new_history,
                 epochs,
                 training_loop: TrainingLoop,
                 update_regrets: 'UpdateRegrets',
                 update_infosets: 'UpdateInfoSets',
                 update_average_strategy: 'UpdateAverageStrategy',
                 update_get_cfv: 'UpdateAndGetCounterFactualValue' = None,
                 n_players=2,
                 track_frequency=10,
                 save_frequency=10):
        self.update_average_strategy = update_average_strategy
        self.training_loop = training_loop
        if update_get_cfv is None:
            update_get_cfv = UpdateAndGetCounterFactualValue()
        self.update_get_cfv = update_get_cfv
        self.update_regrets = update_regrets
        self.update_infosets = update_infosets
        self.save_frequency = save_frequency
        self.track_frequency = track_frequency
        self.n_players = n_players
        self.epochs = epochs
        self.create_new_history = create_new_history
        self.info_sets = {}
        self.tracker = InfoSetTracker()

    def cfr(self, h: History, i: Player, pi_i: float, pi_neg_i: float) -> float:
        if h.is_terminal():
            return h.terminal_utility(i)
        elif h.is_chance():
            a = h.sample_chance()
            return self.cfr(h + a, i, pi_i, pi_neg_i)

        info_set_key = h.info_set_key()
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = h.new_info_set()
        I = self.info_sets[info_set_key]
        v = 0
        va = {}

        for a in I.actions():
            if i == h.player():
                va[a] = self.cfr(h + a, i, pi_i * I.strategy[a], pi_neg_i)
            else:
                va[a] = self.cfr(h + a, i, pi_i, pi_neg_i * I.strategy[a])
            v = v + I.strategy[a] * va[a]

        v = self.update_get_cfv(h, v, i)

        if h.player() == i:
            self.update_average_strategy(I, pi_i)
            self.update_regrets(I, va, v, pi_neg_i)

        return v

    def solve(self):
        for _ in self.training_loop:
            self.update_infosets.clear(self.info_sets)

            for i in range(self.n_players):
                self.cfr(self.create_new_history(), cast(Player, i), 1, 1)

            self.update_infosets(self.info_sets)

            self.tracker(self.info_sets)

            self.update_get_cfv.train()

        logger.inspect(self.info_sets)


class UpdateRegrets:
    def __call__(self, I: InfoSet, va: Dict[Action, float], v: float, pi_neg_i: float):
        for a in I.actions():
            I.regret[a] += pi_neg_i * (va[a] - v)

        I.calculate_strategy()


class UpdateInfoSets:
    def __call__(self, info_sets: Dict[str, InfoSet]):
        pass

    def clear(self, info_sets: Dict[str, InfoSet]):
        pass


class OfflineUpdateRegrets(UpdateRegrets):
    def __call__(self, I: InfoSet, va: Dict[Action, float], v: float, pi_neg_i: float):
        for a in I.actions():
            I.current_regret[a] += pi_neg_i * (va[a] - v)


class OfflineUpdateInfoSets(UpdateInfoSets):
    def __call__(self, info_sets: Dict[str, InfoSet]):
        for k, I in info_sets.items():
            I.update_regrets()
            I.calculate_strategy()

    def clear(self, info_sets: Dict[str, InfoSet]):
        for I in info_sets.values():
            I.clear()


class UpdateAndGetCounterFactualValue:
    def __call__(self, h: History, v: float, i: Player):
        return v

    def train(self):
        pass


class UpdateAverageStrategy:
    def __call__(self, I: InfoSet, pi_i: float):
        for a in I.actions():
            I.average_strategy[a] = I.average_strategy[a] + pi_i * I.strategy[a]


class InfoSetTracker:
    def __init__(self):
        tracker.set_histogram(f'strategy.*')
        tracker.set_histogram(f'average_strategy.*')
        tracker.set_histogram(f'regret.*')
        tracker.set_histogram(f'current_regret.*')

    def __call__(self, info_sets: Dict[str, InfoSet]):
        with monit.section("Track"):
            for I in info_sets.values():
                avg_strategy = I.get_average_strategy()
                for a in I.actions():
                    tracker.add({
                        f'strategy.{I.key}.{a}': I.strategy[a],
                        f'average_strategy.{I.key}.{a}': avg_strategy[a],
                        f'regret.{I.key}.{a}': I.regret[a],
                        f'current_regret.{I.key}.{a}': I.current_regret[a]
                    })


class CFRConfigs(BaseConfigs):
    update_regrets: UpdateRegrets
    update_infosets: UpdateInfoSets
    create_new_history: Callable[[], History]
    epochs: int = 1_00_000
    update_get_cfv: Optional[UpdateAndGetCounterFactualValue] = None
    track_frequency: int = 1_000
    save_frequency: int = 1_000
    cfr: CFR = 'simple_cfr'
    loop: TrainingLoop
    update_average_strategy: UpdateAverageStrategy


@option(CFRConfigs.loop)
def training_loop(c: CFRConfigs):
    return TrainingLoop(
        loop_count=c.epochs,
        loop_step=1,
        is_save_models=True,
        log_new_line_interval=c.track_frequency,
        log_write_interval=c.track_frequency,
        save_models_interval=c.save_frequency,
        is_loop_on_interrupt=True
    )


@option(CFRConfigs.cfr)
def simple_cfr(c: CFRConfigs):
    return CFR(create_new_history=c.create_new_history,
               epochs=c.epochs,
               training_loop=c.loop,
               update_get_cfv=c.update_get_cfv,
               track_frequency=c.track_frequency,
               save_frequency=c.save_frequency,
               update_average_strategy=c.update_average_strategy,
               update_regrets=c.update_regrets,
               update_infosets=c.update_infosets)


@option(CFRConfigs.update_regrets, 'online')
def online_update_regrets():
    return UpdateRegrets()


@option(CFRConfigs.update_infosets, 'online')
def online_update_infosets():
    return UpdateInfoSets()


@option(CFRConfigs.update_regrets, 'offline')
def offline_update_regrets():
    return OfflineUpdateRegrets()


@option(CFRConfigs.update_infosets, 'offline')
def offline_update_infosets():
    return OfflineUpdateInfoSets()


@option(CFRConfigs.update_average_strategy)
def simple_update_average_strategy():
    return UpdateAverageStrategy()
