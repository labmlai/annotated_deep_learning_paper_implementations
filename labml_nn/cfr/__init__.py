"""
---
title: Regret Minimization in Games with Incomplete Information (CFR)
summary: >
  This is an annotated implementation/tutorial of Regret Minimization in Games with Incomplete Information
---

# Regret Minimization in Games with Incomplete Information (CFR)

The paper
[Regret Minimization in Games with Incomplete Information](http://martin.zinkevich.org/publications/regretpoker.pdf)
introduces notion counterfactual regret and how minimizing counterfactual regret through self-play
can be used to reach Nash equilibrium.
The algorithm is called Counterfactural Regret Minization (**CFR**).
The paper uses this technique to solve Texas Hold'em Poker.

We tried to keep our Python implementation easy-to-understand like a tutorial.
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

$$\color{cyan}{\bar{\sigma}^T_i(I)(a)} =
 \frac{\sum_{t=1}^T \pi_i^{\sigma^t}(I)\color{lightgreen}{\sigma^t(I)(a)}}{\sum_{t=1}^T \pi_i^{\sigma^t}(I)}$$

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

### Counterfactual regret

**Counterfactual value** $\color{pink}{v_i(\sigma, I)}$ is the expected utility for player $i$ if the information $I$
is reached multiplied by $\pi^\sigma_{-i}(I)$ the probability of reaching $I$.
without player $i$'s contribution (if player $i$ took the actions leading to $I$ with a probability of $1$).
So this is like the expected utility if player $i$ tried to reach $I$.

$$\color{pink}{v_i(\sigma, I)} = \sum_{z \in Z_I} \pi^\sigma_{-i}(z[I]) \pi^\sigma(z[I], z) u_i(z)$$

where $Z_I$ is the set of terminal histories reachable from $I$ and
$z[I]$ is the prefix of history $z$ upto the information set $I$.
$\pi^\sigma(z[I], z)$ is the probability of reaching z from $z[I]$.

**Immediate counterfactual regret** is,

$$R^T_{i,imm}(I) = \frac{1}{T} \max_{a \in A{I}} \sum_{t=1}^T
\pi^{\sigma^t}_{-i} (I) \Big(
\color{pink}{v_i(\sigma^t |_{I \rightarrow a}, I)} - \color{pink}{v_i(\sigma^t, I)}
\Big)$$

where $\sigma |_{I \rightarrow a}$ is the strategy profile $\sigma$ with the modification
of always taking action $a$ at information set $I$.

The paper proves that (Theorem 3),

$$R^T_i \le \sum_{I \in \mathcal{I}} R^{T,+}_{i,imm}(I)$$
where $$R^{T,+}_{i,imm}(I) = \max(R^T_{i,imm}(I), 0)$$

<a id="RegretMatching"></a>
### Regret Matching

The strategy is calculated using regret matching.

The regret for each information set and action pair $\color{orange}{R^T_i(I, a)}$ is maintained,

\begin{align}
\color{coral}{r^t_i(I, a)} &=
 \pi^{\sigma^t}_{-i} (I) \Big(
 \color{pink}{v_i(\sigma^t |_{I \rightarrow a}, I)} - \color{pink}{v_i(\sigma^t, I)}
 \Big) \\
\color{orange}{R^T_i(I, a)} &=
 \frac{1}{T} \sum_{t=1}^T \color{coral}{r^t_i(I, a)}
\end{align}

and the strategy is calculated with regret matching,

\begin{align}
\color{lightgreen}{\sigma_i^{T+1}(I)(a)} =
\begin{cases}
\frac{\color{orange}{R^{T,+}_i(I, a)}}{\sum_{a'\in A(I)}\color{orange}{R^{T,+}_i(I, a')}},
  & \text{if} \sum_{a'\in A(I)}\color{orange}{R^{T,+}_i(I, a')} \gt 0 \\
\frac{1}{\lvert A(I) \rvert},
 & \text{otherwise}
\end{cases}
\end{align}

where $\color{orange}{R^{T,+}_i(I, a)} = \max \Big(\color{orange}{R^T_i(I, a)}, 0 \Big)$

So we maintain $\color{orange}{R^T_i(I, a)}$ and update
 the strategy $\color{lightgreen}{\sigma_i^{T+1}(I)(a)}$ on each iteration.
Finally we calculate the overall average strategy $\color{cyan}{\bar{\sigma}^T_i(I)(a)}$.

*Let's dive into the code!*
"""
from typing import NewType, Dict, List, Callable, cast

from labml import monit, tracker, logger, experiment
from labml.configs import BaseConfigs, option

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
    # $\sigma_i$, the [strategy](#Strategy) of player $i$
    strategy: Dict[Action, float]
    # Total regret of not taking each action $A(I_i)$,
    #
    # \begin{align}
    # \color{coral}{r^t_i(I, a)} &=
    #  \pi^{\sigma^t}_{-i} (I) \Big(
    #  \color{pink}{v_i(\sigma^t |_{I \rightarrow a}, I)} - \color{pink}{v_i(\sigma^t, I)}
    #  \Big) \\
    # \color{orange}{R^T_i(I, a)} &=
    #  \frac{1}{T} \sum_{t=1}^T \color{coral}{r^t_i(I, a)}
    # \end{align}
    #
    # We maintain $T \color{orange}{R^T_i(I, a)}$ instead of $\color{orange}{R^T_i(I, a)}$
    # since $\frac{1}{T}$ term cancels out anyway when computing strategy
    # $\color{lightgreen}{\sigma_i^{T+1}(I)(a)}$
    regret: Dict[Action, float]
    # We maintain the cumulative strategy
    # $$\sum_{t=1}^T \pi_i^{\sigma^t}(I)\color{lightgreen}{\sigma^t(I)(a)}$$
    # to compute overall average strategy
    #
    # $$\color{cyan}{\bar{\sigma}^T_i(I)(a)} =
    #  \frac{\sum_{t=1}^T \pi_i^{\sigma^t}(I)\color{lightgreen}{\sigma^t(I)(a)}}{\sum_{t=1}^T \pi_i^{\sigma^t}(I)}$$
    cumulative_strategy: Dict[Action, float]

    def __init__(self, key: str):
        """
        Initialize
        """
        self.key = key
        self.regret = {a: 0 for a in self.actions()}
        self.cumulative_strategy = {a: 0 for a in self.actions()}
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
            'average_strategy': self.cumulative_strategy,
        }

    def load_dict(self, data: Dict[str, any]):
        """
        Load data from a saved dictionary
        """
        self.regret = data['regret']
        self.cumulative_strategy = data['average_strategy']
        self.calculate_strategy()

    def calculate_strategy(self):
        """
        ## Calculate strategy

        Calculate current strategy using [regret matching](#RegretMatching).

        \begin{align}
        \color{lightgreen}{\sigma_i^{T+1}(I)(a)} =
        \begin{cases}
        \frac{\color{orange}{R^{T,+}_i(I, a)}}{\sum_{a'\in A(I)}\color{orange}{R^{T,+}_i(I, a')}},
          & \text{if} \sum_{a'\in A(I)}\color{orange}{R^{T,+}_i(I, a')} \gt 0 \\
        \frac{1}{\lvert A(I) \rvert},
         & \text{otherwise}
        \end{cases}
        \end{align}

        where $\color{orange}{R^{T,+}_i(I, a)} = \max \Big(\color{orange}{R^T_i(I, a)}, 0 \Big)$
        """
        # $$\color{orange}{R^{T,+}_i(I, a)} = \max \Big(\color{orange}{R^T_i(I, a)}, 0 \Big)$$
        regret = {a: max(r, 0) for a, r in self.regret.items()}
        # $$\sum_{a'\in A(I)}\color{orange}{R^{T,+}_i(I, a')}$$
        regret_sum = sum(regret.values())
        # if $\sum_{a'\in A(I)}\color{orange}{R^{T,+}_i(I, a')} \gt 0$,
        if regret_sum > 0:
            # $$\color{lightgreen}{\sigma_i^{T+1}(I)(a)} =
            # \frac{\color{orange}{R^{T,+}_i(I, a)}}{\sum_{a'\in A(I)}\color{orange}{R^{T,+}_i(I, a')}}$$
            self.strategy = {a: r / regret_sum for a, r in regret.items()}
        # Otherwise,
        else:
            # $\lvert A(I) \rvert$
            count = len(list(a for a in self.regret))
            # $$\color{lightgreen}{\sigma_i^{T+1}(I)(a)} =
            # \frac{1}{\lvert A(I) \rvert}$$
            self.strategy = {a: 1 / count for a, r in regret.items()}

    def get_average_strategy(self):
        """
        ## Get average strategy

        $$\color{cyan}{\bar{\sigma}^T_i(I)(a)} =
         \frac{\sum_{t=1}^T \pi_i^{\sigma^t}(I)\color{lightgreen}{\sigma^t(I)(a)}}
         {\sum_{t=1}^T \pi_i^{\sigma^t}(I)}$$
        """
        # $$\sum_{t=1}^T \pi_i^{\sigma^t}(I) \color{lightgreen}{\sigma^t(I)(a)}$$
        cum_strategy = {a: self.cumulative_strategy.get(a, 0.) for a in self.actions()}
        # $$\sum_{t=1}^T \pi_i^{\sigma^t}(I) =
        # \sum_{a \in A(I)} \sum_{t=1}^T
        # \pi_i^{\sigma^t}(I)\color{lightgreen}{\sigma^t(I)(a)}$$
        strategy_sum = sum(cum_strategy.values())
        # If $\sum_{t=1}^T \pi_i^{\sigma^t}(I) > 0$,
        if strategy_sum > 0:
            # $$\color{cyan}{\bar{\sigma}^T_i(I)(a)} =
            #  \frac{\sum_{t=1}^T \pi_i^{\sigma^t}(I)\color{lightgreen}{\sigma^t(I)(a)}}
            #  {\sum_{t=1}^T \pi_i^{\sigma^t}(I)}$$
            return {a: s / strategy_sum for a, s in cum_strategy.items()}
        # Otherwise,
        else:
            # $\lvert A(I) \rvert$
            count = len(list(a for a in cum_strategy))
            # $$\color{cyan}{\bar{\sigma}^T_i(I)(a)} =
            # \frac{1}{\lvert A(I) \rvert}$$
            return {a: 1 / count for a, r in cum_strategy.items()}

    def __repr__(self):
        """
        Human readable representation
        """
        raise NotImplementedError()


class CFR:
    """
    ## Counterfactual Regret Minimization (CFR) Algorithm
    """

    # $\mathcal{I}$ set of all information sets.
    info_sets: Dict[str, InfoSet]

    def __init__(self, *,
                 create_new_history: Callable[[], History],
                 epochs: int,
                 n_players: int = 2):
        self.n_players = n_players
        self.epochs = epochs
        self.create_new_history = create_new_history
        self.info_sets = {}
        self.tracker = InfoSetTracker()

    def walk_tree(self, h: History, i: Player, pi_i: float, pi_neg_i: float) -> float:
        if h.is_terminal():
            return h.terminal_utility(i)
        elif h.is_chance():
            a = h.sample_chance()
            return self.walk_tree(h + a, i, pi_i, pi_neg_i)

        info_set_key = h.info_set_key()
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = h.new_info_set()
        I = self.info_sets[info_set_key]
        v = 0
        va = {}

        for a in I.actions():
            if i == h.player():
                va[a] = self.walk_tree(h + a, i, pi_i * I.strategy[a], pi_neg_i)
            else:
                va[a] = self.walk_tree(h + a, i, pi_i, pi_neg_i * I.strategy[a])
            v = v + I.strategy[a] * va[a]

        if h.player() == i:
            for a in I.actions():
                I.cumulative_strategy[a] = I.cumulative_strategy[a] + pi_i * I.strategy[a]
            for a in I.actions():
                I.regret[a] += pi_neg_i * (va[a] - v)

            I.calculate_strategy()

        return v

    def solve(self):
        for t in monit.loop(self.epochs):
            for i in range(self.n_players):
                self.walk_tree(self.create_new_history(), cast(Player, i), 1, 1)

            self.tracker(self.info_sets)
            tracker.save()
            if (t + 1) % 1_000 == 0:
                experiment.save_checkpoint()
                tracker.new_line()

        logger.inspect(self.info_sets)


class InfoSetTracker:
    def __init__(self):
        tracker.set_histogram(f'strategy.*')
        tracker.set_histogram(f'average_strategy.*')
        tracker.set_histogram(f'regret.*')

    def __call__(self, info_sets: Dict[str, InfoSet]):
        with monit.section("Track"):
            for I in info_sets.values():
                avg_strategy = I.get_average_strategy()
                for a in I.actions():
                    tracker.add({
                        f'strategy.{I.key}.{a}': I.strategy[a],
                        f'average_strategy.{I.key}.{a}': avg_strategy[a],
                        f'regret.{I.key}.{a}': I.regret[a],
                    })


class CFRConfigs(BaseConfigs):
    create_new_history: Callable[[], History]
    epochs: int = 1_00_000
    cfr: CFR = 'simple_cfr'


@option(CFRConfigs.cfr)
def simple_cfr(c: CFRConfigs):
    return CFR(create_new_history=c.create_new_history,
               epochs=c.epochs)
