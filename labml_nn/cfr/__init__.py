from typing import NewType, Dict, List, Callable, cast, Optional

from labml import monit, tracker, logger
from labml.configs import BaseConfigs, option
from labml_helpers.training_loop import TrainingLoop

Action = NewType('Action', str)
Player = NewType('Player', int)


class InfoSet:
    key: str
    regret: Dict[Action, float]
    current_regret: Dict[Action, float]
    average_strategy: Dict[Action, float]
    strategy: Dict[Action, float]

    def __init__(self, key: str):
        self.key = key
        self.regret = {a: 0 for a in self.actions()}
        self.current_regret = {a: 0 for a in self.actions()}
        self.average_strategy = {a: 0 for a in self.actions()}
        self.calculate_policy()

    def actions(self) -> List[Action]:
        raise NotImplementedError()

    @staticmethod
    def from_dict(data: Dict[str, any]) -> 'InfoSet':
        raise NotImplementedError()

    def to_dict(self):
        return {
            'key': self.key,
            'regret': self.regret,
            'average_strategy': self.average_strategy,
        }

    def load_dict(self, data: Dict[str, any]):
        self.regret = data['regret']
        self.average_strategy = data['average_strategy']
        self.calculate_policy()

    def calculate_policy(self):
        regret = {a: max(r, 0) for a, r in self.regret.items()}
        regret_sum = sum(regret.values())
        if regret_sum > 0:
            self.strategy = {a: r / regret_sum for a, r in regret.items()}
        else:
            count = len(list(a for a in self.regret))
            self.strategy = {a: 1 / count for a, r in regret.items()}

    def get_average_strategy(self):
        cum_strategy = {a: self.average_strategy.get(a, 0.) for a in self.actions()}
        strategy_sum = sum(cum_strategy.values())
        if strategy_sum > 0:
            return {a: r / strategy_sum for a, r in cum_strategy.items()}
        else:
            count = len(list(a for a in cum_strategy))
            return {a: 1 / count for a, r in cum_strategy.items()}

    def clear(self):
        self.current_regret = {a: 0 for a in self.actions()}

    def update_regrets(self):
        for k, v in self.current_regret.items():
            self.regret[k] += v

    def __repr__(self):
        raise NotImplementedError()


class History:
    def is_terminal(self):
        raise NotImplementedError()

    def terminal_utility(self, i: Player) -> float:
        raise NotImplementedError()

    def is_chance(self) -> bool:
        raise NotImplementedError()

    def __add__(self, action: Action):
        raise NotImplementedError()

    def info_set_key(self) -> str:
        raise NotImplementedError

    def new_info_set(self) -> InfoSet:
        raise NotImplementedError()

    def player(self) -> int:
        raise NotImplementedError()

    def sample_chance(self) -> Action:
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()


class UpdateRegrets:
    def __call__(self, I: InfoSet, va: Dict[Action, float], v: float, pi_neg_i: float):
        for a in I.actions():
            I.regret[a] += pi_neg_i * (va[a] - v)

        I.calculate_policy()


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
            I.calculate_policy()

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


class CFR:
    update_get_cfv: UpdateAndGetCounterFactualValue
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
                 update_regrets: UpdateRegrets,
                 update_infosets: UpdateInfoSets,
                 update_average_strategy: UpdateAverageStrategy,
                 update_get_cfv: UpdateAndGetCounterFactualValue = None,
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
