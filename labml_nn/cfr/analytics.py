from typing import List

import altair as alt
import numpy as np

from labml import analytics
from labml.analytics import IndicatorCollection


def calculate_percentages(means: List[np.ndarray], names: List[List[str]]):
    normalized = []

    for i in range(len(means)):
        total = np.zeros_like(means[i])
        for j, n in enumerate(names):
            if n[-1][:-1] == names[i][-1][:-1]:
                total += means[j]
        normalized.append(means[i] / (total + np.finfo(float).eps))

    return normalized


def plot_infosets(indicators: IndicatorCollection, *,
                  is_normalize: bool = True,
                  width: int = 600,
                  height: int = 300):
    data, names = analytics.indicator_data(indicators)
    step = [d[:, 0] for d in data]
    means = [d[:, 5] for d in data]

    if is_normalize:
        normalized = calculate_percentages(means, names)
    else:
        normalized = means

    common = names[0][-1]
    for i, n in enumerate(names):
        n = n[-1]
        if len(n) < len(common):
            common = common[:len(n)]
        for j in range(len(common)):
            if common[j] != n[j]:
                common = common[:j]
                break

    table = []
    for i, n in enumerate(names):
        for j, v in zip(step[i], normalized[i]):
            table.append({
                'series': n[-1][len(common):],
                'step': j,
                'value': v
            })

    table = alt.Data(values=table)

    selection = alt.selection_multi(fields=['series'], bind='legend')

    return alt.Chart(table).mark_line().encode(
        alt.X('step:Q'),
        alt.Y('value:Q'),
        alt.Color('series:N', scale=alt.Scale(scheme='tableau20')),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.0001))
    ).add_selection(
        selection
    ).properties(width=width, height=height)
