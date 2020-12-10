"""
---
title: Prioritized Experience Replay Buffer
summary: Annotated implementation of prioritized experience replay using a binary segment tree.
---

# Prioritized Experience Replay Buffer

This implements paper [Prioritized experience replay](https://arxiv.org/abs/1511.05952),
using a binary segment tree.
"""

import random

import numpy as np


class ReplayBuffer:
    """
    ## Buffer for Prioritized Experience Replay

    [Prioritized experience replay](https://arxiv.org/abs/1511.05952)
     samples important transitions more frequently.
    The transitions are prioritized by the Temporal Difference error (td error), $\delta$.

    We sample transition $i$ with probability,
    $$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$
    where $\alpha$ is a hyper-parameter that determines how much
    prioritization is used, with $\alpha = 0$ corresponding to uniform case.
    $p_i$ is the priority.

    We use proportional prioritization $p_i = |\delta_i| + \epsilon$ where
    $\delta_i$ is the temporal difference for transition $i$.

    We correct the bias introduced by prioritized replay using
     importance-sampling (IS) weights
    $$w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$$ in the loss function.
    This fully compensates when $\beta = 1$.
    We normalize weights by $\frac{1}{\max_i w_i}$ for stability.
    Unbiased nature is most important towards the convergence at end of training.
    Therefore we increase $\beta$ towards end of training.

    ### Binary Segment Tree
    We use a binary segment tree to efficiently calculate
    $\sum_k^i p_k^\alpha$, the cumulative probability,
    which is needed to sample.
    We also use a binary segment tree to find $\min p_i^\alpha$,
    which is needed for $\frac{1}{\max_i w_i}$.
    We can also use a min-heap for this.
    Binary Segment Tree lets us calculate these in $\mathcal{O}(\log n)$
    time, which is way more efficient that the naive $\mathcal{O}(n)$
    approach.

    This is how a binary segment tree works for sum;
    it is similar for minimum.
    Let $x_i$ be the list of $N$ values we want to represent.
    Let $b_{i,j}$ be the $j^{\mathop{th}}$ node of the $i^{\mathop{th}}$ row
     in the binary tree.
    That is two children of node $b_{i,j}$ are $b_{i+1,2j}$ and $b_{i+1,2j + 1}$.

    The leaf nodes on row $D = \left\lceil {1 + \log_2 N} \right\rceil$
     will have values of $x$.
    Every node keeps the sum of the two child nodes.
    That is, the root node keeps the sum of the entire array of values.
    The left and right children of the root node keep
     the sum of the first half of the array and
     the sum of the second half of the array, respectively.
    And so on...

    $$b_{i,j} = \sum_{k = (j -1) * 2^{D - i} + 1}^{j * 2^{D - i}} x_k$$

    Number of nodes in row $i$,
    $$N_i = \left\lceil{\frac{N}{D - i + 1}} \right\rceil$$
    This is equal to the sum of nodes in all rows above $i$.
    So we can use a single array $a$ to store the tree, where,
    $$b_{i,j} \rightarrow a_{N_i + j}$$

    Then child nodes of $a_i$ are $a_{2i}$ and $a_{2i + 1}$.
    That is,
    $$a_i = a_{2i} + a_{2i + 1}$$

    This way of maintaining binary trees is very easy to program.
    *Note that we are indexing starting from 1*.

    We use the same structure to compute the minimum.
    """

    def __init__(self, capacity, alpha):
        """
        ### Initialize
        """
        # We use a power of $2$ for capacity because it simplifies the code and debugging
        self.capacity = capacity
        # $\alpha$
        self.alpha = alpha

        # Maintain segment binary trees to take sum and find minimum over a range
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        # Current max priority, $p$, to be assigned to new transitions
        self.max_priority = 1.

        # Arrays for buffer
        self.data = {
            'obs': np.zeros(shape=(capacity, 4, 84, 84), dtype=np.uint8),
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_obs': np.zeros(shape=(capacity, 4, 84, 84), dtype=np.uint8),
            'done': np.zeros(shape=capacity, dtype=np.bool)
        }
        # We use cyclic buffers to store data, and `next_idx` keeps the index of the next empty
        # slot
        self.next_idx = 0

        # Size of the buffer
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """
        ### Add sample to queue
        """

        # Get next available slot
        idx = self.next_idx

        # store in the queue
        self.data['obs'][idx] = obs
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        self.data['done'][idx] = done

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity
        # Calculate the size
        self.size = min(self.capacity, self.size + 1)

        # $p_i^\alpha$, new samples get `max_priority`
        priority_alpha = self.max_priority ** self.alpha
        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        """
        #### $\sum_k p_k^\alpha$
        """

        # The root node keeps the sum of all values
        return self.priority_sum[1]

    def _min(self):
        """
        #### $\min_k p_k^\alpha$
        """

        # The root node keeps the minimum of all values
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """
        #### Find largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha  \le P$
        """

        # Start from the root
        idx = 1
        while idx < self.capacity:
            # If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum:
                # Go to left branch of the tree
                idx = 2 * idx
            else:
                # Otherwise go to right branch and reduce the sum of left
                #  branch from required sum
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # We are at the leaf node. Subtract the capacity by the index in the tree
        # to get the index of actual value
        return idx - self.capacity

    def sample(self, batch_size, beta):
        """
        ### Sample from buffer
        """

        # Initialize samples
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        # Get sample indexes
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx

        # $\min_i P(i) = \frac{\min_i p_i^\alpha}{\sum_k p_k^\alpha}$
        prob_min = self._min() / self._sum()
        # $\max_i w_i = \bigg(\frac{1}{N} \frac{1}{\min_i P(i)}\bigg)^\beta$
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]
            # $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            # $w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$
            weight = (prob * self.size) ** (-beta)
            # Normalize by $\frac{1}{\max_i w_i}$,
            #  which also cancels off the $\frac{1}{N}$ term
            samples['weights'][i] = weight / max_weight

        # Get samples data
        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]

        return samples

    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """

        for idx, priority in zip(indexes, priorities):
            # Set current max priority
            self.max_priority = max(self.max_priority, priority)

            # Calculate $p_i^\alpha$
            priority_alpha = priority ** self.alpha
            # Update the trees
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        """
        ### Whether the buffer is full
        """
        return self.capacity == self.size
