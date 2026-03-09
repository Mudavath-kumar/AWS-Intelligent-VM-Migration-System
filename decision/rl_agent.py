"""
rl_agent.py - Q-Learning Reinforcement Learning Agent
========================================================
Learns an optimal VM migration policy through trial-and-error.
State: discretised host CPU utilizations.
Actions: {do_nothing, migrate_from_host_i_to_least_loaded}
Reward: - SLA violations - migration cost + load balance bonus.

Can be compared against RF-based and rule-based strategies.
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.simulator import Simulator
from simulation.migration import migrate_vm, clear_migration_log, get_migration_log
from evaluation.evaluate import evaluate_strategy
from config import get as cfg
from logger import setup_logger

logger = setup_logger(__name__)


class QLearningAgent:
    """
    Tabular Q-learning agent for VM migration scheduling.

    State is a tuple of discretised per-host CPU bins.
    Actions are indexed as:
        0 = do nothing
        1..num_hosts = migrate highest-CPU VM from host i to least-loaded
    """

    def __init__(self, num_hosts=None):
        self.num_hosts = num_hosts or cfg("simulation.num_hosts", 5)
        self.state_bins = cfg("reinforcement_learning.state_bins", 10)
        self.alpha = cfg("reinforcement_learning.alpha", 0.1)
        self.gamma = cfg("reinforcement_learning.gamma", 0.95)
        self.epsilon = cfg("reinforcement_learning.epsilon", 1.0)
        self.epsilon_decay = cfg("reinforcement_learning.epsilon_decay", 0.995)
        self.epsilon_min = cfg("reinforcement_learning.epsilon_min", 0.01)

        self.num_actions = self.num_hosts + 1  # 0=noop, 1..N=migrate from host i
        self.q_table = {}

        self.sla_threshold = cfg("evaluation.sla_threshold", 90)
        self.migration_penalty = cfg("decision.migration_cost_penalty", 5.0)

        self.rewards_history = []

    # ------------------------------------------------------------------ #
    #  State encoding
    # ------------------------------------------------------------------ #
    def _get_state(self, simulator):
        """Discretise host CPUs into bins to form a state tuple."""
        cpus = []
        for host in simulator.hosts:
            cpu = host.get_total_cpu()
            bin_idx = min(int(cpu / (100.0 / self.state_bins)), self.state_bins - 1)
            cpus.append(bin_idx)
        return tuple(cpus)

    # ------------------------------------------------------------------ #
    #  Policy
    # ------------------------------------------------------------------ #
    def _choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        q_vals = [self.q_table.get((state, a), 0.0) for a in range(self.num_actions)]
        return int(np.argmax(q_vals))

    # ------------------------------------------------------------------ #
    #  Reward function
    # ------------------------------------------------------------------ #
    def _calculate_reward(self, simulator, migrated):
        """
        Reward = - (SLA violations) * 10
                 - migration_penalty * int(migrated)
                 + balance_bonus (inverse of load variance)
        """
        sla_violations = sum(
            1 for h in simulator.hosts if h.get_total_cpu() > self.sla_threshold
        )
        cpus = [h.get_total_cpu() for h in simulator.hosts]
        load_var = np.var(cpus)

        reward = -10.0 * sla_violations
        if migrated:
            reward -= self.migration_penalty
        # Bonus for balanced load (max 5 points when perfectly balanced)
        reward += max(0, 5.0 - 0.01 * load_var)

        return reward

    # ------------------------------------------------------------------ #
    #  Action execution
    # ------------------------------------------------------------------ #
    def _execute_action(self, action, simulator):
        """
        Execute migration action.

        Returns:
            bool: True if a migration was performed.
        """
        if action == 0:
            return False

        host_idx = action - 1
        if host_idx >= len(simulator.hosts):
            return False

        host = simulator.hosts[host_idx]
        if len(host.vms) <= 1:
            return False

        vm = max(host.vms, key=lambda v: v.cpu)
        target = simulator.get_least_loaded_host(exclude_host_id=host.host_id)
        if target:
            migrate_vm(host, target, vm.vm_id)
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def train(self, episodes=None, ticks_per_episode=10):
        """
        Train the Q-learning agent over multiple episodes.

        Args:
            episodes (int): Number of training episodes.
            ticks_per_episode (int): Ticks per episode.

        Returns:
            list: Cumulative rewards per episode.
        """
        episodes = episodes or cfg("reinforcement_learning.episodes", 500)

        logger.info("=" * 60)
        logger.info(f"RL TRAINING -- Q-Learning ({episodes} episodes)")
        logger.info("=" * 60)

        seed = cfg("random_seed", 42)
        episode_rewards = []

        for ep in range(1, episodes + 1):
            random.seed(seed + ep)
            np.random.seed(seed + ep)

            sim = Simulator()
            clear_migration_log()
            total_reward = 0.0

            for tick in range(ticks_per_episode):
                # Update VMs
                for host in sim.hosts:
                    for vm in host.vms:
                        vm.update_usage()

                state = self._get_state(sim)
                action = self._choose_action(state)
                migrated = self._execute_action(action, sim)
                reward = self._calculate_reward(sim, migrated)

                next_state = self._get_state(sim)

                # Q-update
                old_q = self.q_table.get((state, action), 0.0)
                next_max = max(
                    self.q_table.get((next_state, a), 0.0)
                    for a in range(self.num_actions)
                )
                new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
                self.q_table[(state, action)] = new_q

                total_reward += reward

            episode_rewards.append(total_reward)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if ep % 50 == 0:
                avg = np.mean(episode_rewards[-50:])
                logger.info(f"  Episode {ep}/{episodes} | Avg Reward (50): {avg:.2f} | eps: {self.epsilon:.4f}")

        self.rewards_history = episode_rewards
        self._plot_training(episode_rewards)
        logger.info(f"  Q-table size: {len(self.q_table)} entries")
        logger.info("RL training complete.")

        return episode_rewards

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, num_ticks=15):
        """
        Evaluate the trained RL policy (greedy, no exploration).

        Returns:
            dict: Evaluation results.
        """
        logger.info("=" * 60)
        logger.info("RL EVALUATION (Greedy Policy)")
        logger.info("=" * 60)

        seed = cfg("random_seed", 42)
        random.seed(seed)
        np.random.seed(seed)

        sim = Simulator()
        clear_migration_log()
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # greedy

        migration_count = 0
        for tick in range(num_ticks):
            for host in sim.hosts:
                for vm in host.vms:
                    vm.update_usage()

            state = self._get_state(sim)
            action = self._choose_action(state)
            migrated = self._execute_action(action, sim)
            if migrated:
                migration_count += 1

        self.epsilon = old_epsilon

        results = evaluate_strategy(sim.hosts, migration_count, "RL (Q-Learning)")
        return results

    # ------------------------------------------------------------------ #
    #  Plotting
    # ------------------------------------------------------------------ #
    def _plot_training(self, rewards):
        """Plot cumulative reward per episode."""
        fig, ax = plt.subplots(figsize=(10, 5))

        # Smoothed curve
        window = min(50, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

        ax.plot(rewards, alpha=0.3, color="steelblue", label="Raw reward")
        ax.plot(range(window - 1, len(rewards)), smoothed,
                color="darkorange", linewidth=2, label=f"Smoothed ({window}-ep)")
        ax.set_title("RL Agent Training Rewards", fontsize=14, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.legend()

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        path = "results/rl_training_rewards.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  RL training plot saved to '{path}'")
