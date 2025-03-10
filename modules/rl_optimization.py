#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reinforcement Learning module for process optimization
"""

import numpy as np
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

class ProcessEnv:
    """
    Environment for process optimization using RL
    The agent chooses (next_activity, resource) pairs and receives rewards
    based on cost, delay, and resource utilization
    """
    def __init__(self, df, le_task, resources):
        self.df = df
        self.le_task = le_task
        self.all_tasks = sorted(df["task_id"].unique())
        self.resources = resources
        self.start_task_id = 0
        self.done = False
        self.current_task = None
        
        # Additional state information could be added here
        self.resource_usage = {r: 0 for r in resources}
        self.total_cost = 0
        self.total_delay = 0
        
        # Enhanced tracking
        self.step_count = 0
        self.history = []
        
        # Precompute transition probabilities for more realistic environment
        self._compute_transition_probabilities()
        
    def _compute_transition_probabilities(self):
        """Calculate transition probabilities from data for more realistic simulation"""
        print("\nComputing transition probabilities from data...")
        self.transition_probs = {}
        
        # Prepare transitions dataframe
        transitions = self.df.copy()
        transitions["next_task_id"] = transitions.groupby("case_id")["task_id"].shift(-1)
        transitions = transitions.dropna(subset=["next_task_id"])
        
        # Calculate probabilities
        for task_id in self.all_tasks:
            task_transitions = transitions[transitions["task_id"] == task_id]
            if len(task_transitions) > 0:
                next_tasks = task_transitions["next_task_id"].value_counts(normalize=True).to_dict()
                self.transition_probs[task_id] = next_tasks
            else:
                self.transition_probs[task_id] = {}
        
        # Calculate task durations for each resource
        self.task_durations = {}
        for task_id in self.all_tasks:
            self.task_durations[task_id] = {}
            for resource in self.resources:
                # In real implementation, would use actual durations from data
                # For now, generate reasonable values
                self.task_durations[task_id][resource] = random.uniform(0.5, 2.0)
        
        print(f"Computed transition probabilities for {len(self.transition_probs)} tasks")
        
    def reset(self):
        """Reset the environment to initial state"""
        self.current_task = self.start_task_id
        self.done = False
        self.resource_usage = {r: 0 for r in self.resources}
        self.total_cost = 0
        self.total_delay = 0
        self.step_count = 0
        self.history = []
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation
        Enhanced with resource availability
        """
        state_vec = np.zeros(len(self.all_tasks) + len(self.resources), dtype=np.float32)
        
        # One-hot encoding of current task
        task_idx = self.all_tasks.index(self.current_task) if self.current_task in self.all_tasks else 0
        state_vec[task_idx] = 1.0
        
        # Resource usage information
        for i, resource in enumerate(self.resources):
            state_vec[len(self.all_tasks) + i] = self.resource_usage[resource]
        
        return state_vec
    
    def step(self, action):
        """
        Take a step in the environment with enhanced feedback
        action = (next_activity_id, resource_id)
        Returns: (next_state, reward, done, info)
        """
        next_task, resource = action
        self.step_count += 1
        
        # Check if next_task is valid based on transition probabilities
        if next_task not in self.all_tasks:
            # Invalid task
            reward = -100.0
            self.done = True
            return self._get_state(), reward, self.done, {"error": "Invalid task"}
        
        # Check if transition is possible with reasonable probability
        if self.current_task in self.transition_probs:
            task_probs = self.transition_probs[self.current_task]
            transition_prob = task_probs.get(next_task, 0)
            
            if transition_prob < 0.01:  # Less than 1% probability
                # Highly unlikely transition
                reward = -50.0 * (0.01 - transition_prob) / 0.01  # Scale penalty by improbability
                self.done = False  # Don't terminate but penalize
                info = {"warning": f"Unlikely transition ({transition_prob:.4f} probability)"}
                
                # We still execute the action but with a penalty
            else:
                # Valid transition
                info = {"transition_probability": transition_prob}
        else:
            # No data for current task
            info = {"warning": "No transition data for current task"}
        
        # Compute costs and delays more realistically
        transition_cost = self._compute_transition_cost(self.current_task, next_task)
        processing_delay = self._compute_processing_delay(next_task, resource)
        resource_efficiency = self._compute_resource_efficiency(resource)
        
        # Update internal state
        self.total_cost += transition_cost
        self.total_delay += processing_delay
        self.resource_usage[resource] += 1
        
        # Track history
        self.history.append({
            "step": self.step_count,
            "from_task": self.current_task,
            "to_task": next_task,
            "resource": resource,
            "cost": transition_cost,
            "delay": processing_delay,
            "efficiency": resource_efficiency
        })
        
        # Compute reward components
        cost_penalty = -transition_cost
        delay_penalty = -processing_delay
        efficiency_bonus = resource_efficiency
        
        # Combined reward
        reward = cost_penalty + delay_penalty + efficiency_bonus
        
        # Move to next state
        self.current_task = next_task
        
        # Check if process should terminate
        if self._should_terminate():
            self.done = True
            # Add completion bonus for successful termination
            reward += 20.0
        elif self.step_count >= 20:
            # Limit maximum steps to prevent infinite loops
            self.done = True
            reward -= 10.0  # Penalty for excessive length
            info["warning"] = "Maximum steps reached"
        
        info.update({
            'transition_cost': transition_cost,
            'processing_delay': processing_delay,
            'resource_efficiency': resource_efficiency,
            'step_count': self.step_count
        })
        
        return self._get_state(), reward, self.done, info
    
    def _compute_transition_cost(self, current_task, next_task):
        """
        Compute cost of transitioning between tasks
        Enhanced with data-based costs
        """
        # In a real implementation, would use historical data
        # For now, use a simple model based on task difference
        base_cost = abs(next_task - current_task) * 0.5
        
        # Add random variation
        variation = random.uniform(0.8, 1.2)
        
        return base_cost * variation
    
    def _compute_processing_delay(self, task, resource):
        """
        Compute processing delay for task-resource pair
        Enhanced with learned patterns
        """
        # Use pre-computed durations if available
        if task in self.task_durations and resource in self.task_durations[task]:
            base_delay = self.task_durations[task][resource]
        else:
            base_delay = random.uniform(0.5, 2.0)
        
        # Factor in resource utilization
        resource_factor = 1.0 + (self.resource_usage[resource] * 0.1)
        
        # Add random variation
        variation = random.uniform(0.9, 1.1)
        
        return base_delay * resource_factor * variation
    
    def _compute_resource_efficiency(self, resource):
        """
        Compute resource utilization efficiency
        Enhanced with more realistic model
        """
        total_usage = sum(self.resource_usage.values())
        if total_usage == 0:
            return 1.0
        
        current_usage = self.resource_usage[resource]
        expected_usage = total_usage / len(self.resources)
        
        # Calculate efficiency - higher when closer to balanced utilization
        if current_usage <= expected_usage:
            efficiency = 1.0
        else:
            # Diminishing efficiency with overutilization
            overuse_ratio = (current_usage - expected_usage) / expected_usage
            efficiency = max(0.0, 1.0 - (overuse_ratio * 0.5))
        
        return efficiency
    
    def _should_terminate(self):
        """
        Determine if the process should terminate
        Enhanced with data-based termination probability
        """
        # Probability increases with step count
        base_prob = 0.05 + (self.step_count * 0.01)
        
        # Certain tasks are more likely to be terminal
        if self.current_task in self.transition_probs:
            # Check if this task often has no further transitions
            task_probs = self.transition_probs[self.current_task]
            if not task_probs or sum(task_probs.values()) < 0.5:
                # Often a terminal task
                base_prob += 0.3
        
        return random.random() < min(0.5, base_prob)

def run_q_learning(env, episodes=30, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q-learning algorithm for process optimization with enhanced monitoring
    """
    print("\n==== Training RL Agent (Q-Learning) ====")
    start_time = time.time()
    
    possible_tasks = env.all_tasks
    possible_resources = env.resources
    
    # All possible actions (task, resource pairs)
    all_actions = []
    for t in possible_tasks:
        for r in possible_resources:
            all_actions.append((t, r))
    num_actions = len(all_actions)
    
    print(f"Action space: {num_actions} possible actions ({len(possible_tasks)} tasks × {len(possible_resources)} resources)")
    
    Q_table = {}
    
    def get_state_key(state):
        """Convert state array to hashable tuple"""
        return tuple(state.round(3))
    
    def get_Q(state):
        """Get Q-values for state, initialize if needed"""
        sk = get_state_key(state)
        if sk not in Q_table:
            Q_table[sk] = np.zeros(num_actions, dtype=np.float32)
        return Q_table[sk]
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    exploration_rates = []
    
    # Training loop with progress bar
    progress_bar = tqdm(
        range(episodes),
        desc="Training RL agent",
        bar_format="{l_bar}{bar:30}{r_bar}",
        ncols=100
    )
    
    for ep in progress_bar:
        s = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_exp_rate = 0
        
        # Dynamic exploration rate - decrease over time
        current_epsilon = max(0.01, epsilon * (1 - ep/episodes))
        exploration_rates.append(current_epsilon)
        
        while not done:
            # ε-greedy action selection
            if random.random() < current_epsilon:
                action_idx = random.randrange(num_actions)
                episode_exp_rate += 1
            else:
                q_values = get_Q(s)
                action_idx = int(np.argmax(q_values))
            
            action = all_actions[action_idx]
            next_state, reward, done, _info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Q-learning update
            current_q = get_Q(s)
            next_q = get_Q(next_state)
            best_next_q = 0.0 if done else np.max(next_q)
            
            # Update Q-value
            current_q[action_idx] += alpha * (
                reward + gamma * best_next_q - current_q[action_idx]
            )
            
            s = next_state
        
        # Update metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Update progress bar
        progress_bar.set_postfix({
            "reward": f"{total_reward:.2f}", 
            "steps": steps,
            "explore": f"{current_epsilon:.2f}"
        })
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards[-5:])  # Average of last 5 episodes
    avg_length = np.mean(episode_lengths[-5:])
    q_coverage = len(Q_table)
    
    # Print summary
    print("\033[1mRL Training Summary\033[0m:")
    print(f"  Final average reward: \033[96m{avg_reward:.2f}\033[0m")
    print(f"  Final average steps: \033[96m{avg_length:.1f}\033[0m")
    print(f"  States discovered: \033[96m{q_coverage}\033[0m")
    print(f"  Final exploration rate: \033[96m{exploration_rates[-1]:.3f}\033[0m")
    print(f"Training completed in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    # Plot learning curves if matplotlib is available
    try:
        # Plot rewards
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        # Plot episode lengths
        plt.subplot(1, 3, 2)
        plt.plot(episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        # Plot exploration rate
        plt.subplot(1, 3, 3)
        plt.plot(exploration_rates)
        plt.title('Exploration Rate')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig('rl_training_curves.png')
        print("Saved RL training curves to rl_training_curves.png")
    except:
        pass
    
    return Q_table

def get_optimal_policy(Q_table, all_actions):
    """
    Extract optimal policy from Q-table with enhanced analysis
    """
    print("\n==== Extracting Optimal Policy ====")
    start_time = time.time()
    
    policy = {}
    q_values = {}
    
    # Extract policy with confidence metrics
    for state in Q_table:
        q_vals = Q_table[state]
        optimal_action_idx = np.argmax(q_vals)
        policy[state] = all_actions[optimal_action_idx]
        
        # Calculate confidence metrics
        max_q = q_vals[optimal_action_idx]
        sorted_q = np.sort(q_vals)[::-1]  # Descending
        
        if len(sorted_q) > 1:
            # Gap between best and second best
            confidence = float(max_q - sorted_q[1]) if max_q > sorted_q[1] else 0.0
        else:
            confidence = 0.0
            
        q_values[state] = {
            'best_q': float(max_q),
            'confidence': confidence,
            'action_distribution': {str(all_actions[i]): float(q) for i, q in enumerate(q_vals) if q > 0}
        }
    
    # Analyze policy statistics
    task_distribution = {}
    resource_distribution = {}
    
    for state, (task, resource) in policy.items():
        task_distribution[task] = task_distribution.get(task, 0) + 1
        resource_distribution[resource] = resource_distribution.get(resource, 0) + 1
    
    print("\033[1mPolicy Statistics\033[0m:")
    print(f"  States with policy: \033[96m{len(policy)}\033[0m")
    print(f"  Unique next tasks: \033[96m{len(task_distribution)}\033[0m")
    print(f"  Resource utilization balance:")
    
    # Print resource distribution
    total_assignments = sum(resource_distribution.values())
    for resource, count in sorted(resource_distribution.items()):
        percentage = count / total_assignments * 100 if total_assignments > 0 else 0
        print(f"    Resource {resource}: \033[96m{percentage:.1f}%\033[0m ({count} assignments)")
    
    print(f"Policy extraction completed in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    # Enhanced return with quality metrics
    return {
        'policy': policy,
        'q_values': q_values,
        'task_distribution': task_distribution,
        'resource_distribution': resource_distribution
    }