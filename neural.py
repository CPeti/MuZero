import torch
import torch.nn as nn
from configs import *
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, config) -> nn.Module:
        layers = []
        for layer_cfg in config:
            layer_type = layer_cfg["type"]
            layer_args = {k: v for k, v in layer_cfg.items() if k != "type"}

            try:
                layer_cls = getattr(nn, layer_type)
                layer = layer_cls(**layer_args)
            except AttributeError:
                raise ValueError(f"Layer type '{layer_type}' is not in torch.nn")
            except TypeError as e:
                raise ValueError(f"Invalid arguments for layer '{layer_type}': {e}")

            layers.append(layer)

        return nn.Sequential(*layers)

class RepresentationNetwork(NeuralNetwork):
    def __init__(self, real_state_shape):
        super().__init__()
        self.config = RepresentationNetworkConfig(real_state_shape)
        self.model = self._build_model(self.config.layers)
    
    def forward(self, x, batched=False):
        # add batch dimension if not present
        # convert to tensor if not already
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        if not batched:
            x = x.unsqueeze(0)
        x = self.model(x)
        return x

class DynamicsNetwork(NeuralNetwork):
    def __init__(self, action_space_size):
        super().__init__()
        self.action_space_size = action_space_size
        self.config = DynamicsNetworkConfig(action_space_size)
        self.state_model = self._build_model(self.config.state_layers)
        self.reward_head = self._build_model(self.config.reward_layers)
                
    def forward(self, state, action, batched=False):
        # One-hot encode the action
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if not batched:
            action = action.unsqueeze(0)
        action_one_hot = torch.zeros(action.size(0), self.action_space_size, device=action.device)
        action_one_hot.scatter_(1, action.unsqueeze(1), 1.0)
        
        x = torch.cat([state, action_one_hot], dim=1)
        next_state = self.state_model(x)
        reward = self.reward_head(next_state)
        
        return next_state, reward
        
class PredictionNetwork(NeuralNetwork):
    def __init__(self, action_space_size):
        super().__init__()
        self.action_space_size = action_space_size
        self.config = PredictionNetworkConfig(action_space_size)
        self.policy_layers = self._build_model(self.config.policy_layers)
        self.value_layers = self._build_model(self.config.value_layers)

    def forward(self, state):
        policy_logits = self.policy_layers(state)
        policy = torch.softmax(policy_logits, dim=1)
        value = self.value_layers(state)
        
        return policy, value 

class NNM:
    def __init__(self, real_state_shape, action_space_size: int):
        flat_state_size = np.prod(real_state_shape) # TODO: add support for resnet
        self.representation = RepresentationNetwork(flat_state_size)
        self.dynamics = DynamicsNetwork(action_space_size)
        self.prediction = PredictionNetwork(action_space_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.representation.to(self.device)
        self.dynamics.to(self.device)
        self.prediction.to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.representation.parameters()) + list(self.dynamics.parameters()) + list(self.prediction.parameters()), 
            lr=0.001,
            weight_decay=1e-4,
        )

    def train_networks(self, episode_buffer, batch_size: int = 64, sample_window: int = 7) -> None:
        """
        Train the networks through BPTT, sampling from the episode buffer.
        
        Args:
            episode_buffer: The buffer containing the episode data.
            batch_size: The size of the batch to sample from the buffer.
        """
        
        batch = episode_buffer.sample(batch_size, window=sample_window)
        losses = {
            'value_loss': [],
            'policy_loss': [],
            'reward_loss': []
        }
        for window in batch:
            real_game_states = torch.tensor(np.array(window['states']), dtype=torch.float32).to(self.device)
            actions = torch.tensor(np.array(window['actions']), dtype=torch.int64).to(self.device)
            target_values = torch.tensor(np.array(window['values']), dtype=torch.float32).to(self.device)
            target_policies = torch.tensor(np.array(window['policies']), dtype=torch.float32).to(self.device)
            target_rewards = torch.tensor(np.array(window['rewards']), dtype=torch.float32).to(self.device)
            v_l, p_l, r_l = self.bptt(real_game_states, actions, target_values, target_policies, target_rewards)
            losses['value_loss'].append(v_l)
            losses['policy_loss'].append(p_l)
            losses['reward_loss'].append(r_l)
        print(f"Value Loss: {np.mean(losses['value_loss'])}, Policy Loss: {np.mean(losses['policy_loss'])}, Reward Loss: {np.mean(losses['reward_loss'])}")

    def bptt(self, real_game_states, actions, target_values, target_policies, target_rewards):
        """
        Backpropagation through time (BPTT) for training the networks.
        
        Args:
            real_game_states: The real game states from the episode buffer.
            actions: The actions taken in the game.
            target_values: The target values for the value head.
            target_policies: The target policies for the policy head.
            target_rewards: The target rewards for the reward head.
            optimizer: The optimizer to use for training.
        """
        self.optimizer.zero_grad()

        abstract_state = self.representation(real_game_states[0], batched=False)
        policy, value = self.prediction(abstract_state)
        policy_pred = [policy]
        value_pred = [value]
        reward_pred = []

        for action in actions:
            # Forward pass through the dynamics network
            abstract_state, reward = self.dynamics(abstract_state, action, batched=False)
            policy, value = self.prediction(abstract_state)
            policy_pred.append(policy)
            value_pred.append(value)
            reward_pred.append(reward)

        # Convert to tensors
        value_pred = torch.stack(value_pred).squeeze(1)
        reward_pred = torch.stack(reward_pred).squeeze(1)
        policy_pred = torch.stack(policy_pred).squeeze(1)

        target_values = target_values.unsqueeze(1)
        target_rewards = target_rewards.unsqueeze(1)

        
        # Compute the loss
        value_loss = nn.MSELoss()(value_pred, target_values)
        # Cross entropy loss for policy
        policy_loss = nn.CrossEntropyLoss()(policy_pred, target_policies.argmax(dim=1))
        # MSE loss for reward
        reward_loss = nn.MSELoss()(reward_pred, target_rewards)

        # Total loss
        loss = value_loss + policy_loss + reward_loss

        loss.backward()
        self.optimizer.step()

        return value_loss.item(), policy_loss.item(), reward_loss.item()
    
    def save_model(self, path: str) -> None:
        """
        Save the model to the specified path.
        
        Args:
            path: The path to save the model to.
        """
        torch.save({
            'representation_state_dict': self.representation.state_dict(),
            'dynamics_state_dict': self.dynamics.state_dict(),
            'prediction_state_dict': self.prediction.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        """
        Load the model from the specified path.
        
        Args:
            path: The path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.representation.load_state_dict(checkpoint['representation_state_dict'])
        self.dynamics.load_state_dict(checkpoint['dynamics_state_dict'])
        self.prediction.load_state_dict(checkpoint['prediction_state_dict'])
        self.representation.to(self.device)
        self.dynamics.to(self.device)
        self.prediction.to(self.device)
        print(f"Model loaded from {path}")


        