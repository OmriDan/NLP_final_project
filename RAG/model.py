import torch
import inspect


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, low_weight=2.0, medium_weight=1.0, high_weight=2.0, medium_range=(0.3, 0.7)):
        super().__init__()
        self.low_weight = low_weight
        self.medium_weight = medium_weight
        self.high_weight = high_weight
        self.medium_range = medium_range

    def forward(self, pred, target):
        # Determine weights based on target values
        weights = torch.ones_like(target)
        weights = torch.where(target < self.medium_range[0], self.low_weight * weights, weights)
        weights = torch.where((target >= self.medium_range[0]) & (target <= self.medium_range[1]),
                              self.medium_weight * weights, weights)
        weights = torch.where(target > self.medium_range[1], self.high_weight * weights, weights)

        # Calculate weighted MSE
        loss = weights * ((pred - target) ** 2)
        return loss.mean()

class RAGQuestionDifficultyRegressor(torch.nn.Module):
    def __init__(self, encoder_model, dropout_rate=0.15):
        super().__init__()
        self.encoder = encoder_model
        hidden_size = self.encoder.config.hidden_size

        # Simplified architecture with fewer layers
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),  # Using ReLU instead of GELU
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size // 2, 1),
            torch.nn.Sigmoid()  # Keep sigmoid for 0-1 range
        )

        # Attention pooling
        self.attention_pool = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get the device of the model
        device = next(self.parameters()).device

        # Ensure inputs are on the same device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Filter kwargs for encoder
        filtered_kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in kwargs.items()
                           if k in inspect.signature(self.encoder.forward).parameters}

        # Forward pass through encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **filtered_kwargs)
        hidden_states = outputs.last_hidden_state

        # Use attention pooling
        attention_scores = self.attention_pool(hidden_states).squeeze(-1)
        attention_weights = torch.softmax(attention_scores.masked_fill(~attention_mask.bool(), -10000.0), dim=1)
        pooled_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)

        score = self.regressor(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            # Simple MSE loss for more stable training
            loss = torch.nn.MSELoss()(score, labels)

        # For compatibility with Trainer
        if loss is not None:
            return loss, score
        else:
            return score