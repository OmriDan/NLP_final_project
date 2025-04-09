import torch
import inspect



class RAGQuestionDifficultyRegressor(torch.nn.Module):
    def __init__(self, encoder_model, dropout_rate=0.4):
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
            # Apply weighted MSE loss based on difficulty ranges
            weights = torch.ones_like(labels)
            # More gentle weighting for low/high difficulty
            weights = torch.where(labels < 0.3, weights * 1.5, weights)
            weights = torch.where(labels > 0.7, weights * 1.5, weights)

            # Weighted MSE loss
            loss = ((score - labels) ** 2).mean()

        # For compatibility with Trainer
        if loss is not None:
            return loss, score
        else:
            return score