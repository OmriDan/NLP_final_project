import torch
import inspect


class RAGQuestionDifficultyRegressor(torch.nn.Module):
    def __init__(self, encoder_model, dropout_rate=0.15):
        super().__init__()
        self.encoder = encoder_model
        hidden_size = self.encoder.config.hidden_size

        # Simplified architecture
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()  # Keep sigmoid for 0-1 range
        )

        # CLS token pooling instead of attention pooling
        self.use_cls = True

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        device = next(self.parameters()).device

        # Move inputs to device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        filtered_kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in kwargs.items()
                           if k in inspect.signature(self.encoder.forward).parameters}

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **filtered_kwargs)

        # Use CLS token for simpler, more stable training
        if self.use_cls:
            pooled_output = outputs.last_hidden_state[:, 0]
        else:
            # Use mean pooling as fallback
            hidden_states = outputs.last_hidden_state
            pooled_output = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask,
                                                                                                       dim=1,keepdim=True)

        score = self.regressor(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            # Use only MSE loss for regression
            loss = torch.nn.MSELoss()(score, labels)

            # Add L2 regularization directly in loss calculation
            l2_lambda = 0.001
            l2_reg = sum(param.pow(2.0).sum() for param in self.regressor.parameters())
            loss += l2_lambda * l2_reg

        if loss is not None:
            return loss, score
        else:
            return score