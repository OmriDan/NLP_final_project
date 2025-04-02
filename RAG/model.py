import torch


class RAGQuestionDifficultyRegressor(torch.nn.Module):
    def __init__(self, encoder_model, dropout_rate=0.2):
        super().__init__()
        self.encoder = encoder_model
        hidden_size = self.encoder.config.hidden_size

        # More sophisticated regressor with dropout
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.GELU(),  # Better than ReLU for transformer outputs
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size // 4, 1),
            torch.nn.Sigmoid()
        )

        # Attention pooling instead of just CLS token
        self.attention_pool = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        accepted_kwargs = ['head_mask', 'inputs_embeds', 'output_attentions',
                           'output_hidden_states', 'return_dict', 'token_type_ids']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **filtered_kwargs)
        hidden_states = outputs.last_hidden_state

        # Use attention pooling instead of CLS token
        attention_scores = self.attention_pool(hidden_states).squeeze(-1)
        attention_weights = torch.softmax(attention_scores.masked_fill(~attention_mask.bool(), -10000.0), dim=1)
        pooled_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)

        score = self.regressor(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            # Combine MSE with Huber loss for robustness
            mse_loss = torch.nn.MSELoss()(score, labels)
            huber_loss = torch.nn.SmoothL1Loss()(score, labels)
            loss = 0.7 * mse_loss + 0.3 * huber_loss

        # For compatibility with Trainer, return score as the first value
        # This fixes the prediction format for compute_metrics
        if loss is not None:
            return loss, score
        else:
            return score