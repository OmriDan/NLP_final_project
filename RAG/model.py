import torch

class RAGQuestionDifficultyRegressor(torch.nn.Module):
    def __init__(self, encoder_model):
        super().__init__()
        self.encoder = encoder_model
        # Single output node with sigmoid activation
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.hidden_size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # Filter out unexpected kwargs that the encoder doesn't accept
        # Keep only standard transformer kwargs
        accepted_kwargs = ['head_mask', 'inputs_embeds', 'output_attentions',
                           'output_hidden_states', 'return_dict', 'token_type_ids']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **filtered_kwargs)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        score = self.regressor(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(score, labels)

        return {"loss": loss, "score": score} if loss is not None else {"score": score}