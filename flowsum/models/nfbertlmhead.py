import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Sigmoid

from transformers import BertLMHeadModel
from flowsum.models.model_output import NFCausalLMOutputWithCrossAttentions


class NFBertLMHeadModel(BertLMHeadModel):
    def __init__(self, config):
        """
        The from_pretrained() function will first initialize based on __init__(), then match the ones
        that are available in the pretrained weights.

        Notes:
            (1) Refine gating mechanism is introduced to handle the gradient vanishing problem.
                    Reference: Improving the gating mechanism of recurrent neural networks (https://arxiv.org/abs/1910.09890).
        """
        super().__init__(config)
        # initialize the gated fusion mechanism
        self.gate = nn.Linear(
            config.hidden_size * 2, config.hidden_size * 2, bias=True
        )  # elementwise gate
        nn.init.constant_(
            self.gate.bias.data[: config.hidden_size], torch.logit(torch.tensor(0.05))
        )  # such that the initial gate score is close to 0.05, and the initial refine score is close to 0.5

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        nf_latent=None,  # latent output by normalizing flows
    ):
        r""" """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # last layer of hidden states, (batch_size, dec_max_length, hidden_size)
        sequence_output = outputs[0]

        # ======== Added by Yu ========
        # add a gated fusion mechanism to the last layer of hidden states
        nf_latent = nf_latent.unsqueeze(1)  # (batch_size, 1, hidden_size)
        sigmoid_fct = Sigmoid()
        gate_score = sigmoid_fct(
            self.gate(
                torch.cat(
                    (
                        sequence_output,
                        nf_latent.repeat(1, sequence_output.shape[1], 1),
                    ),
                    dim=-1,
                ),  # (batch_size, dec_max_length, 2 * hidden_size)
            )  # (batch_size, dec_max_length, 2 * hidden_size)
        )  # (batch_size, dec_max_length, 2 * hidden_size)

        ## refine gating mechanism
        gate_score, refine_score = torch.split(
            gate_score, split_size_or_sections=gate_score.shape[-1] // 2, dim=-1
        )  # (batch_size, dec_max_length, hidden_size)
        gate_score = gate_score + gate_score * (1 - gate_score) * (
            2 * refine_score - 1
        )  # (batch_size, dec_max_length, hidden_size)

        sequence_output = (
            1 - gate_score
        ) * sequence_output + gate_score * nf_latent  # (batch_size, dec_max_length, hidden_size)
        # =============================

        prediction_scores = self.cls(
            sequence_output
        )  # output logits, not probabilities

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = (
                CrossEntropyLoss()
            )  # the input logits don't have to be normalized
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return NFCausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            gate_score=gate_score.mean(),  # one single valued tensor
        )
