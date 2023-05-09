from typing import Union, Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Sigmoid
from transformers import GPT2LMHeadModel


from flowsum.models.model_output import NFCausalLMOutputWithCrossAttentions
from flowsum.models.nflatent import NFLatentModel


class NFGPT2(GPT2LMHeadModel):
    def __init__(
        self,
        config,
        q_input_type="avg_embed",
        q_hidden_dims: List[int] = [200, 200],
        q_act="tanh",
        q_dropout=0.1,
        nf_name="planar",
        nf_latent_size=300,
        nf_num_layers=4,
        nf_loss_weight=0.75,
    ):
        """
        Args:
            q_input_type: the type of input into the latent model, could be one of ["avg_embed", "bows"]
            q_hidden_dims: the hidden dimensions of the q inference network, should be a list of integers.
            nf_num_layers: when set as 0, the nf model degenerates to the traditional VI where the latent distribution
                is characterized by diagonal Gaussian.
        """
        super().__init__(config)

        self.embeddings = self.get_input_embeddings().weight
        self.prompt_token_id = (
            33409  # AutoTokenizer.from_pretrained("gpt2").encode(">>>")
        )

        assert q_input_type in [
            "avg_embed",
            "bows",
        ], "We currently only support ['avg_embed', 'bows'] as q_input_type."
        self.q_input_type = q_input_type
        self.nf_loss_weight = nf_loss_weight
        self.nf_latent_model = NFLatentModel(
            input_size=self.embeddings.shape[-1]
            if q_input_type == "avg_embed"
            else self.embeddings.shape[0],
            q_hidden_dims=q_hidden_dims,
            q_act=q_act,
            q_dropout=q_dropout,
            nf_name=nf_name,
            nf_latent_size=nf_latent_size,
            nf_num_layers=nf_num_layers,
        )

        self.nf_linear = nn.Linear(nf_latent_size, config.n_embd, bias=True)
        self.gate = nn.Linear(config.n_embd * 2, config.n_embd * 2, bias=True)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoder_bows=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        prompt_ids = (
            torch.ones(input_ids.shape[0], 1).type_as(input_ids) * self.prompt_token_id
        )
        input_ids = torch.cat((input_ids, prompt_ids, labels), -1)

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # ======== Added by Yu ========
        nf_input = (
            torch.mm(decoder_bows, self.embeddings)  # (batch_size, embed_size)
            if self.q_input_type == "avg_embed"
            else decoder_bows  # (batch_size, vocab_size)
        )
        nf_latent, log_q, log_prior = self.nf_latent_model(nf_input)

        nf_latent = self.nf_linear(nf_latent)  # (batch_size, decoder_hidden_size)

        # add a gated fusion mechanism to the last layer of hidden states
        nf_latent = nf_latent.unsqueeze(1)  # (batch_size, 1, hidden_size)
        sigmoid_fct = Sigmoid()
        gate_score = sigmoid_fct(
            self.gate(
                torch.cat(
                    (
                        hidden_states,
                        nf_latent.repeat(1, hidden_states.shape[1], 1),
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

        hidden_states = (
            1 - gate_score
        ) * hidden_states + gate_score * nf_latent  # (batch_size, dec_max_length, hidden_size)
        # =============================

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        perplexity = None
        nf_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            perplexity = torch.exp(loss)
            nf_loss = self.nf_loss_weight * (log_q.mean() - log_prior.mean())
            loss += nf_loss

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return NFCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            perplexity=perplexity,
            nf_loss=nf_loss,
            gate_score=gate_score.mean(),
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        decoder_bows = kwargs.get("decoder_bows", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "decoder_bows": decoder_bows,
        }

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        if model_kwargs.get("token_type_ids") is not None:
            model_kwargs["token_type_ids"] = model_kwargs[
                "token_type_ids"
            ].repeat_interleave(expand_size, dim=0)

        if model_kwargs.get("attention_mask") is not None:
            model_kwargs["attention_mask"] = model_kwargs[
                "attention_mask"
            ].repeat_interleave(expand_size, dim=0)

        if model_kwargs.get("decoder_bows") is not None:
            model_kwargs["decoder_bows"] = model_kwargs[
                "decoder_bows"
            ].repeat_interleave(expand_size, dim=0)

        if is_encoder_decoder:
            encoder_outputs = model_kwargs.get("encoder_outputs")
            if encoder_outputs is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            encoder_outputs[
                "last_hidden_state"
            ] = encoder_outputs.last_hidden_state.repeat_interleave(expand_size, dim=0)
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs
