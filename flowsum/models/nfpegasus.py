from typing import Union, Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Sigmoid
from transformers import PegasusForConditionalGeneration, PegasusConfig
from transformers.models.pegasus.modeling_pegasus import shift_tokens_right

from flowsum.models.model_output import NFSeq2SeqLMOutput
from flowsum.models.nflatent import NFLatentModel


class NFPegasus(PegasusForConditionalGeneration):
    """
    Notes:
        (1) from_pretrained() is inherited from PreTrainedModel. It is a class method =>
            def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs). So we only need to modify the __init__().


    """

    def __init__(
        self,
        config: PegasusConfig = None,
        q_input_type="avg_embed",
        q_hidden_dims: List[int] = [200, 200],
        q_act="tanh",
        q_dropout=0.1,
        nf_name="planar",
        nf_latent_size=300,
        nf_num_layers=4,
        nf_loss_weight=0.75,
        beta_vae_constant=0.0,
    ):
        """
        Args:
            q_input_type: the type of input into the latent model, could be one of ["avg_embed", "bows"]
            q_hidden_dims: the hidden dimensions of the q inference network, should be a list of integers.

        config.d_model is the hidden size of the hidden states.
        self.model.shared.num_embeddings == config.vocab_size
        """
        super().__init__(config)

        self.embeddings = self.get_encoder().embed_tokens.weight  # (96103, 1024)

        assert q_input_type in [
            "avg_embed",
            "bows",
        ], "We currently only support ['avg_embed', 'bows'] as q_input_type."
        self.q_input_type = q_input_type
        self.nf_loss_weight = nf_loss_weight
        self.beta_vae_constant = beta_vae_constant
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

        self.nf_linear = nn.Linear(nf_latent_size, config.d_model, bias=True)
        self.gate = nn.Linear(config.d_model * 2, config.d_model * 2, bias=True)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        decoder_bows: torch.FloatTensor = None,  # use the same decoder_ prefix as bert2bert to avoid data regeneration
    ) -> Union[Tuple, NFSeq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # last layer of hidden states, (batch_size, dec_max_length, hidden_size)
        sequence_output = outputs[0]

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

        lm_logits = (
            self.lm_head(sequence_output) + self.final_logits_bias
        )  # (batch_size, dec_max_length, vocab_size)

        masked_lm_loss = None
        perplexity = None
        nf_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )
            perplexity = torch.exp(masked_lm_loss)
            nf_loss = self.nf_loss_weight * torch.abs(
                log_q.mean() - log_prior.mean() - self.beta_vae_constant
            )  # ref: https://aclanthology.org/D19-5612/
            masked_lm_loss += nf_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return NFSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            perplexity=perplexity,
            nf_loss=nf_loss,
            gate_score=gate_score.mean(),
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_bows=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "decoder_bows": decoder_bows,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
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
