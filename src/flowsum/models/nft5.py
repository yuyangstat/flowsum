from typing import Union, Optional, Tuple, Dict, Any, List
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Sigmoid
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput

from nfsummary.models.model_output import NFSeq2SeqLMOutput
from nfsummary.models.nflatent import NFLatentModel


# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class NFT5(T5ForConditionalGeneration):
    def __init__(
        self,
        config: T5Config,
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

        config.d_model is the hidden size of the hidden states.
        self.model.shared.num_embeddings == config.vocab_size

        [#TODO] Similar to BART, vocab_size = 32128, decoder_bows's dim = 32100 (since tokenizer.vocab_size = 32100),
            self.embeddings.shape = self.shared.weight.shape = 32128
            - This problem cannot be solved like in NFBart.
        """
        super().__init__(config)

        self.embeddings = self.shared.weight  #

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

        self.nf_linear = nn.Linear(nf_latent_size, config.d_model, bias=True)
        self.gate = nn.Linear(config.d_model * 2, config.d_model * 2, bias=True)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        decoder_bows: torch.FloatTensor = None,
    ) -> Union[Tuple[torch.FloatTensor], NFSeq2SeqLMOutput]:
        """"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # last layer of hidden states, (batch_size, dec_max_length, hidden_size)
        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

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

        lm_logits = self.lm_head(sequence_output)

        loss = None
        perplexity = None
        nf_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            perplexity = torch.exp(loss)
            nf_loss = self.nf_loss_weight * (log_q.mean() - log_prior.mean())
            loss += nf_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return NFSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            perplexity=perplexity,
            nf_loss=nf_loss,
            gate_score=gate_score.mean(),
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
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
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "decoder_bows": decoder_bows,
            "use_cache": use_cache,
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
