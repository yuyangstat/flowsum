from typing import Union, Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Sigmoid
from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.file_utils import ModelOutput

from flowsum.models.model_output import NFSeq2SeqLMOutput
from flowsum.models.nflatent import NFLatentModel


class NFBart(BartForConditionalGeneration):
    def __init__(
        self,
        config: BartConfig,
        q_input_type="avg_embed",
        q_hidden_dims: List[int] = [200, 200],
        q_act="tanh",
        q_dropout=0.1,
        nf_name="planar",
        nf_latent_size=300,
        nf_num_layers=4,
        nf_loss_weight=0.75,
        beta_vae_constant=0.0,
        output_kld=False,
    ):
        """
        Args:
            q_input_type: the type of input into the latent model, could be one of ["avg_embed", "bows"]
            q_hidden_dims: the hidden dimensions of the q inference network, should be a list of integers.
            nf_num_layers: when set as 0, the nf model degenerates to the traditional VI where the latent distribution
                is characterized by diagonal Gaussian.
            output_kld: whether to output the KL divergence. If True, nf_loss = kld, otherwise, nf_loss = abs(klk - beta_vae_constant)

        Notes:
            (1) self.model.shared.weight.shape[-1] == self.embed_size
            (2) Out of unknown reasons, if we use self.embeddings = self.model.shared.weight, then
                self.embeddings will become one dim smaller, therefore, we directly use self.model.shared.weight
                in forward(). This temporarily solves the problem.
                - Also, note that config.vocab_size = 50264, which is same as the one given by self.embeddings,
                    but the shape of self.model.shared.weight is (50265, 1024) and tokenizer.vocab_size = 50265 (decoder_bow's dim).
                - This issue also leads to problems when loading from checkpoints. If I initialize the model
                    with model_name = "nf-bart-finetune", then it will show size mismatch: "size mismatch
                    for model.shared.weight: copying a param with shape torch.Size([50265, 1024]) from
                    checkpoint, the shape in current model is torch.Size([50264, 1024])." But when I
                    initialize the model with model_name = "nf-bart", the problem is gone. Still need to
                    check the performance before concluding whether this can solve the issue.
                - This issue pops out again when I tried to use BOWs as the input to the q_net. Although
                    self.model.shared.weight.shape[0] gives 50265, but after initializing the model, the input
                    dimension becomes 50264. For now, I just hard code the input_size to be
                    self.model.shared.weight.shape[0] + 1.
                - This issue pops out again when I tried to use BOWs as the input to the q_net on MSI. The
                    chekpoint gives dimension 50265, whereas the model gives 50266 after the hard coding
                    mentioned above. Therefore, on MSI, I remove the hard code and set the input to be
                    self.model.shared.weight.shape[0]. It seems that the problem depends on the system.
        """
        super().__init__(config)

        assert q_input_type in [
            "avg_embed",
            "bows",
        ], "We currently only support ['avg_embed', 'bows'] as q_input_type."
        self.q_input_type = q_input_type
        self.nf_loss_weight = nf_loss_weight
        self.beta_vae_constant = beta_vae_constant
        self.output_kld = output_kld
        self.nf_latent_model = NFLatentModel(
            input_size=self.model.shared.weight.shape[-1]
            if q_input_type == "avg_embed"
            else self.model.shared.weight.shape[0],
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
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        decoder_bows: torch.FloatTensor = None,
        decoder_latent: torch.FloatTensor = None,
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
        if decoder_latent is None:
            assert (
                decoder_bows is not None
            ), "Either decoder_bows or decoder_latent should be provided."
            nf_input = (
                torch.mm(
                    decoder_bows, self.model.shared.weight
                )  # (batch_size, embed_size)
                if self.q_input_type == "avg_embed"
                else decoder_bows  # (batch_size, vocab_size)
            )
            nf_latent, log_q, log_prior = self.nf_latent_model(nf_input)

            decoder_latent = self.nf_linear(
                nf_latent
            )  # (batch_size, decoder_hidden_size)

        # add a gated fusion mechanism to the last layer of hidden states
        decoder_latent = decoder_latent.unsqueeze(1)  # (batch_size, 1, hidden_size)
        sigmoid_fct = Sigmoid()
        gate_score = sigmoid_fct(
            self.gate(
                torch.cat(
                    (
                        sequence_output,
                        decoder_latent.repeat(1, sequence_output.shape[1], 1),
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
        ) * sequence_output + gate_score * decoder_latent  # (batch_size, dec_max_length, hidden_size)
        # =============================

        lm_logits = self.lm_head(sequence_output)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        perplexity = None
        kl_divergence = None
        nf_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )
            perplexity = torch.exp(masked_lm_loss)
            kl_divergence = log_q.mean() - log_prior.mean()
            nf_loss = self.nf_loss_weight * torch.abs(
                kl_divergence - self.beta_vae_constant
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
            nf_loss=kl_divergence if self.output_kld else nf_loss,
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
        decoder_latent=None,
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
            "decoder_latent": decoder_latent,
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

        if model_kwargs.get("decoder_latent") is not None:
            model_kwargs["decoder_latent"] = model_kwargs[
                "decoder_latent"
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

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = (
            model_input_name if model_input_name is not None else self.main_input_name
        )
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        # 4. prepare decoder_latent from model kwargs
        decoder_bows = model_kwargs.get("decoder_bows")
        nf_input = (
            torch.mm(decoder_bows, self.model.shared.weight)  # (batch_size, embed_size)
            if self.q_input_type == "avg_embed"
            else decoder_bows  # (batch_size, vocab_size)
        )
        nf_latent, _, _ = self.nf_latent_model(nf_input)
        decoder_latent = self.nf_linear(nf_latent)  # (batch_size, decoder_hidden_size)
        model_kwargs["decoder_latent"] = decoder_latent

        return model_kwargs
