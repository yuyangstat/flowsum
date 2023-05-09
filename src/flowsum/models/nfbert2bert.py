import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List
from transformers import (
    EncoderDecoderModel,
    EncoderDecoderConfig,
    AutoModel,
    AutoConfig,
)

# from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel


from flowsum.models.nfbertlmhead import NFBertLMHeadModel
from flowsum.models.model_output import NFSeq2SeqLMOutput
from flowsum.models.nflatent import NFLatentModel


class NFBert2Bert(EncoderDecoderModel):
    def __init__(
        self,
        config=None,
        encoder=None,
        decoder=None,
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

        Note:
            (1) The nf_loss is related to nf_latent (z_K) before the linear mapping, so we consider
                self.nf_linear to be part of the decoder instead of the nf_latent_model. This consideration
                matters if we use looped optimization. If we incorporate the nf_linear and the gate mechanism
                into the variational parameters, then they won't be updated since their gradient w.r.t. nf_loss
                is 0.
        """
        assert config is not None or (
            encoder is not None and decoder is not None
        ), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(
                encoder.config, decoder.config
            )
        else:
            assert isinstance(
                config, self.config_class
            ), f"config: {config} has to be of type {self.config_class}"

        super(EncoderDecoderModel, self).__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            decoder = NFBertLMHeadModel(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        assert (
            self.encoder.get_output_embeddings() is None
        ), "The encoder {} should not have a LM Head. Please use a model without LM Head"

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()

        self.embeddings = self.encoder.get_input_embeddings().weight

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
        self.nf_linear = nn.Linear(
            nf_latent_size, self.decoder.config.hidden_size, bias=True
        )

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        r""" """

        kwargs_encoder = {
            argument[len("encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        kwargs_nf = {
            argument: value
            for argument, value in kwargs.items()
            if argument.startswith("nf_") or argument.startswith("q_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            assert (
                encoder_pretrained_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"

            if "config" not in kwargs_encoder:

                encoder_config = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path
                )
                if (
                    encoder_config.is_decoder is True
                    or encoder_config.add_cross_attention is True
                ):

                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder
            )

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            assert (
                decoder_pretrained_model_name_or_path is not None
            ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"

            if "config" not in kwargs_decoder:

                decoder_config = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path
                )
                if (
                    decoder_config.is_decoder is False
                    or decoder_config.add_cross_attention is False
                ):

                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            decoder = NFBertLMHeadModel.from_pretrained(
                decoder_pretrained_model_name_or_path, **kwargs_decoder
            )

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config, **kwargs
        )
        return cls(encoder=encoder, decoder=decoder, config=config, **kwargs_nf)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoder_bows=None,  # normalized bag of words, used to generate average embeddings, (batch_size, vocab_size)
        **kwargs,
    ):
        """ """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("decoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs[0]

        # ============= Added by Yu ========
        # [DONE #2] process input average embeddings with normalizing flows: output z_0, z_k, logdet, prior eval, nf_loss
        # [TODO] compare the cost of using bows vs. all_input_ids and embeddding()
        nf_input = (
            torch.mm(decoder_bows, self.embeddings)  # (batch_size, embed_size)
            if self.q_input_type == "avg_embed"
            else decoder_bows  # (batch_size, vocab_size)
        )
        nf_latent, log_q, log_prior = self.nf_latent_model(nf_input)

        nf_latent = self.nf_linear(nf_latent)  # (batch_size, decoder_hidden_size)
        # ===================================

        # [DONE #3] modify decoder to incorporate z

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            nf_latent=nf_latent,  # latent vector from normalizing flows: z_K
            **kwargs_decoder,
        )

        perplexity = None
        nf_loss = None
        if decoder_outputs.loss is not None:
            perplexity = torch.exp(decoder_outputs.loss)
            nf_loss = self.nf_loss_weight * (log_q.mean() - log_prior.mean())
            decoder_outputs.loss += nf_loss

        if not return_dict:
            return decoder_outputs + encoder_outputs

        # [DONE #4] modify the loss to be decocer_outputs.loss + nf_loss
        return NFSeq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            perplexity=perplexity,
            nf_loss=nf_loss,
            gate_score=decoder_outputs.gate_score,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_bows=None,
        **kwargs,
    ):
        """Used in generate() -> greedy_search(), sample(), beam_search(), beam_sample(),group_beam_search().
        - model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        - **model_inputs will then be passed to forward() to get outputs.
        - model_kwargs represents additional model specific keyword arguments will be forwarded to the
            :obj:`forward` function of the model. If model is an encoder-decoder model the kwargs should
            include :obj:`encoder_outputs`.
        """
        decoder_inputs = self.decoder.prepare_inputs_for_generation(
            input_ids, past=past
        )
        decoder_attention_mask = (
            decoder_inputs["attention_mask"]
            if "attention_mask" in decoder_inputs
            else None
        )
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
            "decoder_bows": decoder_bows,
        }
        return input_dict

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

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        """
        According to https://colab.research.google.com/drive/1Ekd5pUeCX7VOrMx94_czTkwNtLN32Uyu?usp=sharing,
          "because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. "

        This is modified based upone EncoderDecoderModel's implementation, without shifting.
        """
        shifted_input_ids = labels.new_zeros(labels.shape)
        shifted_input_ids[:, :] = labels[:, :].clone()
        if self.config.decoder_start_token_id is None:
            raise ValueError(
                "Make sure to set the decoder_start_token_id attribute of the model's configuration."
            )
        shifted_input_ids[:, 0] = self.config.decoder_start_token_id

        if self.config.pad_token_id is None:
            raise ValueError(
                "Make sure to set the pad_token_id attribute of the model's configuration."
            )
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(
            shifted_input_ids == -100, self.config.pad_token_id
        )

        return shifted_input_ids
