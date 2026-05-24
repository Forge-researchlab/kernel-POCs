"""Patch adapter for Qwen CausalLM fused linear + cross entropy."""
from __future__ import annotations


def _unpack_cross_entropy_result(result):
    if hasattr(result, "loss"):
        return result.loss, getattr(result, "token_accuracy", None), getattr(result, "predicted_tokens", None)
    if isinstance(result, tuple):
        loss = result[0]
        token_accuracy = result[2] if len(result) > 2 else None
        predicted_tokens = result[3] if len(result) > 3 else None
        return loss, token_accuracy, predicted_tokens
    return result, None, None


def make_fused_linear_ce_forward(module, config):
    """Qwen CausalLM.forward -> fused lm_head + CE when training with labels."""
    from forge.kernels.cross_entropy import forge_fused_linear_cross_entropy
    from transformers.modeling_outputs import CausalLMOutputWithPast

    def forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        return_dict = kwargs.pop("return_dict", None)
        if return_dict is None:
            return_dict = getattr(module.config, "use_return_dict", True)
        shift_labels = kwargs.pop("shift_labels", None)
        skip_logits = kwargs.pop("skip_logits", None)
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        ignore_index = kwargs.pop("ignore_index", -100)
        label_smoothing = kwargs.pop("label_smoothing", 0.0)
        final_logit_softcapping = kwargs.pop("final_logit_softcapping", None)
        accum_dtype = kwargs.pop("accum_dtype", None)
        return_token_accuracy = kwargs.pop("return_token_accuracy", False)
        return_predicted_tokens = kwargs.pop("return_predicted_tokens", False)

        output_attentions_ = output_attentions if output_attentions is not None else module.config.output_attentions
        output_hidden_states_ = (
            output_hidden_states if output_hidden_states is not None else module.config.output_hidden_states
        )
        outputs = module.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions_,
            output_hidden_states=output_hidden_states_,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if skip_logits and labels is None and shift_labels is None:
            raise ValueError("skip_logits is True, but labels and shift_labels are None")
        if skip_logits is None:
            skip_logits = module.training and (labels is not None or shift_labels is not None)

        loss = None
        logits = None
        token_accuracy = None
        predicted_tokens = None
        can_use_fused = skip_logits and (isinstance(logits_to_keep, int) and logits_to_keep == 0)

        if can_use_fused and (labels is not None or shift_labels is not None):
            if shift_labels is None:
                shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            else:
                shift_hidden_states = hidden_states
            shift_hidden_states = shift_hidden_states.view(-1, module.config.hidden_size)
            shift_labels = shift_labels.view(-1).to(shift_hidden_states.device)

            reduction = "sum" if num_items_in_batch is not None else "mean"
            result = forge_fused_linear_cross_entropy(
                shift_hidden_states,
                module.lm_head.weight,
                shift_labels,
                bias=getattr(module.lm_head, "bias", None),
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
                reduction=reduction,
                softcap=final_logit_softcapping,
                accum_dtype=accum_dtype,
                return_token_accuracy=return_token_accuracy,
                return_predicted_tokens=return_predicted_tokens,
            )
            loss, token_accuracy, predicted_tokens = _unpack_cross_entropy_result(result)
            if num_items_in_batch is not None:
                loss = loss / num_items_in_batch
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = module.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None or shift_labels is not None:
                loss = module.loss_function(
                    logits=logits,
                    labels=labels,
                    shift_labels=shift_labels,
                    vocab_size=module.config.vocab_size,
                    num_items_in_batch=num_items_in_batch,
                    ignore_index=ignore_index,
                    label_smoothing=label_smoothing,
                    **kwargs,
                )

        if not return_dict:
            output = (logits,) + tuple(outputs[1:])
            output = (loss,) + output if loss is not None else output
            if token_accuracy is not None:
                output = output + (token_accuracy,)
            if predicted_tokens is not None:
                output = output + (predicted_tokens,)
            return output

        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        if token_accuracy is not None:
            setattr(result, "token_accuracy", token_accuracy)
        if predicted_tokens is not None:
            setattr(result, "predicted_tokens", predicted_tokens)
        return result

    return forward
