# Modified from Hugging Face's Diffusers library
# Original source:
# https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
# Modifications by Letian Zhou (CoDA), 2025

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    EXAMPLE_DOC_STRING,
    rescale_noise_cfg,
    retrieve_timesteps,
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.utils.doc_utils import replace_example_docstring

EXAMPLE_DOC_STRING = """
    Examples:
        ```
        ```
"""

class LatentCollector:
    def __init__(self):
        self.latents_history: List[torch.FloatTensor] = []

    def __call__(self, i: int, t: int, latents: torch.FloatTensor):
        # 注意：新版的回调系统更复杂，这里仅保留核心逻辑
        # 如果要使用新版回调，需要适配 callback_on_step_end
        self.latents_history.append(latents.cpu())

    def get_latents_history(self) -> List[torch.FloatTensor]:
        return self.latents_history


class CoDA_SDXL(StableDiffusionXLPipeline):
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable] = None,  # 适配新回调
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],

            # CoDA Custom Parameters
            represent_latent: torch.Tensor = None,
            guideTPercent: float = 0.5,
            CoDA_guidance_scale: float = 0.1,
            conflict_projection_alpha: float = 0.0,
            conflict_projection_kappa_cap: Optional[float] = None,
            guidance_metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Examples:
        """

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct.
        self.check_inputs(
            prompt, prompt_2, height, width, negative_prompt, negative_prompt_2,
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt, prompt_2=prompt_2, device=device, num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance, negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size, negative_crops_coords_top_left, negative_target_size,
                dtype=prompt_embeds.dtype, text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        if self._denoising_end is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self._denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps_end = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps_end]

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        ##################################################################################################
        # Prepare the represent_latent
        ##################################################################################################
        if CoDA_guidance_scale > 0.0 and represent_latent is not None:
            represent_latent = represent_latent.to(device=latents.device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            ##################################################################################################
            # Calculate the stop time to determine the PIS.
            ##################################################################################################
            stop_idx = min(int(len(timesteps) * guideTPercent), len(timesteps) - 1)
            calculated_stop_t_value = timesteps[stop_idx].item()

            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond, cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs, return_dict=False,
                )[0]

                ##################################################################################################
                # CoDA core implementation: Introduce an additional noise correction term.
                ##################################################################################################
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text_ori = noise_pred.chunk(2)
                else:
                    noise_pred_uncond = torch.zeros_like(noise_pred)
                    noise_pred_text_ori = noise_pred

                if CoDA_guidance_scale > 0.0 and t.item() > calculated_stop_t_value:
                    current_latent_cond = latent_model_input.chunk(2)[1] if self.do_classifier_free_guidance else latent_model_input

                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
                    sqrt_one_minus_alpha_prod_t = torch.sqrt(1.0 - alpha_prod_t)

                    ##################################################################################################
                    # Analytically derive the predicted noise-free latent based on the conditional noise prediction.
                    ##################################################################################################
                    current_xstart_cond = (current_latent_cond - sqrt_one_minus_alpha_prod_t * noise_pred_text_ori) / sqrt_alpha_prod_t

                    if current_xstart_cond.shape != represent_latent.shape:
                        raise ValueError(f"Shape mismatch: current_xstart_cond {current_xstart_cond.shape} "
                                         f"vs represent_latent {represent_latent.shape}.")

                    ##################################################################################################
                    # Latent space correction term
                    ##################################################################################################
                    pix_diff = represent_latent - current_xstart_cond
                    current_sigma_t_for_guidance = self.scheduler.sigmas[i]
                    pix_guide_mark = pix_diff * CoDA_guidance_scale * current_sigma_t_for_guidance

                    ##################################################################################################
                    # Convert the latent space correction term to the noise space.
                    ##################################################################################################
                    delta_epsilon_text = - (sqrt_alpha_prod_t / sqrt_one_minus_alpha_prod_t) * pix_guide_mark

                    # Remove only the component of image guidance that opposes text guidance.
                    # alpha=0 preserves CoDA exactly; alpha=1 makes a conflicting image
                    # direction orthogonal to the text direction.
                    text_direction = noise_pred_text_ori - noise_pred_uncond
                    pre_projection_delta = delta_epsilon_text
                    if conflict_projection_alpha > 0.0 or conflict_projection_kappa_cap is not None:
                        reduce_dims = tuple(range(1, text_direction.ndim))
                        dot_product = torch.sum(
                            pre_projection_delta.float() * text_direction.float(), dim=reduce_dims
                        )
                        text_norm_sq = torch.sum(text_direction.float().square(), dim=reduce_dims).clamp_min(1e-12)
                        broadcast_shape = (-1,) + (1,) * (text_direction.ndim - 1)
                        if conflict_projection_kappa_cap is not None:
                            raw_kappa = (-dot_product).clamp_min(0.0) / text_norm_sq
                            excess_kappa = (raw_kappa - conflict_projection_kappa_cap).clamp_min(0.0)
                            delta_epsilon_text = pre_projection_delta + (
                                excess_kappa.reshape(broadcast_shape).to(text_direction.dtype)
                                * text_direction
                            )
                        else:
                            conflicting_coefficient = (dot_product / text_norm_sq).clamp_max(0.0)
                            delta_epsilon_text = pre_projection_delta - (
                                conflict_projection_alpha
                                * conflicting_coefficient.reshape(broadcast_shape).to(text_direction.dtype)
                                * text_direction
                            )
                    noise_pred_text = noise_pred_text_ori + delta_epsilon_text

                    if guidance_metrics_callback is not None:
                        text_flat = text_direction.float().reshape(text_direction.shape[0], -1)
                        image_flat = delta_epsilon_text.float().reshape(delta_epsilon_text.shape[0], -1)
                        pre_image_flat = pre_projection_delta.float().reshape(pre_projection_delta.shape[0], -1)
                        text_norm = torch.linalg.vector_norm(text_flat, dim=1)
                        image_norm = torch.linalg.vector_norm(image_flat, dim=1)
                        pre_image_norm = torch.linalg.vector_norm(pre_image_flat, dim=1)
                        cosine = torch.nn.functional.cosine_similarity(text_flat, image_flat, dim=1)
                        q_value = text_norm / image_norm.clamp_min(1e-12)
                        pre_q_value = text_norm / pre_image_norm.clamp_min(1e-12)
                        dot_product = torch.sum(text_flat * image_flat, dim=1)
                        conflict_projection_ratio = (-dot_product).clamp_min(0.0) / text_norm.square().clamp_min(1e-12)
                        pre_cosine = torch.nn.functional.cosine_similarity(text_flat, pre_image_flat, dim=1)
                        pre_dot_product = torch.sum(text_flat * pre_image_flat, dim=1)
                        pre_conflict_projection_ratio = (
                            (-pre_dot_product).clamp_min(0.0) / text_norm.square().clamp_min(1e-12)
                        )
                        guidance_metrics_callback({
                            "step_index": i,
                            "timestep": int(t.item()),
                            "sigma": float(current_sigma_t_for_guidance.item()),
                            "text_norm_l2": float(text_norm[0].item()),
                            "image_norm_l2": float(image_norm[0].item()),
                            "q_text_over_image": float(q_value[0].item()),
                            "cosine_similarity": float(cosine[0].item()),
                            "conflict_projection_ratio": float(conflict_projection_ratio[0].item()),
                            "pre_projection_cosine_similarity": float(pre_cosine[0].item()),
                            "pre_projection_conflict_ratio": float(pre_conflict_projection_ratio[0].item()),
                            "pre_projection_image_norm_l2": float(pre_image_norm[0].item()),
                            "pre_projection_q_text_over_image": float(pre_q_value[0].item()),
                            "conflict_projection_alpha": float(conflict_projection_alpha),
                            "conflict_projection_kappa_cap": conflict_projection_kappa_cap,
                        })

                else:
                    noise_pred_text = noise_pred_text_ori

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        final_latents_for_return = latents.clone()

        # 9. Post-processing
        if not output_type == "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device,
                                                                                              latents.dtype)
                latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device,
                                                                                            latents.dtype)
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                print(f"!!! WARNING: latents in base pipeline contains NaN or Inf after denoising!")

            image = latents

        if not output_type == "latent":
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, final_latents_for_return)

        return StableDiffusionXLPipelineOutput(images=image), final_latents_for_return
