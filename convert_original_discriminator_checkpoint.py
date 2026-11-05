"""Convert VITS discriminator checkpoint and add it to an already converted VITS checkpoint."""

import argparse
import os

import torch

from transformers.models.vits.modeling_vits import VitsModel
from transformers.models.vits.tokenization_vits import VitsTokenizer

from huggingface_hub import hf_hub_download

from utils.feature_extraction_vits import VitsFeatureExtractor
from utils.configuration_vits import VitsConfig, logging
from utils.modeling_vits_training import VitsDiscriminator, VitsModelForPreTraining

logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.vits")


MAPPING = {
    "conv_post": "final_conv",
}
TOP_LEVEL_KEYS = []
IGNORE_KEYS = []


def fix_weight_norm_loading(generator, generator_checkpoint_path):
    """
    With some torch/transformers versions, the weight-norm weights (flow + posterior encoder)
    are NOT loaded by from_pretrained: the file uses the names weight_g / weight_v while the
    model expects parametrizations.weight.original0 / original1. Those layers then stay random.
    This function reloads them under the right names.
    """
    if not any("parametrizations" in k for k in generator.state_dict()):
        return  # model uses the old weight_norm naming, nothing to fix

    from safetensors.torch import load_file

    if os.path.isdir(generator_checkpoint_path):
        st_file = os.path.join(generator_checkpoint_path, "model.safetensors")
    else:
        st_file = hf_hub_download(repo_id=generator_checkpoint_path, filename="model.safetensors")

    fixed = {}
    for k, v in load_file(st_file).items():
        k = k.replace("weight_g", "parametrizations.weight.original0")
        k = k.replace("weight_v", "parametrizations.weight.original1")
        fixed[k] = v

    res = generator.load_state_dict(fixed, strict=False)
    print("weight-norm fix -> missing:", len(res.missing_keys), "unexpected:", len(res.unexpected_keys))
    if res.missing_keys:
        raise RuntimeError(f"generator weights still missing: {res.missing_keys[:5]}")


@torch.no_grad()
def convert_checkpoint(
    language_code,
    pytorch_dump_folder_path,
    checkpoint_path=None,
    generator_checkpoint_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if language_code is not None:
        checkpoint_path = hf_hub_download(repo_id="facebook/mms-tts", subfolder=f"full_models/{language_code}", filename="D_100000.pth")
        generator_checkpoint_path = f"facebook/mms-tts-{language_code}"

    config = VitsConfig.from_pretrained(generator_checkpoint_path)
    generator = VitsModel.from_pretrained(generator_checkpoint_path)
    fix_weight_norm_loading(generator, generator_checkpoint_path)

    discriminator = VitsDiscriminator(config)

    for disc in discriminator.discriminators:
        disc.apply_weight_norm()

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # load weights

    state_dict = checkpoint["model"]

    for k, v in list(state_dict.items()):
        for old_layer_name in MAPPING:
            new_k = k.replace(old_layer_name, MAPPING[old_layer_name])

        state_dict[new_k] = state_dict.pop(k)

    extra_keys = set(state_dict.keys()) - set(discriminator.state_dict().keys())
    extra_keys = {k for k in extra_keys if not k.endswith(".attn.bias")}
    missing_keys = set(discriminator.state_dict().keys()) - set(state_dict.keys())
    missing_keys = {k for k in missing_keys if not k.endswith(".attn.bias")}
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    discriminator.load_state_dict(state_dict, strict=False)
    n_params = discriminator.num_parameters(exclude_embeddings=True)
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    for disc in discriminator.discriminators:
        disc.remove_weight_norm()

    model = VitsModelForPreTraining(config)

    # load weights
    model.text_encoder = generator.text_encoder
    model.flow = generator.flow
    model.decoder = generator.decoder
    model.duration_predictor = generator.duration_predictor
    model.posterior_encoder = generator.posterior_encoder

    if config.num_speakers > 1:
        model.embed_speaker = generator.embed_speaker

    model.discriminator = discriminator
    tokenizer = VitsTokenizer.from_pretrained(generator_checkpoint_path, verbose=False)
    feature_extractor = VitsFeatureExtractor(sampling_rate=model.config.sampling_rate, feature_size=80)

    model.save_pretrained(pytorch_dump_folder_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        feature_extractor.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language_code",
        default=None,
        type=str,
        help="If set, indicates the language code of the MMS checkpoint to convert. "
        "In that case, it will automatically create the right MMS checkpoint in the HF required format. "
        "If used, `checkpoint_path` and `generator_checkpoint_path` are ignored.",
    )

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Local path to original discriminator checkpoint. Ignored if `language_code` is used."
    )
    parser.add_argument(
        "--generator_checkpoint_path", default=None, type=str, help="Path to the HF generator (VitsModel). Ignored if `language_code` is used."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the HF hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.language_code,
        args.pytorch_dump_folder_path,
        args.checkpoint_path,
        args.generator_checkpoint_path,
        args.push_to_hub,
    )