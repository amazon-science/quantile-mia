import os

os.environ["HF_DATASETS_CACHE"] = "/dataXL/huggingface/datasets"

import torch
import torchvision.models as tvm
import transformers
from cifar_architectures import (
    ResNet10,
    ResNet10ExtraInputs,
    ResNet18,
    ResNet18ExtraInputs,
    ResNet34,
    ResNet34ExtraInputs,
    ResNet50,
    ResNet50ExtraInputs,
    WideResNet,
)

# from transformers import AutoModelForImageClassification, AutoFeatureExtractor, ResNetForImageClassification, \
#     ResNetConfig, ViTConfig, ViTForImageClassification
from transformers import (
    AutoModelForImageClassification,
    ResNetConfig,
    ViTConfig,
    ViTForImageClassification,
)

RESNET18CONFIG = ResNetConfig(
    depths=[2, 2, 2, 2],
    downsample_in_first_stage=False,
    embedding_size=64,
    hidden_act="relu",
    hidden_sizes=[16, 32, 64],
    layer_type="basic",
    num_channels=3,
)


RESNET50CONFIG = ResNetConfig(
    depths=[3, 4, 6, 3],
    # downsample_in_first_stage=False,
    downsample_in_first_stage=True,
    embedding_size=64,
    hidden_act="relu",
    hidden_sizes=[256, 512, 1024, 2048],
    layer_type="bottleneck",
    num_channels=3,
)

transformers.logging.set_verbosity_error()


class HugginFaceTupleWrapper(torch.nn.Module):
    def __init__(self, model_base, hidden_dims=[], extra_inputs=None):
        super().__init__()
        self.model_base = model_base

        # Replaces the linear layer of the default classifier with an MLP
        if isinstance(self.model_base.classifier, torch.nn.Sequential):
            self.classifier = self.model_base.classifier
            self.model_base.classifier = torch.nn.Identity()
        else:
            prev_size = self.model_base.classifier.in_features
            if extra_inputs is not None:
                prev_size += extra_inputs
            num_classes = self.model_base.classifier.out_features
            mlp_list = []
            for hd in hidden_dims:
                mlp_list.append(torch.nn.Linear(prev_size, hd))
                mlp_list.append(torch.nn.LeakyReLU())  # TODO!
                prev_size = hd
            mlp_list.append(torch.nn.Linear(prev_size, num_classes))
            self.classifier = torch.nn.Sequential(*mlp_list)
            self.model_base.classifier = torch.nn.Identity()

        # self.linear =torch.nn.Linear(embedding_size, num_classes)
        super(HugginFaceTupleWrapper, self).add_module("model_base", self.model_base)
        super(HugginFaceTupleWrapper, self).add_module("classifier", self.classifier)

    def forward(self, input, extra_inputs=None):
        embedding = self.model_base(input).logits
        if extra_inputs is not None:
            assert (
                extra_inputs.shape[0] == embedding.shape[0]
                and extra_inputs.ndim == embedding.ndim
            ), "extra inputs and embedding need to have the same batch dimension"
            embedding = torch.concatenate([embedding, extra_inputs], dim=1)
            # print(embedding.shape)
        logits = self.classifier(embedding)
        return logits

    def freeze_base_model(self):
        for p in self.model_base.parameters():
            p.requires_grad = False
        for p in self.model_base.classifier.parameters():
            p.requires_grad = True

    def unfreeze_base_model(self):
        for p in self.model_base.parameters():
            p.requires_grad = True


def get_huggingface_model(
    model_checkpoint, num_classes=10, hidden_dims=[], extra_inputs=None
):
    if model_checkpoint.startswith("base"):
        configuration = ViTConfig(num_labels=num_classes)
        model_base = ViTForImageClassification(configuration)
        # feature_extractor = None
    else:
        model_base = AutoModelForImageClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    model = HugginFaceTupleWrapper(
        model_base, hidden_dims=hidden_dims, extra_inputs=extra_inputs
    )

    return model


def get_torchvision_model(
    model_name="convnext-tiny", num_classes=10, sample_input=None, hidden_dims=[]
):
    model = None
    if model_name == "convnext-tiny":
        model_fn = tvm.convnext_tiny
        model_weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = model_fn(weights=model_weights)
        if len(hidden_dims):
            prev_size = 768
            mlp_list = []
            for hd in hidden_dims:
                mlp_list.append(torch.nn.Linear(prev_size, hd))
                mlp_list.append(torch.nn.ReLU())
                prev_size = hd
            mlp_list.append(torch.nn.Linear(prev_size, num_classes))
            model.classifier = torch.nn.Sequential(*mlp_list)
        else:
            model.classifier[-1] = torch.nn.Linear(
                in_features=768, out_features=num_classes, bias=True
            )
    else:
        raise NotImplementedError
    return model


def get_fresh_resnet_model(
    model="resnet-18", num_classes=10, hidden_dims=[], extra_inputs=None
):
    if model == "resnet-18":
        model_base_config = RESNET18CONFIG

    elif model == "resnet-50":
        model_base_config = RESNET50CONFIG

    else:
        raise NotImplementedError
    model_base_config.num_labels = num_classes
    model_base = AutoModelForImageClassification.from_config(model_base_config)
    model = HugginFaceTupleWrapper(
        model_base, hidden_dims=hidden_dims, extra_inputs=extra_inputs
    )

    # feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    # return model, feature_extractor
    return model


def get_cifar_resnet_model(
    model="cifar-resnet-18", num_classes=10, hidden_dims=[], extra_inputs=None
):
    if extra_inputs is None:
        if model == "cifar-resnet-18":
            return ResNet18(num_classes=num_classes)
        elif model == "cifar-resnet-10":
            return ResNet10(num_classes=num_classes)
        elif model == "cifar-resnet-34":
            return ResNet34(num_classes=num_classes)
        elif model == "cifar-resnet-50":
            return ResNet50(num_classes=num_classes)
        elif model == "cifar-wideresnet":
            return WideResNet(num_classes=num_classes)
    else:
        if model == "cifar-resnet-18":
            return ResNet18ExtraInputs(
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                extra_inputs=extra_inputs,
            )
        elif model == "cifar-resnet-10":
            return ResNet10ExtraInputs(
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                extra_inputs=extra_inputs,
            )
        elif model == "cifar-resnet-34":
            return ResNet34ExtraInputs(
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                extra_inputs=extra_inputs,
            )
        elif model == "cifar-resnet-50":
            return ResNet50ExtraInputs(
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                extra_inputs=extra_inputs,
            )

    raise NotImplementedError


def get_model(
    architecture,
    num_classes,
    image_size=None,
    freeze_embedding=False,
    hidden_dims=[],
    extra_inputs=None,
):
    if architecture.startswith("cifar"):
        model = get_cifar_resnet_model(
            architecture,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            extra_inputs=extra_inputs,
        )
    elif architecture.startswith("resnet"):
        model = get_fresh_resnet_model(
            architecture,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            extra_inputs=extra_inputs,
        )

    elif "/" in architecture:
        model = get_huggingface_model(
            architecture,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            extra_inputs=extra_inputs,
        )
        if freeze_embedding:
            model.freeze_base_model()
    else:
        model = get_torchvision_model(
            model_name=architecture, num_classes=num_classes, hidden_dims=hidden_dims
        )
    return model
