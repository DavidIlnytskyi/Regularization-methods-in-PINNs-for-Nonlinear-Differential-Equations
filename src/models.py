import inspect
from warnings import warn

import torch
from torch import nn

SUPPORTED_WEIGHT_INIT_METHODS = {"xavier", "orthogonal", "small_normal", "default"}


def normalize_weights_init_method(method):
    if method is None:
        return None

    normalized = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "small_normal_weights": "small_normal",
        "normal_small": "small_normal",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in SUPPORTED_WEIGHT_INIT_METHODS:
        supported = ", ".join(sorted(SUPPORTED_WEIGHT_INIT_METHODS))
        raise ValueError(
            f"Unsupported weights_init_method '{method}'. Use one of: {supported}."
        )

    return normalized


def initialize_linear_layer(layer, method="xavier", *, small_normal_std=1e-2):
    method = normalize_weights_init_method(method)
    if method is None:
        return layer

    if small_normal_std <= 0:
        raise ValueError("small_normal_std must be positive")

    if method == "xavier":
        nn.init.xavier_uniform_(layer.weight)
    elif method == "orthogonal":
        nn.init.orthogonal_(layer.weight)
    elif method == "small_normal":
        nn.init.normal_(layer.weight, mean=0.0, std=small_normal_std)

    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

    return layer


def initialize_linear_weights(module, method="xavier", *, small_normal_std=1e-2):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            initialize_linear_layer(
                layer,
                method=method,
                small_normal_std=small_normal_std,
            )
    return module


class FCNN_Dropout(nn.Module):
    def __init__(
        self,
        n_input_units=1,
        n_output_units=1,
        n_hidden_units=None,
        n_hidden_layers=None,
        actv=nn.Tanh,
        hidden_units=None,
        p_drop=0.0,
        drop_last=False,
        bias=True,
        weights_init_method="xavier",
        small_normal_std=1e-2,
    ):
        super(FCNN_Dropout, self).__init__()

        if n_hidden_units is None and n_hidden_layers is not None:
            n_hidden_units = 32
        elif n_hidden_units is not None and n_hidden_layers is None:
            n_hidden_layers = 1

        if n_hidden_units is not None or n_hidden_layers is not None:
            if hidden_units is None:
                hidden_units = tuple(n_hidden_units for _ in range(n_hidden_layers + 1))
                warn(
                    f"`n_hidden_units` and `n_hidden_layers` are deprecated, "
                    f"pass `hidden_units={hidden_units}` instead",
                    FutureWarning,
                )
            else:
                warn(
                    f"Ignoring `n_hidden_units` and `n_hidden_layers` in favor of `hidden_units={hidden_units}`",
                    FutureWarning,
                )

        if hidden_units is None:
            hidden_units = (32, 32)

        if not isinstance(hidden_units, tuple):
            hidden_units = tuple(hidden_units)

        units = (n_input_units,) + hidden_units
        layers = []
        for i in range(len(units) - 1):
            layers.append(nn.Linear(units[i], units[i + 1], bias=bias))
            layers.append(actv())
            if p_drop and (drop_last or i < len(units) - 2):
                layers.append(nn.Dropout(p_drop))

        layers.append(nn.Linear(units[-1], n_output_units, bias=bias))
        self.NN = nn.Sequential(*layers)
        initialize_linear_weights(
            self,
            method=weights_init_method,
            small_normal_std=small_normal_std,
        )

    def forward(self, t):
        return self.NN(t)
    

class FourierFeatureEncoding(nn.Module):
    def __init__(
        self,
        n_input_units,
        mapping_size=32,
        scale=3,
        include_input=False,
        trainable=False,
    ):
        super().__init__()

        if mapping_size <= 0:
            raise ValueError("mapping_size must be positive")

        self.n_input_units = n_input_units
        self.mapping_size = mapping_size
        self.scale = scale
        self.include_input = include_input

        B = torch.randn(n_input_units, mapping_size) * scale
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    @property
    def output_dim(self):
        encoded_dim = 2 * self.mapping_size
        if self.include_input:
            encoded_dim += self.n_input_units
        return encoded_dim

    def forward(self, x):
        projection = x @ self.B
        encoded = [torch.sin(projection), torch.cos(projection)]
        if self.include_input:
            encoded.insert(0, x)
        return torch.cat(encoded, dim=-1)


class EncodedBody(nn.Module):
    def __init__(self, encoder, body):
        super().__init__()
        self.encoder = encoder
        self.body = body
        self.NN = body.NN

    def forward(self, x):
        return self.body(self.encoder(x))


def _normalize_hidden_units(hidden_units):
    if hidden_units is None:
        return ()

    if isinstance(hidden_units, int):
        hidden_units = (hidden_units,)
    elif not isinstance(hidden_units, tuple):
        hidden_units = tuple(hidden_units)

    if any(unit <= 0 for unit in hidden_units):
        raise ValueError("Hidden layer sizes must be positive integers")

    return hidden_units


def _resolve_network_bias(network_config, attr_name):
    attr_value = getattr(network_config, attr_name)
    if attr_value is None:
        return bool(network_config.use_bias)
    return bool(attr_value)


def build_body_model(network_config, *, n_input_units, basis_length):
    encoding = network_config.input_encoding.lower()
    body_units = _normalize_hidden_units(network_config.body_units)
    body_bias = _resolve_network_bias(network_config, "use_bias_in_body")

    if encoding not in {"identity", "fourier"}:
        raise ValueError(
            f"Unsupported input_encoding '{network_config.input_encoding}'. "
            "Use 'identity' or 'fourier'."
        )

    if encoding == "fourier":
        encoder = FourierFeatureEncoding(
            n_input_units=n_input_units,
            mapping_size=network_config.fourier_mapping_size,
            scale=network_config.fourier_scale,
            include_input=network_config.fourier_include_input,
            trainable=network_config.fourier_trainable,
        )
        body = FCNN_Dropout(
            n_input_units=encoder.output_dim,
            hidden_units=body_units,
            n_output_units=basis_length,
            bias=body_bias,
            weights_init_method=network_config.weights_init_method,
            small_normal_std=network_config.small_normal_std,
        )
        return EncodedBody(encoder, body)

    return FCNN_Dropout(
        n_input_units=n_input_units,
        hidden_units=body_units,
        n_output_units=basis_length,
        bias=body_bias,
        weights_init_method=network_config.weights_init_method,
        small_normal_std=network_config.small_normal_std,
    )


def build_head_model(network_config, *, basis_length):
    head_units = _normalize_hidden_units(network_config.head_units)
    head_bias = _resolve_network_bias(network_config, "use_bias_in_heads")
    units = (basis_length,) + head_units + (1,)
    layers = []

    for i in range(len(units) - 2):
        layers.append(
            nn.Linear(
                units[i],
                units[i + 1],
                bias=head_bias,
            )
        )
        layers.append(nn.Tanh())

    layers.append(
        nn.Linear(
            units[-2],
            units[-1],
            bias=head_bias,
        )
    )

    head = nn.Sequential(*layers)
    initialize_linear_weights(
        head,
        method=network_config.weights_init_method,
        small_normal_std=network_config.small_normal_std,
    )
    return head


class NET(nn.Module):
    def __init__(self, H_model, head_model):
        super(NET, self).__init__()
        self.H_model = H_model
        self.head_model = head_model

    def forward(self, x):
        x = self.H_model(x)
        x = self.head_model(x)
        return x


class NET_FREEZE(nn.Module):
    def __init__(self, H_model, head_model):
        super(NET_FREEZE, self).__init__()

        for param in H_model.parameters():
            param.requires_grad = False
        self.H_model = H_model
        self.head_model = head_model
        # Freeze the parameters of H_model
        # for param in self.H_model.parameters():
        # param.requires_grad = False

    def forward(self, x):
        x = self.H_model(x)
        x = self.head_model(x)
        return x


def _requires_closure(optimizer):
    # starting from torch v1.13, simple optimizers no longer have a `closure` argument
    closure_param = inspect.signature(optimizer.step).parameters.get("closure")
    return closure_param and closure_param.default == inspect._empty