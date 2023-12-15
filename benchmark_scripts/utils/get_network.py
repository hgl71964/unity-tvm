from typing import Dict

import os
import json
import numpy as np
from pathlib import Path

import tvm, tvm.relay, tvm.relay.testing
from tvm import relay
from . import benchmark_configs as Configs


def get_relay_network(name: str,
                      configs: Configs,
                      batch_size: int = 1,
                      layout="NHWC",
                      dtype="float32",
                      workload_shape=(224, 224, 3),
                      override_workload=False):
    """Get the symbol definition and random weight of a network

    Parameters
    ----------
    name: str
        name of the model. If you are relying on the existing scripts
        to create a relay model and parameters, please follow the model
        names listed in configs.py
    batch_size: int
        number of inputs that will be fed into the compiled model in each
        batch.
    layout: str
        data layout of the model. Models from :fun:`~tvm.relay.testing~`
        may allow conversion between different data layouts. Otherwise
        this parameter doesn't really matter (especially if a custom
        function is supplied in configs.py to generate the relay function).
    dtype: str
        data type of the input. Follow model specification to decide
    workload_shape: Tuple[int]
        specifies the shape of the input, excluding the dimension describing
        the batch size.
    override_workload: bool
        If set to True, this function will always regenerate the relay model
        and relevant parameters from scratch, either by converting from
        another model format, or by using some existing functions provided
        by tvm. If set to False, this function will load cached models that has
        already been converted whenever possible.

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains a ResNet network.

    params : Dict[str, NDArray]
        The parameters.

    input_shape:  Tuple[int]
        The shape of the input to the model, where the first number always indicates the
        batch size.

    output_shape: Tuple[int]
        The shape of the output of the model.
    """

    # If a custom relay model generator is supplied, override all current behaviour
    if configs.custom_get_relay_network.get(name):
        return configs.custom_get_relay_network[name](name, batch_size, layout,
                                                      dtype, workload_shape,
                                                      override_workload)

    input_shape = (batch_size, ) + workload_shape
    output_shape = (batch_size, 1000)

    mod: tvm.IRModule
    params: Dict[str, np.ndarray]

    model_json: str = f"{name}-{layout}-{dtype}-{workload_shape}-{configs.batch_size}_model.json"
    params_json: str = f"{name}-{layout}-{dtype}-{workload_shape}-{configs.batch_size}_params.tvmbytes"

    dir_path = configs.RELAY_MODELS_PATH
    model_json_path: Path = configs.RELAY_MODELS_PATH.joinpath(model_json)
    params_bytes_path: Path = configs.RELAY_MODELS_PATH.joinpath(params_json)

    if not override_workload \
        and model_json_path.is_file() \
        and params_bytes_path.is_file():
        print(
            f"Using existing model configurations: {model_json_path} & {params_bytes_path}"
        )
        with open(model_json_path, 'r') as istream:
            model_string = json.dumps(json.load(istream))
            mod = tvm.ir.load_json(model_string)
        params = tvm.runtime.load_param_dict_from_file(str(params_bytes_path))
        return mod, params, input_shape, output_shape

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = tvm.relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=workload_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = tvm.relay.testing.resnet_3d.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=workload_shape)
    elif name == "toy":
        shape = (1, 64, 54, 54)
        c_data = np.empty(shape).astype("float32")
        c = relay.const(c_data)
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        x = relay.var("x", relay.TensorType((1, 64, 56, 56), "float32"))
        conv = relay.nn.conv2d(x, weight)
        y = relay.add(c, c)
        y = relay.multiply(y, relay.const(2, "float32"))
        y = relay.add(conv, y)
        z = relay.add(y, c)
        z1 = relay.add(y, c)
        z2 = relay.add(z, z1)
        f = relay.Function([x, weight], z2)
        mod = tvm.IRModule.from_expr(f)
        raise RuntimeError("Not implemented yet")

    elif name == "mobilenet":
        mod, params = tvm.relay.testing.mobilenet.get_workload(
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=workload_shape)
    elif name == "dcgan":
        input_shape = (batch_size, 3, 64,
                       64) if layout == "NCHW" else (batch_size, 64, 64, 3)
        mod, params = tvm.relay.testing.dcgan.get_workload(
            batch_size=batch_size,
            dtype=dtype,
            oshape=(3, 64, 64),
            ngf=128,
            layout=layout,
            random_len=100)
    elif name.startswith("vgg-"):
        n_layer = int(name.split("-")[1])
        mod, params = tvm.relay.testing.vgg.get_workload(
            batch_size=batch_size,
            num_classes=1000,
            image_shape=workload_shape,
            dtype=dtype,
            num_layers=n_layer,
            batch_norm=False)
    elif name == "densenet":
        mod, params = tvm.relay.testing.densenet.get_workload(
            batch_size=batch_size,
            dtype=dtype,
            image_shape=workload_shape,
        )
    elif name == "dqn":
        mod, params = tvm.relay.testing.dqn.get_workload(
            batch_size=batch_size,
            dtype=dtype,
            image_shape=workload_shape,
        )
    elif name == "inception-v3":
        mod, params = tvm.relay.testing.inception_v3.get_workload(
            batch_size=batch_size,
            dtype=dtype,
            image_shape=workload_shape,
        )
    elif name == "lstm":
        mod, params = tvm.relay.testing.lstm.get_workload(
            batch_size=batch_size,
            iterations=workload_shape[0],
            num_hidden=workload_shape[1],
            dtype=dtype,
        )
    elif name.startswith("bertsquad-"):
        import onnx
        onnx_model = onnx.load(
            str(
                Path(configs.MODELS_PATH).joinpath("bert-squad",
                                                   f"{name}.onnx")))
        input_tup = (batch_size, ) + workload_shape
        shape_dict = {
            "input_ids:0": input_tup,
            "segment_ids:0": input_tup,
            "input_mask:0": input_tup,
            "unique_ids_raw_output___9:0": (batch_size, )
        }
        mod, params = tvm.relay.frontend.from_onnx(onnx_model,
                                                   shape=shape_dict)
        input_shape = input_tup
    elif name == "bert":
        # In reference to https://gist.github.com/merrymercy/c48da9e09822101453a2870260678dc7#file-tune_bert_x86-py-L197
        # import torch
        # import transformers  # pip3 install transfomers==3.0

        # os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # model_class = transformers.BertModel
        # tokenizer_class = transformers.BertTokenizer

        # # You can also download them manualy
        # #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
        # #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
        # #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
        # # Then rename to pytorch_model.bin, vocab.txt & config.json
        # # weight = 'path to downloaded model dir'
        # weight = 'bert-base-uncased'
        # model = model_class.from_pretrained(
        #     weight, return_dict=False)  # Modification is needed here
        # model.eval()

        # # tokenizer = tokenizer_class.from_pretrained(weight)
        # # A = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])
        # # There is 30522 words in bert-base-uncased's vocabulary list
        # input_shape = [batch_size,
        #                256]  # Changed input size = 256 to maintain consistency
        # input_name = 'input_ids'
        # input_dtype = 'int64'
        # rand_input = torch.randint(30000, input_shape)
        # scripted_model = torch.jit.trace(model, [rand_input], strict=False)
        # shape_list = [('input_ids', input_shape)]
        # mod, params = tvm.relay.frontend.from_pytorch(scripted_model,
        #                                               shape_list)

        # mod = tvm.relay.transform.FastMath()(mod)
        # mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        # BindPass = tvm.relay.transform.function_pass(
        #     lambda fn, new_mod, ctx: tvm.relay.build_module.
        #     bind_params_by_name(fn, params),
        #     opt_level=1)
        # mod = BindPass(mod)
        # mod = tvm.relay.transform.FoldConstant()(mod)
        # mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        # mod = tvm.relay.transform.FoldConstant()(mod)
        # input_shape = tuple(input_shape)

        import torch
        from transformers import BertForSequenceClassification
        from tvm.relay.transform import ToMixedPrecision

        model = BertForSequenceClassification.from_pretrained(
            'bert-large-uncased')

        batch_size = 8
        inputs = (torch.ones(batch_size, 128, dtype=torch.int64),
                  torch.ones(batch_size, 128, dtype=torch.int64),
                  torch.ones(batch_size, 128, dtype=torch.int64))

        input_shapes = [("input_ids", (inputs[0].shape, "int64")),
                        ("attention_mask", (inputs[1].shape, "int64")),
                        ("token_type_ids", (inputs[2].shape, "int64"))]

        with torch.no_grad():
            out = model(*inputs)

        class TraceWrapper(torch.nn.Module):

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, *inp):
                out = self.model(*inp)
                return out["logits"]

        input_shape = (list(inp.shape) for inp in inputs)
        script_module = torch.jit.trace(TraceWrapper(model), inputs).eval()
        mod, params = relay.frontend.from_pytorch(script_module, input_shapes)
        mod = ToMixedPrecision("float16")(mod)

    elif name == "vit":
        import torch
        import transformers  # pip3 install transfomers==3.0

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        model_class = transformers.ViTModel
        preprocessor_class = transformers.ViTImageProcessor

        # You can also download them manualy
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
        # Then rename to pytorch_model.bin, vocab.txt & config.json
        # weight = 'path to downloaded model dir'
        weight = 'google/vit-base-patch16-224-in21k'
        image_shape = workload_shape

        model = model_class.from_pretrained(
            weight, return_dict=False)  # Modification is needed here
        model.eval()

        # tokenizer = tokenizer_class.from_pretrained(weight)
        # A = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])
        # There is 30522 words in bert-base-uncased's vocabulary list
        input_shape = (batch_size, ) + image_shape
        input_name = 'pixel_values'
        rand_input = torch.FloatTensor(*input_shape).uniform_(-1.0, 1.0)
        print(rand_input.shape)
        # WARNING: torch.jit.trace cannot handle models with conditionals
        scripted_model = torch.jit.trace(model, rand_input, strict=False)
        shape_list = [(input_name, input_shape)]
        mod, params = tvm.relay.frontend.from_pytorch(scripted_model,
                                                      shape_list)
    elif name == "bert-mx":
        # This is the version of bert used in the benchmark script used in the Ansor
        # "Ansor : Generating High-Performance Tensor Programs for Deep Learning" paper
        import gluonnlp

        seq_length = 128

        # Instantiate a BERT classifier using GluonNLP
        model_name = "bert_12_768_12"
        dataset = "book_corpus_wiki_en_uncased"
        model, _ = gluonnlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            use_classifier=False,
        )

        # Convert the MXNet model into TVM Relay format
        shape_dict = {
            "data0": (batch_size, seq_length),
            "data1": (batch_size, seq_length),
            "data2": (batch_size, ),
        }
        mod, params = tvm.relay.frontend.from_mxnet(model, shape_dict)
        input_shape = (shape_dict["data0"], shape_dict["data1"],
                       shape_dict["data2"])

        # mod = tvm.relay.transform.FastMath()(mod)
        # mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        BindPass = tvm.relay.transform.function_pass(
            lambda fn, new_mod, ctx: tvm.relay.build_module.
            bind_params_by_name(fn, params),
            opt_level=1,
        )
        mod = BindPass(mod)
        # mod = tvm.relay.transform.FoldConstant()(mod)
        # mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        # mod = tvm.relay.transform.FoldConstant()(mod)
    else:
        raise ValueError(f"Model not supported: {name}")

    # Write model to file
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    with open(model_json_path, 'w') as f:
        mod_json = json.loads(tvm.ir.save_json(mod))
        json.dump(mod_json, f, indent=4)
    tvm.runtime.save_param_dict_to_file(params, str(params_bytes_path))

    return mod, params, input_shape, output_shape
