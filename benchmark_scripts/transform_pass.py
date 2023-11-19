import numpy as np

import tvm
from tvm import te
import tvm.relay as relay
from tvm.relay.testing import resnet
from tvm.contrib.download import download_testdata
from tvm.relay.build_module import bind_params_by_name
from tvm.ir.instrument import (
    PassTimingInstrument,
    pass_instrument,
)

print("""
# ###########
# pass instruments
# ###########
""")

# get a workload
batch_size = 1
num_of_image_class = 1000
image_shape = (3, 224, 224)
output_shape = (batch_size, num_of_image_class)
relay_mod, relay_params = resnet.get_workload(num_layers=18,
                                              batch_size=1,
                                              image_shape=image_shape)

# print("Printing the IR module...")
# print(relay_mod.astext(show_meta_data=False))

timing_inst = PassTimingInstrument()
with tvm.transform.PassContext(instruments=[timing_inst]):
    relay_mod = relay.transform.InferType()(relay_mod)
    relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
    # before exiting the context, get profile results.
    profiles = timing_inst.render()

print()
print("Printing results of timing profile...")
print(profiles)
print()

# the current PassContext interface
#
# cur_pass_ctx = tvm.transform.PassContext.current()
# cur_pass_ctx.override_instruments([timing_inst])
# relay_mod = relay.transform.InferType()(relay_mod)
# relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
# profiles = timing_inst.render()
# print("Printing results of timing profile...")
# print(profiles)

print("""
# ###########
# pass
# ###########
""")


def example():
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
    return relay.Function([x, weight], z2)


f = example()
mod = tvm.IRModule.from_expr(f)

# Glob the interested passes.
seq = tvm.transform.Sequential([
    relay.transform.FoldConstant(),
    relay.transform.EliminateCommonSubexpr(),
    relay.transform.FuseOps(fuse_opt_level=2),
])

# by default, tvm only runs pass with optimization level less or equal to 2
with tvm.transform.PassContext(opt_level=3):
    mod2 = seq(mod)

print()
print(mod2)
print()

print("""
# ###########
# Custom pass
# ###########
""")


@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    """Simple test function to replace one argument to another."""
    def __init__(self, multiplier):
        self.multiplier = multiplier

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
        obj = self

        class ReplaceConstant(tvm.relay.ExprMutator):
            def visit_constant(self, c):
                return relay.multiply(obj.multiplier, c)

        return ReplaceConstant().visit(func)


f = example()
mod = tvm.IRModule.from_expr(f)
custom_pass = CustomPipeline(multiplier=relay.const(3, "float32"))
assert custom_pass.info.name == "CustomPipeline"
mod3 = custom_pass(mod)
print()
print(mod3)
print()
