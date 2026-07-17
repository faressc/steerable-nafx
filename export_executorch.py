"""Export the steerable-nafx TCN models to ExecuTorch .pte programs.

Rebuilds the eager TCN from the notebook (steerable-nafx-pytorch.ipynb) for each
shipped variant (4 / 3 / 2 blocks), loads its weights out of the corresponding
TorchScript export (steerable-nafx[-N_blocks]-dynamic.pt) so the .pte carries
exactly the same parameters, cross-checks the eager reconstruction against the
TorchScript module, and lowers to ExecuTorch with the XNNPACK partitioner using
a dynamic sample axis (like the ONNX exports).

Requires: torch + executorch (pip install executorch).
"""

import os
import sys

import torch

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_0")

# All shipped variants share these hyperparameters (see the training notebook);
# they differ only in block count.
KERNEL_SIZE = 13
DILATION_GROWTH = 10
N_CHANNELS = 32

VARIANTS = {
    # infix -> n_blocks ("" is the full 4-block model)
    "": 4,
    "-3_blocks": 3,
    "-2_blocks": 2,
}


class TCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation=True):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=0, bias=True)
        if activation:
            self.act = torch.nn.PReLU()
        self.res = torch.nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        x_in = x
        x = self.conv(x)
        if hasattr(self, "act"):
            x = self.act(x)
        x_res = self.res(x_in)
        x_res = x_res[..., (self.kernel_size - 1) * self.dilation:]
        return x + x_res


class TCN(torch.nn.Module):
    def __init__(self, n_inputs=1, n_outputs=1, n_blocks=4, kernel_size=KERNEL_SIZE,
                 n_channels=N_CHANNELS, dilation_growth=DILATION_GROWTH):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks
        self.dilation_growth = dilation_growth
        self.blocks = torch.nn.ModuleList()
        for n in range(n_blocks):
            in_ch = n_inputs if n == 0 else n_channels
            out_ch = n_outputs if (n + 1) == n_blocks else n_channels
            self.blocks.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation_growth ** n, activation=True))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def compute_receptive_field(self):
        rf = self.kernel_size
        for n in range(1, self.n_blocks):
            rf += (self.kernel_size - 1) * (self.dilation_growth ** n)
        return rf


def export_variant(infix, n_blocks):
    ts_path = os.path.join(MODEL_DIR, f"steerable-nafx{infix}-dynamic.pt")
    out_path = os.path.join(MODEL_DIR, f"steerable-nafx{infix}-executorch.pte")

    ts_model = torch.jit.load(ts_path, map_location="cpu")
    ts_model.eval()

    model = TCN(n_blocks=n_blocks)
    model.load_state_dict(ts_model.state_dict())
    model.eval()

    rf = model.compute_receptive_field()
    example_len = rf + 2048 - 1  # the input size anira uses for a 2048-sample output
    example = torch.linspace(-1.0, 1.0, example_len).reshape(1, 1, example_len)

    with torch.no_grad():
        eager_out = model(example)
        ts_out = ts_model(example)
    max_diff = (eager_out - ts_out).abs().max().item()
    print(f"{os.path.basename(out_path)}: receptive field {rf}, "
          f"eager vs torchscript max abs diff {max_diff:.3e}")
    assert max_diff < 1e-6, "reconstructed TCN does not match the TorchScript model"

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge, to_edge_transform_and_lower

    # Dynamic sample axis, like the ONNX exports: any input length > the receptive
    # field works; the output length is input length - (receptive field - 1). The
    # upper bound drives ExecuTorch's worst-case memory planning, so keep it just
    # large enough for anira's use (max 8192-sample buffers + the receptive field).
    sample_dim = torch.export.Dim("samples", min=rf + 1, max=1 << 15)
    exported = torch.export.export(model, (example,), dynamic_shapes=({2: sample_dim},))
    try:
        program = to_edge_transform_and_lower(
            exported, partitioner=[XnnpackPartitioner()]).to_executorch()
        print("  lowered with the XNNPACK partitioner")
    except Exception as e:
        # The XNNPACK partitioner cannot lower the TCN's dilated-conv + residual
        # slice pattern on all versions; fall back to portable/optimized CPU ops.
        print(f"  XNNPACK partitioning failed ({type(e).__name__}); using portable ops")
        program = to_edge(exported).to_executorch()
    with open(out_path, "wb") as f:
        f.write(program.buffer)
    print(f"  wrote {out_path} ({len(program.buffer)} bytes)")

    # Reference blob for the C++ runtime cross-check (raw float32 input/output).
    example.numpy().tofile(out_path + ".input.raw")
    eager_out.numpy().tofile(out_path + ".expected.raw")


def main():
    for infix, n_blocks in VARIANTS.items():
        export_variant(infix, n_blocks)
    print("OK")


if __name__ == "__main__":
    sys.exit(main())
