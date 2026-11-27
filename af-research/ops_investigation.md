# CDNN operator metadata and graph constraints

This investigation covers only the two digest-pinned S5 V2.9 model components. It validates static constraints; it does not compare against an original DSP execution.

Run from `af-research/` after installing the Python 3.12+ research environment described in the [project README](README.md). Supply an external directory containing the two complete, SHA-256-pinned original model components, or a parent directory containing `unpacked-v29/`:

```sh
.venv/bin/python extract.py --firmware-root /path/to/original-s5-v29
.venv/bin/python ops_investigation.py --firmware-root /path/to/original-s5-v29
```

Extraction creates the model output directories used by this command. `S5_AF_FIRMWARE_ROOT` can replace the repeated option; `--source /path/to/components` explicitly selects a component directory. On Windows use `.venv\Scripts\python.exe` instead of `.venv/bin/python`. The original update package and historical ZIP/TSV are optional extraction cross-checks; missing evidence is recorded, not marked passed. No firmware input is included in this research source directory.

The command regenerates `artifacts/<model>/operator-investigation.json`. The module exposes `operator_metadata(data, node)` and `analyze(data)`.

All offsets below are relative to `q = decode_parameters.tail_offset(node)`, the end of the input connection table. `u32` means little-endian, possibly unaligned.

| Operator | Observed fields | Candidate interpretation |
| --- | --- | --- |
| convolution 1 | bytes 61,62; u32 63,67; u32 71,75; u32 79,83 | kernel H/W; stride H/W; dilation H/W; symmetric padding H/W |
| convolution 1 | byte 100; byte 101; u32 335 | transpose flag; activation flag; redundant activation flag |
| pooling 5 | six u32 from 61 | kernel H/W, stride H/W, padding H/W |
| concat 9 | byte 57 = 1 | concatenate channels |
| slice 10 | byte 69 = 1; u32 70 = 1; u32 74 | channel axis; one cut; cut position |
| reshape 8 | four u32 from 6 | explicit N,C,H,W output shape |
| softmax-like 29 | byte 62 = 1 | class/channel axis |
| proposal 24 | u32 61,65 = 16; u32 69 = 16 | feature strides; anchor base size |
| proposal 24 | u32 73,77 = 5; u32 81 = 3000; u32 85 = 30; float32 89 ≈ 0.3 | ratio/scale count; pre/post NMS top-N; NMS threshold |

The former `padding_four_fields_candidate` conflated dilation and padding. For 1×1 convolutions those fields are `[1,1,0,0]`. Interpreting all four as padding would grow spatial dimensions and break the graph. Dilation `[1,1]` and padding `[0,0]` fit every 1×1 layer.

All pooling records have 3×3 kernel, stride 2, padding 0. A ceil-mode output size is required to merge with the convolution branches. For example the first inception pool consumes 32×48 and must produce 16×24; floor mode produces 15×23 and fails concat. Pool maximum vs average remains unproven by static dimensions.

Both files pass all 12 concat branch shape checks. The main spatial chain is 256×384 → 128×192 → 64×96 → 32×48 → inception3 16×24 → inception4 8×12 → upsample 16×24. The fusion inputs have 96,192,256 channels, giving 544. The RPN fusion has 128+256+64=448 channels.

All 65 / 61 ordinary convolution records have a packed weight element count consistent with their graph input channels. `upsample` is excluded from this particular check because its coefficient duplication/padding is a different format. All 12 / 4 encoded reshape records preserve element counts. Classifier heads produce 50×16×24, reshape to 2×400×24, apply the axis-1 operation and reshape back. Regression produces 100×16×24. The five ratio and five scale entries give 25 anchors per cell, exactly 50 class channels and 100 box coordinates; the grid contains 9600 anchors before filtering.

Activation bytes 101 and u32 335 agree in all 128 convolution records. They are 1 on the backbone and RPN feature convolutions and 0 on upsample/classification/regression output layers. This is consistent with fused ReLU, but clipping, exact activation implementation and quantization effects need runtime evidence.

Proposal fields have conventional Faster-RCNN-like values, but field semantics, coordinate conventions, output structures, filtering order and NMS implementation remain hypotheses. The explicit shape checks establish compatibility of the proposed graph; they do not establish numerical equivalence or recognition accuracy.
