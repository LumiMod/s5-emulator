# Original runtime consumer investigation

This work establishes where the shipped model runtime lives and how ARM code loads and registers it. It does not establish original DSP numerical parity. The investigation recorded on 2026-11-27 had no connected S5 device, CEVA SDK, or original input/output captures.

## Reproduce

Run from `af-research/`, using the Python environment in the [project README](README.md):

```sh
.venv/bin/python runtime_investigation.py --firmware-root /path/to/original-s5-v29
```

The external input directory must contain `03_program.bin`, `39_hm_c_prog.bin`, `40_hm_d_prog.bin`, `41_hm_c_ddr.bin` and `42_hm_d_ddr.bin`, either directly or under `unpacked-v29/`. Use `--source /path/to/components` for an explicit component directory, or set `S5_AF_FIRMWARE_ROOT` instead of the command option. Windows uses `.venv\Scripts\python.exe`. These runtime files are optional for the overall model workflow: `prepare_offline.py` records this investigation as skipped when they are absent. Running `runtime_investigation.py` directly requires all five.

The script uses Capstone from the selected Python environment only to disassemble known ARM Thumb consumer regions. It exports `artifacts/runtime-investigation.json` and, when Capstone is available, `artifacts/runtime-arm-consumer.txt`. Without Capstone it still verifies the binary data references and exports the JSON with `arm_disassembly.available=false`; ARM disassembly is then not claimed. The script records input digests and structural relationships; its standalone execution does not independently prove firmware-package provenance.

## Verified ARM-side loading and registration

In `03_program.bin`, loaded at ARM VA `0x06280000`:

- VA `0x067e05c8` loads the literal `0x84268000` from file offset `0x560858`; VA `0x067e05ce` stores it into the runtime state at offset `+4`.
- VA `0x067e05d2` loads `0x84568000` from file offset `0x56085c`; VA `0x067e05d6` stores it at state offset `+8`.
- The loader at `0x067e0702` requests a `0x300000`-byte transfer from `0x43a00000` into state `+4` (call at `0x067e073c`) and a `0x20000`-byte transfer from `0x43d00000` into state `+8` (call at `0x067e0752`). The source-address interpretation as flash mapping is not independently validated here; the argument values and destinations are observed instructions.
- The 3 MiB size matches `41_hm_c_ddr.bin`, and `0x84268000 + 0x300000 = 0x84568000`. The 128 KiB data transfer matches the non-FF prefix of `42_hm_d_ddr.bin`.
- Both model transfers request 4 MiB. The code checks the `CDNN` magic and compares the halfword at header `+0x10` against 6 for the multiclass model and 2 for the human model.
- The caller at `0x0679764c` collects both model pointers via `0x067e0832` and `0x067e083e`, sets each size to 4 MiB, and passes its assembled descriptor to `0x067d1770` at callsite `0x067976ae`.

An early guess of code base `0x84280000` was discarded. The ARM consumer establishes `0x84268000`; future reference offsets must use that base.

## Concrete runtime class references

`42_hm_d_ddr.bin` contains RTTI structures whose name pointers consistently equal `0x84568000 + string_file_offset`. Their vtables point into the code range established above. The script independently checks the name reference, vtable RTTI back-reference and each method pointer's mapped nonempty code bytes.

Ten selected classes yield 86 method-pointer entries. Individual method names are not recovered; each listed address is a vtable-slot candidate, not a confirmed `forward()` or preprocessing function.

Priority classes for later instruction decoding/capture are `InputLayer`, `XM4InputKernel`, `ConvLayer`, `XM4ConvKernel`, `XM4PoolKernel` and `XM4ProposalKernel`. JSON includes every pointer, corresponding `41_hm_c_ddr.bin` offset, first 64 bytes and snippet digest. This links the class identities to actual shipped code, instead of relying only on free-standing strings.

Examples:

- `InputLayer`: name file `0x33e4`, RTTI `0x33f4`, vtable methods at `0x3408`; first method VA `0x842c0d42`, code file offset `0x58d42`.
- `XM4InputKernel`: name `0x5c00`, RTTI `0x5c14`, methods at `0x5c28`; method VAs include `0x84321e04`, `0x84321d60`, `0x84321c40`, `0x84321400`, `0x843213e0`.
- `XM4ConvKernel`: name `0x60a0`, RTTI `0x60b0`, methods at `0x60c4`; entries include `0x8432c7c2`, `0x8432c784`, `0x84329800`.
- `XM4ProposalKernel`: name `0x59dc`, RTTI `0x59f0`, methods at `0x5a04`; entries include `0x843209e0`, `0x843209a0`, `0x84320642`.

`DataMean` occurs at data file offset `0x3538`, near the InputLayer structures. This supports investigating input centering there, but its use, order of subtraction/scaling and image conversion remain undecoded. The nearby ARM image rotation routine `0x067e005c` has a caller at `0x067c219e`; adjacency alone does not connect that routine to CDNN preprocessing, so no such claim is made.

## ISA/tool availability and remaining parity gap

The model target marker `XM4`, compiled `XM4*Kernel` RTTI/vtables and the embedded `CEVA_DEV/.../dev_xm4/.../CDNNTarget` source path jointly support CEVA-XM4 runtime identity. Exact instruction encoding/revision was not decoded or executed. Raw code should not be treated as ARM merely because an ARM disassembler produces occasional valid-looking instructions.

The bounded local inventory at the time of this investigation found Capstone but no `cevatools`, XM4 decoder libraries or CEVA simulator. This is not an exhaustive proof that no copy exists elsewhere, nor a check of a new contributor's environment.

The CEVA XM4 white paper consulted for this investigation describes instruction-set and cycle-accurate simulators for Windows/Linux. The referenced Lauterbach CEVA-X manual says XM4 disassembly uses CEVA-provided libraries, mapping the CPU name to `xm4`; its examples specify `cevaxasmsrv.dll` and `<cpu>db.dll` or the corresponding Linux libraries. These documents describe a reference-tool route, not an available installation in this project. Sources: [CEVA-XM4 white paper](https://www.ceva-ip.com/wp-content/uploads/CEVA-XM4-White-Paper.pdf), [Lauterbach CEVA-X debugger manual, pages 11–12](https://www2.lauterbach.com/pdf/debugger_cevax.pdf).

The remaining reference task would capture tensor values before/after the original input kernel and each first convolution, including quantization/activation, then compare recovered computation. Vtable entry points and code/data bases are now available for that work; no original tensor capture has been performed. Until a compatible decoder/simulator or device capture is available, candidate-network execution on the host cannot establish original-runtime equivalence.

## Secondary extracted files audit

The previously audited secondary `.cdnn` files were exact prefixes of the pinned components, but each omitted 16 zero bytes inside the header-declared parameter range. They are historical local artifacts, not required inputs or files shipped by this source project:

| File | Actual length | Header-declared end |
| --- | --- | --- |
| `s5_af_multi_6cls.cdnn` | `0x33bf50` | `0x33bf60` |
| `s5_af_human_2cls.cdnn` | `0x30c800` | `0x30c810` |

They therefore do not supply a complete new reference model or a runnable SDK. The accompanying parsed JSON labels weight bits as 8 but includes no executable unpacker/runtime proof for that field. Continue using the original full components and the independently verified parameter-byte accounting; retain the JSON as a secondary claim source only.

To repeat this optional audit with copies you already hold, pass `--secondary-models /path/to/secondary-files` to `runtime_investigation.py`; the default performs no secondary-file comparison. The optional older parsed-JSON comparison is separately enabled with `packing_investigation.py --prior-manifest /path/to/s5_af_models_parsed.json` after model extraction and parameter recovery. Neither command assumes a contributor's home directory.
