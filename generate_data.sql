CREATE TABLE IF NOT EXISTS default.library_updates
(
    library_name String,
    old_version  String,
    new_version  String,
    release_date Date,
    update_type  String,
    update_notes String
)
    engine = MergeTree ORDER BY library_name
        SETTINGS index_granularity = 8192;



INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Highligts', 'We are excited to announce the release of PyTorch® 2.3! PyTorch 2.3 offers support for user-defined Triton kernels in torch.compile, allowing for users to migrate their own Triton kernels from eager without experiencing performance complications or graph breaks. As well, Tensor Parallelism improves the experience for training Large Language Models using native PyTorch functions, which has been validated on training runs for 100B parameter models.

         This release is composed of 3393 commits and 426 contributors since PyTorch 2.2. We want to sincerely thank our dedicated community for your contributions. As always, we encourage you to try these out and report any issues as we improve 2.3. More information about how to get started with the PyTorch 2-series can be found at our Getting Started page.

         Stable  Beta  Prototype  Performance Improvements
         User-defined Triton kernels in torch.compile  torch.export adds new API to specify dynamic_shapes  Weight-Only-Quantization introduced into Inductor CPU backend
         Tensor parallelism within PyTorch Distributed  Asynchronous checkpoint generation
         Support for semi-structured sparsity    ');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Backwards Incompatible Changes
         ', 'Change default torch_function behavior to be disabled when torch_dispatch is defined (#120632)
         Defining a subclass with a torch_dispatch entry will now automatically set torch_function to be disabled. This aligns better with all the use cases we’ve observed for subclasses. The main change of behavior is that the result of the torch_dispatch handler will not go through the default torch_function handler anymore, wrapping it into the current subclass. This allows in particular for your subclass to return a plain Tensor or another subclass from any op.

         The original behavior can be recovered by adding the following to your Tensor subclass:

         @classmethod
         def torch_function(cls, func, types, args=(), kwargs=None):
               return super().torch_function(func, types, args, kwargs)');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Tracked Regressions', 'torch.compile on MacOS is considered unstable for 2.3 as there are known cases where it will hang (#124497)
         torch.compile imports many unrelated packages when it is invoked (#123954)
         This can cause significant first-time slowdown and instability when these packages are not fully compatible with PyTorch within a single process.

         torch.compile is not supported on Python 3.12 (#120233)
         PyTorch support for Python 3.12 in general is considered experimental. Please use Python version between 3.8 and 3.11 instead. This is an existing issue since PyTorch 2.2.');
INSERT
INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Backwards Incompatible Changes
         ', 'ProcessGroupNCCL removes multi-device-per-thread support from C++ level (#119099, #118674)
         Python level support was removed in 2.2.
         To simplify ProcessGroupNCCL’s code, we remove support for multiple cuda devices per thread. To our knowledge, this is not an active use case, but it adds a large burden to our codebase. If you are relying on this, there is no workaround other than rewriting your pytorch program to use one device per process or one device per thread (multi-threads per process is still supported).');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Backwards Incompatible Changes
         ', 'Removes no_dist and coordinator_rank from public DCP API\'s (#121317)
         As part of an overall effort to simplify our public facing API\'s for Distributed Checkpointing, we\'ve decided to deprecate usage of the coordinator_rank and no_dist parameters under torch.distributed.checkpoint. In our opinion, these parameters can lead to confusion around the intended effect during API usage, and have limited value to begin with. One concrete example is here, #118337, where there is ambiguity in which Process Group is referenced by the coordinator rank (additional context: #118337). In the case of the no_dist parameter, we consider this an implementation detail which should be hidden from the user. Starting in this release, no_dist is inferred from the initialized state of the process group, assuming the intention is to use collectives if a process group is initialized, and assuming the opposite in the case it is not.

         2.2  2.3
         # Version 2.2.2
         import torch.distributed.checkpoint as dcp

         dcp.save(
           state_dict={"model": model.state_dict()},
                checkpoint_id="path_to_model_checkpoint"
                no_dist=True,
                coordinator_rank=0
         )
         # ...
         dcp.load(
           state_dict={"model": model.state_dict()},
                checkpoint_id="path_to_model_checkpoint"
                no_dist=True,
                coordinator_rank=0
         )
         # Version 2.2.3
         # no dist is assumed from pg state, and rank 0 is always coordinator.
         import torch.distributed.checkpoint as dcp

         dcp.save(
           state_dict={"model": model.state_dict()},
                checkpoint_id="path_to_model_checkpoint"
         )
         # ...
         dcp.load(
           state_dict={"model": model.state_dict()},
                checkpoint_id="path_to_model_checkpoint"
         )
         ');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Backwards Incompatible Changes
         ', 'Remove deprecated tp_mesh_dim arg (#121432)
         Starting from PyTorch 2.3, parallelize_module API only accepts a DeviceMesh (the tp_mesh_dim argument has been removed). If having a N-D DeviceMesh for multi-dimensional parallelism, you can use mesh_nd["tp"] to obtain a 1-D DeviceMesh for tensor parallelism.

         ');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Backwards Incompatible Changes
         ', 'torch.export
         Users must pass in an nn.Module to torch.export.export. The reason is that we have several invariants the ExportedProgram that are ambiguous if the top-level object being traced is a function, such as how we guarantee that every call_function node has an nn_module_stack populated, and we offer ways to access the state_dict/parameters/buffers of the exported program. We\'d like torch.export to offer strong invariants—the value proposition of export is that you can trade flexibility for stronger guarantees about your model. (#117528)
         Removed constraints in favor of dynamic_shapes (#117573, #117917, #117916, #120981, #120979)
         ExportedProgram is no longer a callable. Instead users will need to use .module() to call the ExportedProgram. This is to prevent users from treating ExportedPrograms as torch.nn.Modules as we do not plan to support all features that torch.nn.Modules have, like hooks. Instead users can create a proper torch.nn.Module through exported_program.module() and use that as a callable. (#120019, #118425, #119105)
         Remove equality_constraints from ExportedProgram as it is not used or useful anymore. Dimensions with equal constraints will now have the same symbol. (#116979)
         Remove torch._export.export in favor of torch.export.export (#119095)
         Remove CallSpec (#117671)');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Backwards Incompatible Changes
         ', 'Enable fold_quantize by default in PT2 Export Quantization (#118701, #118605, #119425, #117797)
         Previously, the PT2 Export Quantization flow did not generate quantized weight by default, but instead used fp32 weight in the quantized model in this pattern: fp32 weight -> q -> dq -> linear. Setting fold_quantize=True produces a graph with quantized weights in the quantized model in this pattern by default after convert_pt2e, and users will see a reduction in the model size: int8 weight -> dq -> linear.

         2.2  2.3
         folded_model = convert_pt2e(model, fold_quantize=True)
         non_folded_model = convert_pt2e(model)
         folded_model = convert_pt2e(model)
         non_folded_model = convert_pt2e(model, fold_quantize=False)');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Backwards Incompatible Changes
         ', 'Remove deprecated torch.jit.quantized APIs (#118406)
         All functions and classes under torch.jit.quantized will now raise an error if called/instantiated. This API has long been deprecated in favor of torch.ao.nn.quantized.

         2.2  2.3
         # torch.jit.quantized APIs

         torch.jit.quantized.quantize_rnn_cell_modules

         torch.jit.quantized.quantize_rnn_modules
         torch.jit.quantized.quantize_linear_modules

         torch.jit.quantized.QuantizedLinear
         torch.jit.QuantizedLinearFP16

         torch.jit.quantized.QuantizedGRU
         torch.jit.quantized.QuantizedGRUCell
         torch.jit.quantized.QuantizedLSTM
         torch.jit.quantized.QuantizedLSTMCell
         # Corresponding torch.ao.quantization APIs

         torch.ao.nn.quantized.dynamic.RNNCell

         torch.ao.quantization.quantize_dynamic APIs

         torch.ao.nn.quantized.dynamic.Linear

         torch.ao.nn.quantized.dynamic.GRU
         torch.ao.nn.quantized.dynamic.GRUCell
         torch.ao.nn.quantized.dynamic.LSTM');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Backwards Incompatible Changes
         ', 'Remove deprecated fbgemm operators (#112153)
         TorchScript models that were exported with the deprecated torch.jit.quantized API will no longer be loadable, as the required internal operators have been removed. Please re-export your models using the newer torch.ao.quantization API instead.

         ');


INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Backwards Incompatible Changes
         ', 'Other
         Make List::get() const match List::operator[]() const (#117568)
         Delete C10_IS_TRIVIALLY_COPYABLE (#120120)
         Fix synchronization behavior for copies with type change (#121341)');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Deprecations', 'torch.autograd.Function: Using the torch.autograd.function.traceable decorator and getting/setting torch.autograd.Function\'s is_traceable is now deprecated (#121413)
         These decorators were previously marked for internal use only. They will be removed in version 2.4.

         ');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Deprecations', 'torch.utils.checkpoint: not passing use_reentrant explicitly to activation checkpoint and checkpoint_sequential is deprecated (#116710)
         torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.

         (Note that this was already deprecated in a previous release. In this version, we improve the deprecation message.)

         ');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Deprecations', 'Deprecated torch.backends.cuda.sdp_kernel and replace with torch.nn.attention.sdpa_kernel (#114689)
         This PR deprecated torch.backends.cuda.sdp_kernel, users can now use torch.nn.attention.sdpa_kernel instead. The old code will raise the following warning: FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated. In the future, this context manager will be removed. Please see, torch.nn.attention.sdpa_kernel() for the new context manager, with updated signature.

         2.2  2.3
         import torch
         from torch.backends.cuda import sdp_kernel

         with sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
             torch.nn.functional.scaled_dot_product_attention(...)
         import torch
         from torch.nn.attention import sdpa_kernel, SDPBackend

         with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
             torch.nn.functional.scaled_dot_product_attention(...)');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Deprecations', 'Distributed API
         [c10d] Deprecate Work.result() (#117565)
         Deprecate torch.distributed.pipeline in favor of the PiPPy library (#121464)
         Add deprecation msg for NO_SHARD (#119553)
         Add composable API fully_shard deprecation warning (#120929)
         [DTensor] Change distribute_module input/output_fn to accept module, deprecating the input/output_fn that does not accept a module (#120895)
         ');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Deprecations', 'Releng
Removal of macOS x86 binaries build jobs following their deprecation in 2.2 (#116726)
         CircleCI removed (#115701)');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'New Features', 'Autograd API
         Add basic autograd TORCH_LOGS support. (#115438)
         Autograd attaches logging hooks only in debug level (#116522)
         Update torch.autograd.graph logging to not print out grad_output (#116523)
         Added "any" mode to register_multi_grad_hook (#117984)
         CUDA
         Add bfloat16 CUDA support for smoothl1loss (#116933), RNN (#116927), gamma unary functions (#116929), binomial distribution (#116932), and multinomial (#116951)
         Add float16 support to CUDA logaddexp2 (#116948)
         Add bfloat16 + fp16 support to fractional_max_pool for CUDA and CPU (#116950)
         Make torch.cuda.has_magma a build time check (#116299)
         Distributed API
         C10d:
         Install an excepthook which annotates exceptions with rank information when distributed is initialized (#118190)
         Flight recorder for debugging failed collectives: (#119837, #116905, #114817, #118044, #115139, #118046, #119249, #118047 , #118142, #115771, #120063, #120331, #120544, #120262, #120724, #120270, #120893, #120502)
         explicitly abort communicators in destroy_process_group call (#119250)
         ProcessGroupGloo::allgather_into_tensor_coalesced (#118910)
         ProcessGroupGloo::reduce_scatter_tensor_coalesced (#118911)
         Create a python c10d API _set_pg_timeout to set timeout (#115453)
         Introduce 3 low-latency, intra-node allreduce algorithms for small messages to PyTorch (#114001) (#116125)
         Add complex support for P2P (#121240)
         Distributed Checkpointing (DCP):
         Adds Checkpointer Wrapper for DCP [3/N] (#114603)
         Adds async save, makes checkpointer private (#116293)
         Makes async_save public (#121325)
         Adds storage reader and planner classes for online loading/sharding of models in torch.save format (#119816)
         Adds support for meta tensor loading for DCP.load_state_dict() (#113319)
         [state_dict] Implement pin_memory and shared_memory copy for _offload_state_dict_to_cpu (#120378)
         Add tests to demonstrate DCP checkpoint conversion (#117773)
         Adds utility for converting dcp to torch save format (#119814), converting torch save to dcp (#119815)
         Add distributed checkpoint support for custom device (#120201)
         Add a profiler function for benchmarking save and load (#116007)
         Enables load balancing duplicates in DCP (#116469)
         Enable filesystem/fsspec auto detection (#118888)
         DTensor:
         Add op support for aten.gather.default (#118513)
         Add op support for nll_loss_backward (#119256)
         Implement scaled dot product attention (flash-attention, forward only) (#120298)
         Add async_op option to redistribute and some refactor (#121477)
         [TP] add support for loss parallel (#119877)
         [TP] Introduce Sequence Parallel Style for Laynorm/RMSNorm/Dropout (#121295)
         allow OpStrategy to represent ops whose return type is a tuple (#115682)
         Add op support for nll_loss_forward (#118917)
         Supported foreach=False for clip_grad_norm_ (#120238)
         Supported foreach=True for clip_grad_norm_ (#120910)
         Add layer norm backward support (#115683)
         Add Silu operator support (#118702)
         Enable adadelta foreach optimizer (#115564)
         Enable radam foreach optimizer (#115566)
         Enable Adamax foreach optimizer (#119850)
         Make add_.Tensor/div_.Scalar to be linear pointwise instead (#121294)
         [DeviceMesh] Allow 1d slice from 1d mesh (#118895)
         Add a implicit replication flag (#115297)
         Adds tool to visualize sharding (#114307)
         Initialized RNG tracker if needed (#121328)
         get CommsDebugMode to work with DTensor (#118769)
         [DTensor][XLA] support XLA backend in distribute_module API (#121355)
         DDP
         Use compiled_autograd to trace DDP backward allreduce (#110662)
         Functional Collectives
         Support broadcast in native funcol (#119229)
         Allow using native c10d_functional via _functional_collectives (#113057)
         Implements permute_tensor in functional collectives (#115078)
         Support tracing native functional collective via python APIs (#119103)
         Directly import DeviceMesh to avoid circular dependency (#115649)
         Port all_to_all_single to native c10d_functional (#113438)
         Make native c10d_functional ops work with AOTInductor (#113735)
         fix an issue where mutation on views fails in inductor (#118333)
         Run device mesh tests with native funcol enabled (#118437)
         Change the .clone() in native funcol\'s all_reduce to use at::MemoryFormat::Contiguous (#120042 )
         Make tests using CommDebugMode work for both legacy and native funcol (#120070)
         Temporarily support ranks + tag as pg identifier in native funcol (#120226)
         don\'t import torchdynamo when running torchdeploy (#120900)
         Preliminary DeviceMesh + native c10d functional integration (#118423)
         Change native funcol inductor tests to use fake pg (#119104)
         Prepare test_inductor_collectives.py for native funcol migration (#120025)
         Prepare test_dtensor.py for native funcol migration (#120043)
         Disable GroupRegistry\'s thread isolation by default (#121457)
         FX
         Experimental non-strict mode (#114658)
         Add _assert_scalar and teach Inductor to codegen it (#114148)
         Add symbol_guard_limit_before_specialize (#119347)
         Add _assert_scalar and teach Inductor to codegen it (#114148)
         torch.compile
         Dynamo
         Mutable variable trackers - TorchDynamo variable trackers are now mutable. Therefore, we can just mutate the variable tracker during symbolic tracing, instead of creating a new variable tracker. This improves compilation time for models with frequent mutations.
         (beta feature) Improved automatic deletion of compilation units - TorchDynamo caches the compilation units internally on every compilation event. Earlier, this cache was not cleared automatically when the original nn module went out of scope, holding on to the GPU memory. Therefore, users needed to call torch._dynamo.reset() to clear the cache. Now, we automatically clear the cache as soon as the nn module goes out of scope.
         (beta feature) torch.compile works with User defined triton kernels - Allows for PyTorch code that contains a triton kernel to be executed natively using torch.compile. This will allow users to migrate code containing triton kernels from eager PyTorch to using torch.compile without running into performance complications or graph breaks. A good entry point to how it works is on https://github.com/pytorch/pytorch/blob/main/torch/_higher_order_ops/triton_kernel_wrap.py
         (beta feature) Improved tracing rules infrastructure - Added a central infrastructure to control whether to inline or skip TorchDynamo tracing.
         Create a sentinel file for each dynamo test skips (Part 1) (#120500)
         Windows Dynamo Error Removal CI Check (#115969)
         Add TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED (#118750)
         feat: Add min, max ranges to mark_dynamic API (#119737)
         Add support for dynamic shapes in round (#115259)
         Add support for torch.cond in vmap (#114523)
         Added dynamic shapes support for math trigo ops: sin(h), cos(h), tan(h) ... (#114866)
         Add dynamo support for operator.abs (#117442)
         Add support for operator.truth (#117463)
         Add support for compiling SDPAParams (#117207)
         Add more dynamo call_methods and getattr support or Placement (#117733)
         Add size(), get_coordinate() support for DeviceMesh in dynamo (#117710)
         Initial torchbind support in PT2 (#117697)
         Add compilable foreach RAdam support (#117912)
         [HigherOrderOp] support while_loop in dynamo (#116913)
         Add Support for CausalBias to torch compile (#116071)
         support dict.clear() (#119197)
         Add support for labels to ttir analysis (#119836)
         Add jacrev support in torch.compile (#121146)
         Introduce size oblivious guards (#118579)
         Inductor
         Some basic support for uint{16,32,64} codegen in CPU inductor (#116810)
         Enable for MacOS (#118076)
         Add torch.cond support to JIT Inductor (#119759) and AOT Inductor (#121120)
         Double buffering for Weights (#114446)
         Add Runtime Constant-folding for AOTInductor (#118765)
         Intel GPU backend Upstream: Generalize device-bias code in code generation (#116020), Register and add Intel GPU Inductor backend (#116330)
         torch.export
         Introduced pre-dispatch IR through torch.export._trace._export(..., predispatch=True). This returns a graph that contains ATen operators at a higher level (aten.linear.default will not become decomposed) and are also safe for training (#117278)
         Support higher order op functionalization in predispatch IR (#115314)
         Ignore autograd ops for predispatch export (#116527)
         Introduced non-strict export (#118607, #118609, #119297, #119446, #119602)
         Transformed global state mutating operators torch._C.set_grad_enabled into a higher order operator (#119732, #119736, #119810, #119913, #119915)
         Added while_loop support (#116823)
         A very preliminary torchbind object tracing support was added. (#116985, #117978, #117979, #118158, #118684)
         Linalg
         Add HIPSolver backend to linalg.eigh (#115177)
         Add MKLDNN backend to matmul for bfloat32 (#116015)
         Support torch.mm with conjugate transposed inputs (#117238)
         MPS
         Add complex_out to MPS backend (#110851)
         Add Conv3D support for MPS (#114183)
         Enable select/[broad]cast ops for complex dtypes (#115727)
         Add torch.fft. support (#119670)
         Add support for 64-bit index operations (#116942)
         torch.nn API
         Added swap_tensors path to nn.Module._apply (#117167)
         Integrated swap_tensors into nn.Module.load_state_dict ((#117913)
         Added assign argument to torch.Tensor.module_load (#121158)
         Added an attention bias subclass for a lower right causal masking (#114823)
         Added config to disable TransformerEncoder/MultiHeadAttention fastpath (#112212)
         Profiler
         Enable profiler initialization to enable on-demand profiling for CPU only mode (#118320)
         Add execution_trace_observer as an optional argument to Profiler API (#119912)
         Python API
         Add new torch.utils.swap_tensors function that can be used to exchange Tensor while preserving references from other objects (#111747)
         Add unsigned integer dtypes with full interop with third party (#116594, #116805, #116806, #116807, #116808, #116803, #116804)
         Add {Untyped,Typed}Storage.resizable() to check if a Storage can be resized (#119286)
         add Half support for flash attention (#119247)
         Sparse
         Add bfloat16 support to torch.sparse.addmm for CPU (#115535)
         Add tune_bsr_dense_addmm as an API to find optimal triton kernel parameters for bsr_dense_addmm (#115499)
         Add values backward support for sparse CSR, CSC, BSR, and BSC tensors (#115586)
         Add batched sparse CSR/CSC/BSR/BSC to sparse COO conversion support (#116206)
         Add meta device support to sparse compressed tensors (#120498)
         Add sparse compressed meta tensor support (#120707)
         Add sparse compressed fake tensor support (#120920)
         [semi-structured] Enable fp32 support, separate sparse and dense constraints (#115550)
         Support add(sparse_compressed, dense) (#115432)
         Support csc layout for add sparse/dense. (#115433)
         Add in out_dtype support (i8i8->bf16, i32) for cusparselt (#119296)
         Vulkan
         Added Vulkan support for 1D for the following ops: Convolution (#117780, #118660, #118833, #118834, #118835) Linear (#118690)
         XPU
         Intel GPU Runtime Upstreaming for Device (#116019, #116833, #116850, #116869, #117611, #117619), Event (#117734), Device Allocator (#118091), Guard (#118523), and Generator (#118528, #118613)
         Other
         Enable compiled Adam in the benchmarks (#116093)
         [ROCm] TunableOp (#114894)
         [pytree] Add access api (#117771)');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Improvements', 'Autograd API
         Out-variant of ops for which autograd also does not have a formula now produce an improved error message when the out= argument is a leaf tensor (#121089)
         Support custom autograd Function forward AD return non-Tensor in forward (#118234)
         In-place foreach to returns the input list as-is instead of None (#121405)
         Composability
         FakeTensors and meta tensors are used to perform shape propagating when tracing out a graph in torch.compile. There were a number of op coverage improvements this release:
         New metas
         _foreach_norm (#119927)
         _upsample_bicubic2d_aa (#117347)
         Fixes to metas
         _efficient_attention_forward for jagged inputs (#118657)
         _flash_attention_forward() (#119812)
         efficient_attention_forward fix for NT inputs (#120594)
         We have python “reference” decompositions for many aten operators. These are used during the tracing step of torch.compile in a few ways: sometimes they are used to directly decompose operators in the captured graph. Other times, they are used as an alternative to a shape-propagation rule for an operator. There were several improvements to operator coverage in this release
         New decomps
         reflection_pad{1, 2, 3}d (#115100)
         replication_pad (#115113)
         torch.block_diag (#115096)
         torch.take (#114813)
         rsub (#118288)
         roll (#119857)
         frexp (#119217)
         pad_sequence (#116285)
         linalg.cross (#119809)
         Fixes to some decomps:
         SDPA decomp: actually use attn_mask (#117579)
         aten.diag_embed (#120549)
         Isin (#120821)
         linalg norm (#120993)
         Im2col: Remove opmath cast (#121363)
         Index_copy: fix 0-dim Index (#117065)
         Rrelu_with_noise: add default parameters (#117141)
         _refs.linalg.svd: don’t check is_conj() (#117972)
         Weight norm interface decomp: fix missing default dim param (#118762)
         SDPA decomposition: add tests for different dtypes (#119239)
         _to_copy fix (#119868)
         torch.Tensor decomp fix to support sequences of tensors (#120872)
         Avoid mkldnn ops appearing in graph in some cases (#115448)
         Decomps relevant to the Core ATen opset defined here.
         Add decomp pixel_shuffle/unshuffle (#118239), (#118921), (#120092)
         Modify SDPA decomposition to decompose _scaled_dot_product_flash_attention_for_cpu (#117097)
         Dynamic shapes:

         Improved unbacked SymInt support
         Support symbolic min/max on unbacked SymInt (#118953)
         Rewrite maybe_reduce more carefully for unbacked SymInt (#119562)
         Size oblivious test for slice optimization (#119625)
         If data dependent, check if guard_size_oblivious would fix problem and report if so (#121011)
         Prevent unbacked symbol reallocation by forcing unification for unbacked symbol def sites (#114368)
         Improved dynamic shapes support
         Always accept 0-d scalar tensors as int, even if index fails (#117451)
         SymIntify prod_backward (#120776)
         Add is_integer to SymFloat (#114703)
         Dedupe symbolic shapes in tracing (#116158)
         [Dynamic] Fix dynamic shape size inspection bug (#120341)
         Docs / debugging tools
         Add basic reference documentation for symbolic_shapes.py (#118997)
         Rename unbacked SymInt prefix to u (#117859)
         Augment create_symbol with user/infra backtrace fragment (#118215)
         CPP API
         Add TensorIteratorConfig::add_const_input to avoid COW materialize (#118053)
         Reserve sizes in c10::VaryingShape::concrete_sizes(), c10::TensorType::computeStrideProps() (#119189)
         CUDA
         Faster gc_count update for CUDACachingAllocator (and avoid nullptr de… (#117064)
         Reduce register usage of fused adam(w) (#118361)
         Improve CUDACachingAllocator lock contention (#118550)
         Back scalar value to pinned memory for .item() (#119202)
         Avoid COW materialization for TensorInfo with const type (#119502)
         [CUDA graphs] Pool argument for make_graphed_callables (#121475)
         [CUDA Caching Allocator] Export sync-stream-and-free-HBM counter in memory_stats for performance debugging (#120050)
         Distributed API
         c10d:
         ProcessGroup/NCCL logging improvements: (#115801, #116059, #116060, #116520, #117291, #117868, #118335, #113238, #118455, #118582, #118924, #116489 )
         NCCL ProcessGroup watchdog improvements: (#115403, #115577, #116702, #116267, #116717, #116545, #117312, #117093, #117682, #117297, #118016, #115770, #116661, #118344, #117699, #117168, #117738, #121132
         Use TCPStore to record NCCL timeout and dump debug info (#115226)
         Extend NCCL communicator splitting to more use cases (#114916)
         Let all_reduce_coalesced accept one tensor as well (#115650)
         Add stream info during nccl comm abort call (#116076)
         Pass group global rank information to NCCL PG (#114736)
         Store PG global rank information in tracing logs (#115730)
         Only open NCCL dump pipe file once per process (#115798)
         Dynamo + collectives: allreduce remap (#115950), all_gather_into_tensor remap (#117224)
         Expose check method to Python for store via pybind (#116144)
         Refactor all_reduce variants as private methods (#120855)
         To make ProcessGroupNCCL to use globalStore for coordination (#117075)
         Allow nonblocking wrap of ncclCommInitRankConfig (#118256)
         Do not print NCCL_DEBUG before NCCL init (#117328)
         Update the work progress of PG periodically (#120438)
         Add NCCL work sequence number to work info (#120596)
         [UCC] Retain CUDA context in progress_loop (#121446)
         Change watchdog log from "NCCL" to "Process group" (#118121)
         [IntraNodeComm] accept P2P buffer size as constructor argument (#120856)
         FSDP:
         [torch.compile] FSDP changes (#115497)
         Remove unused flat_param_part_view (#117082)
         Replace acc_grad hooking with register_post_accumulate_grad_hook on flat_param (#112184)
         Fix deprecation warning on typed storage (#116714)
         Pass DTensor shape/stride during tensor unflatten in 2D (#117340)
         Cloned unsharded tensor slice in optim state dict load (#117261)
         Drop all gather stats to debug not warning (#117669)
         Idempotent reshard (#117997)
         Removed .detach in clip_grad_norm_ (#120612)
         Vlean up unwanted _fsdp_wrapped_module FQNs (#120600)
         Distributed Checkpointing (DCP):
         Automatically set no_dist if distributed is unavailable (#119813)
         Let distributed_state_dict filter out the compiler prefix (#119830)
         Only wait on AsyncCollectiveTensor after DTensor-based state dict loading (#119716)
         Improve the readability of filesystem and fsspec filesystem (#116246)
         Uses Serial Loader for DCP.save when more then one thread is used. (#118114)
         Skip log line if no tensors were dedupped (DCP) (#119742)
         Removes Checkpoint Wrapped Prefix from state dict fqns (#118119)
         [DCP] Replaced storage() with untyped_storage() (#121538)
         Let _offload_state_dict_to_cpu to return the companion_obj if it exist. (#121273)
         Asserts CPU backend for async_save (#120241)
         Allow users to save and load without creating storage reader and writer (#117772)
         Passes process group to _all_gather_keys in dcp.load (#118301)
         DTensor:
         [DeviceMesh] Ensure mesh tensor is a cpu tensor (#120046)
         Standardize tuple strategy handling for foreach ops (#120695)
         Add mesh_dim_names to DeviceMesh repr if it exists (#115579)
         Remove assert to allow tensor sharding dimension < Shard(x).ndim (#115114)
         Refactor sharding cost model to count for latency (#119897)
         Make tensor_flatten more compatible for dynamo getattr (#118209)
         DTensor: use memory_format in the hash for all aten ops that use that arg (e.g. aten.clone) (#118667)
         [DeviceMesh] Removed print of self._dim_group_infos (#118527)
         Account for empty list when turning to OpStrategy (#115298)
         [debug] have visualize_sharding correctly print for sub-mesh DTensor (#121216)
         switch softmax backward ops to OpStrategy (#119255)
         Make DTensor from_local backward partial() to replicate() pass through (#115967)
         Refactor partial redistribution logic (#113334)
         Refactor redistribute and fix uneven sharding redistribution (#115525)
         Switch softmax forward ops to OpStrategy (#117723)
         Simplify outputs wrapping handling (#120297)
         Relaxed to_local requires_grad warning (#118186)
         Make local_shard_size_on_dim be staticmethod (#118078)
         TorchElastic:
         Support for overprovisioning in C10 based rendezvous (#117066)
         [rendezvous] Add option to enable libuv for TCPStore based rendezvous backend (#118944)
         Create root log directory by default (#121257)
         Refactoring to support non-default logging strategy (#120691)
         [Logging] Pluggable logsspecs using python entrypoints and option to specify one by name. (#120942)
         Refactor SubprocessHandler to separate module for easier subclass (#120373)
         torch.compile
         Dynamo
         Fewer graph breaks for dicts - ConstDictVariableTracker now supports many more types of keys, earlier it was just string and ints.
         Improved TorchDynamo reliability by fixing many bugs, increasing the pass rate of TorchDynamo wrapped PyTorch tests to 90%.
         Improve TORCHDYNAMO_EXTENDED_DEBUG for GuardOnDataDependentSymNode (#119412)
         Make torch._dynamo.mark_static work inside graph (#118962)
         Expand dynamic dims support for traceable subclasses (#114311)
         Extend auto_functionalized to support ops that return Tensors (#115135)
         Improve support for dynamic shapes str.format and _assert (#115203)
         make lookup_backend return None when cache misses (#114766)
         Make subclass type instances constants (like UserDefinedClasses) (#115323)
         [HigherOrderOp] make MapHigherOrder use should_flatten_output=True (#115204)
         Move the shape env symint cache to a symbol cache, better routing for subclass fakification [re-pr 115227] (#115396)
         Remove always restore (#115317)
         Remove replace_all and make VTs mutable (#113725)
         Support torch function user objects (#111765)
         Check tensor subclass when using torch.compile + SAC (#115960)
         Ensure wrapping subclasses with as_subclass is supported (#116091)
         Add CALL_FINALLY opcode (#116159)
         Add a wrapper to transform a NumPy function into a PyTorch function (#114610)
         [HigherOrderOp] set set_subgraph_inputs to flatten_manual for map (#115853)
         Graphbreak when creating a map with unsupported keys (#116460)
         Specialize SymNodeVariable when used as module index (#114377)
         Impl. call_hasattr for BaseUserFunctionVariable (#116049)
         [HigherOrderOp] change signature of map_impl (#117161)
         Error if compiled nondeterministic backward called in deterministic mode (#114780)
         Add hasattr support for TupleVariable (#117694)
         Break on unsupported keys for dicts / elements for sets (#117630)
         Implement set in terms of dict (#110524)
         Add DictView variable tracker (#108420)
         add common methods to DistributedVariable (#117590)
         make ConstantSource propagate through built-in ops for TensorVariable (#117704)
         Add compilable and capturable foreach adamax with tests (#117835)
         Enhance torch.vmap support from inside torch.compile (#116050)
         Install module globals per output_graph (#117998)
         Remove optimizer.step patching for profiler hook (#115772)
         add username in debug path (#117820)
         avoid graph break on tensor.element_size() (#118229)
         avoid graph break on torch.backends.cuda.matmul.allow_tf32 (#118236)
         move torch._C._get_cublas_allow_tf32 to constant_fold_functions (#118342)
         inline torch.jit._unwrap_optional (#118434)
         support inference_mode with no arguments (#118427)
         constant fold torch.cuda.get_device_properties to avoid graph break (#118422)
         Faster empty LIST_LENGTH guard (#118542)
         Expose dynamic_shapes api at multiple levels (#118695)
         graph break on isinstance calls if we don\'t know the type (#118778)
         Print the malformed guard when there\'s a guard error. (#117982)
         Use SourcelesBuilder in BuiltinVariable (#118098)
         [optim] Place guards on the args before assuming they exist (#117983)
         Make variables in dict LazyTrackers (not lazily guarded yet) and avoid using DICT_KEYS guard (#117625)
         Make dict guards amenable to the CSE pass (#118194)
         Add Typing variable to possible dict keys (#118003)
         Don\'t assume all subclasses of BaseUserFunctionVariable have a fn attribute (#118208)
         Add functools.partial and UserDefinedFunction to dict keys (#118199)
         [optim] Use the actual sources from the parameters when tracing "params" in an optimizer (#118535)
         Optimize dict keys guard when all the keys are constant (#118855)
         bypass graph break due to masking if inference mode (#119056)
         decrease logging level for graph break in higher order op. (#119079)
         Add torch.backends.mha.get_fastpath_enabled to FUNC_INLINELIST (#118979)
         support comparing stream with constant (#119199)
         Functools partial reconstruct (#118583)
         Print the value of constants in str (#119276)
         inlining into iter of user defined object (#119243)
         Support kwargs for lazy module (#119445)
         In dynamo tracing for index() use None as the default indicator for end and not -1 (#119151)
         Capture untyped_storage().resize_() (#119647)
         Respect autograd.Function + multiple save_for_backward calls (#117
         Support attribute access on tensor subclasses without sources (#117666)
         [functional_collectives] Add all_to_all_single, all_gather_list, reduce_scatter_list to dynamo remapping (#119683)
         [Optimus] Log the optimus graph transformation to the scuba (#119745)
         Update tracing rules for new cudnn functions (#120268)
         [guards-cpp-refactor] WEAKREF_ALIVE guard (#120344), DictGuardManager (#120359)
         derived dim (#118729)
         Let torch dynamo inline torch.func.grad (#118407)
         Teach dynamo about vjp (#119405)
         Support module backwards hooks (#120685)
         DICT_CONTAINS guard (#120673)
         KeyValueDictGuardManager (#121147)
         Re-dispatch torch.Tensor.new into torch.Tensor.new_empty method. (#121075)
         guard on grads being None in compiled optimizers (#121291)
         Use type check for also is_not (#113859)
         Relax missing symbols runtime assert (#121339)
         Add operator length hint support (#121495)
         Improve Dynamo support for torch function and class methods in general (#121365)
         Switch cudagraph backend to cudagraph trees (#121019)
         Support _unsafe_set_version_counter (#121086)
         Inductor
         [custom ops] Add tag to custom ops to preserve stride orders in inductor (#117298)
         Fix argument unused during compilation warning (#118077)
         Add a decomposition for isin() (#115390)
         Add no weight change version of fuse_parallel_linear (#115791)
         Allow user to explicitly specify Device to run on (#117413)
         Add torch._export.aot_load (#117610)
         Support .item() in the ABI-compatible mode (#117989)
         Add _scaled_dot_product_efficient_attention to C shim (#118169)
         Support scalar to tensor in the ABI-compatible mode (#118024)
         Replicate split_cat from torch IR to predispatch IR" (#118590)
         Change the cpp wrapper codegen for sdpa (#120592)
         Store OpOverload in ir.ExternKernel (#120629)
         Use torchgen to generate C shim functions (#120513)
         Update cpp wrapper codegen to use v2 C shim (#120714)
         Reuse generated kernels between constant graph and main graph (#121564)
         Update AOTI runner util (#116971)
         Remove caching for compiled model.so (#117087)
         Retrieve original FQNs for weights (#116157)
         Enable Dequant Promotion when Linear input dimension size exceeds 2 (#113912)
         Enable QLinear weight prepack when input dimension size exceeds 2 (#113928)
         Make some improvements to FX graph caching (#117888)
         Fuse parallel linear based on pre grad aten IR (#114776)
         Fuse pointwise operators in the post grad (#114778)
         Enable mkldnn op weight pre-packing on aarch64 (#115037)
         Add sub and div pointwise ops to the post grad fusion (#115389)
         [optimus] enable smart fusion (#115471)
         Parameterize ir.Scan on combine_fn (#109132)
         Add input numel assert for minimal arrayref interface (#113577)
         Remove ArrayRefTensor::dtype (#113578)
         Added non-integer expr support for floordiv in triton codegen (#115751)
         Updated upsample_bilinear2d decomposition (#104182)
         Consolidate usage of fp8 linears for inference models (#115808)
         Add lowerings for reflection_pad{1, 3}d_backward (#115645)
         Support Predispatch functionalization (#113728)
         Support sym exprs in lowering constant promotion (#116196)
         Add Support For Symbolic Shapes in Register_replacement, SDPA Pattern Matching (#115441)
         Handle more edge cases in slice and slice_scatter (#117377)
         Add statically_known_true utility for SymBool (#117359)
         Allow explicit shutdown of the compile-worker pools (#117664)
         Add runtime numeric check for pt2 Optimus in the pre grad pass (#115142)
         Allow reinplacing before meta-only users (#117121)
         Allow reinplacing functionalized scatter ops (#116899)
         Handle cum{sum,prod} on zero-dim tensors (#117990)
         Use an op counter to decide when to realize a kernel (#117030)
         Remove follow_imports = skip from sympy (#118469)
         Complete decomposition for aten.round (#118635)
         Never reuse accumulated gradients\' buffers (#119334)
         Vectorization support for int32/int64 (#119001)
         Update the compile options for CppPythonBindingsCodeCache (#119415)
         Add CUDAEvent recording for constant folding to show up. (#119216)
         Decompose torch.ops.higher_order.auto_functionalized in Inductor (#118673)
         Support storage resizing (#119749)
         Use torch.cuda.clock_rate instead of triton.testing.nvsmi (#118662)
         Handle aliases correctly in foreach (#119508)
         Add Runtime Constant-Folding function of AOTInductor for AOTInductorModels used internally. (#119823)
         Simplify indexing when doing ModularIndexing + index propagation. (#119863)
         Add split cat pattern to remove cat nodes (#115004)
         Decompose memory bound mm (#120047)
         [cond] make sure subgraphs in cond are decomposed according to current decomp table (#120366)
         Use a dtype property in torch inductor nodes (#119227)
         Add unbind node normalization (#120253)
         Reinplace auto_functionalized (#120829)
         Change the split cat log to debug (#120823)
         Add decompostition for mm in backward (#120933)
         Do not use warm_pool() if TorchTnT is used (#121047)
         Triage the remaining fallbacks (#121312)
         Enable ABI-compatible mode for cpp-wrapper JIT (#121309)
         Change assertion throw to error message for const_run_impl call. (#121396)
         Split predispatch pass into multiple passes (#121592)
         Port remove_split_ops to PT2 pre-grad passes (#121674)
         Replace lld with the default ld linker (#115478)
         Emit static constexpr int array vars when possible (#112174)
         Avoid aoti_torch_data_ptr calls for constants at inference time (#112405)
         Use static_cast, not dynamic_cast (#112798)
         Add minimal arrayref interface (#112800)
         Add updaing constant buffer to active buffer. (#116001)
         Add aoti_torch_view_dtype in C shim (#118705)
         Support _embedding_bag in C shim (#118706)
         Skip launching kernels with zero grid in AOTInductor when using backed symints (#118654)
         Support copy_, _fft_c2c and view_as_real in C shim (#119125)
         Migrate fuse_split_linear_add from dper_pass to AOTI based on predispatch IR (#118983)
         Add C-shim for index_put (#116667)
         Port fuse_parallel_linear (without changing weights) to PT2 pre-grad (#121617)
         [OAT] move matmul precision out of system info (#115242)
         [OAT] toggle for forcing matmul precision matching (#115326)
         Remove hashing of tensor data for constants (#115356)
         [Triton] Replace triton.runtime.jit.get_cuda_stream with torch.cuda.c… (#115397)
         De-duplicate triton helper functions (#115546)
         Don\'t print disable_cudagraphs_reason when cudagraphs is disabled (#115489)
         Implement a deduplist data structure for name to user tracking (#115609)
         SDPA extend backward realized tensor alignment checking to forward realized tensors (#116069)
         Serve multistream graph captures from correct pool (#116199)
         Add input shape check for quantized conv binary lowering (#115247)
         Preserve strides of custom Triton kernel args (#116219)
         Add ABI shim function for torch.scatter_reduce (#116700)
         Use max sm clock when calculating device tflops (#116754)
         Replace recursive stable_topological_sort() with iterative. (#116761)
         Remove the float16 restriction for cpu cpp_wrapper (#116205)
         Sort unbacked symbols before iterating on them (#116421)
         Decompose bmm if batch2\'s last dim size is 1 and coordinate_descent_tuning is enabled (#116582)
         Control the cpp_wrapper mode with an env variable (#116615)
         Add shape checks to ExpandView (#113839)
         Add remaining user check for qconv binary fusion (#115809)
         add predispatch_pass to hold pass functions to be run when config.is_predispatch is true (#116788)
         [ROCm] Add opt-in option for inductor\'s layout optimisation on ROCm (#116329)
         Disable pointwise_cat on CPU (#116313)
         Don\'t access cluster_dims for too old version of triton (#117192)
         Check nan/inf for graph inputs (#117189)
         Iterative percolate tags (#117306)
         Update JIT Inductor cpp wrapper entry function signature (#119280)
         Make auto_functionalized HOP fallback in inductor (#117084)
         Realize non-ReinterpretView Views in custom Triton kernel args (#117468)
         Add torch.complex128 and torch.complex32 to DTYPE_TO_ATEN dictionary. (#117929)
         correctly retrieve the "shared" attribute from a Triton binary (#120666)
         Add lowering for adaptive_max_pool2d (#120254)
         Track constant\'s original_fqn mapping (#120524)
         enable fp8 cast for inductor CPU (#117737)
         Express y grid > 2^16 in terms of z grid (#121554)
         Use codegen reference for buffer to string (#117838)
         move disable_cudagraph_reason disabling after codecache is accessed (#117823)
         Add new pattern matchers for SDPA (#113004)
         sympy.Symbol is a subclass of sympy.Expr (#117857)
         For View.create(x, sizes) call realize_input() instead of realize() when handling unbacked symints (#117013)
         [ac][pattern matcher] Do not percolate tags beyond the inputs of matched portion (#118034)
         Add lowering to special.bessel_j0 (2nd try) (#118565)
         Dont fuse write into read if indexing differs (#118210)
         Handle special values correctly in ir.Scan codegen (#118788)
         Limit reductions into pointwise cat fusion (#118452)
         Update pointwise concat heuristics (#118453)
         Add lowering to special.bessel_j1 (#118992)
         Support ProxyExecutor argument codegen for sympy.Expr (#119166)
         Implementing missing magic methods on IR values. (#118933)
         Support sympy.expr in user-defined Triton kernel grid fn (#119165)
         Add lowering to special.modified_bessel_i0 (#118993)
         Add split scan kernel (#117992)
         Add lowerings to special functions (#119187)
         Use list comprehension to initialize unused_views. (#119618)
         Rewrite group_batch_fusion.find_independent_subset_greedy() to be iterative. (#118324)
         Recursivly unwrap_storage_for_input when convert_to_reinterpret_view fails (#119867)
         Replace generators with map. (#119818)
         Reorder if check to avoid more expensive check. (#119817)
         [scheduler] Use set for origin (#119861)
         Allow padding mm/bmm/addmm in the presence of dynamic dims (#120073)
         Enhance next_power_of_2 function (#120153)
         Always allow 64 bit in next_power_of_2 (#120164)
         Use two pass reduction for deterministic reduction order (#115620)
         Remove redundant to_dtype in Fused Schedular Nodes (#118365)
         Remove dependency of triton during inductor codegen (#120193)
         Pass device_str for async_compile.triton function (#120202)
         Colorization improvements for bandwidth profiler (#120343)
         Disable masked load for non-fp data types (#120558)
         Apply fx passes recursively to nested subgraphs (#120665)
         Make triton_meta be part of user defined triton kernel cache (#120809)
         Emit grid wrapper inlined with the user defined triton kernel (#120824)
         Add lowering for fraction_max_pool2d (#120460)
         Fix accuracy failure for a few models under freezing (#121054)
         Add a decomposition for torch.put, 2. (#120179)
         Skip foreach kernel for benchmark fusion (#121168)
         Move JK check to on-demand (#121182)
         Correctly read the cache key for remote cache (#121151)
         Make configs hash part of remote cache key (#121152)
         Fuse nodes with sizes (s0_s1_...,) and (s0, s1, s2, ...) (#120077)
         Use indices for constants in triton_meta (#121427)
         Skip welford combine on first reduciton loop iteration (#121488)
         Changes to support newer triton pin (#121267)
         torch.export
         Added effect token to export (#121424)
         Preserve constant fqn (#120664)
         Require pytree serialized_type_name (#120636)
         Added \'is_lifted_tensor_constant\' and \'get_lifted_tensor_constant\' utils (#120546)
         Use forward hooks to capture module signatures. (#120468)
         Allow str inputs in non-strict tracing (#120536)
         Support output types that are non tensors (#120804)
         Allow None as the meta value for tensor output. (#116664)
         Make spec comparison indifferent to fx collections (#118718)
         Support non-tensor tuple hoo outputs (#119402)
         Make control flow operators respect global decomp table (#120412)
         Make balance_gradient preserved in export (#120332)
         Ensure optional fields in the schema always have default value. (#121163)
         FX
         Ignore ill-formed solution of reduce_inequalities (#117310)
         Add an option to not retrace when doing op fusion (#118120)
         [minimizer] Defined traverse (#118889)
         [pytree] Properly register immutable collections (#120036)
         Skip less replacements (#119570)
         Refine value ranges on inequalities (#120800)
         Remove dead get_shape_groups (#120813)
         Simplify guards using info from previous guards (#121463)
         Inspect get_attr nodes for _decline_if_input_dtype (#118760)
         Optimize recursive_add_node in fx splitter (#117969)
         Slightly faster FX graph iterator (#121611)
         More strong typed codegen for partial specialized code on boolean (#117201)
         Add torch.fx.interpreter to uninteresting_files (#117460)
         Report function name in stack trace annotations (#117459)
         Cache dfs path in propose_partitions and re-use that later when trying to find cycles in the graph (#115943)
         [pytree] update treespec num_children access (#116370)
         [pytree] Allow tree_map_only to support predicate function as filter (#119974)
         Register torch.return_types in torch.fx._pytree (#120027)
         [pytree] Add key path api (#116786)
         [pytree] Reuse flatten_fn in flatten_with_keys_fn to ensure consistency (#117656)
         JIT
         Release the GIL in serialization when it is safe to do so (#120818)
         Improve support for boolean inputs to operators in TorchScript (#113835)
         NestedTensors
         Support ragged_idx != 1 on aten::is_same_size, aten::_to_copy (#118442)
         view: basic support for ragged_idx != 1 and _unsafe_view (#118317)
         Support nested tensor in check_trace (#121039)
         Add is_nested_int() (#119975)
         Rename singleton int to nested int (#119661)
         Linalg
         Avoid COW materialization in at::parallel_for/parallel_reduce (#120455), TensorAccessors with const type (#119501), and input materialization in more forward ops (#121070)
         [executorch] Run llama in xplat (#118831)
         MPS
         Add MacOS 14 runtime check (#115512)
         Add support for MPSDataTypeComplexFloat[16|32] (#115513)
         Fix sum and prod for complex types (#115554)
         Add support for MPSDataTypeComplexFloat[16|32] (#115513)
         Fix sum and prod for complex types (#115554)
         Enable torch.rand[n] for complex types (#115514)
         Implement aten::upsample_linear1d on mps (#115031)
         Speedup addmm (#116548)
         Enable bfloat16 support on MacOS 14 (#119641)
         Add naive std_mean implementation (#119777)
         Implement aten::upsample_linear1d on mps (#115031)
         Use dispatch with rethrow for indexing (#116903)
         Add function to materialize COW storages (#117053)
         Make addmm support empty matmul (#117223)
         torch.nn API
         Add python and C++ support for LPPool3d (#114199)
         Add compatibility with channels_last_3d for conv3d (#114790)
         Add Half support for interpolate operators on CPU (#105648)
         Add nn.Module.to_empty() suggestion in the error message (#119353)
         Explicitly set nn.Module.set_extra_state return type to None (#120161)
         Updated nn.Module._apply to not gate on should_use_set_data when swap_tensors is set (#120659)
         Add Half support for AvgPool2d on CPU (#109578)
         Fix AdaptiveAvgPool1D to account for shmem limit for certain input sizes (#115231)
         Add back initial Flash Attention support on ROCM (#115981)
         Add 64-bit indexing for CUDA avg_pool_backward (#114193)
         Add Half support for layer_norm on CPU (#99590)
         Add Half support for flash attention on CPU (#118368)
         ONNX
         Introduce decomposition skips using custom operator (#117314)
         Apply Modularizarion to ExportedProgram during ONNX Export (#119498)
         Disable opmath type promotion for div (#119112)
         Prevent instance_norm decomp for export (#120866)
         Allow ONNXProgram.save to use torch.load(..., mmap=True) for large models (#117295)
         Enable llama attention with dynamic shapes for onnxrt backend (#117009)
         Add bfloat16 support for scaled_dot_product_attention (#117878)
         Require Module to be passed to export (#117528)
         Improve support to mmap for ONNXProgram.save (#117863)
         Use environment variable ONNXRT_DUMP_PATH to dump onnx models created by onnxrt backend (#117551)
         Remove monkey-patch for torch.utils._rebuild_tensor (#120446)
         Enable custom ONNX model transforms in onnxrt dynamo backend (#120854)
         Add Float8 support to onnx exporter (#121281)
         Add support to save safetensors checkpoint directly into onnx (#121001)
         Update submodule onnx==1.16.0 (#123125)
         Optimizer
         Allow torch.float64 scalars (in addition to torch.float32) for forloop + foreach implementations (#115841)
         Add beta1 support to CyclicLR momentum (#113548)
         Add guardrails preventing complex params in SparseAdam and add complex support for L-BFGS (#118161, #118184)
         Add capturable API for the forloop/single tensor (foreach=False) implementation in Adamax, RAdam, and ASGD (#121183, #121260, #121264)
         Profiler
         Clean up first line of element text for readability (#120245)
         Add Total memory used after allocation in Trace View (#120339)
         Track context for SEGMENT_FREE and SEGMENT_UNMAP (#118055)
         Add CUDAAllocatorConfig details into snapshot metadata (#119404)
         Log process group config information in GPU trace’s distributedInfo field (#119443)
         Support GPU annotations for auto-trace jobs similar on-demand support (#114638)
         Record nccl version in distributed info (#121044)
         Add a function to allow adding preset user-defined metadata to traces (#121487)
         Python API
         Enable eye on CPU for bfloat16 dtype (#116616)
         Quantization
         PT2 Export Quantization Flow:

         Relax constraints on dtype and qscheme to allow for customizations (#116287)
         Skip conv-bn folding when there are no batchnorm ops (#116440)
         Add move_exported_model_to_train (#113492)
         Allow users to override train/eval behavior (#119091)
         Add convert callback to Observer module (#115001)
         Fix _disallow_eval_train error message (#119694)
         Add model_is_exported util function (#119726)
         Relax model_is_exported input (#120720)
         Add the operator of decomposed fake quant per channel (#121297)
         Add error check for input_edge annotation in Quantizer (#121536)
         Call sub-quantizers\' transform_for_annotation in ComposableQuantizer (#121548)
         XNNPACKQuantizer:

         XNNPACKQuantizer skip inserting observers for non-float Tensors (#114999)
         Add support for linear_relu in XNNPACKQuantizer (#117052)
         Support custom qmin/qmax for activation and weight for XNNPACKQuantizer (#117305)
         Fix module name filter for underscores (#119344)
         X86 CPU Inductor Backend:

         Enable QLinear input with multi dims in x86 CPU inductor backend (#113733)
         Add int8 linear op gelu for x86 CPU Inductor backend (#114852)
         Add dynamic quantization config for x86 inductor backend (#115337)
         Enable QConv2d with hardswish post op (#117487)
         Add Hardswish Conv2d Unary Annotation (#117488)
         DTypes:

         Add uint1 to uint7 dtypes (#117208)
         Add float8 types to dtypes table (#117375)
         Enable cat for cuda torch.bits types (#115044)
         Others:

         Skip privateuse1\'s checkZeroPoints (#114117)
         Support lowering for operator.matmul in fx graph mode quantization (#113954)
         Add quantized gelu (#119935)
         Update Quantizable LSTM to support QAT (#121448)
         Releng
         Addition of linux cpu test for 3.12 (#117853)
         Bazel CUDA tests timeout increased to 480s (#120443)
         Update torchbench commit pin, add sam_fast benchmark (#121420)
         Trigger a mergability check on ghstack prs (#115944)
         Use matrix generate script for docker release workflows (#115949)
         [CI] Addition of initial inductor cpu smoketest for performance (#116456)
         [CI] Addition of python test skip logic for XPU (#117621)
         [ROCm] upgrade CI to 6.0 (#119495)
         Improved Dynamo testing convenience (#116173)
         Improved the error message when a PR lacks the necessary approvals (#116161)
         [CI] Addition of initial ci build test for XPU (#116100)
         Addition of CPU inductor merge rule (#116679)
         ROCm
         Add hipblaslt support (#114329)
         Initial Flash Attention support on ROCM (#114309)
         Make ATen-cpu cuda/rocm agnostic (#121082)
         Autocast RNN Support (#121539)
         Add Flash Attention support on ROCM (#121561)
         Initial ir.Scan/aten.cumsum lowering support on ROCm (#119369)
         Other
         Multiprocessing api to use sigkill if sigterm doesn\'t kill the process (#115219)
         Explicitly error out if CuDNN older than 8.5 (#118235)
         Set maximum supported version of Python as 3.12 (#119743)
         Update cslt to 0.5.2.1 (#115988)
         Update cutlass from 3.3.0 to 3.4.1 (#120434)
         Preserve metadata for MutableMapping and MutableSequence in pin_memory and collate_fn (#120553)
         Add inf norm support for _foreach_norm (#118441)
         Expose aggressive_recomputation as an inductor config (#118943)
         Disables denormal floating numbers on ARM CPU (#115184)
         Enable TORCH_TRACE by default in all Tupperware like environments (#120915)
         Upgrade submodule oneDNN to v3.3.6 (#122164)
         Pin protobuf to 3.20.2 on macOS (#121918)
         Make PyTorch compilable against upcoming Numpy-2.0 (#121880)
         Upgrade submodule pybind to 2.12.0 (#122899)');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Bug fixes', 'Autograd API
         Fast gradcheck bug no longer ignores specified eps argument when recomputing in slow mode to produce the error message (#115634)
         Properly handle retains_grad hook on the base when view of it is mutated (#117552)
         Composability
         Fixed hash issue in fx_graph_cse graph pass (#119567)
         Better support for fakeifying torch_dispatch tensor subclasses that are views (#118405)
         Bugfix for where output of a compiled graph is a subclass, but is a view of a graph input (#118191)
         CPP API
         Add the bound check for flatten with out_dim (#120894)
         torch check the division by zero in batch_norm_update_stats (#120882)
         Fixed crash when calling pad_packed_tensor when packed with cuda tensors and ensure_sorted=false due to indexing with tensors on different devices (#115028)
         [Caffe2] Fix bug in str on wide types (#117531)
         Try creating a bf16 tensor as a last resort of is_bf16_supported(). (#115924)
         Fix admm over empty tensors and broadcastable input (#118619)
         Distributed API
         C10d:
         [c10d] Fix Store check condition in NCCL PG watchdog (#115475)
         [c10d] Fix compilation of NCCL_EXP path (#119805)
         Don\'t add NCCL backend by default without CUDA (#119149)
         [IntraNodeComm] fix a hybridCubeMeshAllReduceKernel breakage caused by a recent refactor (#121575)
         Add partial read test for libuv backend and fix an error which only happens when partially reading a buffer (#116141)
         Fix get_rank under a non-default group. (#120481)
         Fix a bug where nn.functional._AllGather.backward produces wrong gradients (#120582)
         Fix default world_size when running on 1 or 0 GPU (#119372)
         Fix distributed debug w/ non-equal split (#115483)
         [AMD] Fix build for intra_node_comm (#116291)
         Fix timeout dump path write path overlap when there are multiple PGs (#116218)
         Handle unwaited work objects on process termination (#119881)
         Guarantee init cuda before attaching hooks (#120052)
         Remove backend_id from pg_info (#120038)
         Fix logic for default group=None in _set_pg_timeout (#120686)
         Make _set_pg_timeout work with DeviceMesh PG (#120850)
         Fix false positive ‘errors’ due to ‘reason’ string (#120863)
         Fix the hang issue in store.check(TIMEOUT_DUMP) (#116297)
         FSDP:
         Sharded grad scaler: copy found_inf after waiting on async reduce_all (#115710)
         Fix FSDP + TP state dict in param unflattening (#115105)
         Fixed device_mesh and auto wrap (#119064)
         Added warning about unsupported double backwards (#120926)
         Distributed Checkpointing (DCP):
         Call os.sync if os.fsync does not work for fsspec (#119287)
         Fixes expected behavior when no_dist=True in state_dict_loader.load (#115660)
         Fix no shard state dict loading (#120367)
         DTensor:
         Force re-compute sharding when normalized_shape differs in fwd layer norm (#115250)
         Fix is_tensor_shardable to correctly handle Replicate placement (#117726)
         Fix unnecessary redistribute in new_factory_strategy (#118037)
         Make input contiguous for DTensor reduce scatter to fix the incorrect numerical values (#115847)
         DTensor + dynamo: fix is_shard/replicate always inlining to False (#118668)
         to_local backward grad placement passthrough (#121474)
         nn.Module: use swap_tensors for Tensor subclasses (#122755)
         Fix swap_tensors path in _apply for modules that inherit from RNNBase (RNN, GRU, LSTM) (#122800)
         DeviceMesh
         Fix fsdp device mesh depenency issue (#121061)
         Cache and reuse sliced result (#122975)
         DDP
         Pass inductor strides forward in ddp optimizer (#120523)
         Ignore gradient sync if the gradient is not defined (#120419)
         DistributedDataParallel._post_forward, fix return (#114678)
         Lazily compile submodules - to propagate real tensor strides to backend compiler (#114154)
         Fix wrong behavior of is_alias_of and c10d::reducer on MTIA (#115553)
         TorchElastic:
         Correctly detect SystemExit with code == 0 when using –run-path #119697
         Misc
         Fix ChunkShardingSpec metadata offsets for empty shards (#121002)
         torch.compile
         Dynamo
         Fix F632 bug in dynamo (if statement is always false) (#116867)
         Properly trace into mark_static (#120232)
         Handle guard_size_oblivious in user code (#120379)
         Do not attempt to make nditer spawned arrays writable (#120868)
         Fix autograd.Function x enum input x torch.compile (#115206)
         Fix handling of one_hot (#116338)
         Fix functools.reduce() function with None as initial (#116398)
         Fix sum() function with start argument (#116389)
         Fix torch function kwarg dispatch (#117083)
         Fix several bugs related to unbacked SymInt codegen in inductor (#117862)
         Fix Auto Functionalize to handle specified default values (#118331)
         Fix __name on a reconstructed NestedUserFunctionVariable (#118768)
         [HigherOrderOp] fix stack trace to report user stack (#118826)
         Fix dupe deprecated warning in dynamo export (#120896)
         Fix support for nn.Parameter constructor (part 1) (#120163)
         Fix gradient refcounts in pybind and compiled autograd (#118817)
         Inductor
         Fix torch.bernoulli decomposition return type (#115699)
         Fix angle decomposition return type (#115700)
         Allow sympy expressions to participate in type promotion (#115676)
         Do variance calculation in opmath type (#115181)
         Properly unwrap_storage tensors sent to DynamicScalar (#117444)
         Catch some missing unbacked symbol dependencies (#117650)
         Don\'t try to directly compare symbols, it won\'t work (#117674)
         Changed return type of randint64_cpu to int64_t to prevent codegen is… (#117443)
         Realize inputs to DynamicScalar before unwrapping storage (#118125)
         Prevent DCE\'ing unbacked SymInt for view outputs (#119552)
         Exclude operators that produce unbacked symbols (#120917)
         Fix guards for code objects (#120909)
         Fix a bug in batch linear fusion in the post grad (#115061) (#115131)
         Add missing include to model.h (#118075)
         Fix a bug in the torch._export.aot_load API (#118039)
         Fix a None as index codegen issue (#118187)
         Forward fix #117989 (#118291)
         Fix a strict-aliasing warning (#120628)
         Fix broadcast_tensors with unbacked symints when translation validation is off (#118066)
         Fixed issue with true div on integer input with dyn shapes (#115920)
         Fix QConv Binary Inplace Layout Issue (#115613)
         [Optimus] Fix batch layernorm numerical issue (#117404)
         Fix a bug in merge_splits (#117707)
         [Optimus] Fix a bug in gradients computation for runtime numeric check (#118105)
         Fix key error in pre_grad fx_passes_numeric_check (#118325)
         [Runtime numeric check] Fix compatibility issue (#118578)
         Fix a RAIIAtenTensorHandle premature deallocation bug (#118963)
         Fix FallbackKernel behavior on mutable ops (#118649)
         Fix compile error on scan with no mask (#119555)
         Fix a bug in merge_splits (#119956)
         Fix lint after #105590 (#120461)
         Fix "example_value" absent for stack nodes (#120655)
         Fix missing "example_value" for nodes introduced by group batch fusion (#120974)
         Forward fix lint after 121202 (#121425)
         Fix cudagraph check message (#115664)
         Fix constant folding and extern kernel mutation tracking bugs (#115908)
         Fix cpp_wrapper inputs mismatch (#116197), codegen for ir.ComplexView (#116481), cumsum codegen (#116171)
         Fix Conv Binary Inplace Fusion issue (#115153
         Fixed issue in upsample_nearestnd lowering with scales (#117538)
         Fix inductor pattern match error for qlinear with bmm (#117633)
         Fix cpp backend relu codegen with inf input (#117622)
         Fix CPP wrapper codegen for ExternKernel args (#117931)
         Fix sympy_subs to preserve integer and non-negative properties. (#118150)
         Fix Argmax codegen with Nan input (#118358)
         Fix constant folding bug with sym size tensor (#118411)
         Fix a typo in should_pad_bench (#118598)
         Fix codegen bug with Native Triton kernels with ReinterpretView args (#118569)
         Fix an internal test issue (#118903)
         Fix a cpp kernel missing arg type issue (#119021)
         Fix Inductor CSE Across Separate Reductions (#119410)
         Fix bandwidth extimation for StarDep (#120266)
         Fix bug around out of order constexprs in inductor (#120287)
         Fix compiler check (#120492)
         Fix q/dq per channel lowering with 64-bit qparams (#120984)
         Fix the layout problem for nll_loss2d_backward (#121173)
         Fix for Wait kernel lowering in inductor not accepting MultiOutputs from non-collective calls (#121428)
         Correct index propagation for % (#119864)
         Fix a missing declaration for the result of item() (#115175)
         Make sure bitcast input and target type have the same bitwidth (#115619)
         Avoid inplace for ComplexView (#115166)
         SDPA extend backward realized tensor alignment checking to forward realized tensors (#116069)
         Ignore SIGINT in codecache workers (#116380)
         Add more alias and mutation check for other input of Conv Binary Inplace fusion (#117330)
         Fail Conv Binary Inplace check when act and accum are same tensor (#117331)
         [CPU] Disable floating-point contraction when compiling (#116318)
         Use wait stream instead of synchronize() in cudagraph warmup (#117578)
         Place .lrodata later in the binary (#117575)
         Correctly generate grid info for benchmark_kernel (#118202)
         Do not reuse buffers across scopes in mem planning (#120777)
         Fix profiler (#119959)
         Wrap remote cache creation with a try-catch (#121340)
         Inductor cpp wrapper: fix dtype of ShapeAsConstantBuffer (#122297)
         torch.export
         Handle transposition pattern seen in SDPA with unbacked SymInts (#121005)
         Fix bug removing node from wrong graph (#121574)
         Fix graph signature for primitive outputs (#118818)
         Fixed bug with user input mutations (#118942)
         Prevent specialization on backends (#118683)
         Fixed nn_module_stack in retracing (#121423)
         Fixed accidental specialization with faketensor input checks (#121460)
         Fixed name collision on constant name (#121145)
         Don\'t error if nn_module_stack doesn\'t contain a class (#119753)
         Fixed tuple return with symints (#119829)
         Fixed canonicalization for input mutations (#119533)
         Fixed getting meta["val"] (#117313)
         Add pass to remove auto functionalized hop (#122246)
         Fix auto_functionalize (#121990)
         Hack skip index_put_ in DCE (#122683)
         Various fixes to .module() (#118272)
         Do not rewrite state dict when unlifting (#118611)
         FX
         Fix pass_manager type annotation (#119499)
         Suggested fixes for congruences (#121418)
         Fix: set codegen in _SplitterBase partitioner (#120361)
         Fixed FxGraphDrawer compat constructor (#119767)
         [sigmoid] Fix for FX tracing unflattened modules (#115708)
         Fix for subgraph rewriter (#119052)
         Fix F821 error in torch/fx/experimental (#116587)
         Support printing storage while FakeTensorMode is enabled (#118780)
         Don\'t guard if there are unbacked SymInts (#119312)
         Avoid performing replacements when it would unrefine ranges (#117356)
         Fix none type comparison (#116399)
         JIT
         Fix RuntimeError: NYI: Named tensors are not supported with the tracer errors when using torch.jit.trace (#118393)
         Fix LLVM18 build (#115652, #117086)
         Fix handling of broadcasted inputs in Linear-BN Fusion (#119264)
         Linalg
         Fix mm accuracy in ROCm for some inputs (#116537)
         Update matmul heuristics in the presence of gradients (#117067) (#118617)
         Workaround a cusolver bug on CUDA < 12.1 in triangular_solve (#117636)
         MPS
         Fix addmm (#116547)
         Fix SegFault when torch.all/any dispatched to mps or other backends (#116457)
         Increase metal language support to 2.3 (#117472)
         Fix torch.mm correctness for large matrices (#117549)
         Fix lintear for 5D tensors (#117837)
         Fix use_metal_mm condition (#118830)
         Add support for complex scalars (#119318)
         Use dyspatch_sync_with_rethrow in searchsorted (#119646)
         Fix cfloat->chalf conversion on MacOS13 (#119681)
         Enable conj and conj_physical (#119669)
         Fix out resize logic in torch.where (#121476)
         Fix CrossEntropyLoss for float16 (#116597)
         Do not crash if Metal function can not be found (#116938)
         Fix float32 error on mps, in linalg.matrix_rank and linalg.pinv (#114771)
         Fix placeholder tensor is empty for relu in mps (#118965)
         Fix boundary checks in generateKernelOffsets (#116915)
         Fix torch.clamp in MPS to handle NaN correctly (#121381)
         Fwd-fix for clamp regression (#122148)
         Fix naive matmul for BFloat16 (#121731)
         Fix for MPS regression in #122016 and #123178 (#123234)
         torch.nn API
         Fixed numpy warning when importing torch without numpy installed (#115867)
         Fixed edge case for size 1 channels dim in AdaptiveMaxPool (#116482)
         Fixed module pre bw hooks when input doesn\'t require grad but gradients are changed by the user (#116454)
         Fixed TransformerEncoderLayer for bias=False (#116760)
         Fixed error checking for LSTM with wrong input shape (#115542)
         Removed an incorrect type specification from AdaptiveMaxPool1d (#118162)
         Fixed an illegal memory access in cross entropy loss when using an index that is not a valid class (#117561)
         Fixed pool padding assertion to account for dilation (#118897)
         Fixed flash_attn_bw impl to match meta implementation when k and v have different strides (#119500)
         Fixed nonlinearity arg issue in RNN (#120234)
         Fixed requires_grad preservation for nn.Module.load_state_dict(assign=True) (#121157)
         Fixed last_dim stride check for singleton dimensions (#117001)
         Fixed gradients on cuda for interpolate::trilinear on non-contiguous grad output (#117373)
         Added Half support for masked_softmax on CPU (#117028)
         Fixed an issue where nn.Linear would cause an internal int underflow (#119221)
         Fixed segfault in torch.native_channel_shuffle when input is empty (#121199)
         Nested Tensors
         Proper view support for jagged layout NestedTensor (#113279)
         ONNX
         Add decomposition for upsample_linear{1d, 3d} (#114774)
         Fix upsample_bilinear2d decomp skip with output shape (#118823)
         Fix ONNXRT torch.compile backend running with OrtValueVector (#116124)
         Fix output mismatch issue of repeat_interleave when dim is None (#116689)
         Update initializer path for ONNXProgram.save due to onnx.checker limitation (#117294)
         Set proper fqn in lift constant tensor pass (#115222)
         Fix type promotion pass (#118246)
         Perform implicit casting of constants for the onnx::where operator (#118733) (#120619)
         Fix onnxrt backends with inputs on mix devices (#121159)
         Fix breaking changes for ONNX Runtime Training (#122000)
         beartype to emit warning instead of error by default (#123205)
         Optimizer
         Rectify capturable testing and fix load_state_dict bugs with capturable! (#118326)
         Use torch.no_grad decorator for clip_grad_norm APIs vs local detaches (#120638)
         ReduceLROnPlateau allow get_last_lr to not error (#119556)
         Profiler
         Stop clearing history when changing context (#120436)
         Fix conversion of max memory allocated and reserved from GB to GiB (#120172)
         Fix the missing device string in _memory_profiler (#119751)
         Log process group id instead of backend id in GPU traces (#120475)
         Add kineto init delay when used in daemon mode (#120276)
         Fix recorded profiler step number (#121127)
         [ET] Fix deadlock in ExecutionTraceObserver (#119242) (#119398)
         Python API
         Fix index range checks when index is the minimum int value (#116062)
         Fix slots handling in torch.utils.swap_tensor (#116128)
         Fix NaN bug in torch.signal.windows.kaiser (#116470)
         Fix handling of empty inputs in torch.fft.fftn (#117368)
         Fix and/or ops on torch.uint8 tensors only return 0x00 or 0x01 (#117827)
         Fix inf handling in torch.nn.functional.scaled_dot_product_attention (#119577)
         Fix serialization of torch.complex32 dtype (#120388)
         Fix torch.gradient check for spacing arg list length (#115686)
         Fix type hints on torch.nn.attention.sdpa_kernel (#119140)
         Fix dimension checks in torch.distributions.MixtureSameFamily (#118947)
         Quantization
         Make HistogramObserver handle torch.inf and closeby values (#103467)
         Fix equal_quantized_cpu for QUInt4x2 and QUInt2x4 (#116307)
         Fix XNNPACKQuantizer set_module_type issue (#115252)
         Fix a segfault when calling topk on a quantized scalar tensor. (#116337)
         Fix a segfault issue when passing an empty kernel to quantized_max_pool1d (#116342)
         Fix batchnorm folding in pt2e quantization (#118720)
         Update PerChannelMinMaxObserver default _load_from_state_dict (#118659)
         Releng
         Fix for sparse windows on CPU with MKL (#102604)
         Fix for ExecuTorch pinned commit update failure (#117518)
         Sparse
         Fix a crash in sparse compressed tensor invariants check when nnz == 0 (#115825)
         Fix sparse compressed tensor invariants checks when nnz==0 (#115826)
         Fix segfault when trying to permute empty tensor (#116335)
         Other
         Remove compute capability 3.5 for CUDA 12 (#114930)
         VSX: Fix overflow in complex division (#116972)
         VSX: Fix vectorized abs function for complex tensors (#116859)
         Add complex support to parametrizations.spectral_norm (#121452)
         Fix crash in SymInt unary minus (#116160)
         Fix for out of bounds read in mobile interpreter INTERFACE_CALL opcode handler (#110301)
         Fix for out of bounds registers_ access in mobile TorchScript interpreter (#110300)
         ');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Performance', 'Composability
         In torch.compile, avoid allocating extra buffers unnecessarily in cases where the compiled function returns a mutated input directly (#120514)
         Min-cut partitioner always saves tensors that are returned as-is in backward (#114970)
         CUDA
         Speed up triu_tril_kernel (#115013)
         Inductor
         Add an autotune cache for inductor generated kernels (#120963)
         Add ArrayRefTensor (#112115)
         Replace cached thread_locals with stack allocation in AOTI (#112116)
         Autotune with matrix_instr_nonkdim for AMDGPU (#120742)
         Enable lowering of dynamic qlinear for X86Inductor (#120605)
         [NFC][Autotune] Use device_prop.regsPerMultiprocessor instead of hardcoded reg number. (#115094)
         [Autotune] Enable register pressure handling logic for H100. (#115295)
         Support vectorization for index_expr that depends on tiling itervar or with indirect indexing (#114545)
         Load as scalar for the index invariant in the vector range (#116387)
         Inductor qlinear int8_fp32 with bmm (#116599)
         Inductor qlinear int8_bf16 with bmm (#116604)
         Enable fp16 mkldnn fusion/prepack in inductor (#117206)
         Improve vector contiguous checks for FloorDiv and ModularIndexing (#117221)
         Load as scalar for the index invariant in the vector range (#116387)
         Use sleef implementation for CPP backend acosh codegen (#118350)
         Don\'t skip register-spilling configs in custom Triton kernel auto-tuning (#119634)
         be more consrevative until regression is debugged (#119583)
         Check alignment of ReinterpretView args of custom Triton kernels (#119649)
         Apply simplify_index_in_vec_range in select_tiling_indices to enable more contiguous vec load (#117260)
         Apply simplify_index_in_vec_range to vector store and vector transpose (#117263)
         Multi-kernel support (#103469)
         Enable the Inductor Lowering of QConv2d post op hardswish (#117489)
         Change CppWrapperCodeCache to use faster python binding (#117693)
         Slightly faster memory allocation on CPU (#118171)
         Slightly faster memory allocation on CUDA (#118255)
         Add Thread Number Checker in scatter_reduce_ fallback for CPP backend (#118278)
         Enable vectorization with constant bool (#118380)
         Support scalar value in vec reduction (#118511)
         Use at::detail::empty_strided_* in cpp_wraper mode (#118490)
         Add equal_to_1 to triton_meta for user-written Triton kernels (#120579)
         Add mask_convert_to_lp to support bool->fp16/bf16 convert (#117830)
         Optimize transpose_mxn with bf16 data type (#117958)
         Add Int8 data type into Inductor CPP backend vectorized code generation (#119179)
         [Autotune] Multithreaded Precompilation (#119386)
         Add SDPA pattern for HuggingFace models BF16 (#121202)
         Support auto-tuned custom PT ops in abi compatible mode (#120877)
         Benchmark templates (#118880)
         MPS
         Add native lerp support (#119036)
         Optimizer
         Replace new().resize_as_() by torch.full_like() in Rprop (#119978)
         clip_grad_norm can use fast foreach path for inf norm (#120623)
         Profiler
         Only profile when JIT is enabled. (#121404)
         Python API
         Speed up fp16<->fp32 conversion on ARMV8 platforms (#120012)
         ROCm
         CatArrayBatchedCopy performance improvement (#118685)
         Fix performance regression and memory storage handling of Flash Attention on ROCM (#122857)
         Other
         Add NEON accelerated torch.mv kernel (#119992)
         ');

INSERT INTO default.library_updates (library_name, old_version, new_version, release_date, update_type, update_notes)
VALUES ('pytorch', '2.2.2', '2.3.0', '2024-04-26', 'Documentation', 'Autograd API
         Autograd doc cleanup (#118500)
         Deduplicate docs between global and non-global full backward hooks (#119708)
         Add missing words to torch.utils.checkpoint doc (#120196)
         CUDA
         Include a print for _get_cuda_arch_flags (#118503)
         Clarify how to get extra link flags when building CUDA/C++ extension (#118743)
         Test seo torch cuda (#119324)
         Distributed API
         C10d
         Add documentation for the device_id parameter for init_process_group (#116222)
         Add docstrings and tests for src / dst (#118593)
         Add device for distributed examples (#118867)
         Add Work to distributed docs (#115172)
         DDP:
         Fix docstring errors in model_averaging (#117038)
         Fix docstring errors in ddp_comm_hooks (#116866)
         Update DDP dynamo debug docs (#118295)
         FSDP
         Fix optim_state_dict_to_load doc errors (#118195)
         Distributed Checkpointing (DCP):
         Fix the documents for distributed_state_dict (#121276)
         Update the distributed state_dict document (#121290)
         DTensor
         Update README to make all example runnable (#115365)
         Add torchrec even row-wise sharding example
         Add clarification to doc and improve TP examples (#121431, #117618)
         Add torch.float64 precision support to the transformer test suite in TP/SP (#116436)
         Misc:
         Add doc for torch.distributed.breakpoint (#115656)
         FX
         Reduce create_env log level to DEBUG (#120772)
         torch.compile
         Inductor
         Document and type torch._inductor.virtualized (#117658)
         Document OpsHandler protocol (#117790)
         torch.export
         Added TORCH_LOGS=export (#116993)
         Update _constrain_as_size docs (#120728)
         Added docs for 2.3 release (#121466)
         Updated docs to not export raw functions (#121272)
         Add comments about runtime_var_to_range. (#118539)
         Linalg
         Fix error in examples of torch.linalg.lu_factor (#120484)
         Add links to _ex variants in all linalg functions that support them (#121451)
         torch.nn API
         Updated documentation for the constraints of FractionalMaxPool2d (#116261)
         Updated BCEWithLogitsLoss documentation regarding pos_weight (#117046)
         Fixed typo in register_state_dict_pre_hook doc (#118571)
         Change the parameter type from int to float in torch.nn.Softplus (#120183)
         Documented special case in AvgPool (#120335)
         Added hyperlink to Transformer documentation in Transformer-related modules (#120565)
         Added type hints to TransformerEncoder/Decoder (#120550)
         Added a note in Transformer about difference in masking semantic with torch.nn.functional.scaled_dot_product_attention (#120668)
         Fixed documentation for mask broadcasting behavior in torch.nn.functional.scaled_dot_product_attention (#120859)
         Fixed math display in ChannelShuffle documentation (#121247)
         Documented padding size constraint in nn.ReflectionPad2d (#115995)
         Fixed documentation of nn.functional.scaled_dot_product_attention to indicate scale is a keyword only arg (#119129)
         Optimizer
         Added example regarding weight_decay distinction with per-parameter API (#117436)
         Fix optim.lr_scheduler examples in doc to use optimizer vs self.opt (#119563)
         Clarify decay vs multiply by a constant factor in the constantLR doc (#120852)
         Clarify the patience in ReduceLROnPlateau (#119872)');

INSERT INTO FUNCTION
   s3(
       's3_file_url',
       's3_access_key',
       's3_secret_key'
    )
SELECT library_name, old_version, new_version, release_date, update_type, update_notes
FROM default.library_updates;