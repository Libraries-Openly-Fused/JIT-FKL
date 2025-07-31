/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_JIT_OPERATION_EXECUTOR_H
#define FK_JIT_OPERATION_EXECUTOR_H

#include <fused_kernel/core/execution_model/executors.h>

namespace fk {
    template <enum TF TFEN>
    struct Executor<TransformDPP<ParArch::GPU_NVIDIA_JIT, TFEN, void>> {
        FK_STATIC_STRUCT(Executor, Executor)
    private:
        using Child = Executor<TransformDPP<ParArch::GPU_NVIDIA_JIT, TFEN>>;
        using Parent = BaseExecutor<Child>;
        template <typename... IOps>
        FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA_JIT>& stream, const IOps&... iOps) {
            constexpr ParArch PA = ParArch::GPU_NVIDIA;
            const auto tDetails = TransformDPP<PA, TFEN>::build_details(iOps...);
            using TDPPDetails = std::decay_t<decltype(tDetails)>;
            std::string detailsType = fk::typeToString<TDPPDetails>();
            std::string kernelName{ "launchTransformDPP_Kernel<ParArch::GPU_NVIDIA, " };
            std::string tfi;
            ActiveThreads activeThreads;
            std::string threadDivisible;
            if constexpr (TDPPDetails::TFI::ENABLED) {
                tfi = std::string("TF::ENABLED");
                activeThreads = tDetails.activeThreads;
                if (!tDetails.threadDivisible) {
                    threadDivisible = std::string("false");
                }
                else {
                    threadDivisible = std::string("true");
                }
            }
            else {
                tfi = std::string("TF::DISABLED");
                activeThreads = get<0>(iOps...).getActiveThreads();
                threadDivisible = std::string("true");
            }
            const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

            const dim3 block{ ctx_block.x, ctx_block.y, 1 };
            const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                             static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                             activeThreads.z };

            std::string kernelNameWithDetails = kernelName + tfi + ", " + threadDivisible + ", " + typeToString<TDPPDetails>() + ", ";
            std::vector<JIT_Operation_pp> pipeline = jit_internal::buildOperationPipeline(iOps...);
            CUfunction kernelFunc = JITExecutorCache::getInstance().addKernel(kernelNameWithDetails, pipeline);
            std::vector<void*> args = jit_internal::buildKernelArguments(pipeline);
            args.insert(args.begin(), (void*)&tDetails);
            gpuErrchk(cuLaunchKernel(kernelFunc, grid.x, grid.y, grid.z,
                block.x, block.y, block.z, 0,
                reinterpret_cast<CUstream>(stream.getCUDAStream()), args.data(), nullptr));
        }
        FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA_JIT>& stream, const std::vector<JIT_Operation_pp>& iOps) {
            constexpr ParArch PA = ParArch::GPU_NVIDIA;
            /*const auto tDetails = TransformDPP<PA, TFEN>::build_details(iOps...);
            using TDPPDetails = std::decay_t<decltype(tDetails)>;
            std::string detailsType = fk::typeToString<TDPPDetails>();
            std::string kernelName{ "launchTransformDPP_Kernel<ParArch::GPU_NVIDIA, " };
            std::string tfi;
            ActiveThreads activeThreads;
            std::string threadDivisible;
            if constexpr (TDPPDetails::TFI::ENABLED) {
                tfi = std::string("TF::ENABLED");
                activeThreads = tDetails.activeThreads;
                if (!tDetails.threadDivisible) {
                    threadDivisible = std::string("false");
                } else {
                    threadDivisible = std::string("true");
                }
            } else {
                tfi = std::string("TF::DISABLED");
                activeThreads = get<0>(iOps...).getActiveThreads();
                threadDivisible = std::string("true");
            }
            const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

            const dim3 block{ ctx_block.x, ctx_block.y, 1 };
            const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                             static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                             activeThreads.z };

            std::string kernelNameWithDetails = kernelName + tfi + ", " + threadDivisible + ", " + typeToString<TDPPDetails>() + ", ";
            std::vector<JIT_Operation_pp> pipeline = jit_internal::buildOperationPipeline(iOps...);
            CUfunction kernelFunc = JITExecutorCache::getInstance().addKernel(kernelNameWithDetails, pipeline);
            std::vector<void*> args = jit_internal::buildKernelArguments(pipeline);
            args.insert(args.begin(), &tDetails);
            gpuErrchk(cuLaunchKernel(kernelFunc, grid.x, grid.y, grid.z,
                block.x, block.y, block.z, 0,
                reinterpret_cast<CUstream>(stream.getCUDAStream()), args.data(), nullptr));*/
        }
    public:
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::GPU_NVIDIA_JIT;
        }
        DECLARE_EXECUTOR_PARENT_IMPL
    };
} // namespace fk

#endif // FK_JIT_OPERATION_EXECUTOR_H