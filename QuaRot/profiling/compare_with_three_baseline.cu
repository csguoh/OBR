#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <memory>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_uncompress.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"


class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};


template<typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
class CutlassSparseGemmConfig {
public:
    using ElementInputA = ElementA;
    using ElementInputB = ElementB;
    using ElementOutput = ElementC;
    using ElementComputeEpilogue = ElementAccumulator;
    

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
    

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 256>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 256>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 128>;
    

    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue>;
    

    static constexpr int NumStages = 3;
    

    using GemmKernel = cutlass::gemm::device::SparseGemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        MMAOp, SmArch,
        ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
        EpilogueOp, SwizzleThreadBlock, NumStages>;
    

    using ElementInputE = typename GemmKernel::ElementE;
    using LayoutInputE = cutlass::layout::RowMajor;
    using ReorderedLayoutInputE = typename GemmKernel::LayoutE;
    

    static constexpr int kSparse = GemmKernel::kSparse;
    static constexpr int kElementsPerElementE = GemmKernel::kElementsPerElementE;
    static constexpr int kMetaSizeInBits = GemmKernel::kMetaSizeInBits;
};


template<typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
class CutlassDenseGemmConfig {
public:
    using ElementInputA = ElementA;
    using ElementInputB = ElementB;
    using ElementOutput = ElementC;
    using ElementComputeEpilogue = ElementAccumulator;
    

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
    

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 128>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 64>;
    

    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue>;

    static constexpr int NumStages = 3;

    using GemmKernel = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        MMAOp, SmArch,
        ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
        EpilogueOp, SwizzleThreadBlock, NumStages>;
};


using INT4SparseGemmConfig = CutlassSparseGemmConfig<
    cutlass::int4b_t,   // ElementA
    cutlass::int4b_t,   // ElementB  
    int32_t,            // ElementC
    int32_t             // ElementAccumulator
>;

using INT4DenseGemmConfig = CutlassDenseGemmConfig<
    cutlass::int4b_t,   // ElementA
    cutlass::int4b_t,   // ElementB  
    int32_t,            // ElementC
    int32_t             // ElementAccumulator
>;


class SparseGemmWrapper {
private:
    using Config = INT4SparseGemmConfig;
    using GemmKernel = typename Config::GemmKernel;
    

    std::unique_ptr<cutlass::HostTensor<typename Config::ElementInputA, typename Config::LayoutInputA>> tensor_a_;
    std::unique_ptr<cutlass::HostTensor<typename Config::ElementInputA, typename Config::LayoutInputA>> tensor_a_uncompressed_;
    std::unique_ptr<cutlass::HostTensor<typename Config::ElementInputB, typename Config::LayoutInputB>> tensor_b_;
    std::unique_ptr<cutlass::HostTensor<typename Config::ElementOutput, typename Config::LayoutOutput>> tensor_c_;
    std::unique_ptr<cutlass::HostTensor<typename Config::ElementOutput, typename Config::LayoutOutput>> tensor_d_;
    std::unique_ptr<cutlass::HostTensor<typename Config::ElementInputE, typename Config::LayoutInputE>> tensor_e_;
    std::unique_ptr<cutlass::HostTensor<typename Config::ElementInputE, typename Config::ReorderedLayoutInputE>> tensor_e_reordered_;

    cutlass::gemm::GemmCoord problem_size_;
    

    int batch_size_;
    int seq_len_;
    int cin_;
    int cout_;

    std::unique_ptr<GemmKernel> gemm_op_;
    std::unique_ptr<cutlass::device_memory::allocation<uint8_t>> workspace_;

    PerformanceTimer timer_;
    
public:

    SparseGemmWrapper(int batch_size, int seq_len, int cin, int cout) 
        : batch_size_(batch_size), seq_len_(seq_len), cin_(cin), cout_(cout),
          problem_size_(cout, batch_size * seq_len, cin) {
        allocate_tensors();
        initialize_gemm();
    }

    SparseGemmWrapper(int m, int n, int k) 
        : batch_size_(1), seq_len_(m), cin_(k), cout_(n),
          problem_size_(n, m, k) {
        allocate_tensors();
        initialize_gemm();
    }

    ~SparseGemmWrapper() = default;
    

    void allocate_tensors() {
        const int m = problem_size_.m();  // cout
        const int n = problem_size_.n();  // batch * seq_len
        const int k = problem_size_.k();  // cin
        

        tensor_a_ = std::make_unique<cutlass::HostTensor<typename Config::ElementInputA, typename Config::LayoutInputA>>(
            cutlass::make_Coord(m, k / Config::kSparse));
        

        tensor_a_uncompressed_ = std::make_unique<cutlass::HostTensor<typename Config::ElementInputA, typename Config::LayoutInputA>>(
            cutlass::make_Coord(m, k));
        

        tensor_b_ = std::make_unique<cutlass::HostTensor<typename Config::ElementInputB, typename Config::LayoutInputB>>(
            cutlass::make_Coord(k, n));
        

        tensor_c_ = std::make_unique<cutlass::HostTensor<typename Config::ElementOutput, typename Config::LayoutOutput>>(
            cutlass::make_Coord(m, n));
        

        tensor_d_ = std::make_unique<cutlass::HostTensor<typename Config::ElementOutput, typename Config::LayoutOutput>>(
            cutlass::make_Coord(m, n));
        

        tensor_e_ = std::make_unique<cutlass::HostTensor<typename Config::ElementInputE, typename Config::LayoutInputE>>(
            cutlass::make_Coord(m, k / Config::kSparse / Config::kElementsPerElementE));
        

        tensor_e_reordered_ = std::make_unique<cutlass::HostTensor<typename Config::ElementInputE, typename Config::ReorderedLayoutInputE>>(
            cutlass::make_Coord(m, k / Config::kSparse / Config::kElementsPerElementE));
    }

    void initialize_gemm() {
        gemm_op_ = std::make_unique<GemmKernel>();
    }
    

    void fill_random_data(int seed = 2024) {
        cutlass::reference::host::TensorFillRandomUniform(
            tensor_a_->host_view(), 1,
            typename Config::ElementInputA(2),
            typename Config::ElementInputA(-2), seed);

        cutlass::reference::host::TensorFillRandomUniform(
            tensor_b_->host_view(), 1,
            typename Config::ElementInputB(2),
            typename Config::ElementInputB(-2), seed + 1);
        

        cutlass::reference::host::TensorFill(tensor_c_->host_view());
        

        cutlass::reference::host::TensorFillRandomSparseMeta(
            tensor_e_->host_view(), 1, Config::kMetaSizeInBits);
        

        cutlass::reference::host::TensorFill(tensor_d_->host_view());
        

        cutlass::reorder_meta(
            tensor_e_reordered_->host_ref(),
            tensor_e_->host_ref(),
            {problem_size_.m(), problem_size_.n(), 
             problem_size_.k() / Config::kSparse / Config::kElementsPerElementE});
    }

    void sync_to_device() {
        tensor_a_->sync_device();
        tensor_b_->sync_device();
        tensor_c_->sync_device();
        tensor_d_->sync_device();
        tensor_e_reordered_->sync_device();
    }

    bool run_gemm(float alpha = 1.0f, float beta = 0.0f) {
        typename GemmKernel::Arguments arguments{
            problem_size_,
            tensor_a_->device_ref(),
            tensor_b_->device_ref(),
            tensor_c_->device_ref(),
            tensor_d_->device_ref(),
            tensor_e_reordered_->device_ref(),
            {typename Config::ElementComputeEpilogue(alpha), 
             typename Config::ElementComputeEpilogue(beta)},
            1  // split_k_slices
        };
        

        size_t workspace_size = GemmKernel::get_workspace_size(arguments);
        workspace_ = std::make_unique<cutlass::device_memory::allocation<uint8_t>>(workspace_size);
        

        cutlass::Status status = gemm_op_->can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "can_implement failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return false;
        }
        

        status = gemm_op_->initialize(arguments, workspace_->get());
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "initialize failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return false;
        }
        

        status = gemm_op_->operator()();
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "operator() failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return false;
        }
        

        cudaDeviceSynchronize();
        
        return true;
    }
    

    std::vector<double> benchmark(int warmup_iterations = 5, int benchmark_iterations = 10) {
        std::vector<double> times;
        

        for (int i = 0; i < warmup_iterations; ++i) {
            if (!run_gemm()) {
                std::cerr << "Failed" << std::endl;
                return times;
            }
        }

        times.reserve(benchmark_iterations);
        
        for (int i = 0; i < benchmark_iterations; ++i) {
            timer_.start();
            if (!run_gemm()) {
                std::cerr << "Failed" << std::endl;
                return times;
            }
            double elapsed_ms = timer_.stop();
            times.push_back(elapsed_ms);
        }
        
        return times;
    }
    

    std::vector<int> get_problem_size() const {
        return {batch_size_, seq_len_, cin_, cout_};
    }
    

    std::vector<int> get_gemm_size() const {
        return {problem_size_.m(), problem_size_.n(), problem_size_.k()};
    }
    

    std::vector<int> get_sparse_info() const {
        return {Config::kSparse, Config::kElementsPerElementE, Config::kMetaSizeInBits};
    }
};


class DenseGemmWrapper {
private:
    using Config = INT4DenseGemmConfig;
    using GemmKernel = typename Config::GemmKernel;
    

    std::unique_ptr<cutlass::HostTensor<typename Config::ElementInputA, typename Config::LayoutInputA>> tensor_a_;
    std::unique_ptr<cutlass::HostTensor<typename Config::ElementInputB, typename Config::LayoutInputB>> tensor_b_;
    std::unique_ptr<cutlass::HostTensor<typename Config::ElementOutput, typename Config::LayoutOutput>> tensor_c_;
    std::unique_ptr<cutlass::HostTensor<typename Config::ElementOutput, typename Config::LayoutOutput>> tensor_d_;

    cutlass::gemm::GemmCoord problem_size_;
    

    int batch_size_;
    int seq_len_;
    int cin_;
    int cout_;

    std::unique_ptr<GemmKernel> gemm_op_;

    PerformanceTimer timer_;
    
public:
    DenseGemmWrapper(int batch_size, int seq_len, int cin, int cout)
        : batch_size_(batch_size), seq_len_(seq_len), cin_(cin), cout_(cout),
          problem_size_(cout, batch_size * seq_len, cin) {
        allocate_tensors();
        initialize_gemm();
    }
    
    DenseGemmWrapper(int m, int n, int k)
        : batch_size_(1), seq_len_(m), cin_(k), cout_(n),
          problem_size_(n, m, k) {
        allocate_tensors();
        initialize_gemm();
    }
    
    ~DenseGemmWrapper() = default;
    
    void allocate_tensors() {
        const int m = problem_size_.m();  // cout
        const int n = problem_size_.n();  // batch * seq_len
        const int k = problem_size_.k();  // cin
        
        tensor_a_ = std::make_unique<cutlass::HostTensor<typename Config::ElementInputA, typename Config::LayoutInputA>>(
            cutlass::make_Coord(m, k));
        
        tensor_b_ = std::make_unique<cutlass::HostTensor<typename Config::ElementInputB, typename Config::LayoutInputB>>(
            cutlass::make_Coord(k, n));
        
        tensor_c_ = std::make_unique<cutlass::HostTensor<typename Config::ElementOutput, typename Config::LayoutOutput>>(
            cutlass::make_Coord(m, n));
        
        tensor_d_ = std::make_unique<cutlass::HostTensor<typename Config::ElementOutput, typename Config::LayoutOutput>>(
            cutlass::make_Coord(m, n));
    }
    
    void initialize_gemm() {
        gemm_op_ = std::make_unique<GemmKernel>();
    }
    
    void fill_random_data(int seed = 2024) {
        cutlass::reference::host::TensorFillRandomUniform(
            tensor_a_->host_view(), 1,
            typename Config::ElementInputA(7),
            typename Config::ElementInputA(-8), seed);
        
        cutlass::reference::host::TensorFillRandomUniform(
            tensor_b_->host_view(), 1,
            typename Config::ElementInputB(7),
            typename Config::ElementInputB(-8), seed + 1);
        
        cutlass::reference::host::TensorFill(tensor_c_->host_view());
        
        cutlass::reference::host::TensorFill(tensor_d_->host_view());
    }
    
    void sync_to_device() {
        tensor_a_->sync_device();
        tensor_b_->sync_device();
        tensor_c_->sync_device();
        tensor_d_->sync_device();
    }
    
    bool run_gemm(float alpha = 1.0f, float beta = 0.0f) {
        typename GemmKernel::Arguments arguments{
            problem_size_,
            tensor_a_->device_ref(),
            tensor_b_->device_ref(),
            tensor_c_->device_ref(),
            tensor_d_->device_ref(),
            {typename Config::ElementComputeEpilogue(alpha), 
             typename Config::ElementComputeEpilogue(beta)}
        };
        
        cutlass::Status status = gemm_op_->can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "can_implement failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return false;
        }
        
        status = gemm_op_->initialize(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "initialize failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return false;
        }
        
        status = gemm_op_->operator()();
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "operator() failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return false;
        }
        
        cudaDeviceSynchronize();
        return true;
    }
    
    std::vector<double> benchmark(int warmup_iterations = 5, int benchmark_iterations = 10) {
        std::vector<double> times;
        
        for (int i = 0; i < warmup_iterations; ++i) {
            if (!run_gemm()) {
                std::cerr << "Failed" << std::endl;
                return times;
            }
        }
        
        times.reserve(benchmark_iterations);
        
        for (int i = 0; i < benchmark_iterations; ++i) {
            timer_.start();
            if (!run_gemm()) {
                std::cerr << "Failed" << std::endl;
                return times;
            }
            double elapsed_ms = timer_.stop();
            times.push_back(elapsed_ms);
        }
        
        return times;
    }
    
    std::vector<int> get_problem_size() const {
        return {batch_size_, seq_len_, cin_, cout_};
    }
    
    std::vector<int> get_gemm_size() const {
        return {problem_size_.m(), problem_size_.n(), problem_size_.k()};
    }
};

std::shared_ptr<SparseGemmWrapper> create_sparse_gemm_batch(int batch_size, int seq_len, int cin, int cout) {
    return std::make_shared<SparseGemmWrapper>(batch_size, seq_len, cin, cout);
}

std::shared_ptr<SparseGemmWrapper> create_sparse_gemm(int m, int n, int k) {
    return std::make_shared<SparseGemmWrapper>(m, n, k);
}

std::shared_ptr<DenseGemmWrapper> create_dense_gemm_int4_batch(int batch_size, int seq_len, int cin, int cout) {
    return std::make_shared<DenseGemmWrapper>(batch_size, seq_len, cin, cout);
}

std::shared_ptr<DenseGemmWrapper> create_dense_gemm_int4(int m, int n, int k) {
    return std::make_shared<DenseGemmWrapper>(m, n, k);
}

bool check_gpu_compatibility() {
    cudaDeviceProp props;
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    
    if (error != cudaSuccess) {
        std::cerr << "Failed to request GPU: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    int compute_capability = props.major * 10 + props.minor;
    return compute_capability >= 80;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS INT4 Sparse/Dense GEMM PyBind Interface with Batch Support";
    
    pybind11::class_<SparseGemmWrapper, std::shared_ptr<SparseGemmWrapper>>(m, "SparseGemmWrapper")
        .def("fill_random_data", &SparseGemmWrapper::fill_random_data, 
             pybind11::arg("seed") = 2024)
        .def("sync_to_device", &SparseGemmWrapper::sync_to_device)
        .def("run_gemm", &SparseGemmWrapper::run_gemm,
             pybind11::arg("alpha") = 1.0f, pybind11::arg("beta") = 0.0f)
        .def("benchmark", &SparseGemmWrapper::benchmark,
             pybind11::arg("warmup_iterations") = 5, pybind11::arg("benchmark_iterations") = 10)
        .def("get_problem_size", &SparseGemmWrapper::get_problem_size)
        .def("get_gemm_size", &SparseGemmWrapper::get_gemm_size)
        .def("get_sparse_info", &SparseGemmWrapper::get_sparse_info);
    
    pybind11::class_<DenseGemmWrapper, std::shared_ptr<DenseGemmWrapper>>(m, "DenseGemmWrapper")
        .def("fill_random_data", &DenseGemmWrapper::fill_random_data, 
             pybind11::arg("seed") = 2024)
        .def("sync_to_device", &DenseGemmWrapper::sync_to_device)
        .def("run_gemm", &DenseGemmWrapper::run_gemm,
             pybind11::arg("alpha") = 1.0f, pybind11::arg("beta") = 0.0f)
        .def("benchmark", &DenseGemmWrapper::benchmark,
             pybind11::arg("warmup_iterations") = 5, pybind11::arg("benchmark_iterations") = 10)
        .def("get_problem_size", &DenseGemmWrapper::get_problem_size)
        .def("get_gemm_size", &DenseGemmWrapper::get_gemm_size);

    m.def("create_sparse_gemm_batch", &create_sparse_gemm_batch, 
          "Create a sparse GEMM wrapper with batch support",
          pybind11::arg("batch_size"), pybind11::arg("seq_len"), 
          pybind11::arg("cin"), pybind11::arg("cout"));
    m.def("create_sparse_gemm", &create_sparse_gemm, 
          "Create a sparse GEMM wrapper (legacy interface)");
    m.def("create_dense_gemm_int4_batch", &create_dense_gemm_int4_batch, 
          "Create a dense INT4 GEMM wrapper with batch support",
          pybind11::arg("batch_size"), pybind11::arg("seq_len"), 
          pybind11::arg("cin"), pybind11::arg("cout"));
    m.def("create_dense_gemm_int4", &create_dense_gemm_int4, 
          "Create a dense INT4 GEMM wrapper (legacy interface)");
    m.def("check_gpu_compatibility", &check_gpu_compatibility, "Check GPU compatibility");
}