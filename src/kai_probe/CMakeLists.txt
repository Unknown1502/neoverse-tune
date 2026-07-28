cmake_minimum_required(VERSION 3.16)
project(kai_probe CXX C ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# This probe measures Arm microkernel selection. There is nothing meaningful to
# build elsewhere, so fail early and clearly rather than at link time.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  message(FATAL_ERROR
    "kai_probe targets arm64 (found '${CMAKE_SYSTEM_PROCESSOR}').\n"
    "Run it on an Arm64 Linux host — see docs/reproducing.md.")
endif()

# --- KleidiAI --------------------------------------------------------------
# Vendored by scripts/05_kai_probe.sh so the exact revision is recorded.
set(KAI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/kleidiai"
    CACHE PATH "Path to a KleidiAI checkout")

if(NOT EXISTS "${KAI_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR
    "KleidiAI not found at ${KAI_ROOT}\n"
    "Run: bash scripts/05_kai_probe.sh   (it vendors KleidiAI for you)")
endif()

add_subdirectory("${KAI_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/kleidiai" EXCLUDE_FROM_ALL)

# --- generated variant table ----------------------------------------------
# tools/gen_variants.py writes kai_variants.inc by scanning the checkout, so no
# kernel name is ever hardcoded here.
set(KAI_VARIANTS_INC "${CMAKE_CURRENT_BINARY_DIR}/generated/kai_variants.inc")
if(NOT EXISTS "${KAI_VARIANTS_INC}")
  message(FATAL_ERROR
    "Missing ${KAI_VARIANTS_INC}\n"
    "Generate it first:\n"
    "  python3 tools/gen_variants.py ${KAI_ROOT} ${KAI_VARIANTS_INC}")
endif()

add_executable(kai_probe kai_probe.cpp)

# KleidiAI headers are not flattened into one include root; each microkernel
# family and the packing routines live in their own directory.
target_include_directories(kai_probe PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/generated"
  "${KAI_ROOT}"
  "${KAI_ROOT}/kai"
  "${KAI_ROOT}/kai/ukernels/matmul"
  "${KAI_ROOT}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp"
  "${KAI_ROOT}/kai/ukernels/matmul/pack"
)

# Our translation unit only calls into KleidiAI; the kernels themselves are
# compiled by KleidiAI's own build with their required per-file -march flags.
# We still need the baseline features so the headers' inline helpers compile.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=armv8.2-a+dotprod+i8mm" HAS_DOTPROD_I8MM)
if(HAS_DOTPROD_I8MM)
  target_compile_options(kai_probe PRIVATE -march=armv8.2-a+dotprod+i8mm)
else()
  message(WARNING
    "Compiler rejected -march=armv8.2-a+dotprod+i8mm; falling back to -march=native. "
    "i8mm variants may be unavailable.")
  target_compile_options(kai_probe PRIVATE -march=native)
endif()

target_compile_options(kai_probe PRIVATE -O3 -Wall -Wextra)

# KleidiAI's library target has been named both ways across releases.
if(TARGET kleidiai)
  target_link_libraries(kai_probe PRIVATE kleidiai)
elseif(TARGET kai)
  target_link_libraries(kai_probe PRIVATE kai)
else()
  message(FATAL_ERROR
    "Neither a 'kleidiai' nor a 'kai' CMake target was defined by ${KAI_ROOT}. "
    "Inspect that project's CMakeLists.txt for its library target name.")
endif()

install(TARGETS kai_probe RUNTIME DESTINATION bin)
