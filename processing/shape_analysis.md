# CUDA 和 FlagGems 的 aten::mm 输入 Shape 分析

## CUDA aten::mm 算子（按总时间排序）

| 框架算子名 | 执行算子名 | 输入 shape | CUDA调用次数 | CUDA总时间(ms) | CUDA占比 |
|---|---|---|---|---|---|
| aten::mm | sm90_fp8_gemm_1d2d_impl | grid(78,1,1) block(384,1,1); grid(64,1,1) block(256,1,1) | 542744 | 82331.982 | 12.72% |
| aten::mm | fp8_gemm_kernel | grid(78,1,1) block(384,1,1); grid(78,1,1) block(256,1,1) | 111600 | 41971.106 | 6.49% |
| aten::mm | nvjet_tst_128x160_64x5_2x1_v_bz_TNT | 128x160, 64x5, 2x1 | 5424 | 2023.644 | 0.31% |
| aten::mm | fp8_gemm_kernel_swapAB | grid(78,1,1) block(384,1,1) | 24000 | 1052.423 | 0.16% |
| aten::mm | nvjet_tst_128x192_64x5_2x1_v_bz_coopB_NNN | 128x192, 64x5, 2x1 | 3744 | 955.440 | 0.15% |
| aten::mm | nvjet_tst_128x256_64x4_2x1_v_bz_coopA_TNN | 128x256, 64x4, 2x1 | 5616 | 737.514 | 0.11% |
| aten::mm | nvjet_tst_32x128_64x10_1x2_h_bz_splitK_TNN | 32x128, 64x10, 1x2 | 5616 | 302.130 | 0.05% |
| aten::mm | nvjet_tst_256x104_64x4_2x1_v_bz_coopA_TNT | 256x104, 64x4, 2x1 | 984 | 181.650 | 0.03% |
| aten::mm | nvjet_tst_64x64_64x13_1x2_h_bz_splitK_TNT | 64x64, 64x13, 1x2 | 9600 | 114.492 | 0.02% |
| aten::mm | nvjet_tst_64x32_64x16_1x2_h_bz_splitK_TNT | 64x32, 64x16, 1x2 | 14784 | 108.054 | 0.02% |
| aten::mm | nvjet_tst_64x16_64x16_1x2_h_bz_splitK_TNT | 64x16, 64x16, 1x2 | 13680 | 87.637 | 0.01% |
| aten::mm | nvjet_tst_64x32_64x16_1x4_h_bz_splitK_TNT | 64x32, 64x16, 1x4 | 12480 | 81.445 | 0.01% |
| aten::mm | nvjet_tst_32x64_64x16_2x1_v_bz_splitK_TNN | 32x64, 64x16, 2x1 | 6000 | 54.235 | 0.01% |
| aten::mm | nvjet_tst_64x16_64x16_1x4_h_bz_splitK_TNT | 64x16, 64x16, 1x4 | 8736 | 45.362 | 0.01% |
| aten::mm | nvjet_tst_128x32_64x10_2x1_v_bz_splitK_TNT | 128x32, 64x10, 2x1 | 3600 | 43.597 | 0.01% |
| aten::mm | nvjet_tst_96x64_64x10_1x2_h_bz_splitK_TNN | 96x64, 64x10, 1x2 | 2400 | 42.701 | 0.01% |
| aten::mm | nvjet_tst_128x184_64x5_2x1_v_bz_coopA_splitK_TNT | 128x184, 64x5, 2x1 | 600 | 35.638 | 0.01% |
| aten::mm | nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | 64x8, 64x16, 4x1 | 6000 | 34.411 | 0.01% |
| aten::mm | nvjet_tst_64x80_64x11_2x1_v_bz_splitK_TNT | 64x80, 64x11, 2x1 | 2400 | 33.242 | 0.01% |
| aten::mm | nvjet_tst_64x8_64x16_1x4_h_bz_splitK_TNT | 64x8, 64x16, 1x4 | 6240 | 31.835 | 0.00% |
| aten::mm | nvjet_tst_128x160_64x5_2x1_v_bz_splitK_TNT | 128x160, 64x5, 2x1 | 600 | 31.581 | 0.00% |
| aten::mm | nvjet_tst_64x32_64x16_4x1_v_bz_splitK_TNT | 64x32, 64x16, 4x1 | 3600 | 28.517 | 0.00% |
| aten::mm | nvjet_tst_288x128_64x4_1x2_h_bz_coopA_TNN | 288x128, 64x4, 1x2 | 16 | 28.099 | 0.00% |
| aten::mm | nvjet_tst_64x16_64x16_4x1_v_bz_splitK_TNT | 64x16, 64x16, 4x1 | 4800 | 27.808 | 0.00% |
| aten::mm | nvjet_tst_64x8_64x16_1x1_v_bz_splitK_TNT | 64x8, 64x16, 1x1 | 4992 | 26.034 | 0.00% |
| aten::mm | nvjet_tst_128x128_64x6_2x1_v_bz_NNT | 128x128, 64x6, 2x1 | 624 | 25.682 | 0.00% |
| aten::mm | nvjet_tst_128x160_64x5_2x1_v_bz_NNT | 128x160, 64x5, 2x1 | 624 | 24.410 | 0.00% |
| aten::mm | nvjet_tst_64x8_64x16_1x2_h_bz_splitK_TNT | 64x8, 64x16, 1x2 | 3744 | 21.703 | 0.00% |
| aten::mm | nvjet_tst_64x40_64x16_1x2_h_bz_splitK_TNT | 64x40, 64x16, 1x2 | 2400 | 21.644 | 0.00% |
| aten::mm | nvjet_tst_64x96_64x10_4x1_v_bz_splitK_TNT | 64x96, 64x10, 4x1 | 1200 | 21.088 | 0.00% |
| aten::mm | nvjet_tst_128x200_64x5_2x1_v_bz_coopA_TNT | 128x200, 64x5, 2x1 | 624 | 19.902 | 0.00% |
| aten::mm | nvjet_tst_176x128_64x5_1x1_v_bz_TNN | 176x128, 64x5, 1x1 | 1872 | 17.552 | 0.00% |
| aten::mm | nvjet_tst_176x128_64x5_1x2_h_bz_TNN | 176x128, 64x5, 1x2 | 624 | 17.356 | 0.00% |
| aten::mm | nvjet_tst_128x128_64x6_1x2_h_bz_NNT | 128x128, 64x6, 1x2 | 1248 | 16.829 | 0.00% |
| aten::mm | nvjet_tst_128x80_64x8_2x1_v_bz_splitK_TNT | 128x80, 64x8, 2x1 | 600 | 16.748 | 0.00% |
| aten::mm | nvjet_tst_64x48_64x15_2x1_v_bz_splitK_TNT | 64x48, 64x15, 2x1 | 1200 | 16.610 | 0.00% |
| aten::mm | nvjet_tst_128x64_64x8_2x1_v_bz_TNT | 128x64, 64x8, 2x1 | 1872 | 16.192 | 0.00% |
| aten::mm | nvjet_tst_64x24_64x16_1x2_h_bz_splitK_TNT | 64x24, 64x16, 1x2 | 2400 | 16.136 | 0.00% |
| aten::mm | nvjet_tst_64x64_64x13_4x1_v_bz_splitK_TNT | 64x64, 64x13, 4x1 | 1200 | 15.213 | 0.00% |
| aten::mm | nvjet_tst_32x64_64x16_1x2_h_bz_splitK_TNN | 32x64, 64x16, 1x2 | 1248 | 13.140 | 0.00% |
| aten::mm | nvjet_tst_64x64_64x13_2x1_v_bz_splitK_TNT | 64x64, 64x13, 2x1 | 1200 | 12.921 | 0.00% |
| aten::mm | nvjet_tst_64x48_64x15_1x2_h_bz_splitK_TNT | 64x48, 64x15, 1x2 | 1200 | 12.623 | 0.00% |
| aten::mm | nvjet_tst_64x56_64x14_1x2_h_bz_splitK_TNT | 64x56, 64x14, 1x2 | 1200 | 12.000 | 0.00% |
| aten::mm | nvjet_tst_64x40_64x16_4x1_v_bz_splitK_TNT | 64x40, 64x16, 4x1 | 1200 | 11.378 | 0.00% |
| aten::mm | nvjet_tst_64x128_64x8_2x1_v_bz_NNN | 64x128, 64x8, 2x1 | 1248 | 10.118 | 0.00% |
| aten::mm | nvjet_tst_64x120_64x9_2x1_v_bz_NNT | 64x120, 64x9, 2x1 | 1248 | 9.781 | 0.00% |
| aten::mm | nvjet_tst_64x112_64x9_2x1_v_bz_NNT | 64x112, 64x9, 2x1 | 1248 | 9.369 | 0.00% |
| aten::mm | nvjet_tst_128x120_64x6_1x2_h_bz_TNT | 128x120, 64x6, 1x2 | 1248 | 9.280 | 0.00% |
| aten::mm | nvjet_tst_64x8_64x16_4x1_v_bz_NNT | 64x8, 64x16, 4x1 | 2496 | 9.180 | 0.00% |
| aten::mm | nvjet_tst_64x104_64x10_2x1_v_bz_NNT | 64x104, 64x10, 2x1 | 1248 | 8.997 | 0.00% |
| aten::mm | nvjet_tst_64x96_64x10_2x1_v_bz_NNT | 64x96, 64x10, 2x1 | 1248 | 8.525 | 0.00% |
| aten::mm | nvjet_tst_64x32_64x16_4x2_v_bz_splitK_TNT | 64x32, 64x16, 4x2 | 1200 | 8.476 | 0.00% |
| aten::mm | nvjet_tst_64x8_64x16_2x1_v_bz_TNT | 64x8, 64x16, 2x1 | 2496 | 8.348 | 0.00% |
| aten::mm | nvjet_tst_32x64_64x16_1x1_v_bz_splitK_TNN | 32x64, 64x16, 1x1 | 1248 | 8.284 | 0.00% |
| aten::mm | nvjet_tst_32x64_64x16_4x1_v_bz_splitK_TNN | 32x64, 64x16, 4x1 | 1200 | 8.194 | 0.00% |
| aten::mm | nvjet_tst_64x88_64x11_2x1_v_bz_NNT | 64x88, 64x11, 2x1 | 1248 | 8.173 | 0.00% |
| aten::mm | nvjet_tst_128x120_64x6_1x2_h_bz_NNT | 128x120, 64x6, 1x2 | 624 | 8.038 | 0.00% |
| aten::mm | nvjet_tst_64x80_64x11_2x1_v_bz_TNT | 64x80, 64x11, 2x1 | 1248 | 8.013 | 0.00% |
| aten::mm | nvjet_tst_64x240_64x5_2x1_v_bz_NNT | 64x240, 64x5, 2x1 | 624 | 7.951 | 0.00% |
| aten::mm | nvjet_tst_64x80_64x11_2x1_v_bz_NNT | 64x80, 64x11, 2x1 | 1248 | 7.947 | 0.00% |
| aten::mm | nvjet_tst_176x64_64x7_1x1_v_bz_TNN | 176x64, 64x7, 1x1 | 1248 | 7.881 | 0.00% |
| aten::mm | nvjet_tst_64x24_64x16_4x1_v_bz_splitK_TNT | 64x24, 64x16, 4x1 | 1200 | 7.454 | 0.00% |
| aten::mm | nvjet_tst_64x72_64x12_2x1_v_bz_NNT | 64x72, 64x12, 2x1 | 1248 | 7.368 | 0.00% |
| aten::mm | nvjet_tst_64x16_64x16_1x1_h_bz_splitK_TNT | 64x16, 64x16, 1x1 | 1248 | 7.353 | 0.00% |
| aten::mm | nvjet_tst_64x208_64x6_2x1_v_bz_NNT | 64x208, 64x6, 2x1 | 624 | 7.165 | 0.00% |
| aten::mm | nvjet_tst_64x8_64x16_1x1_h_bz_splitK_TNT | 64x8, 64x16, 1x1 | 1248 | 6.963 | 0.00% |
| aten::mm | nvjet_tst_64x200_64x6_2x1_v_bz_NNT | 64x200, 64x6, 2x1 | 624 | 6.949 | 0.00% |
| aten::mm | nvjet_tst_64x64_64x13_1x2_h_bz_NNT | 64x64, 64x13, 1x2 | 1248 | 6.925 | 0.00% |
| aten::mm | nvjet_tst_128x96_64x7_1x2_h_bz_NNT | 128x96, 64x7, 1x2 | 624 | 6.823 | 0.00% |
| aten::mm | nvjet_tst_128x48_64x9_2x1_v_bz_NNT | 128x48, 64x9, 2x1 | 624 | 6.609 | 0.00% |
| aten::mm | nvjet_tst_64x56_64x14_1x2_h_bz_NNT | 64x56, 64x14, 1x2 | 1248 | 6.575 | 0.00% |
| aten::mm | nvjet_tst_64x16_64x16_1x1_v_bz_splitK_TNT | 64x16, 64x16, 1x1 | 1248 | 6.530 | 0.00% |
| aten::mm | nvjet_tst_64x184_64x6_2x1_v_bz_NNT | 64x184, 64x6, 2x1 | 624 | 6.503 | 0.00% |
| aten::mm | nvjet_tst_64x64_64x13_2x1_v_bz_NNT | 64x64, 64x13, 2x1 | 624 | 6.381 | 0.00% |
| aten::mm | nvjet_tst_64x176_64x7_2x1_v_bz_NNT | 64x176, 64x7, 2x1 | 624 | 6.329 | 0.00% |
| aten::mm | nvjet_tst_64x48_64x15_1x2_h_bz_NNT | 64x48, 64x15, 1x2 | 1248 | 6.172 | 0.00% |
| aten::mm | nvjet_tst_64x168_64x7_2x1_v_bz_NNT | 64x168, 64x7, 2x1 | 624 | 6.119 | 0.00% |
| aten::mm | nvjet_tst_128x208_64x5_2x1_v_bz_coopA_TNT | 128x208, 64x5, 2x1 | 624 | 6.059 | 0.00% |
| aten::mm | nvjet_tst_128x80_64x8_1x2_h_bz_NNT | 128x80, 64x8, 1x2 | 624 | 6.017 | 0.00% |
| aten::mm | nvjet_tst_64x40_64x16_1x2_h_bz_NNT | 64x40, 64x16, 1x2 | 1248 | 5.817 | 0.00% |
| aten::mm | nvjet_tst_64x152_64x7_2x1_v_bz_NNT | 64x152, 64x7, 2x1 | 624 | 5.700 | 0.00% |
| aten::mm | nvjet_tst_64x144_64x8_2x1_v_bz_NNT | 64x144, 64x8, 2x1 | 624 | 5.512 | 0.00% |
| aten::mm | nvjet_tst_128x152_64x6_2x1_v_bz_TNT | 128x152, 64x6, 2x1 | 624 | 5.306 | 0.00% |
| aten::mm | nvjet_tst_64x136_64x8_2x1_v_bz_NNT | 64x136, 64x8, 2x1 | 624 | 5.284 | 0.00% |
| aten::mm | nvjet_tst_64x32_64x16_1x2_h_bz_NNT | 64x32, 64x16, 1x2 | 1248 | 5.275 | 0.00% |
| aten::mm | nvjet_tst_128x144_64x6_2x1_v_bz_TNT | 128x144, 64x6, 2x1 | 624 | 5.244 | 0.00% |
| aten::mm | nvjet_tst_128x136_64x6_2x1_v_bz_TNT | 128x136, 64x6, 2x1 | 624 | 5.093 | 0.00% |
| aten::mm | nvjet_tst_64x24_64x16_1x2_h_bz_NNT | 64x24, 64x16, 1x2 | 1248 | 4.939 | 0.00% |
| aten::mm | nvjet_tst_128x128_64x6_2x1_v_bz_TNT | 128x128, 64x6, 2x1 | 624 | 4.851 | 0.00% |
| aten::mm | nvjet_tst_128x128_64x6_1x2_h_bz_TNT | 128x128, 64x6, 1x2 | 624 | 4.789 | 0.00% |
| aten::mm | nvjet_tst_64x16_64x16_1x2_h_bz_NNT | 64x16, 64x16, 1x2 | 1248 | 4.702 | 0.00% |
| aten::mm | nvjet_tst_128x48_64x9_2x1_v_bz_TNT | 128x48, 64x9, 2x1 | 624 | 4.606 | 0.00% |
| aten::mm | nvjet_tst_128x112_64x7_1x2_h_bz_TNT | 128x112, 64x7, 1x2 | 624 | 4.475 | 0.00% |
| aten::mm | nvjet_tst_64x216_64x6_2x1_v_bz_TNT | 64x216, 64x6, 2x1 | 624 | 4.318 | 0.00% |
| aten::mm | nvjet_tst_128x104_64x7_1x2_h_bz_TNT | 128x104, 64x7, 1x2 | 624 | 4.289 | 0.00% |
| aten::mm | nvjet_tst_64x200_64x6_2x1_v_bz_TNT | 64x200, 64x6, 2x1 | 624 | 4.123 | 0.00% |
| aten::mm | nvjet_tst_128x88_64x7_1x2_h_bz_TNT | 128x88, 64x7, 1x2 | 624 | 3.906 | 0.00% |
| aten::mm | nvjet_tst_64x168_64x7_2x1_v_bz_TNT | 64x168, 64x7, 2x1 | 624 | 3.720 | 0.00% |
| aten::mm | nvjet_tst_128x80_64x8_1x2_h_bz_TNT | 128x80, 64x8, 1x2 | 624 | 3.678 | 0.00% |
| aten::mm | nvjet_tst_64x152_64x7_2x1_v_bz_TNT | 64x152, 64x7, 2x1 | 624 | 3.518 | 0.00% |
| aten::mm | nvjet_tst_128x72_64x8_1x2_h_bz_TNT | 128x72, 64x8, 1x2 | 624 | 3.478 | 0.00% |
| aten::mm | nvjet_tst_64x136_64x8_2x1_v_bz_TNT | 64x136, 64x8, 2x1 | 624 | 3.309 | 0.00% |
| aten::mm | nvjet_tst_128x64_64x8_1x2_h_bz_TNT | 128x64, 64x8, 1x2 | 624 | 3.207 | 0.00% |
| aten::mm | nvjet_tst_64x120_64x9_2x1_v_bz_TNT | 64x120, 64x9, 2x1 | 624 | 3.119 | 0.00% |
| aten::mm | nvjet_tst_128x56_64x9_1x2_h_bz_TNT | 128x56, 64x9, 1x2 | 624 | 3.106 | 0.00% |
| aten::mm | nvjet_tst_128x48_64x9_1x2_h_bz_TNT | 128x48, 64x9, 1x2 | 624 | 2.943 | 0.00% |
| aten::mm | nvjet_tst_64x104_64x10_2x1_v_bz_TNT | 64x104, 64x10, 2x1 | 624 | 2.931 | 0.00% |
| aten::mm | nvjet_tst_64x88_64x11_2x1_v_bz_TNT | 64x88, 64x11, 2x1 | 624 | 2.762 | 0.00% |
| aten::mm | nvjet_tst_64x72_64x12_2x1_v_bz_TNT | 64x72, 64x12, 2x1 | 624 | 2.619 | 0.00% |
| aten::mm | nvjet_tst_256x88_64x4_2x1_v_bz_TNT | 256x88, 64x4, 2x1 | 16 | 2.523 | 0.00% |
| aten::mm | nvjet_tst_64x64_64x13_2x1_v_bz_TNT | 64x64, 64x13, 2x1 | 624 | 2.496 | 0.00% |
| aten::mm | nvjet_tst_64x56_64x14_2x1_v_bz_TNT | 64x56, 64x14, 2x1 | 624 | 2.495 | 0.00% |
| aten::mm | nvjet_tst_64x48_64x15_2x1_v_bz_TNT | 64x48, 64x15, 2x1 | 624 | 2.436 | 0.00% |
| aten::mm | nvjet_tst_64x40_64x16_2x1_v_bz_TNT | 64x40, 64x16, 2x1 | 624 | 2.413 | 0.00% |
| aten::mm | nvjet_tst_64x8_64x16_1x2_h_bz_NNT | 64x8, 64x16, 1x2 | 624 | 2.311 | 0.00% |
| aten::mm | nvjet_tst_64x32_64x16_2x1_v_bz_TNT | 64x32, 64x16, 2x1 | 624 | 2.297 | 0.00% |
| aten::mm | nvjet_tst_64x24_64x16_2x1_v_bz_TNT | 64x24, 64x16, 2x1 | 624 | 2.234 | 0.00% |
| aten::mm | nvjet_tst_64x16_64x16_2x1_v_bz_TNT | 64x16, 64x16, 2x1 | 624 | 2.178 | 0.00% |
| aten::mm | nvjet_tst_256x72_64x5_2x1_v_bz_TNT | 256x72, 64x5, 2x1 | 16 | 2.084 | 0.00% |
| aten::mm | nvjet_tst_256x56_64x5_2x1_v_bz_TNT | 256x56, 64x5, 2x1 | 16 | 1.648 | 0.00% |
| aten::mm | nvjet_tst_256x8_64x6_2x1_v_bz_TNT | 256x8, 64x6, 2x1 | 24 | 1.505 | 0.00% |
| aten::mm | nvjet_tst_256x40_64x5_2x1_v_bz_TNT | 256x40, 64x5, 2x1 | 16 | 1.228 | 0.00% |
| aten::mm | nvjet_tst_256x24_64x6_2x1_v_bz_TNT | 256x24, 64x6, 2x1 | 16 | 1.100 | 0.00% |

## FlagGems aten::mm 算子（按总时间排序）

| 框架算子名 | 执行算子名 | 输入 shape | FlagGems调用次数 | FlagGems总时间(ms) | FlagGems占比 |
|---|---|---|---|---|---|
| aten::mm | sm90_fp8_gemm_1d2d_impl | grid(78,1,1) block(384,1,1); grid(64,1,1) block(256,1,1) | 542744 | 82454.329 | 10.41% |
| aten::mm | fp8_gemm_kernel | grid(78,1,1) block(384,1,1); grid(78,1,1) block(256,1,1) | 111600 | 41916.409 | 5.29% |
| aten::mm | mm_kernel_general_host_tma | grid(8,1,1) block(128,1,1); grid(24,1,1) block(128,1,1) | 96982 | 7434.742 | 0.94% |
| aten::mm | bmm_kernel | grid(8,8,8) block(128,1,1); grid(2,8,8) block(128,1,1) | 109357 | 2952.014 | 0.37% |
| aten::mm | mm_kernel_general | grid(1,1,1) block(128,1,1); grid(256,1,1) block(128,1,1) | 74652 | 2369.495 | 0.30% |
| aten::mm | fp8_gemm_kernel_swapAB | grid(78,1,1) block(384,1,1) | 24000 | 1052.674 | 0.13% |
