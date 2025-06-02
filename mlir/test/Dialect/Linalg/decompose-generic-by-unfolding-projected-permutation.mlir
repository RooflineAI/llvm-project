// RUN: mlir-opt %s -split-input-file --linalg-specialize-generic-ops | FileCheck %s

#projection = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d1)>
#identity   = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @transpose_and_broadcast(%x : tensor<7x8x9xf32>, %y:  tensor<5x9x7x8x10xf32>, %z :  tensor<5x9x7x8x10xf32>) ->  tensor<5x9x7x8x10xf32> {
  %res = linalg.generic
     { indexing_maps = [#projection, #identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
     ins(%x, %y : tensor<7x8x9xf32>, tensor<5x9x7x8x10xf32>) outs(%z : tensor<5x9x7x8x10xf32>) {
     ^bb0(%in: f32, %in_1: f32, %out: f32):
       %div = arith.divf %in, %in_1 : f32
       linalg.yield %div : f32
  } -> tensor<5x9x7x8x10xf32>
  return %res : tensor<5x9x7x8x10xf32>
}

// CHECK-LABEL: transpose_and_broadcast
// CHECK-SAME: %[[X:.+]]: tensor<7x8x9xf32>, %[[Y:.+]]: tensor<5x9x7x8x10xf32>, %[[Z:.+]]: tensor<5x9x7x8x10xf32>) -> tensor<5x9x7x8x10xf32> {
// CHECK: %[[E0:.+]] = tensor.empty() : tensor<9x7x8xf32>
// CHECK: %[[X_trans:.+]] = linalg.transpose ins(%[[X]] : tensor<7x8x9xf32>) outs(%[[E0]] : tensor<9x7x8xf32>) permutation = [2, 0, 1]
// CHECK: %[[E1:.+]] = tensor.empty() : tensor<5x9x7x8x10xf32>
// CHECK: %[[X_trans_bc:.+]] = linalg.broadcast ins(%[[X_trans]] : tensor<9x7x8xf32>) outs(%[[E1]] : tensor<5x9x7x8x10xf32>) dimensions = [0, 4]
// CHECK: {{.*}} = linalg.div ins(%[[X_trans_bc]], %[[Y]] : tensor<5x9x7x8x10xf32>, tensor<5x9x7x8x10xf32>) outs(%[[Z]] : tensor<5x9x7x8x10xf32>) -> tensor<5x9x7x8x10xf32>
// CHECK-NOT: linalg.generic

// -----

#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#transposed = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

func.func @transpose_only(%x : tensor<32x2x16xf32>, %y:  tensor<2x16x32xf32>, %z :  tensor<2x16x32xf32>) ->  tensor<2x16x32xf32> {
  %res = linalg.generic
     { indexing_maps = [#transposed, #identity, #identity], iterator_types = ["parallel", "parallel", "parallel"]}
     ins(%x, %y : tensor<32x2x16xf32>, tensor<2x16x32xf32>)
     outs(%z : tensor<2x16x32xf32>) {
     ^bb0(%in: f32, %in_1: f32, %out: f32):
       %div = arith.divf %in, %in_1 : f32
       linalg.yield %div : f32
  } -> tensor<2x16x32xf32>
  return %res : tensor<2x16x32xf32>
}

// CHECK-LABEL: transpose_only
// CHECK-SAME: %[[X:.+]]: tensor<32x2x16xf32>, %[[Y:.+]]: tensor<2x16x32xf32>, %[[Z:.+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK: %[[E0:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK: %[[X_trans:.+]] = linalg.transpose ins(%[[X]] : tensor<32x2x16xf32>) outs(%[[E0]] : tensor<2x16x32xf32>) permutation = [1, 2, 0]
// CHECK: {{.*}} = linalg.div ins(%[[X_trans]], %[[Y]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>) outs(%[[Z]] : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
// CHECK-NOT: linalg.generic

// -----

#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast = affine_map<(d0, d1, d2) -> (d0, d2)>
func.func @broadcast_only(%x : tensor<2x16x32xf32>, %y:  tensor<2x32xf32>, %z :  tensor<2x16x32xf32>) ->  tensor<2x16x32xf32> {
  %res = linalg.generic
     { indexing_maps = [#identity, #broadcast, #identity], iterator_types = ["parallel", "parallel", "parallel"]}
     ins(%x, %y : tensor<2x16x32xf32>, tensor<2x32xf32>)
     outs(%z : tensor<2x16x32xf32>) {
     ^bb0(%in: f32, %in_1: f32, %out: f32):
       %div = arith.divf %in, %in_1 : f32
       linalg.yield %div : f32
  } -> tensor<2x16x32xf32>
  return %res : tensor<2x16x32xf32>
}

// CHECK-LABEL: broadcast_only
// CHECK-SAME: %[[X:.+]]: tensor<2x16x32xf32>, %[[Y:.+]]: tensor<2x32xf32>, %[[Z:.+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK: %[[E0:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK: %[[X_bc:.+]] = linalg.broadcast ins(%[[Y]] : tensor<2x32xf32>) outs(%[[E0]] : tensor<2x16x32xf32>) dimensions = [1]
// CHECK: {{.*}} = linalg.div ins(%[[X]], %[[X_bc]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>) outs(%arg2 : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
// CHECK-NOT: linalg.generic


// -----

#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast = affine_map<(d0, d1, d2) -> ()>
func.func @scalar_broadcast(%x: tensor<1x8x16xf32>, %y:  f32) ->  tensor<1x8x16xf32> {
  %empty = tensor.empty() : tensor<1x8x16xf32>
  %res = linalg.generic
     { indexing_maps = [#identity, #broadcast, #identity], iterator_types = ["parallel", "parallel", "parallel"]}
     ins(%x, %y : tensor<1x8x16xf32>, f32)
     outs(%empty : tensor<1x8x16xf32>) {
     ^bb0(%in: f32, %in2: f32,  %out: f32):
       %add = arith.addf %in, %in2 : f32
       linalg.yield %add : f32
  } -> tensor<1x8x16xf32>
  return %res : tensor<1x8x16xf32>
}

// CHECK-LABEL: scalar_broadcast
// CHECK-SAME:      %[[INPUT:.+]]: tensor<1x8x16xf32>
// CHECK-SAME:      %[[SCALAR:.+]]: f32
// CHECK-DAG:     %[[EMPTY_ADD:.+]] = tensor.empty() : tensor<1x8x16xf32>
// CHECK-DAG:     %[[EMPTY_FILL:.+]] = tensor.empty() : tensor<1x8x16xf32>
// CHECK-DAG:     %[[FILL:.+]] = linalg.fill
// CHECK-SAME:                    ins(%[[SCALAR]] : f32)
// CHECK-SAME:                    outs(%[[EMPTY_FILL]] : tensor<1x8x16xf32>)
// CHECK:         %[[ADD:.+]] = linalg.add
// CHECK-SAME:                    ins(%[[INPUT]], %[[FILL]] : tensor<1x8x16xf32>, tensor<1x8x16xf32>) 
// CHECK-SAME:                    outs(%[[EMPTY_ADD]] : tensor<1x8x16xf32>)
// CHECK:         return %[[ADD]] : tensor<1x8x16xf32>

// -----

#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast = affine_map<(d0, d1, d2) -> (d2)>
func.func @ignore_non_ranked_tensor_types(%x: memref<1x8x16xf32>, %y:  memref<16xf32>) {
  %empty = memref.alloc() : memref<1x8x16xf32>
  linalg.generic
     { indexing_maps = [#identity, #broadcast, #identity], iterator_types = ["parallel", "parallel", "parallel"]}
     ins(%x, %y : memref<1x8x16xf32>,  memref<16xf32>)
     outs(%empty : memref<1x8x16xf32>) {
     ^bb0(%in: f32, %in2: f32,  %out: f32):
       %add = arith.addf %in, %in2 : f32
       linalg.yield %add : f32
  }
  func.return
}

// CHECK-LABEL: ignore_non_ranked_tensor_types
// CHECK-SAME:      %[[X:.+]]: memref<1x8x16xf32>
// CHECK-SAME:      %[[Y:.+]]: memref<16xf32>
// CHECK:         %[[EMPTY:.+]] = memref.alloc() : memref<1x8x16xf32>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[X]], %[[Y]] : memref<1x8x16xf32>, memref<16xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : memref<1x8x16xf32>)
// CHECK:         return
