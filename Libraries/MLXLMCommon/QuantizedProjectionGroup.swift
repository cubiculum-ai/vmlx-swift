// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN

/// One packed affine projection standing in for several compatible same-input ones.
///
/// Affine quantization metadata is independent for every OUTPUT ROW: `scales` and `biases` are
/// `[out, in / groupSize]`, one row per output. Projections that read the same input can therefore
/// be concatenated on the output-row axis and evaluated by a single `quantizedMM` **without
/// changing their math** — each row still meets exactly the scale and bias it had before. What the
/// packing removes is the per-projection command launch, and, where the caller was concatenating
/// the results anyway, the concatenate too.
///
/// The win is dispatch count, not arithmetic. A KDA decode step issues `q_proj`, `k_proj` and
/// `v_proj` against the same hidden state and immediately concatenates them into the `[B, T, 3P]`
/// buffer the fused convolution wants; that is four commands to produce one buffer that one command
/// can produce.
///
/// This type does not decide policy. It reports whether a group is packable and builds it if asked;
/// a caller that cannot use the packed form keeps its own projections and loses nothing.
///
/// ## Why the compatibility test is shape-only
///
/// `weight` is `[out, in * bits / 32]` and `scales` is `[out, in / groupSize]`. For projections that
/// share an input, equal `weight.dim(1)` across the group forces equal `bits`, and equal
/// `scales.dim(1)` forces equal `groupSize` — so agreement on those two trailing dimensions is
/// exactly agreement on the quantization parameters, with no need to read a config. That matters
/// because the caller runs before per-layer quantization overrides have been resolved, and a test
/// phrased in terms of declared metadata would be testing something the weights had not yet been
/// reconciled with.
public enum QuantizedProjectionGroup {

    /// The packed parameters of a projection group.
    ///
    /// Deliberately NOT a `Module` and deliberately not holding one. `Module` discovers its children
    /// by reflection, collecting `MLXArray`, `Module`, and collections of those; a plain final class
    /// falls through that match and stays invisible. Holding the packed arrays here therefore keeps
    /// them out of `parameters()`, out of `update(parameters:)`, and out of `verify:` — which is what
    /// we want, because these arrays are DERIVED at run time and correspond to no checkpoint key. A
    /// module that appears in the tree but binds no key is the shape of the phantom-MTP-head bug:
    /// it consumes no keys, so `.noUnusedKeys` cannot see it, and it costs full resident memory.
    public final class Packed {
        public let weight: MLXArray
        public let scales: MLXArray
        public let biases: MLXArray?
        public let groupSize: Int
        public let bits: Int
        public let mode: QuantizationMode
        /// Output widths of the original projections, in order — how a caller splits the result.
        public let outputWidths: [Int]

        init(
            weight: MLXArray, scales: MLXArray, biases: MLXArray?,
            groupSize: Int, bits: Int, mode: QuantizationMode, outputWidths: [Int]
        ) {
            self.weight = weight
            self.scales = scales
            self.biases = biases
            self.groupSize = groupSize
            self.bits = bits
            self.mode = mode
            self.outputWidths = outputWidths
        }

        /// Apply the packed projection, returning the concatenation the originals would have
        /// produced: `[..., outputWidths.reduce(0, +)]`, in the order the group was built.
        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            let inputDType = x.dtype
            let y = quantizedMM(
                x, weight, scales: scales, biases: biases, transpose: true,
                groupSize: groupSize, bits: bits, mode: mode)
            // `quantizedMM` promotes; the originals returned the input dtype, and a caller that
            // concatenates our output with an unpromoted tensor would otherwise get a surprise.
            return y.dtype == inputDType ? y : y.asType(inputDType)
        }
    }

    /// Why a group cannot be packed, or `nil` when it can.
    ///
    /// Returned as a sentence rather than a `Bool` so a caller that expected packing to apply can
    /// say what stopped it. Silence here would make a lost optimisation indistinguishable from an
    /// applied one.
    public static func reasonGroupIsNotPackable(_ linears: [Module]) -> String? {
        guard linears.count >= 2 else {
            return "a group needs at least two projections, got \(linears.count)"
        }
        var quantized: [QuantizedLinear] = []
        quantized.reserveCapacity(linears.count)
        for (i, layer) in linears.enumerated() {
            guard let q = layer as? QuantizedLinear else {
                return "projection \(i) is \(type(of: layer)), not QuantizedLinear"
            }
            // A bias on the Linear itself would have to be concatenated too. None of the projections
            // this exists for carry one, so rather than write an untested path, decline.
            if q.bias != nil {
                return "projection \(i) carries a Linear bias, which this packing does not handle"
            }
            quantized.append(q)
        }
        let first = quantized[0]
        for (i, q) in quantized.enumerated().dropFirst() {
            if q.weight.ndim != 2 || first.weight.ndim != 2 {
                return "projection \(i) weight is rank \(q.weight.ndim), expected 2"
            }
            if q.weight.dim(1) != first.weight.dim(1) {
                return
                    "projection \(i) packs \(q.weight.dim(1)) input words vs \(first.weight.dim(1)) "
                    + "— different bit widths or input sizes"
            }
            if q.scales.dim(1) != first.scales.dim(1) {
                return
                    "projection \(i) has \(q.scales.dim(1)) scale groups vs \(first.scales.dim(1)) "
                    + "— different group sizes"
            }
            if q.groupSize != first.groupSize || q.bits != first.bits {
                return
                    "projection \(i) is \(q.bits)b/g\(q.groupSize) vs \(first.bits)b/g\(first.groupSize)"
            }
            if q.mode != first.mode {
                return "projection \(i) uses a different quantization mode"
            }
            if q.weight.dtype != first.weight.dtype || q.scales.dtype != first.scales.dtype {
                return "projection \(i) has a different storage dtype"
            }
            if (q.biases == nil) != (first.biases == nil) {
                return "projection \(i) disagrees with the group about having biases"
            }
        }
        return nil
    }

    /// The outcome of a packing attempt: the packed group, or the sentence explaining the refusal.
    ///
    /// An enum rather than an optional so the refusal reason cannot be dropped by a caller that
    /// writes `if let`. A group silently declining to pack looks exactly like one that packed.
    public enum Outcome {
        case packed(Packed)
        case refused(String)
    }

    /// Pack a group, or refuse with the reason.
    ///
    /// Concatenation COPIES: the packed arrays are new storage, and the originals stay alive until
    /// the caller drops them. A caller under memory pressure should pack one group at a time and
    /// release as it goes rather than packing a whole model and then releasing.
    public static func pack(_ linears: [Module]) -> Outcome {
        if let reason = reasonGroupIsNotPackable(linears) { return .refused(reason) }
        // `reasonGroupIsNotPackable` already established every element is a QuantizedLinear.
        let quantized = linears.compactMap { $0 as? QuantizedLinear }
        let first = quantized[0]
        return .packed(
            Packed(
                weight: concatenated(quantized.map(\.weight), axis: 0),
                scales: concatenated(quantized.map(\.scales), axis: 0),
                biases: first.biases == nil
                    ? nil : concatenated(quantized.compactMap(\.biases), axis: 0),
                groupSize: first.groupSize,
                bits: first.bits,
                mode: first.mode,
                outputWidths: quantized.map { $0.weight.dim(0) }))
    }
}
