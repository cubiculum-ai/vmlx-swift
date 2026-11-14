//
//  DoRA+Layers.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 02.06.2025.
//

import Foundation
import MLX
import MLXNN

/// Performs the forward pass for a DoRA linear layer.
/// DoRA's per-output-channel rescaling, `magnitude / norm(W + scale * Bt At, axis: 1)`.
///
/// Split out of `forward` so a layer that can prove its inputs are fixed may hydrate it ONCE. It is
/// by far the expensive half: materializing `adapted` is a dense [out, in] outer-product expansion
/// plus an add, and the norm is a full reduction over it — all three per token, against a forward
/// whose own matmul is a single [out, in] pass.
private func doraDirectionScale(
    weight: MLXArray, loraA: MLXArray, loraB: MLXArray,
    scale: Float, magnitude: MLXArray
) -> MLXArray {
    let adapted = weight + matmul(scale * loraB.T, loraA.T)
    return magnitude / norm(adapted, axis: 1)
}

private func forward(
    x: MLXArray, y: MLXArray,
    bias: MLXArray?,
    loraA: MLXArray, loraB: MLXArray,
    scale: Float, directionScale: MLXArray
) -> MLXArray {
    let z = matmul(matmul(x, loraA), loraB)
    var out = y + (scale * z).asType(x.dtype)

    out *= directionScale.asType(x.dtype)

    return if let bias {
        out + bias
    } else {
        out
    }
}

/// Fuses the base weights with the DoRA parameters.
private func fuse(
    weight: MLXArray,
    loraA: MLXArray, loraB: MLXArray,
    scale: Float, magnitude: MLXArray
) -> MLXArray {
    let loraA = loraA.T.asType(weight.dtype)
    let loraB = (scale * loraB.T).asType(weight.dtype)

    var adapted = weight + matmul(loraB, loraA)
    let denom = norm(adapted, axis: 1)
    adapted *= (magnitude / denom).reshaped([-1, 1])

    return adapted
}

/// Filters out DoRA-specific parameters from a list of module keys.
private func filterFreezeKeys(from module: Module, keys: [String]?) -> [String] {
    return
        (keys
        ?? module.filterMap(filter: type(of: module).filterLocalParameters)
        .flattened()
        .map { $0.0 })
        .filter { !["lora_a", "lora_b", "m"].contains($0) }
}

/// Implementation of DoRA `Linear` replacement layer.
///
/// This layer implements DoRA (Weight-Decomposed Low-Rank Adaptation) for `Linear` layers.
///
/// ``QDoRALinear`` is the equivalent class for `QuantizedLinear`.
public class DoRALinear: Linear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray
    @ParameterInfo(key: "m") var magnitude: MLXArray

    required public init(linear: Linear, rank: Int = 8, scale: Float = 20.0) {
        let (outputDimensions, inputDimensions) = linear.shape
        let loraScale = 1 / sqrt(Float(inputDimensions))

        self.scale = scale
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])
        self._magnitude.wrappedValue = MLXLinalg.norm(linear.weight, axis: 1)

        super.init(weight: linear.weight, bias: linear.bias)

        freeze()
    }

    public static func from(linear: Linear, rank: Int = 8, scale: Float = 20.0) -> LoRALayer {
        if let linear = linear as? QuantizedLinear {
            QDoRALinear(linear: linear, rank: rank, scale: scale)
        } else {
            DoRALinear(linear: linear, rank: rank, scale: scale)
        }
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        let keys = filterFreezeKeys(from: self, keys: keys)
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    public func fused() -> Module {
        Linear(
            weight: fuse(
                weight: weight, loraA: loraA, loraB: loraB, scale: scale, magnitude: magnitude),
            bias: bias
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = matmul(x, weight.T)
        return forward(
            x: x, y: y,
            bias: bias,
            loraA: loraA, loraB: loraB,
            scale: scale,
            directionScale: doraDirectionScale(
                weight: weight, loraA: loraA, loraB: loraB,
                scale: scale, magnitude: magnitude)
        )
    }
}

/// Implementation of DoRA `QuantizedLinear` replacement layer.
///
/// See ``DoRALinear`` (equivalent class for `Linear` layers) for more information.
///
/// ### See Also
/// - ``DoRALinear``
public class QDoRALinear: QuantizedLinear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray
    @ParameterInfo(key: "m") var magnitude: MLXArray

    required public init(linear: QuantizedLinear, rank: Int = 8, scale: Float = 20.0) {
        let (outputDimensions, inputDimensions) = linear.shape
        let loraScale = 1 / sqrt(Float(inputDimensions))

        self.scale = scale
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])
        self._magnitude.wrappedValue = MLXLinalg.norm(linear.dequantizedWeight, axis: 1)

        super.init(
            weight: linear.weight, bias: linear.bias,
            scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits,
            mode: linear.mode
        )

        freeze()
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        let keys = filterFreezeKeys(from: self, keys: keys)
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    /// The dequantized base weight, hydrated on first use and then kept.
    ///
    /// DoRA needs the DENSE base weight in its forward and cannot be rewritten to avoid it: the
    /// output is scaled by `magnitude / norm(W + scale * Bt At, axis: 1)`, the per-row norms of the
    /// ADAPTED matrix, which depend on `W` entry by entry. But `dequantizedWeight` is a COMPUTED
    /// property on `QuantizedLinear` — every read reruns `dequantized` over the whole matrix. Reading
    /// it from `callAsFunction` therefore rebuilt the full dense weight of every adapted layer on
    /// EVERY token. Peak memory does not rise from keeping it, because an array that size was already
    /// being materialized each time; this only stops discarding it.
    ///
    /// WHY THERE IS NO INVALIDATION, AND WHY THAT IS SAFE HERE. The obvious guard — compare the
    /// source array's identity and re-hydrate when it changes — DOES NOT WORK in this runtime, and
    /// looks like it does. `Module.update(parameters:)` writes through `MLXArray._updateInternal`,
    /// which calls `mlx_array_set`, and that assigns the underlying C++ array IN PLACE
    /// (`*static_cast<mlx::core::array*>(d.ctx) = s`). Neither the Swift object nor the `ctx` handle
    /// changes, so an identity check silently reports "unchanged" across a real parameter update, and
    /// mlx-c exposes no array id to key on instead. What makes the cache sound is narrower and
    /// checkable: `weight` is frozen here (`freeze()` in `init`), so no optimizer writes it, and a
    /// DoRA adapter file carries `lora_a`, `lora_b` and `m` — never the base weight. Nothing in the
    /// load path rewrites `weight` after the first forward. If that ever changes, this cache goes
    /// stale silently, so the invariant is worth keeping in mind rather than assuming.
    ///
    /// IT MUST START NIL. `Module` reflects over stored properties and `ModuleValue.build` maps any
    /// NON-NIL `MLXArray` ivar to `.value(.parameters(...))`, while a nil optional maps to `.none`;
    /// `_items` is then memoized on first use. Populated lazily, after loading, this stays out of
    /// `parameters()`. Assigned in `init` it would become a phantom parameter that gets saved into
    /// checkpoints and breaks key matching on load.
    private var denseBaseWeight: MLXArray?

    private func baseWeight() -> MLXArray {
        if let cached = denseBaseWeight { return cached }
        let dense = dequantizedWeight
        eval(dense)
        denseBaseWeight = dense
        return dense
    }

    public func fused() -> Module {
        QuantizedLinear(
            weight: fuse(
                weight: baseWeight(), loraA: loraA, loraB: loraB, scale: scale,
                magnitude: magnitude),
            bias: bias, groupSize: groupSize, bits: bits
        )
    }

    /// DoRA's rescaling vector, hydrated once for inference and recomputed while training.
    ///
    /// Unlike `denseBaseWeight` this depends on `loraA`, `loraB` and `magnitude`, which DO change —
    /// an optimizer rewrites them every step, and loading an adapter writes all three. So it is
    /// cached only in EVAL mode, and `didSetTrain` drops it on any mode change.
    ///
    /// Caching it while training would be worse than stale: `magnitude / norm(W + scale * Bt At)` is
    /// exactly where DoRA's gradient with respect to the low-rank factors flows, so a constant here
    /// does not merely use old numbers, it severs that gradient and silently trains a different
    /// model. That is why the gate is the training flag and not a performance switch.
    ///
    /// Loading a DIFFERENT adapter is safe without any hook: `LoRAContainer.load(into:)` goes
    /// through `replaceLayers`, which constructs new layer instances, so the cache cannot outlive
    /// the parameters it was derived from.
    private var cachedDirectionScale: MLXArray?

    public override func didSetTrain(_ mode: Bool) {
        super.didSetTrain(mode)
        cachedDirectionScale = nil
    }

    private func directionScale() -> MLXArray {
        if training {
            return doraDirectionScale(
                weight: baseWeight(), loraA: loraA, loraB: loraB,
                scale: scale, magnitude: magnitude)
        }
        if let cached = cachedDirectionScale { return cached }
        let computed = doraDirectionScale(
            weight: baseWeight(), loraA: loraA, loraB: loraB,
            scale: scale, magnitude: magnitude)
        eval(computed)
        cachedDirectionScale = computed
        return computed
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = quantizedMM(
            x, weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits,
            mode: mode)
        return forward(
            x: x, y: y,
            bias: bias,
            loraA: loraA, loraB: loraB,
            scale: scale, directionScale: directionScale()
        )
    }
}
