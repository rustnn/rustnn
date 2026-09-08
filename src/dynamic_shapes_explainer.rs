// SPDX-FileCopyrightText: 2026 Nvidia
//
// SPDX-License-Identifier: Apache-2

use crate::{
    mlcontext::MLOperand,
    operator_options::{
        MLOperatorOptions, MLResample2dDynamicOptions, MLReshapeTo2dOptions, MLSliceDynamicOptions,
        MLSplitOptions, MLSqueezeOptions,
    },
};

// From https://github.com/webmachinelearning/webnn/pull/945.
pub trait DynamicShapeBuilder {
    // Read an operand's shape as a runtime uint32 1-D tensor.
    fn shape(&mut self, input: MLOperand) -> MLOperand {
        self.shape_with_options(input, MLOperatorOptions::default())
    }
    fn shape_with_options(&mut self, input: MLOperand, options: MLOperatorOptions) -> MLOperand;

    // Shape generators / arithmetic on shape tensors.
    fn range(&mut self, start: MLOperand, limit: MLOperand, delta: MLOperand) -> MLOperand {
        self.range_with_options(start, limit, delta, MLOperatorOptions::default())
    }
    fn range_with_options(
        &mut self,
        start: MLOperand,
        limit: MLOperand,
        delta: MLOperand,
        options: MLOperatorOptions,
    ) -> MLOperand;
    fn modulus_floor(&mut self, a: MLOperand, b: MLOperand) -> MLOperand {
        self.modulus_floor_with_options(a, b, MLOperatorOptions::default())
    }
    fn modulus_floor_with_options(
        &mut self,
        a: MLOperand,
        b: MLOperand,
        options: MLOperatorOptions,
    ) -> MLOperand;
    fn modulus_truncate(&mut self, a: MLOperand, b: MLOperand) -> MLOperand {
        self.modulus_truncate_with_options(a, b, MLOperatorOptions::default())
    }
    fn modulus_truncate_with_options(
        &mut self,
        a: MLOperand,
        b: MLOperand,
        options: MLOperatorOptions,
    ) -> MLOperand;

    // Rank-changing operators (the seam where dynamic rank originates).
    fn squeeze(&mut self, input: MLOperand) -> MLOperand {
        self.squeeze_with_options(input, MLSqueezeOptions::default())
    }
    fn squeeze_with_options(&mut self, input: MLOperand, options: MLSqueezeOptions) -> MLOperand;
    fn unsqueeze(&mut self, input: MLOperand, axes: &[u32]) -> MLOperand {
        self.unsqueeze_with_options(input, axes, MLOperatorOptions::default())
    }
    fn unsqueeze_with_options(
        &mut self,
        input: MLOperand,
        axes: &[u32],
        options: MLOperatorOptions,
    ) -> MLOperand;
    fn reshape_to_2d(&mut self, input: MLOperand) -> MLOperand {
        self.reshape_to_2d_with_options(input, MLReshapeTo2dOptions::default())
    }
    fn reshape_to_2d_with_options(
        &mut self,
        input: MLOperand,
        options: MLReshapeTo2dOptions,
    ) -> MLOperand;

    // Dynamic variants: shape parameters are operands, evaluated at dispatch.
    fn reshape_dynamic(&mut self, input: MLOperand, new_shape: MLOperand) -> MLOperand {
        self.reshape_dynamic_with_options(input, new_shape, MLOperatorOptions::default())
    }
    fn reshape_dynamic_with_options(
        &mut self,
        input: MLOperand,
        new_shape: MLOperand,
        options: MLOperatorOptions,
    ) -> MLOperand;
    fn expand_dynamic(&mut self, input: MLOperand, new_shape: MLOperand) -> MLOperand {
        self.expand_dynamic_with_options(input, new_shape, MLOperatorOptions::default())
    }
    fn expand_dynamic_with_options(
        &mut self,
        input: MLOperand,
        new_shape: MLOperand,
        options: MLOperatorOptions,
    ) -> MLOperand;
    fn slice_dynamic(
        &mut self,
        input: MLOperand,
        starts: MLOperand,
        sizes: MLOperand,
    ) -> MLOperand {
        self.slice_dynamic_with_options(input, starts, sizes, MLSliceDynamicOptions::default())
    }
    fn slice_dynamic_with_options(
        &mut self,
        input: MLOperand,
        starts: MLOperand,
        sizes: MLOperand,
        options: MLSliceDynamicOptions,
    ) -> MLOperand;
    fn pad_dynamic(
        &mut self,
        input: MLOperand,
        beginning_padding: MLOperand,
        ending_padding: MLOperand,
    ) -> MLOperand {
        self.pad_dynamic_with_options(
            input,
            beginning_padding,
            ending_padding,
            MLOperatorOptions::default(),
        )
    }
    fn pad_dynamic_with_options(
        &mut self,
        input: MLOperand,
        beginning_padding: MLOperand,
        ending_padding: MLOperand,
        options: MLOperatorOptions,
    ) -> MLOperand;

    fn split_dynamic(&mut self, input: MLOperand, splits: MLOperand) -> Vec<MLOperand> {
        self.split_dynamic_with_options(input, splits, MLSplitOptions::default())
    }
    fn split_dynamic_with_options(
        &mut self,
        input: MLOperand,
        splits: MLOperand,
        options: MLSplitOptions,
    ) -> Vec<MLOperand>;
    fn resample_2d_dynamic(&mut self, input: MLOperand) -> MLOperand {
        self.resample_2d_dynamic_with_options(input, MLResample2dDynamicOptions::default())
    }
    fn resample_2d_dynamic_with_options(
        &mut self,
        input: MLOperand,
        options: MLResample2dDynamicOptions,
    ) -> MLOperand;
    fn tile_dynamic(&mut self, input: MLOperand, repetitions: MLOperand) -> MLOperand {
        self.tile_dynamic_with_options(input, repetitions, MLOperatorOptions::default())
    }
    fn tile_dynamic_with_options(
        &mut self,
        input: MLOperand,
        repetitions: MLOperand,
        options: MLOperatorOptions,
    ) -> MLOperand;
}

//impl DynamicShapeBuilder for
