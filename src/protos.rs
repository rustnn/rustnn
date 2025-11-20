pub mod coreml {
    pub mod core_ml_models {
        include!(concat!(
            env!("OUT_DIR"),
            "/core_ml.specification.core_ml_models.rs"
        ));
    }
    pub mod mil_spec {
        include!(concat!(
            env!("OUT_DIR"),
            "/core_ml.specification.mil_spec.rs"
        ));
    }
    pub mod specification {
        // Bring sibling modules into scope to satisfy generated references.
        pub use super::core_ml_models;
        pub use super::mil_spec;

        include!(concat!(env!("OUT_DIR"), "/core_ml.specification.rs"));
    }

    // Re-export to satisfy super::super::StringVector lookups in generated code.
    pub use specification::StringVector;
}

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
