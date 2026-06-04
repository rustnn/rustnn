//! JSON **wire format** for [`crate::operators::Operation`]: `Serialize` / `Deserialize` and
//! interchange helpers. Domain types ([`crate::operators::Operation`], [`crate::operator_options::OperatorOptions`])
//! stay free of this module’s serde details; decoding builds operations via
//! [`Operation::from_json_attributes`](crate::operators::Operation::from_json_attributes).

mod operation;
