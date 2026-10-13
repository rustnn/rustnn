/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Spans for a [`tracing`](https://docs.rs/tracing) subscriber.
//!
//! A span is declared on the function it measures, in that function's own file:
//!
//! ```ignore
//! #[tracing::instrument(skip_all, level = "debug", fields(bytes = data.len()))]
//! pub fn write_tensor(..) { .. }
//! ```
//!
//! The span opens when the function is entered and closes when it returns. It takes the
//! function's name and belongs to the function's module. `skip_all` keeps the arguments out of
//! the span; `fields(..)` says what to report instead. A fallible function adds `err`, so a
//! failure is reported as an `ERROR` event inside its span — the cache reads are the exception,
//! because a miss arrives as an `Err` too.
//!
//! `INFO` marks the work that happens once per context, graph or file — creating, building,
//! saving, loading, validating. `DEBUG` marks the work that happens on every call: dispatching,
//! reading and writing tensors, the shape checks and the cache lookups.
//!
//! A program has to install a subscriber; without one, entering a span costs a quick check and
//! its fields are never evaluated.
//!
//! Values known only once the work is done — a tensor's id, the final graph sizes, what a cache
//! read found — are emitted as an event inside the span instead of being recorded on it: an
//! attribute's span cannot be reached from the function body, while an event already belongs to
//! the span it is emitted in. An event is reported under the module it is emitted from, so those
//! live next to the code they measure.
//!
//! A message that repeats what the span already carries as a field is not logged again at
//! `DEBUG`; the verbose dumps stay at `TRACE`.
//!
//! What is left here is the field types those spans share.

use std::fmt;

use crate::mlcontext::{MLNamedOperands, MLNamedTensors};

/// The tensors bound to a dispatch, printed for the `inputs` and `outputs` fields as
/// `{lhs: #0 Float32[2, 2], sum: #1 Float32[2, 2]}`.
pub(crate) struct TensorBindings<'a, 'names>(pub(crate) &'a MLNamedTensors<'names>);

impl fmt::Debug for TensorBindings<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        for (index, (name, tensor)) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            // The id is the one the create_tensor event and the read/write spans report.
            write!(
                f,
                "{name}: #{id} {:?}{:?}",
                tensor.data_type(),
                tensor.shape(),
                id = tensor.id
            )?;
        }
        f.write_str("}")
    }
}

/// The output names of a build or save, printed for the `output_names` field as `["sum"]`.
pub(crate) struct Names<'a, 'names>(pub(crate) &'a MLNamedOperands<'names>);

impl fmt::Debug for Names<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.keys()).finish()
    }
}

/// Test helpers for the modules that emit spans and events: a subscriber that records what
/// reaches it — including the span an event was emitted inside — and what it captured.
#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::HashMap;
    use std::fmt;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Level, Metadata, Subscriber};

    /// One event, as a subscriber receives it.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct Reported {
        pub(crate) level: Level,
        pub(crate) target: String,
        pub(crate) fields: Vec<String>,
        /// The span the event was emitted inside, if any.
        pub(crate) within: Option<String>,
    }

    impl Reported {
        /// The `name=value` text a field carried, if the event recorded it.
        pub(crate) fn field(&self, name: &str) -> Option<&str> {
            let wanted = format!("{name}=");
            self.fields
                .iter()
                .find(|field| field.starts_with(&wanted))
                .map(String::as_str)
        }
    }

    #[derive(Clone, Default)]
    pub(crate) struct Recorder {
        events: Arc<Mutex<Vec<Reported>>>,
        /// Each span's name by id, and the ids entered on this thread.
        names: Arc<Mutex<HashMap<u64, &'static str>>>,
        entered: Arc<Mutex<Vec<u64>>>,
        next_id: Arc<AtomicU64>,
    }

    struct FieldValues(Vec<String>);

    impl Visit for FieldValues {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.push(format!("{}={value}", field.name()));
        }

        fn record_u64(&mut self, field: &Field, value: u64) {
            self.0.push(format!("{}={value}", field.name()));
        }

        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.0.push(format!("{}={value:?}", field.name()));
        }
    }

    impl Subscriber for Recorder {
        fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, span: &Attributes<'_>) -> Id {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
            self.names
                .lock()
                .unwrap()
                .insert(id, span.metadata().name());
            Id::from_u64(id)
        }

        fn record(&self, _span: &Id, _values: &Record<'_>) {}
        fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

        fn event(&self, event: &Event<'_>) {
            let mut fields = FieldValues(Vec::new());
            event.record(&mut fields);
            let within = self
                .entered
                .lock()
                .unwrap()
                .last()
                .and_then(|id| self.names.lock().unwrap().get(id).copied())
                .map(str::to_string);
            self.events.lock().unwrap().push(Reported {
                level: *event.metadata().level(),
                target: event.metadata().target().to_string(),
                fields: fields.0,
                within,
            });
        }

        fn enter(&self, span: &Id) {
            self.entered.lock().unwrap().push(span.into_u64());
        }

        fn exit(&self, _span: &Id) {
            self.entered.lock().unwrap().pop();
        }
    }

    /// Run `f` under a recording subscriber and return the events it emitted.
    pub(crate) fn events(f: impl FnOnce()) -> Vec<Reported> {
        let recorder = Recorder::default();
        tracing::subscriber::with_default(recorder.clone(), f);
        let captured = recorder.events.lock().unwrap();
        captured.clone()
    }
}

#[cfg(test)]
mod tests {
    use tracing::Level;

    use super::*;
    use crate::instrumentation::test_support::events;

    /// The dispatch bindings report each tensor's context-scoped id next to its dtype and shape,
    /// so a tensor can be followed from one dispatch into the next.
    #[test]
    fn dispatch_bindings_report_tensor_ids() {
        use crate::mlcontext::{MLNamedTensors, MLTensor, MLTensorDescriptor};
        use crate::operator_enums::MLOperandDataType;

        let tensor = MLTensor {
            id: 7,
            constant: false,
            descriptor: MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]),
        };
        let bindings = MLNamedTensors::from([("lhs", &tensor)]);

        assert_eq!(
            format!("{:?}", TensorBindings(&bindings)),
            "{lhs: #7 Float32[2, 2]}"
        );
    }

    /// `err` on a failing function turns into an `ERROR` event inside its span.
    #[test]
    fn a_failing_function_reports_its_error() {
        let reported = events(|| {
            // Three elements for a `[2, 2]` shape: this has to fail.
            let failed = crate::runtime_checks::validate_shape_data_length("lhs", &[2, 2], 3);
            assert!(failed.is_err());
        });

        assert_eq!(reported.len(), 1, "{reported:?}");
        assert_eq!(reported[0].level, Level::ERROR);
        assert_eq!(reported[0].target, "rustnn::runtime_checks");
        assert_eq!(
            reported[0].within.as_deref(),
            Some("validate_data_length"),
            "the error is reported inside its own span: {reported:?}"
        );
        assert!(
            reported[0].field("error").is_some(),
            "the event carries the error: {reported:?}"
        );
    }
}
