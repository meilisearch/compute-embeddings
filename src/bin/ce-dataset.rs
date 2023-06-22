use clap::{Parser, ValueEnum};
use std::env::var;
use std::io::{self, BufReader, BufWriter};
use std::time::Instant;

use compute_embeddings::{openai_vectors, SemanticApi};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use itertools::Itertools;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The style of the output documents
    #[arg(long)]
    documents_style: DocumentStyle,

    /// Number of documents processed at the same time.
    #[arg(long, default_value_t = 4)]
    batched_documents: usize,

    /// The API you want to use to compute the embeddings.
    #[arg(long)]
    semantic_api: SemanticApi,

    /// The fields to concatenate in this specific order to generate the embeddings.
    documents_fields: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    let Args {
        documents_style,
        batched_documents,
        semantic_api,
        documents_fields,
    } = Args::parse();

    let reader = BufReader::new(io::stdin());
    let documents: Vec<Input> = serde_json::from_reader(reader)?;

    let progress_style = ProgressStyle::with_template("{wide_bar} {pos}/{len} {eta}").unwrap();
    let progress_bar = ProgressBar::new(documents.len() as u64).with_style(progress_style);

    // Set-up sentence embeddings model
    let now = Instant::now();
    let mut model = None;

    eprintln!("It took {:.02?} to initialize the model.", now.elapsed());

    let mut output = Vec::new();
    for chunk in documents
        .into_iter()
        .enumerate()
        .progress_with(progress_bar)
        .chunks(batched_documents)
        .into_iter()
    {
        let chunk: Vec<_> = chunk.collect();
        let sentences: Vec<_> = chunk
            .iter()
            .map(|(_, payload)| payload.text(&documents_fields))
            .collect();

        let vectors = match semantic_api {
            SemanticApi::OpenAi => {
                let api_key = var("OPENAI_API_KEY").expect("missing OPENAI_API_KEY env variable");
                openai_vectors(sentences, &api_key)?
            }
            SemanticApi::AllMiniLmL6V2 => {
                let model = match model.as_ref() {
                    Some(model) => model,
                    None => {
                        let m = SentenceEmbeddingsBuilder::remote(
                            SentenceEmbeddingsModelType::AllMiniLmL6V2,
                        )
                        .create_model()?;
                        model.get_or_insert(m)
                    }
                };

                model.encode(&sentences)?
            }
        };

        for entry in chunk.into_iter().zip(vectors) {
            output.push(entry);
        }
    }

    match documents_style {
        DocumentStyle::Meilisearch => {
            let output: Vec<_> = output
                .into_iter()
                .map(|((_, mut payload), vector)| {
                    payload._vectors = Some(vector);
                    payload
                })
                .collect();
            let writer = BufWriter::new(io::stdout());
            serde_json::to_writer_pretty(writer, &output)?;
        }
        DocumentStyle::Qdrant => {
            let points = output
                .into_iter()
                .map(|((id, payload), vector)| Point {
                    id,
                    vector,
                    payload,
                })
                .collect();
            let output = Output { points };
            let writer = BufWriter::new(io::stdout());
            serde_json::to_writer_pretty(writer, &output)?;
        }
    }

    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct Input {
    #[serde(flatten)]
    fields: Map<String, Value>,
    #[serde(skip_deserializing, skip_serializing_if = "Option::is_none")]
    _vectors: Option<Vec<f32>>,
}

impl Input {
    fn text<I>(&self, fields: I) -> String
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut internal_buffer = String::new();
        let mut text = String::new();
        for field_name in fields {
            if let Some(value) = self.fields.get(field_name.as_ref()) {
                internal_buffer.clear();
                if let Some(t) = json_to_string(value, &mut internal_buffer) {
                    text.push_str(t);
                    text.push(' ');
                }
            }
        }

        text
    }
}

/// Transform a JSON value into a string that can be indexed.
fn json_to_string<'a>(value: &'a Value, buffer: &'a mut String) -> Option<&'a str> {
    fn inner(value: &Value, output: &mut String) -> bool {
        use std::fmt::Write;
        match value {
            Value::Null | Value::Object(_) => false,
            Value::Bool(boolean) => write!(output, "{}", boolean).is_ok(),
            Value::Number(number) => write!(output, "{}", number).is_ok(),
            Value::String(string) => write!(output, "{}", string).is_ok(),
            Value::Array(array) => {
                let mut count = 0;
                for value in array {
                    if inner(value, output) {
                        output.push_str(" ");
                        count += 1;
                    }
                }
                // check that at least one value was written
                count != 0
            }
        }
    }

    if let Value::String(string) = value {
        Some(string)
    } else if inner(value, buffer) {
        Some(buffer)
    } else {
        None
    }
}

// {
//   "points": [
//     {"id": 1, "vector": [0.05, 0.61, 0.76, 0.74], "payload": {"city": "Berlin" }}
//   ]
// }
#[derive(Debug, Serialize)]
struct Output {
    points: Vec<Point>,
}

#[derive(Debug, Serialize)]
struct Point {
    id: usize,
    vector: Vec<f32>,
    payload: Input,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum DocumentStyle {
    Meilisearch,
    Qdrant,
}
