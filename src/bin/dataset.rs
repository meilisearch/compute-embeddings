use clap::{Parser, ValueEnum};
use std::io::{self, BufReader, BufWriter};
use std::time::Instant;

use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use itertools::Itertools;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Serialize, Deserialize)]
struct Input {
    name: String,
    description: String,
    brand: String,
    categories: Vec<String>,
    #[serde(rename = "hierarchicalCategories")]
    hierarchical_categories: Map<String, Value>,
    #[serde(rename = "type")]
    _type: String,
    price: f32,
    price_range: String,
    image: String,
    url: String,
    free_shipping: bool,
    popularity: u32,
    rating: u32,
    #[serde(rename = "objectID")]
    object_id: String,
    #[serde(skip_deserializing, skip_serializing_if = "Option::is_none")]
    _vector: Option<Vec<f32>>,
}

impl Input {
    fn text(&self) -> String {
        let Input {
            name,
            description,
            brand,
            categories,
            _type,
            object_id,
            ..
        } = self;

        let categories = categories.join(" ");
        format!("{name} {description} {brand} {categories} {_type} {object_id}")
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The style of the output documents
    #[arg(long)]
    documents_style: DocumentStyle,

    /// Number of documents processed at the same time.
    #[arg(long, default_value_t = 4)]
    batched_documents: usize,
}

fn main() -> anyhow::Result<()> {
    let Args {
        documents_style,
        batched_documents,
    } = Args::parse();

    let reader = BufReader::new(io::stdin());
    let documents: Vec<Input> = serde_json::from_reader(reader)?;

    let progress_style = ProgressStyle::with_template("{wide_bar} {pos}/{len} {eta}").unwrap();
    let progress_bar = ProgressBar::new(documents.len() as u64).with_style(progress_style);

    // Set-up sentence embeddings model
    let now = Instant::now();
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
        .create_model()?;

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
        let sentences: Vec<_> = chunk.iter().map(|(_, payload)| payload.text()).collect();
        let vectors = model.encode(&sentences)?;
        for entry in chunk.into_iter().zip(vectors) {
            output.push(entry);
        }
    }

    match documents_style {
        DocumentStyle::Meilisearch => {
            let output: Vec<_> = output
                .into_iter()
                .map(|((_, mut payload), vector)| {
                    payload._vector = Some(vector);
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
