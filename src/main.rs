use std::io::{self, BufReader, BufWriter};

use indicatif::ProgressIterator;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
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

fn main() -> anyhow::Result<()> {
    let reader = BufReader::new(io::stdin());
    let documents: Vec<Input> = serde_json::from_reader(reader)?;

    // Set-up sentence embeddings model
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
        .create_model()?;

    let mut points = Vec::new();
    for (id, document) in documents.into_iter().enumerate().progress() {
        let vector = compute_embedding(&model, &document)?;
        points.push(Point {
            id,
            vector,
            payload: document,
        });
    }

    let output = Output { points };
    let writer = BufWriter::new(io::stdout());
    serde_json::to_writer_pretty(writer, &output)?;

    Ok(())
}

fn compute_embedding(
    model: &SentenceEmbeddingsModel,
    document: &Input,
) -> anyhow::Result<Vec<f32>> {
    let Input {
        name,
        description,
        brand,
        categories,
        _type,
        object_id,
        ..
    } = document;

    let categories = categories.join(" ");
    let sentence = format!("{name} {description} {brand} {categories} {_type}, {object_id}");
    Ok(model.encode(&[&sentence])?.remove(0))
}
