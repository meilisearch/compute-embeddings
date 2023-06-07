use std::time::Instant;
use std::{env, io};

use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

fn main() -> anyhow::Result<()> {
    let query = env::args().nth(1).unwrap_or_default();
    let now = Instant::now();
    // Set-up sentence embeddings model
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
        .create_model()?;

    eprintln!("It took {:.02?} to initialize the model.", now.elapsed());

    let vector = model.encode(&[&query])?.remove(0);
    let writer = io::stdout();
    serde_json::to_writer(writer, &vector)?;

    Ok(())
}
