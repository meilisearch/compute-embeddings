use std::{env, io};

use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

fn main() -> anyhow::Result<()> {
    if let Some(query) = env::args().nth(1) {
        // Set-up sentence embeddings model
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
            .create_model()?;

        let vector = model.encode(&[&query])?.remove(0);
        let writer = io::stdout();
        serde_json::to_writer(writer, &vector)?;
    }

    Ok(())
}
