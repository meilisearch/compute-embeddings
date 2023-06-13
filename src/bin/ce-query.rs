use std::env::var;
use std::time::Instant;
use std::{io};

use clap::Parser;
use compute_embeddings::{openai_vectors, SemanticApi};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The API you want to use to compute the embeddings.
    #[arg(long)]
    semantic_api: SemanticApi,

    /// Generate the embeddings of this query.
    query: String,
}

fn main() -> anyhow::Result<()> {
    let Args {
        semantic_api,
        query,
    } = Args::parse();

    let vector = match semantic_api {
        SemanticApi::OpenAi => {
            let now = Instant::now();
            let api_key = var("OPENAI_API_KEY").expect("missing OPENAI_API_KEY env variable");
            let vector = openai_vectors(vec![query], &api_key)?.remove(0);
            eprintln!("It took {:.02?} to encode the query.", now.elapsed());
            vector
        }
        SemanticApi::AllMiniLmL6V2 => {
            let now = Instant::now();
            // Set-up sentence embeddings model
            let model =
                SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                    .create_model()?;
            eprintln!("It took {:.02?} to initialize the model.", now.elapsed());

            let now = Instant::now();
            let vector = model.encode(&[&query])?.remove(0);
            eprintln!("It took {:.02?} to encode the query.", now.elapsed());
            vector
        }
    };

    let writer = io::stdout();
    serde_json::to_writer(writer, &vector)?;

    Ok(())
}
