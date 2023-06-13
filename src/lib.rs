use std::thread;
use std::time::Duration;

use anyhow::bail;
use clap::ValueEnum;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum SemanticApi {
    OpenAi,
    AllMiniLmL6V2,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    data: Vec<Embedding>,
}

#[derive(Debug, Deserialize)]
struct Embedding {
    embedding: Vec<f32>,
    // object: String,
    // index: usize,
}

pub fn openai_vectors(
    sentences: Vec<String>,
    openai_api_key: &str,
) -> anyhow::Result<Vec<Vec<f32>>> {
    use ureq::Error;

    let mut sentences = sentences;
    let mut wait_for = Duration::from_secs(2);
    for _ in 0..100 {
        let result = ureq::post("https://api.openai.com/v1/embeddings")
            .set("Authorization", &format!("Bearer {openai_api_key}"))
            .send_json(ureq::json!({
                "model": "text-embedding-ada-002".to_string(),
                "input": sentences,
            }));
        match result {
            Err(Error::Status(503 | 429, response)) => {
                let response = response.into_string()?;
                eprintln!("Retrying after {:.02?}: {}", wait_for, response);
                thread::sleep(wait_for);
                wait_for *= 2;
            }
            Err(Error::Status(400, response)) => {
                // Most of the time it is due to the OpenAI 8191 max tokens
                let max_length = sentences.iter().map(|s| s.len()).max().unwrap();
                let cut_at = max_length * 80 / 100;
                eprintln!(
                    "Seeing error cutting sentences from max {max_length} to {cut_at}: {}",
                    response.into_string()?,
                );
                sentences = sentences
                    .into_iter()
                    .map(|mut s| {
                        s.truncate(cut_at);
                        s
                    })
                    .collect();
            }
            Err(Error::Status(_, resp)) => {
                bail!(
                    "Cannot query OpenAI due to a {} status code. {}",
                    resp.status(),
                    resp.into_string()?,
                )
            }
            Err(transport) => bail!("Cannot query OpenAI due to {}", transport),
            Ok(response) => {
                let response: OpenAiResponse = response.into_json()?;
                return Ok(response.data.into_iter().map(|d| d.embedding).collect());
            }
        }
    }
    bail!("Cannot query OpenAI, too many retry")
}
