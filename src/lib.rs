use webhook_flows::{create_endpoint, request_handler, send_response};
use openai_flows::{
    embeddings::{EmbeddingsInput},
    chat::{ChatModel, ChatOptions, ChatRole, chat_history},
    OpenAIFlows,
};
use store_flows::{get, set};
use vector_store_flows::*;
use flowsnet_platform_sdk::logger;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use serde_json::json;
use serde_json::Value;
use regex::Regex;

static SOFT_CHAR_LIMIT : usize = 30000;

#[derive(Debug)]
struct ContentSettings {
    title: String,
    author: String,
    publisher: String,
    link: String,
    system_prompt: String,
    post_prompt: String,
    error_mesg: String,
    no_answer_mesg: String,
    random_mesg: String,
    collection_name: String,
}

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn on_deploy() {
    create_endpoint().await;
}

#[request_handler]
async fn handler(headers: Vec<(String, String)>, _qry: HashMap<String, Value>, body: Vec<u8>) {
    logger::init();
    let cs = &ContentSettings {
        title: std::env::var("title").unwrap_or("".to_string()),
        author: std::env::var("author").unwrap_or("".to_string()),
        publisher: std::env::var("publisher").unwrap_or("".to_string()),
        link: std::env::var("link").unwrap_or("".to_string()),
        system_prompt: std::env::var("system_prompt").unwrap_or("".to_string()),
        post_prompt: std::env::var("post_prompt").unwrap_or("".to_string()),
        error_mesg: std::env::var("error_mesg").unwrap_or("".to_string()),
        no_answer_mesg: std::env::var("no_answer_mesg").unwrap_or("No answer".to_string()),
        random_mesg: std::env::var("random_mesg").unwrap_or("".to_string()),
        collection_name: std::env::var("collection_name").unwrap_or("".to_string()),
    };
    log::info!("The system prompt is {} lines", cs.system_prompt.lines().count());

    let help = format!("Ask any question about the '{}' book, by {} from {}. Get the book: {}", cs.title, cs.author, cs.publisher, cs.link);

    log::info!("Headers -- {:?}", headers);
    let mut chat_id = "".to_string();
    for header in headers {
        if header.0.eq_ignore_ascii_case("x-conversation-name") {
            chat_id = first_x_chars(&alpha_numeric(&header.1), 48);
            break;
        }
    }

    let body_string = String::from_utf8(body).unwrap_or("".to_string());
    let mut text = body_string.as_str();
    let mut random_q = String::new();
    if text.eq_ignore_ascii_case("/help") {
        reply(&help);
        return;

    } else if text.eq_ignore_ascii_case("") {
        reply(&help);
        set(&chat_id, json!(true), None);
        log::info!("Started converstion for {}", chat_id);
        return;

    } else if text.eq_ignore_ascii_case("/new") {
        reply(&help);
        set(&chat_id.to_string(), json!(true), None);
        log::info!("Restarted converstion for {}", chat_id);
        return;

    } else if text.eq_ignore_ascii_case("/feeling_lucky") {
        let mut qs = Vec::<String>::new();
        let mut current = String::new();
        for line in cs.random_mesg.lines() {
            if line.starts_with("------") {
                qs.push(current.clone());
                current.clear();
            } else {
                current.push_str(line);
                current.push_str("\n");
            }
        }
        let q = qs.choose(&mut rand::thread_rng()).unwrap();

        reply(q);
        set(&chat_id.to_string(), json!(true), None);
        log::info!("Restarted converstion for {}", chat_id);
        random_q = q.to_string();
    }

    if !random_q.is_empty() {
        text = &random_q;
    }

    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);
                
    let restart = match get(&chat_id.to_string()) {
        Some(v) => v.as_bool().unwrap_or_default(),
        None => false,
    };

    let mut question_history = String::new();
    if !restart {
        match chat_history(&chat_id.to_string(), 8) {
            Some(v) => {
                for m in v.into_iter() {
                    if let ChatRole::User = m.role {
                        question_history.push_str(&m.content);
                        question_history.push_str("\n");
                    }
                }
            },
            None => (),
        };
    }
    question_history.push_str(&text);
    log::debug!("The question history is {}", question_history);

    // Compute embedding for the question
    let question_vector = match openai.create_embeddings(EmbeddingsInput::String(question_history)).await {
        Ok(r) => {
            if r.len() < 1 {
                log::error!("OpenAI returned no embedding for the question");
                reply(&cs.no_answer_mesg);
                return;
            }
            r[0].iter().map(|n| *n as f32).collect()
        }
        Err(e) => {
            log::error!("OpenAI returned an error: {}", e);
            reply(&cs.error_mesg);
            return;
        }
    };

    // Search for embeddings from the question
    let p = PointsSearchParams {
        vector: question_vector,
        limit: 5,
    };
    let mut system_prompt_updated = String::from(&cs.system_prompt);
    match search_points(&cs.collection_name, &p).await {
        Ok(sp) => {
            for p in sp.iter() {
                if system_prompt_updated.len() > SOFT_CHAR_LIMIT { break; }
                log::debug!("Received vector score={} and text={}", p.score, first_x_chars(p.payload.as_ref().unwrap().get("text").unwrap().as_str().unwrap(), 256));
                let p_text = p.payload.as_ref().unwrap().get("text").unwrap().as_str().unwrap();
                if p.score > 0.75 && !system_prompt_updated.contains(p_text) {
                    system_prompt_updated.push_str("\n");
                    system_prompt_updated.push_str(p_text);
                }
            }
        }
        Err(e) => {
            log::error!("Vector search returns error: {}", e);
            reply(&cs.error_mesg);
            return;
        }
    }
    // log::debug!("The prompt is {} chars starting with {}", system_prompt_updated.len(), first_x_chars(&system_prompt_updated, 256));
    
    match system_prompt_updated.eq(&cs.system_prompt) {
        true =>  {
            log::info!("No relevant context for question");
            reply(&cs.no_answer_mesg);
            return;
        },
        _ => (),
    }

    let co = ChatOptions {
        // model: ChatModel::GPT4,
        model: ChatModel::GPT35Turbo16K,
        restart: restart,
        system_prompt: Some(&system_prompt_updated),
        post_prompt: Some(&cs.post_prompt),
        ..Default::default()
    };

    match openai.chat_completion(&chat_id.to_string(), &text, &co).await {
        Ok(r) => {
            reply(&r.choice);
        }
        Err(e) => {
            reply(&cs.error_mesg);
            log::error!("OpenAI returns error: {}", e);
            return;
        }
    }

    // A successful restart. The new message will NOT be a restart
    if restart {
        log::info!("Detected restart = true");
        set(&chat_id.to_string(), json!(false), None);
    }
}

fn alpha_numeric (s: &str) -> String {
    let regex_pattern = Regex::new(r"[^a-zA-Z0-9]").unwrap();
    regex_pattern.replace_all(s, "-").to_string()
}

fn first_x_chars (s: &str, x: usize) -> String {
    s.chars().take(x).collect()
}

fn reply (s: &str) {
    send_response(
        200,
        vec![(String::from("content-type"), String::from("text/html"))],
        s.as_bytes().to_vec(),
    );
}
