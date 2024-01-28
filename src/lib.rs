use anyhow;
use flowsnet_platform_sdk::logger;
use itertools::Itertools;
use llmservice_flows::{ chat::{ chat_history, ChatOptions, ChatRole }, LLMServiceFlows };
use openai_flows::{ embeddings::EmbeddingsInput, OpenAIFlows };
use regex::Regex;
use serde_json::{ from_str, json, Value };
use std::collections::HashMap;
use store_flows::{
    get,
    set,
    // Expire, ExpireKind
};
use vector_store_flows::*;
use webhook_flows::{ create_endpoint, request_handler, send_response };

#[derive(Debug, Clone)]
pub struct ContentSettings {
    initial_system_prompt: String,
    pub system_prompt: String,
    post_prompt: String,
    error_mesg: String,
    no_answer_mesg: String,
    collection_name: String,
}

impl ContentSettings {
    pub fn new(
        initial_system_prompt: String,
        system_prompt: String,
        post_prompt: String,
        error_mesg: String,
        no_answer_mesg: String,
        collection_name: String
    ) -> Self {
        Self {
            initial_system_prompt,
            system_prompt,
            post_prompt,
            error_mesg,
            no_answer_mesg,
            collection_name,
        }
    }

    pub fn mutate(&mut self, new_prompt: String) {
        self.system_prompt = new_prompt;
    }

    pub fn update(&mut self, new_content: String) {
        self.system_prompt = self.system_prompt.clone() + &new_content;
    }

    pub fn reset(&mut self) {
        self.system_prompt = self.initial_system_prompt.clone();
    }

    pub fn post_prompt(&self) -> &str {
        &self.post_prompt
    }

    pub fn error_mesg(&self) -> &str {
        &self.error_mesg
    }

    pub fn no_answer_mesg(&self) -> &str {
        &self.no_answer_mesg
    }

    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }
}

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn on_deploy() {
    create_endpoint().await;
}

#[request_handler]
async fn handler(headers: Vec<(String, String)>, _qry: HashMap<String, Value>, body: Vec<u8>) {
    logger::init();
    let llm_endpoint = std::env::var("llm_endpoint").unwrap_or("".to_string());
    let mut cs = ContentSettings {
        initial_system_prompt: std::env::var("system_prompt").unwrap_or("".to_string()),
        system_prompt: std::env::var("system_prompt").unwrap_or("".to_string()),
        post_prompt: std::env::var("post_prompt").unwrap_or("".to_string()),
        error_mesg: std::env::var("error_mesg").unwrap_or("".to_string()),
        no_answer_mesg: std::env::var("no_answer_mesg").unwrap_or("No answer".to_string()),
        collection_name: std::env::var("collection_name").unwrap_or("".to_string()),
    };

    // log::info!("Headers -- {:?}", headers);
    let mut chat_id = "".to_string();
    for header in headers {
        if header.0.eq_ignore_ascii_case("x-conversation-name") {
            chat_id = first_x_chars(&alpha_numeric(&header.1), 48);
            break;
        }
    }

    let body_string = String::from_utf8(body).unwrap_or("".to_string());
    let mut text = body_string.as_str();
    if text.eq_ignore_ascii_case("/new") {
        set(&chat_id.to_string(), json!(true), None);
        log::info!("Restarted converstion for {}", chat_id);
        return;
    }

    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);

    let llm_api_key = std::env::var("LLM_API_KEY").unwrap_or("".to_string());
    let mut llm = LLMServiceFlows::new(&llm_endpoint);
    llm.set_api_key(&llm_api_key);

    let restart = match get(&chat_id.to_string()) {
        Some(v) => v.as_bool().unwrap_or_default(),
        None => false,
    };

    let mut user_prompt = String::new();
    if !restart {
        let hypo_answer = create_hypothetical_answer(&text).await.unwrap_or(String::new());
        let last_3_relevant_answers = last_3_relevant_answers(&hypo_answer, &chat_id).await;
        log::info!("The answer history is {:?}", last_3_relevant_answers);

        match
            is_relevant(text, "This source material is a technical document on Kubernetes.").await
        {
            true => {
                cs.update(last_3_relevant_answers.clone());

                let rag_content = get_rag_content(text, &hypo_answer, &cs).await.unwrap_or(
                    String::new()
                );
                user_prompt = format!(
                    "Given the context: `{rag_content}`, Here is the question you're to reply now: `{text}`. Please provide a concise answer, stay truthful and factual."
                );
            }
            false => {
                cs.mutate(
                    String::from("You're a question and answer bot.") + &last_3_relevant_answers
                );

                user_prompt = format!(
                    "Here is the question you're to reply now: `{text}`. Please provide a concise answer."
                );
            }
        }
    }

    let co = ChatOptions {
        model: Some("mistralai/Mixtral-8x7B-Instruct-v0.1"),
        restart: restart,
        system_prompt: Some(cs.system_prompt.as_str()),
        post_prompt: Some(&cs.post_prompt),
        token_limit: 2048,
        ..Default::default()
    };

    match llm.chat_completion(&chat_id.to_string(), &user_prompt, &co).await {
        Ok(r) => {
            reply(&r.choice);
        }
        Err(e) => {
            reply(&cs.error_mesg);
            log::error!("LLM returns error: {}", e);
            return;
        }
    }

    // A successful restart. The new message will NOT be a restart
    if restart {
        log::info!("Detected restart = true");
        set(&chat_id.to_string(), json!(false), None);
        cs.reset();
    }
}

fn alpha_numeric(s: &str) -> String {
    let regex_pattern = Regex::new(r"[^a-zA-Z0-9]").unwrap();
    regex_pattern.replace_all(s, "-").to_string()
}

fn first_x_chars(s: &str, x: usize) -> String {
    s.chars().take(x).collect()
}

fn reply(s: &str) {
    send_response(
        200,
        vec![(String::from("content-type"), String::from("text/html"))],
        s.as_bytes().to_vec()
    );
}

pub async fn create_hypothetical_answer(question: &str) -> anyhow::Result<String> {
    // let llm_endpoint = std::env::var("llm_endpoint").unwrap_or("".to_string());
    // let llm = LLMServiceFlows::new(&llm_endpoint);

    let openai = OpenAIFlows::new();
    let sys_prompt_1 = format!(
        "You're an assistant bot with expertise in all domains of human knowledge."
    );

    let usr_prompt_1 = format!(
        "You're preparing to answer questions about a specific source material, before ingesting the source material, you need to answer the question based on the knowledge you're trained on, here it is: `{question}`, please provide a concise answer in one paragraph, stay truthful and factual."
    );
    let co = openai_flows::chat::ChatOptions {
        model: openai_flows::chat::ChatModel::GPT4Turbo,
        restart: true,
        system_prompt: Some(&sys_prompt_1),
        max_tokens: Some(128),
        ..Default::default()
    };

    if let Ok(r) = openai.chat_completion("create-hypo-answer", &usr_prompt_1, &co).await {
        return Ok(r.choice);
    }
    Err(anyhow::anyhow!("LLM generation went sideway"))
}

pub async fn search_collection(
    question: &str,
    collection_name: &str
) -> anyhow::Result<Vec<(u64, String)>> {
    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);

    let question_vector = match
        openai.create_embeddings(EmbeddingsInput::String(question.to_string())).await
    {
        Ok(r) => {
            if r.len() < 1 {
                log::error!("LLM returned no embedding for the question");
                return Err(anyhow::anyhow!("LLM returned no embedding for the question"));
            }
            r[0]
                .iter()
                .map(|n| *n as f32)
                .collect()
        }
        Err(_e) => {
            log::error!("LLM returned an error: {}", _e);
            return Err(anyhow::anyhow!("LLM returned no embedding for the question"));
        }
    };

    let p = PointsSearchParams {
        vector: question_vector,
        limit: 5,
    };
    let mut rag_content = Vec::new();

    match search_points(&collection_name, &p).await {
        Ok(sp) => {
            for p in sp.iter() {
                log::debug!(
                    "Received vector score={} and text={}",
                    p.score,
                    first_x_chars(
                        p.payload.as_ref().unwrap().get("text").unwrap().as_str().unwrap(),
                        256
                    )
                );
                let p_text = p.payload.as_ref().unwrap().get("text").unwrap().as_str().unwrap();
                let p_id = match p.id {
                    PointId::Num(i) => i,
                    _ => 0,
                };
                if p.score > 0.75 {
                    rag_content.push((p_id, p_text.to_string()));
                }
            }
        }
        Err(e) => {
            log::error!("Vector search returns error: {}", e);
        }
    }
    Ok(rag_content)
}

pub async fn get_rag_content(
    text: &str,
    hypo_answer: &str,
    cs: &ContentSettings
) -> anyhow::Result<String> {
    let raw_found_vec = search_collection(&text, &cs.collection_name).await?;

    let mut raw_found_combined = raw_found_vec.into_iter().collect::<HashMap<u64, String>>();

    // use the additional source material found to update the context for answer generation
    let found_vec = search_collection(&hypo_answer, &cs.collection_name).await?;

    for (id, text) in found_vec {
        raw_found_combined.insert(id, text);
    }

    let found_combined = raw_found_combined
        .into_iter()
        .map(|(_, v)| v)
        .collect::<Vec<String>>()
        .join("\n");

    Ok(found_combined)
}

pub async fn is_relevant(current_q: &str, previous_q: &str) -> bool {
    use nalgebra::DVector;

    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);

    let embedding_input = EmbeddingsInput::Vec(vec![current_q.to_string(), previous_q.to_string()]);

    let (current_q_vector, previous_q_vector) = match
        openai.create_embeddings(embedding_input).await
    {
        Ok(r) if r.len() >= 2 =>
            r
                .into_iter()
                .map(|v|
                    v
                        .iter()
                        .map(|&n| n as f32)
                        .collect::<Vec<f32>>()
                )
                .take(2)
                .collect_tuple()
                .unwrap_or((Vec::<f32>::new(), Vec::<f32>::new())),
        _ => {
            log::error!("LLM returned an error");
            return false;
        }
    };

    let q1 = DVector::from_vec(current_q_vector);
    let q2 = DVector::from_vec(previous_q_vector);
    let score = q1.dot(&q2);

    let head = current_q.chars().take(100).collect::<String>();
    let tail = previous_q.chars().take(100).collect::<String>();
    log::debug!("similarity: {score} between {head} and {tail}");

    score > 0.75
}

pub async fn last_3_relevant_answers(hypo_answer: &str, chat_id: &str) -> String {
    let mut accumulated_answers = String::new();
    let mut count = 0;
    if let Some(v) = chat_history(&chat_id.to_string(), 8) {
        for m in v.into_iter() {
            match m.role {
                ChatRole::Assistant => {
                    if is_relevant(hypo_answer, &m.content).await {
                        let one_round_anwser = format!(
                            "User asked a question, you answered: `{}` \n",
                            m.content
                        );
                        accumulated_answers.push_str(&one_round_anwser);
                        count += 1;
                    }

                    if count > 2 {
                        break;
                    }
                }
                _ => (),
            }
        }
    }

    accumulated_answers
}
