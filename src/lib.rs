use anyhow;
use flowsnet_platform_sdk::logger;
use llmservice_flows::{ chat::{ chat_history, ChatOptions, ChatRole }, LLMServiceFlows };
use openai_flows::{ embeddings::EmbeddingsInput, OpenAIFlows };
use regex::Regex;
use serde_json::{ from_str, json, Value };
use std::collections::HashMap;
use store_flows::{ get, set, Expire, ExpireKind };
use vector_store_flows::*;
use webhook_flows::{ create_endpoint, request_handler, send_response };
use itertools::Itertools;

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
        let accumulated_chat = last_2_chats(text, &chat_id).await;
        log::info!("The question history is {:?}", accumulated_chat);

        match
            is_relevant(text, "This source material is a technical document on Kubernetes.").await
        {
            true => {
                cs.update(accumulated_chat.clone());

                let rag_content = get_rag_content(text, &cs).await.unwrap_or(String::new());
                user_prompt = format!(
                    "Given the source material retrieved: `{rag_content}`, Here is the question you're to reply now: `{text}`. Please provide a concise answer, stay truthful and factual."
                );
            }
            false => {
                cs.mutate(String::from("You're a question and answer bot.") + &accumulated_chat);

                user_prompt = format!(
                    "Here is the question you're to reply now: `{text}`. Please provide a concise answer."
                );
            }
        }
    }

    let co = ChatOptions {
        // model: ChatModel::GPT4,
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
    let llm_endpoint = std::env::var("llm_endpoint").unwrap_or("".to_string());

    let llm = LLMServiceFlows::new(&llm_endpoint);
    let sys_prompt_1 = format!(
        "You're an assistant bot with expertise in all domains of human knowledge."
    );

    let usr_prompt_1 = format!(
        "You're preparing to answer questions about a specific source material, before ingesting the source material, you need to answer the question based on the knowledge you're trained on, here it is: `{question}`, please provide a concise answer in one paragraph, stay truthful and factual."
    );
    let co = ChatOptions {
        // model: ChatModel::GPT4,
        restart: true,
        system_prompt: Some(&sys_prompt_1),
        token_limit: 2048,
        ..Default::default()
    };

    if let Ok(r) = llm.chat_completion("create-hypo-answer", &usr_prompt_1, &co).await {
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

pub fn parse_questions_from_json(input: &str) -> Vec<String> {
    let mut questions: Vec<String> = Vec::new();

    let parsed_result: Result<Value, serde_json::Error> = from_str(input);

    match parsed_result {
        Ok(parsed) => {
            for (key, value) in parsed.as_object().unwrap().iter() {
                if key.starts_with("question_") {
                    if let Some(question) = value.as_str() {
                        questions.push(question.to_string());
                    } else {
                        log::error!("Value for '{}' key is not a string", key);
                    }
                }
            }
        }
        Err(e) => {
            log::error!("Error parsing JSON: {:?}", e);
            let re = Regex::new(r#""question_\d+":\s*"([^"]*)""#).expect(
                "Failed to compile regex pattern"
            );

            for cap in re.captures_iter(input) {
                if let Some(question) = cap.get(1) {
                    questions.push(question.as_str().to_string());
                }
            }
        }
    }
    questions
}

pub async fn chain_of_chat(
    sys_prompt_1: &str,
    usr_prompt_1: &str,
    chat_id: &str,
    gen_len_1: u32,
    usr_prompt_2: &str,
    gen_len_2: u32,
    error_tag: &str
) -> anyhow::Result<String> {
    let llm_endpoint = std::env::var("llm_endpoint").unwrap_or("".to_string());

    let co_1 = ChatOptions {
        restart: true,
        system_prompt: Some(&sys_prompt_1),
        token_limit: gen_len_1,
        ..Default::default()
    };

    let llm = LLMServiceFlows::new(&llm_endpoint);
    match llm.chat_completion("step_1", &usr_prompt_1, &co_1).await {
        Ok(res_1) => {
            let sys_prompt_2 =
                serde_json::json!([{"role": "system", "content": sys_prompt_1},
    {"role": "user", "content": usr_prompt_1},
    {"role": "assistant", "content": &res_1.choice}]).to_string();

            let co_2 = ChatOptions {
                restart: false,
                system_prompt: Some(&sys_prompt_2),
                token_limit: gen_len_2,
                ..Default::default()
            };
            match llm.chat_completion("step_2", &usr_prompt_2, &co_2).await {
                Ok(res_2) => {
                    if res_2.choice.len() < 10 {
                        log::error!(
                            "{}, LLM generation went sideway: {:?}",
                            error_tag,
                            res_2.choice
                        );
                        return Err(anyhow::anyhow!("LLM generation went sideway"));
                    }
                    return Ok(res_2.choice);
                }
                Err(_e) => log::error!("{}, Step 2 LLM generation error {:?}", error_tag, _e),
            };
        }
        Err(_e) => log::error!("{}, Step 1 LLM generation error {:?}", error_tag, _e),
    }

    Err(anyhow::anyhow!("LLM generation went sideway"))
}

pub async fn get_rag_content(text: &str, cs: &ContentSettings) -> anyhow::Result<String> {
    let raw_found_vec = search_collection(&text, &cs.collection_name).await?;

    let mut raw_found_combined = raw_found_vec.into_iter().collect::<HashMap<u64, String>>();

    // use LLM's existing knowledge to answer the question, use the answer
    // that contains more information to retrieve source material that may missed the first search
    let hypo_answer = create_hypothetical_answer(&text).await?;

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

    log::debug!("Cosine similarity between current question and previous question: {score}");

    score > 0.75
}

pub async fn last_2_chats(current_q: &str, chat_id: &str) -> String {
    let mut accumulated_chat = String::new();
    match chat_history(&chat_id.to_string(), 2) {
        Some(v) => {
            let question_list = v
                .into_iter()
                .filter_map(|m| {
                    let mut tup: (String, String) = ("".to_string(), "".to_string());
                    match m.role {
                        ChatRole::User => {
                            tup.0 = m.content;
                        }
                        ChatRole::Assistant => {
                            tup.1 = m.content;
                        }
                    }

                    if !tup.0.is_empty() || !tup.1.is_empty() {
                        Some(tup)
                    } else {
                        None
                    }
                })
                .collect::<Vec<(String, String)>>();

            for (q, a) in question_list {
                if is_relevant(current_q, &q).await {
                    let one_round_chat = format!("User asked: `{}` \n You answered: `{}` \n", q, a);
                    accumulated_chat.push_str(&one_round_chat);
                }
            }
        }
        None => (),
    }

    accumulated_chat
}
