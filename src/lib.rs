use webhook_flows::{ create_endpoint, request_handler, send_response };
use openai_flows::{ embeddings::{ EmbeddingsInput }, OpenAIFlows };
use llmservice_flows::{ chat::{ ChatOptions, ChatRole, chat_history }, LLMServiceFlows };
use store_flows::{ get, set, Expire, ExpireKind };
use vector_store_flows::*;
use flowsnet_platform_sdk::logger;
use std::collections::HashMap;
use serde_json::{ json, Value, from_str };
use regex::Regex;

static SOFT_CHAR_LIMIT: usize = 512;

#[derive(Debug)]
struct ContentSettings {
    system_prompt: String,
    post_prompt: String,
    error_mesg: String,
    no_answer_mesg: String,
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
    let llm_endpoint = std::env::var("llm_endpoint").unwrap_or("".to_string());
    let cs = &(ContentSettings {
        system_prompt: std::env::var("system_prompt").unwrap_or("".to_string()),
        post_prompt: std::env::var("post_prompt").unwrap_or("".to_string()),
        error_mesg: std::env::var("error_mesg").unwrap_or("".to_string()),
        no_answer_mesg: std::env::var("no_answer_mesg").unwrap_or("No answer".to_string()),
        collection_name: std::env::var("collection_name").unwrap_or("".to_string()),
    });
    log::info!("The system prompt is {} lines", cs.system_prompt.lines().count());

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
    if text.eq_ignore_ascii_case("/new") {
        set(&chat_id.to_string(), json!(true), None);
        log::info!("Restarted converstion for {}", chat_id);
        return;
    }

    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);
    let mut llm = LLMServiceFlows::new(&llm_endpoint);

    let restart = match get(&chat_id.to_string()) {
        Some(v) => v.as_bool().unwrap_or_default(),
        None => false,
    };

    let mut question_history = String::new();
    if !restart {
        match chat_history(&chat_id.to_string(), 8) {
            Some(v) => {
                let question_list = v
                    .into_iter()
                    .filter_map(|m| {
                        if let ChatRole::User = m.role { Some(m.content) } else { None }
                    })
                    .collect::<Vec<String>>();
                question_history = chat_history_smart(question_list, restart).await.join("\n");
            }
            None => (),
        };
    }
    log::debug!("The question history is {}", question_history);

    let question_vector = match
        openai.create_embeddings(EmbeddingsInput::String(question_history)).await
    {
        Ok(r) => {
            if r.len() < 1 {
                log::error!("OpenAI returned no embedding for the question");
                reply(&cs.no_answer_mesg);
                return;
            }
            r[0]
                .iter()
                .map(|n| *n as f32)
                .collect()
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
                if system_prompt_updated.len() > SOFT_CHAR_LIMIT {
                    break;
                }
                log::debug!(
                    "Received vector score={} and text={}",
                    p.score,
                    first_x_chars(
                        p.payload.as_ref().unwrap().get("text").unwrap().as_str().unwrap(),
                        256
                    )
                );
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
        true => {
            log::info!("No relevant context for question");
            reply(&cs.no_answer_mesg);
            return;
        }
        _ => (),
    }

    // use LLM's existing knowledge to answer the question, use the answer
    // that contains more information to retrieve source material that may missed the first search
    let hypo_answer = create_hypothetical_answer(&text).await;

    // use the additional source material found to enrich the context for answer generation
    let _ = search_collection(&hypo_answer, &cs.collection_name, &mut system_prompt_updated).await;

    let co = ChatOptions {
        // model: ChatModel::GPT4,
        restart: restart,
        system_prompt: Some(&system_prompt_updated),
        post_prompt: Some(&cs.post_prompt),
        token_limit: 2048,
        ..Default::default()
    };

    match llm.chat_completion(&chat_id.to_string(), &text, &co).await {
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

pub async fn create_hypothetical_answer(question: &str) -> String {
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

    match llm.chat_completion("create-hypo-answer", &usr_prompt_1, &co).await {
        Ok(r) => r.choice,

        Err(_e) => "".to_owned(),
    }
}

pub async fn search_collection(
    question: &str,
    collection_name: &str,
    system_prompt_updated: &mut String
) {
    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);

    let question_vector = match
        openai.create_embeddings(EmbeddingsInput::String(question.to_string())).await
    {
        Ok(r) => {
            if r.len() < 1 {
                log::error!("OpenAI returned no embedding for the question");
                return;
            }
            r[0]
                .iter()
                .map(|n| *n as f32)
                .collect()
        }
        Err(e) => {
            log::error!("OpenAI returned an error: {}", e);
            return;
        }
    };

    let p = PointsSearchParams {
        vector: question_vector,
        limit: 5,
    };
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
                if p.score > 0.75 && !system_prompt_updated.contains(p_text) {
                    system_prompt_updated.push_str("\n");
                    system_prompt_updated.push_str(p_text);
                }
            }
        }
        Err(e) => {
            log::error!("Vector search returns error: {}", e);
            return;
        }
    }
}

pub async fn chat_history_smart(question_list: Vec<String>, restart: bool) -> Vec<String> {
    // let mut chat_history: Vec<String> = if restart {
    //     vec![current_q.to_string()]
    // } else {
    //     get("chat_history")
    //         .and_then(|v|
    //             v.as_array().map(|arr| {
    //                 arr.iter()
    //                     .filter_map(|val| val.as_str().map(String::from))
    //                     .collect()
    //             })
    //         )
    //         .unwrap_or_else(Vec::new)
    // };

    // if !restart {
    //     chat_history.push(current_q.to_string());
    //     if chat_history.len() > 8 {
    //         chat_history.remove(0);
    //     }
    // }

    if restart {
        return vec![question_list.last().unwrap().to_string()];
    }
    let llm_endpoint = std::env::var("llm_endpoint").unwrap_or("".to_string());

    let question_list_str = question_list
        .iter()
        .enumerate()
        .map(|(i, q)| format!("{}: {}", i + 1, q))
        .collect::<Vec<String>>()
        .join("\n");

    let llm = LLMServiceFlows::new(&llm_endpoint);
    let sys_prompt_1 = format!("You're an assistant bot with strong logical thinking.");

    let usr_prompt_1 = format!(
        r#"You are an AI bot that assists a simpler bot, which is tasked with answering questions based on given source material. Your primary duty is to review the history of questions asked, retain the most recent question regardless of its content, and determine the relevance of preceding questions to the source material. Follow these steps:
1. Always retain the most recent question asked, as it is the current priority for response or action.
2. Evaluate all preceding questions to determine if they are directly relevant to the source material.
3. Retain any preceding questions that are directly relevant to the source material and likely to provide context or additional information needed to answer the most recent question.
4. Remove any preceding questions that are meta-questions, unrelated to the source material, or less than 50% likely to be relevant.
5. Maintain the order of the retained questions, ensuring the most recent question is listed last.
When analyzing the provided list of questions, produce a JSON object that includes the retained questions, formatted according to RFC8259 standards. The JSON object should be correctly structured, with proper escaping of special characters, and contain no additional content or formatting.
Here's how to proceed with the list of questions:
1. Include the most recent question at the end of the JSON object, regardless of its nature.
2. Include only relevant preceding questions about the source material in the JSON object before the most recent question.
3. If no preceding questions are relevant, the JSON object should only contain the most recent question.The list of questions is as follows: 
{question_list_str}
Reply with a JSON object in the following format:
{{
\"question_1\": \"original question 1\",
\"question_2\": \"original question 2\",
...
}}
Ensure the JSON is properly formatted and ready to be parsed by a JSON parser. Do not add any additional content or formatting outside of what is specified."#
    );
    let co = ChatOptions {
        // model: ChatModel::GPT4,
        restart: true,
        system_prompt: Some(&sys_prompt_1),
        token_limit: 1024,
        ..Default::default()
    };

    match llm.chat_completion("filter-question-list", &usr_prompt_1, &co).await {
        Ok(r) => {
            let questions = parse_questions_from_json(&r.choice);

            set(
                "chat_history",
                serde_json::json!(questions),
                Some(Expire {
                    kind: ExpireKind::Ex,
                    value: 120,
                })
            );
            return questions;
        }
        Err(_e) => {
            log::error!("LLM returns error");

            vec![]
        }
    }
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
