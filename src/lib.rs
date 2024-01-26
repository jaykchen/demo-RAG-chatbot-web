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

#[derive(Debug, Clone)]
pub struct ContentSettings {
    pub system_prompt: String,
    pub post_prompt: String,
    pub error_mesg: String,
    pub no_answer_mesg: String,
    pub collection_name: String,
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
    let mut llm = LLMServiceFlows::new(&llm_endpoint);

    let restart = match get(&chat_id.to_string()) {
        Some(v) => v.as_bool().unwrap_or_default(),
        None => false,
    };

    let mut no_rag = false;
    let mut question_history = Vec::new();
    if !restart {
        match chat_history(&chat_id.to_string(), 8) {
            Some(v) => {
                let question_list = v
                    .into_iter()
                    .filter_map(|m| {
                        if let ChatRole::User = m.role { Some(m.content) } else { None }
                    })
                    .collect::<Vec<String>>();
                (question_history, no_rag) = chat_history_smart(question_list, restart).await;
            }
            None => (),
        };
    }
    log::info!("The question history is {:?}", question_history);
    let mut system_prompt = String::from("You're an AI assistant bot");
    if !no_rag {
        let mut found_content = process_with_rag(
            question_history.join("\n"),
            text,
            &cs
        ).await.expect("Failed to process with RAG");

        system_prompt = cs.system_prompt.to_string() + &found_content;
    }

    let co = ChatOptions {
        // model: ChatModel::GPT4,
        restart: restart,
        system_prompt: Some(system_prompt.as_str()),
        post_prompt: Some(&cs.post_prompt),
        token_limit: 1024,
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
    let mut found_content = Vec::new();

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
                    found_content.push((p_id, p_text.to_string()));
                }
            }
        }
        Err(e) => {
            log::error!("Vector search returns error: {}", e);
        }
    }
    Ok(found_content)
}

pub async fn chat_history_smart(question_list: Vec<String>, restart: bool) -> (Vec<String>, bool) {
    if restart {
        return (vec![question_list.last().unwrap().to_string()], false);
    }

    let question_list_str = question_list
        .iter()
        .enumerate()
        .map(|(i, q)| format!("{}: {}", i + 1, q))
        .collect::<Vec<String>>()
        .join("\n");

    let sys_prompt_1 = format!("You're an assistant bot with strong logical thinking.");

    let usr_prompt_1 = format!(
        r#"You are reviewing a list of questions. Your task is to evaluate the relevance of each question to the most recent question. You must determine if each question is 'relevant', i.e. your confidence is greater than 80%, or 'irrelevant' otherwise. This relevance assessment is crucial because, in the next step, we will create a JSON object and questions deemed 'irrelevant' will be removed from the JSON, except for the final question in the list which will be included irrespective of its relevance.
Please format your response by listing each question followed by its relevance status. Make sure to explicitly identify any 'irrelevant' questions as these will need to be excluded in the JSON object creation process. Also, be sure to clearly indicate which question is the last in the list.
Here is the list of questions: `{question_list_str}`.
Provide your analysis of the relevance of each question and remember to clearly mark the last question. Keep in mind that the next step involves removing any questions that are not fully relevant, with the exception of the last question.
Please respond with your analysis of the relevance of each question."#
    );

    let usr_prompt_2 = format!(
        r#"Based on the relevance analysis from Step 1, you are tasked with creating a JSON object that includes only the questions identified as 'relevant'. The only exception is the last question in the list, which should be included in the JSON object under the key "question_last" regardless of its relevance. 
Please ensure that you:  
- Include only 'relevant' questions in the JSON object, under sequential keys starting with "question_1".
- Do not include 'irrelevant' or 'partially relevant' questions, except for the last question.
- Clearly mark the last question with the key "question_last", even if it was not deemed 'relevant'.
- Use the exact text from the questions as they appear in the list.
- Make sure the JSON structure is valid, with all keys and string values enclosed in double quotes.
The JSON object should be formatted as follows:   
{{
\"question_1\": \"actual_text_of_first_relevant_question\",
\"question_2\": \"actual_text_of_second_relevant_question\",
...
\"question_last\": \"actual_text_of_the_last_question_from_the_list regardless of its relevance\"
}}
Confirm that the irrelevant questions have been omitted, and provide the correctly formatted JSON object now based on these refined instructions.
"#
    );

    match
        chain_of_chat(
            &sys_prompt_1,
            &usr_prompt_1,
            "chained-questions-filter",
            1024,
            &usr_prompt_2,
            768,
            "question-filter-failed"
        ).await
    {
        Ok(r) => {
            let questions = parse_questions_from_json(&r);

            log::info!("Parsed questions from JSON: {:?}", questions.clone());
            set(
                "chat_history",
                serde_json::json!(questions),
                Some(Expire {
                    kind: ExpireKind::Ex,
                    value: 120,
                })
            );

            if question_list.len() > 1 && questions.len() == 1 {
                return (questions, true);
            } else {
                return (questions, false);
            }
        }
        Err(_e) => {
            log::error!("LLM returns error");

            (vec![], false)
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

pub async fn process_with_rag(
    question_history: String,
    text: &str,
    cs: &ContentSettings
) -> anyhow::Result<String> {
    let raw_input_question = format!(
        "This is the list of questions you were asked: `{question_history}`, here is the question you're to reply now: `{text}`."
    );
    let raw_found_vec = search_collection(&raw_input_question, &cs.collection_name).await?;

    let mut raw_found_combined = raw_found_vec.into_iter().collect::<HashMap<u64, String>>();

    // log::debug!("The prompt is {} chars starting with {}", system_prompt_updated.len(), first_x_chars(&system_prompt_updated, 256));

    // use LLM's existing knowledge to answer the question, use the answer
    // that contains more information to retrieve source material that may missed the first search
    let hypo_answer = create_hypothetical_answer(&text).await?;

    // use the additional source material found to enrich the context for answer generation
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
