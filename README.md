# Use saved embeddings in a web chatbot

This variant of the chatbot showcases several techniques to improve its RAG performance.


## Use hypothetical answers to enhance retrieval from the source

The author learned about this technique from other projects. The technique is backed by research in [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496).

Please note that this code uses OpenAI to generate hypothetical answers, no intention to "cheat", but aims to isolate the side effect that such action may have on the main LLM currently engaged in the question/answer task. 


## Mannully keep most recent 3 answers revelant to the current question

To answer questions wtih RAG, the bot uses long context from the source/carrying a heavy chat history, considering that the bot's historical answers have denser information about the user's intention and what has been found in the source. It's more cost-effective to use the answers as part of the context to enrich the bot's knowledge. Because the user may switch topic/diverge to unrelated topics, this code filters out historical answers that are not very relevant to the current question. Again, a hypothetical answer is used to make the filtering more precise.


## Mannully calculate similarities between questions, answers, source material, etc.

It's very efficient to get embeddings for short chunk of texts from an API, and it's also very efficient to 
- do the filtering of historical answers by calculating the cosine distnaces of the embeddings of each answer and the current question
- evaluate whether current question is about the source materials, so as to engage the RAG logic

without making additional round trips to LLM.

## Below are almost the same with the main branch.

Execpt for `LLM_API_KEY` additionally needed for this branch.

If you want to build locally, make sure you have installed Rust and added the `wasm32-wasi` target.

```
cargo build --target wasm32-wasi --release
```

Settings when you deploy the flow function on flows.network.
Click on "Advanced" when you import the GitHub repo to flows.network to enter the setting names and values below.

| Name             | Value                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------- |
| llm_endpoint | https://e238-216-9-110-13.ngrok-free.app/v1/ |
| LLM_API_KEY | what-ever-it-looks-like-for-your-api | 
| collection_name | The collection name you used to store the embeddings, e.g., `my_kb` |
| system_prompt | Please answer the question based on the context provided. CONTEXT:  |
| post_prompt | Only answer questions in the context. Don't provide any information unrelated to the subject. |
| error_mesg | Sorry, an error has occurred or your question is not related to the subject. Please try again later. |
| no_answer_mesg | Sorry, I cannot answer questions that are not related to the subject. |
| RUST_LOG | debug  |

After a successful deployment, you will receive a webhook URL in the Webhook Endpoint section. Copy and paste this URL after `https://flows-chat-ui.vercel.app/?chat_url=` in your browser's address bar. This will allow you to start a chat with your RAG chatbot.

```
https://flows-chat-ui.vercel.app/?chat_url=https://code.flows.network/webhook/a-z
```





