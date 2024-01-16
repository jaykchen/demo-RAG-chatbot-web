# Use saved embeddings in a web chatbot

If you want to build locally, make sure you have installed Rust and added the `wasm32-wasi` target.

```
cargo build --target wasm32-wasi --release
```

Settings when you deploy the flow function on flows.network.
Click on "Advanced" when you import the GitHub repo to flows.network to enter the setting names and values below.

| Name             | Value                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------- |
| llm_endpoint | https://e238-216-9-110-13.ngrok-free.app/v1/ |
| collection_name | The collection name you used to store the embeddings, e.g., `my_kb` |
| system_prompt | Please answer the question based on the context provided. CONTEXT:  |
| post_prompt | Only answer questions in the context. Don't provide any information unrelated to the subject. |
| error_mesg | Sorry, an error has occurred or your question is not related to the subject. Please try again later. |
| no_answer_mesg | Sorry, I cannot answer questions that are not related to the subject. |
| RUST_LOG | debug  |

After a successful deployment, you will receive a webhook URL in the Webhook Endpoint section. Copy and paste this URL after 'https://flows-chat-ui.vercel.app/?chat_url=' in your browser's address bar. This will allow you to start a chat with your RAG chatbot.

```
https://flows-chat-ui.vercel.app/?chat_url=https://code.flows.network/webhook/a-z
```





