# Use saved embeddings in a telegram chat

If you want to build locally, make sure you have installed Rust and added the `wasm32-wasi` target.

```
cargo build --target wasm32-wasi --release
```

Settings when you deploy the flow function on flows.network.
Click on "Advanced" when you import the GitHub repo to flows.network to enter the setting names and values below.

| Name             | Value                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------- |
| collection_name | The collection name you used to store the embeddings, e.g., `my_kb` |
| error_mesg | Sorry, an error has occurred or your question is not related to the Rust programming language. Please try again later. |
| system_prompt | You are an expert in the Rust programming language. Please answer the question based on the context provided. CONTEXT:  |
| RUST_LOG | debug  |

After a successful deployment, you will receive a webhook URL in the Webhook Endpoint section. Copy and paste this URL after 'https://flows-chat-ui.vercel.app/?chat_url=' in your browser's address bar. This will allow you to start a chat with your RAG chatbot.

```
https://flows-chat-ui.vercel.app/?chat_url=https://code.flows.network/webhook/a-z
```




