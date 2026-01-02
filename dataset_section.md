## Dataset Structure (E-Commerce Chatbot – Canonical SFT Format)

This experiment uses a **canonicalized instruction–response dataset** derived from the public  
**`bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset`**.

All examples are normalized into a consistent JSONL format with **integer IDs**, ensuring:

- Deterministic shuffling and splits  
- Clean experiment reproducibility  
- Easy comparison across fine-tuning configurations  

The dataset represents **realistic retail and customer-support conversations**, suitable for supervised fine-tuning (SFT).

---

### 1) Training Examples (`train.final.jsonl`)

Each line is one **instruction–response pair** with a unique ID.

Each JSON object contains:

- `example_id` (integer)
- `instruction` (string) — customer question or request
- `response` (string) — ideal assistant reply

```jsonl
{"example_id": 101, "instruction": "Where is my order?", "response": "You can track your order by visiting the Orders section of your account. If you provide your order number, I can help check its status."}
{"example_id": 102, "instruction": "Can I return an item after 30 days?", "response": "Our standard return window is 30 days. However, some items may qualify for extended returns depending on the seller and condition."}
