import torch
import torch.nn.functional as F

def retrieve_context(
    dpr_ctx_tokenizer,
    dpr_ctx_encoder, 
    dpr_q_tokenizer,
    dpr_q_encoder, 
    prompt,
):

    with torch.no_grad():
        c_inputs = dpr_ctx_tokenizer(passages, return_tensors="pt", padding=True, truncation=True)
        print(c_inputs)
        c_embed = dpr_ctx_encoder(**c_inputs).pooler_output
        print(c_embed.shape)

    print(c_embed[:, :10])

    # Encode query
    query = "Where is the Eiffel Tower?"
    with torch.no_grad():
        q_inputs = dpr_q_tokenizer(query, return_tensors="pt")
        q_embed = dpr_q_encoder(**q_inputs).pooler_output

    # Search
    c_normalized = F.normalize(c_embed)
    q_normalized = F.normalize(q_embed)

    search = c_normalized @ torch.transpose(q_normalized, 0, 1)
    vals, indices = torch.sort(torch.squeeze(search), 0, descending = True)
    print(indices)

    # # Display results
    print(f"\n🔍 Query: {query}\n")
    print("Top matching passages:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {passages[idx]} (Score: {vals[i]:.4f})")
        # print(f"tokens: {c_normalized[idx,:10]}")