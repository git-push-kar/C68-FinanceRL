"""Actual selected Nifty LoRA inference. No stored-probability fallback."""
import hashlib
import time
from pathlib import Path
import numpy as np
from finance.paper_engine import panel


def predict(service):
    if service._stub_backend:
        raise RuntimeError('Paper trading requires the real selected model; stubs are disabled')
    import torch
    import torch.nn.functional as F
    started = time.perf_counter()
    meta = panel().manifest
    if not getattr(service, '_finance_verified_base', False):
        base_weights = Path(service.base_dir)/'model.safetensors'
        with base_weights.open('rb') as source:
            actual_base_hash = hashlib.file_digest(source, 'sha256').hexdigest()
        if actual_base_hash != meta['base_weight_sha256']:
            raise RuntimeError('Pristine base weights do not match the pinned research snapshot')
        service._finance_verified_base = True
    path = Path(service._lora_dir('finance_policy'))/'adapter_model.safetensors'
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['adapter_sha256']:
        raise RuntimeError('Selected trading adapter integrity failure')
    with service.lease('finance_policy') as lease:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = service.residency.model
        base = model.get_base_model()
        lm = base.language_model
        tok = service._ensure_tokenizer()
        ids = tok.apply_chat_template([{'role': 'user', 'content': meta['prompt']}],
                                     tokenize=True, add_generation_prompt=True, return_tensors='pt')
        labels = [tok.encode(x, add_special_tokens=False) for x in 'ABCD']
        if any(len(x) != 1 for x in labels):
            raise ValueError('Policy action tokens changed')
        labels = [x[0] for x in labels]
        with torch.inference_mode():
            embeddings = lm.get_input_embeddings()(ids.to(lm.get_input_embeddings().weight.device))
            hidden = lm.model(inputs_embeds=embeddings, attention_mask=torch.ones_like(ids).to(embeddings.device),
                              use_cache=False, return_dict=True).last_hidden_state[:, -1].float()
            head = lm.lm_head
            logits = F.linear(hidden, head.get_base_layer().weight[labels].float())
            name = 'finance_policy'
            a, b = head.lora_A[name], head.lora_B[name]
            z = a(head.lora_dropout[name](hidden.to(a.weight.dtype)))
            logits += F.linear(z, b.weight[labels], None if b.bias is None else b.bias[labels])*head.scaling[name]
            probabilities = (.95*torch.softmax(logits/2., -1)+.05/4)[0].double().cpu().numpy()
        probabilities /= probabilities.sum()
        expected = np.asarray(meta['recorded_probabilities']['seed42_64'], dtype=float)
        expected /= expected.sum()
        error = float(np.max(np.abs(probabilities-expected)))
        if error > 2e-6:
            raise RuntimeError(f'Trading policy differs from the selected research artifact ({error:.3g})')
        return probabilities, {'mode': 'real_model', 'adapter': meta['adapter'],
                               'adapter_sha256': meta['adapter_sha256'], 'input_tokens': ids.shape[1],
                               'base_weight_sha256': meta['base_weight_sha256'],
                               'max_probability_error': error, 'elapsed_seconds': round(time.perf_counter()-started, 3),
                               'loaded': lease['swap']['loaded']}
