# Verifier receipt

Verifier model family: independent external verifier (not the generator family).

I labeled all 100 records in `inputs.jsonl` directly from:
- `policy.md`
- `output_schema.json`
- `inputs.jsonl`

I did not edit `inputs.jsonl` and did not consult any candidate label file, family manifest, or README siblings beyond what was provided.

Output:
- `judgements.jsonl` — 100 schema-valid judgement lines, one per `packet_id`.

Route summary:
- DEEPFAKE: 42
- BASE: 58

Ambiguity:
- `ambiguous: true` on 13 records where the current request was genuinely underspecified, had a missing/expired/unsupported target, mixed non-authenticity intent with an image, or could be read as either a definition/clarification and an implicit verdict request.

Reason codes used:
- direct_verdict_question
- explicit_reanalysis_request
- implicit_authenticity_request
- implicit_target_request
- explanation_request_on_existing_verdict
- followup_clarification_only
- mixed_request
- ambiguous_target_question
- missing_target_image
- expired_target_image
- unsupported_target_image
- non_authenticity_image_task
- definition_question
- no_target_image

Honest flags:
- Several image-bearing requests are only indirectly authenticities, for example appearance, OCR, captioning, lighting, or "is my hair okay" questions. I routed those to BASE and marked the unclear ones ambiguous.
- Two requests have no image at all but are phrased like authenticity prompts; I routed them to BASE as missing-target and marked them ambiguous.
- One direct deepfake question targets an available image that is marked unsupported, so I kept it BASE and flagged it ambiguous.
- One authenticity followup targets an expired asset, so I kept it BASE and flagged it ambiguous.

Checks I did:
- Confirmed every output line is valid JSON with exactly the five required keys and no extras.
- Confirmed every `packet_id` matches an input `packet_id`.
- Confirmed every `route` is either `BASE` or `DEEPFAKE`.
- Confirmed every `ambiguous` is a boolean.
- Confirmed every `reason_code` and `explanation` is a non-empty string.

I stopped after writing the one judgement per packet and this receipt.
