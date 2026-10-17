"""Shared SFT serialization; no model, optimizer or corpus writes.

Headers/content are tokenized as separate segments on purpose. Training and chat
inference MUST call these helpers so BPE cannot merge across loss-mask boundaries.
No vocabulary entries are added; only existing EOS id 1 ends assistant turns.
"""
from __future__ import annotations

TEMPLATE_VERSION = 'english_plain_roles_v1'
EOS_ID = 1
PAD_ID = 0
SPECIAL_TEXT = ('<|pad|>', '<|endoftext|>', '<|unk|>', '<|beginoftext|>')
HEADERS = {'system': 'System:\n', 'user': 'User:\n', 'assistant': 'Assistant:\n'}
SEPARATOR = '\n\n'


def validate_messages(messages, *, complete=True):
    if not isinstance(messages, list) or not messages:
        raise ValueError('Conversation must be a nonempty message list')
    expected = 'user'
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {'role', 'content'}:
            raise ValueError('Messages require exactly role/content')
        role, content = message['role'], message['content']
        if role not in HEADERS or not isinstance(content, str) or not content.strip():
            raise ValueError('Invalid or empty role/content')
        if any(marker in content for marker in SPECIAL_TEXT):
            raise ValueError('Reserved token text inside message content')
        if role == 'system':
            if index != 0:
                raise ValueError('System message must be first')
            continue
        if role != expected:
            raise ValueError('User and assistant roles must alternate')
        expected = 'assistant' if role == 'user' else 'user'
    if messages[-1]['role'] != ('assistant' if complete else 'user'):
        raise ValueError('Complete conversation must end with assistant; prompt with user')


def _encode(messages, tokenizer, *, prompt=False):
    validate_messages(messages, complete=not prompt)
    ids, mask, text_parts = [], [], []

    def append(text, supervised):
        tokens = tokenizer.encode(text, add_special_tokens=False).ids
        if any(token in (0, 1, 2, 3) for token in tokens):
            raise ValueError('Unexpected special/unknown token in a text segment')
        ids.extend(tokens)
        mask.extend([int(supervised)]*len(tokens))
        text_parts.append(text)

    for index, message in enumerate(messages):
        role = message['role']
        append(HEADERS[role], False)
        append(message['content'], role == 'assistant')
        if role == 'assistant':
            ids.append(EOS_ID)
            mask.append(1)
            text_parts.append(SPECIAL_TEXT[1])
        if index < len(messages)-1 or prompt:
            append(SEPARATOR, False)
    if prompt:
        append(HEADERS['assistant'], False)
    return {'input_ids': ids, 'assistant_mask': mask,
            'assistant_targets': sum(mask[1:]), 'text': ''.join(text_parts)}


def encode_conversation(messages, tokenizer):
    return _encode(messages, tokenizer)


def encode_prompt(messages, tokenizer):
    return _encode(messages, tokenizer, prompt=True)['input_ids']
