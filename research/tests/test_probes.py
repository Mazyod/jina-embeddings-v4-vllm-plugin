from jinav4_vllm.common.probes import TEXT_PROBES, IMAGE_PROBES, build_text_prompt, build_image_prompt

def test_text_probes_cover_required_cases():
    kinds = {p.kind for p in TEXT_PROBES}
    assert {"query", "passage"} <= kinds
    langs = {p.lang for p in TEXT_PROBES}
    assert len(langs) >= 2                      # multilingual coverage
    assert any(len(p.text) > 200 for p in TEXT_PROBES)   # a long one

def test_build_text_prompt_prefixes():
    assert build_text_prompt("hello", "query") == "Query: hello"
    assert build_text_prompt("world", "passage") == "Passage: world"

def test_build_image_prompt_is_exact_template():
    assert build_image_prompt() == (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Describe the image.<|im_end|>\n"
    )

def test_image_probes_exist_and_have_paths():
    assert len(IMAGE_PROBES) >= 2
    for p in IMAGE_PROBES:
        assert p.path.endswith((".png", ".jpg", ".jpeg"))
