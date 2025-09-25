from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_ = model.encode(["프리로딩 테스트"], convert_to_tensor=True)

# 한국어 NER 파이프라인 (Leo97/KoELECTRA-small-v3-modu-ner)
ner_model_name = "Leo97/KoELECTRA-small-v3-modu-ner"
tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

ner_pipeline = pipeline(
    "token-classification",
    model=ner_model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# 본문 정리: 특정 패턴 포함 문장 삭제
def _clean_content(text: str):
    if not isinstance(text, str):
        return ""

    text = text.replace("\u200b", "").strip()
    if not text:
        return ""

    # 문장 단위로 분리
    sentences = sent_tokenize(text)

    # 제거할 패턴 정의
    patterns = [
        r"신문구독신청",
        r"로그인",
        r"회원가입",
        r"공유",
        r"카테고리",
        r"기자",
        r"Update\d{4}\.\d{2}\.\d{2}.*",
        r"http[s]?://\S+",
        r"(페이스북|트위터|구글플러스|싸이월드|라인|네이버블로그|네이버밴드|네이트온쪽지|구글북마크|스크랩하기|프린트하기|이메일보내기|글자확대|글자축소).*",
        r"기자\s?\S+\s?기자의 다른 기사 보기",
        r"기자\s?\S+",
        r"기사 수정",
        r"많이 본 뉴스",
        r"퍼가",
        r"All rights reserved"
    ]

    cleaned_sentences = []
    for sent in sentences:
        if not any(re.search(pat, sent) for pat in patterns):
            cleaned_sentences.append(sent)

    # 문장들을 다시 합침
    cleaned_text = "\n".join(cleaned_sentences).strip()
    return cleaned_text


# 빈값/공백/None 방어 함수
def _to_sentences(text: str):
    if not isinstance(text, str):
        return []
    
    cleaned = text.replace("\u200b", "").strip()
    if not cleaned:
        return []
    
    return [s.strip() for s in sent_tokenize(cleaned) if s.strip()]

# NER
def extract_credible_phrases(sentences):
    credible_phrases = []
    keywords = ["연구", "보고서", "발표", "통계", "조사", "전문가", "기관", "정부", "장관", "교수"]

    for sent in sentences:
        entities = ner_pipeline(sent)
        has_org_or_person = any(e['entity_group'] in ["ORG", "PER"] for e in entities)
        has_number = any(e['entity_group'] == "NUM" or e['word'].isdigit() for e in entities)
        has_keyword = any(k in sent for k in keywords)

        if has_org_or_person or has_number or has_keyword:
            credible_phrases.append(sent)

    return credible_phrases

# 코사인 유사도 계산
def _similarity_from_embeddings(main_embeddings, article_embeddings):
    if getattr(article_embeddings, "shape", [0])[0] == 0 or getattr(main_embeddings, "shape", [0])[0] == 0:
        return 0.0

    sim_matrix = util.cos_sim(main_embeddings, article_embeddings)
    if sim_matrix.numel() == 0:
        return 0.0

    return round(sim_matrix.max().item(), 4)

# 메인 함수
def free_analyze_and_summarize(dto, threshold=0.5):
    main_news = dto.get("mainNewsDto", {})
    cleaned_main_content = _clean_content(main_news.get("content") or "")
    main_sentences = _to_sentences(cleaned_main_content)
    credible_main = extract_credible_phrases(main_sentences)

    if not main_sentences:
        # 본문 없을 때 모든 유사도 0
        comparison_results = [
            {
                "newsWithSimilarityDto": {
                    "similarity": 0.0,
                    "newsDto": {
                        "url": article.get("url", ""),
                        "title": article.get("title", ""),
                        "date": article.get("date", ""),
                        "phrases": [],
                        "content": article.get("content", ""),
                    }
                }
            }
            for article in dto.get("newsDtos", [])
        ]
        comparison_results.sort(key=lambda x: x["newsWithSimilarityDto"]["similarity"], reverse=True)
        return {
            "category": dto.get("category", "NORMAL"),
            "mainNewsDto": {**main_news, "phrases": credible_main, "content": cleaned_main_content},
            "newsComparisonDtos": comparison_results
        }

    main_embeddings = model.encode(main_sentences, convert_to_tensor=True)

    article_sentences_list = [_to_sentences(article.get("content") or "") for article in dto.get("newsDtos", [])]
    flattened_sentences = [s for sentences in article_sentences_list for s in sentences]
    all_embeddings = model.encode(flattened_sentences, convert_to_tensor=True)

    sliced_embeddings = []
    idx = 0
    for sentences in article_sentences_list:
        n = len(sentences)
        sliced_embeddings.append(all_embeddings[idx:idx+n])
        idx += n

    comparison_results = []

    for i, article in enumerate(dto.get("newsDtos", [])):
        cleaned_article_content = _clean_content(article.get("content") or "")
        article_sentences = _to_sentences(cleaned_article_content)

        credible_article = extract_credible_phrases(article_sentences)
        article_embeddings = model.encode(article_sentences, convert_to_tensor=True)
        similarity = _similarity_from_embeddings(main_embeddings, article_embeddings)

        comparison_results.append({
            "newsWithSimilarityDto": {
                "similarity": similarity,
                "newsDto": {
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                    "date": article.get("date", ""),
                    "content": cleaned_article_content,
                    "phrases": credible_article

                }
            }
        })

    comparison_results.sort(key=lambda x: x["newsWithSimilarityDto"]["similarity"], reverse=True)

    return {
        "category": dto.get("category"),
        "plan": dto.get("plan", ""),
        "mainNewsDto": {**main_news, "phrases": credible_main, "content": cleaned_main_content},
        "newsComparisonDtos": comparison_results
    }
