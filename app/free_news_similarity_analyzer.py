from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


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
    main_sentences = _to_sentences(main_news.get("content") or "")
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
            "category": dto.get("category", "free"),
            "mainNewsDto": {**main_news, "phrases": credible_main},
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
        credible_article = extract_credible_phrases(article_sentences_list[i])
        similarity = _similarity_from_embeddings(main_embeddings, sliced_embeddings[i])
        comparison_results.append({
            "newsWithSimilarityDto": {
                "similarity": similarity,
                "newsDto": {
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                    "date": article.get("date", ""),
                    "phrases": credible_article,
                    "content": article.get("content", "")

                }
            }
        })

    comparison_results.sort(key=lambda x: x["newsWithSimilarityDto"]["similarity"], reverse=True)

    return {
        "category": dto.get("category", "NORMAL"),
        "mainNewsDto": {**main_news, "phrases": credible_main},
        "newsComparisonDtos": comparison_results
    }
