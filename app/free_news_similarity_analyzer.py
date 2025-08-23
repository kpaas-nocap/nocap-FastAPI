from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_ = model.encode(["프리로딩 테스트"], convert_to_tensor=True)

# 빈값/공백/None 방어 함수
def _to_sentences(text: str):
    if not isinstance(text, str):
        return []
    
    cleaned = text.replace("\u200b", "").strip()
    if not cleaned:
        return []
    
    return [s.strip() for s in sent_tokenize(cleaned) if s.strip()]

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

    if not main_sentences:
        # 본문 없을 때 모든 유사도 0
        comparison_results = [
            {
                "newsWithSimilarityDto": {
                    "similarity": 0.0,
                    "newsDto": {
                        "url": article.get("url", ""),
                        "title": article.get("title", ""),
                        "content": article.get("content", "")
                    }
                }
            }
            for article in dto.get("newsDtos", [])
        ]
        comparison_results.sort(key=lambda x: x["newsWithSimilarityDto"]["similarity"], reverse=True)
        return {
            "category": dto.get("category", "free"),
            "mainNewsDto": main_news,
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
        similarity = _similarity_from_embeddings(main_embeddings, sliced_embeddings[i])
        comparison_results.append({
            "newsWithSimilarityDto": {
                "similarity": similarity,
                "newsDto": {
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                    "content": article.get("content", "")
                }
            }
        })

    comparison_results.sort(key=lambda x: x["newsWithSimilarityDto"]["similarity"], reverse=True)

    return {
        "category": dto.get("category", "free"),
        "mainNewsDto": main_news,
        "newsComparisonDtos": comparison_results
    }
