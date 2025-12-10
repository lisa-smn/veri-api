from app.llm.openai_client import OpenAIClient
from app.services.agents.factuality_agent import FactualityAgent

def evaluate_pairs(pairs):
    llm = OpenAIClient(model_name="gpt-4o-mini")
    agent = FactualityAgent(llm)

    total = 0
    avg_score = 0.0

    for article, summary in pairs:
        result = agent.run(article, summary, meta={})
        total += 1
        avg_score += result.score

        # optional: reporte num_errors, error_types
        print("Score:", result.score, "num_errors:", result.details["num_errors"])

    if total > 0:
        avg_score /= total
    print("Durchschnittlicher Factuality-Score:", avg_score)


if __name__ == "__main__":
    pairs = [
        (
            "Lisa wohnt in Berlin und studiert Softwareentwicklung.",
            "Lisa wohnt in Berlin und studiert Softwareentwicklung.",
        ),
        (
            "Lisa wohnt in Berlin und studiert Softwareentwicklung.",
            "Lisa wohnt in Paris und studiert Medizin.",
        ),
        # später: echte Beispiele
    ]
    evaluate_pairs(pairs)
