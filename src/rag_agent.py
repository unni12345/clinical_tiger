# src/rag_agent.py
import logging
from tools import (
    classify_intents,
    generate_answer,
    generate_citations,
    generate_verified_answer,
    retrieve_image_path,
    retrieve_relevant_chunks,
    plan_verification_questions,
    execute_verifications,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:
    """
    Basic agent to classify, retrieve context, generate citations, and produce an answer.
    """
    def __init__(self, top_k_per_intent: int = 3):
        self.top_k_per_intent = top_k_per_intent

    def run(self, user_query: str) -> tuple:
        logger.info("Agent.run() - Starting classification for query.")
        classification = classify_intents.invoke({"query": user_query})
        intents = classification.get("intents", [])
        drug_source = classification.get("drug_source", "unknown")
        logger.info("Classified intents: %s", intents)
        logger.info("Identified drug source: %s", drug_source)

        if drug_source == "other":
            logger.info("Drug source 'other' detected, generating answer without context.")
            final_answer = generate_answer.invoke({
                "user_query": user_query,
                "context": "None",
                "references": "None"
            })
            return final_answer, None

        logger.info("Retrieving context chunks (allowed_sources=%s).", drug_source)
        relevant_chunks = retrieve_relevant_chunks.invoke({
            "query": user_query,
            "intents": intents,
            "top_k_per_intent": self.top_k_per_intent,
            "allowed_sources": drug_source
        })

        logger.info("Generating citations from retrieved chunks.")
        context, references = generate_citations.invoke({"chunks": relevant_chunks})

        modified_query = f"{user_query}\n(Note: The relevant drug is {drug_source})"
        logger.info("Producing final answer using modified query.")
        final_answer = generate_answer.invoke({
            "user_query": modified_query,
            "context": context,
            "references": references
        })

        logger.info("Agent.run() - Final answer generated.")
        return final_answer, references


class VerifiedAgent:
    """
    Extended agent that runs a chain-of-verification.
    """
    def __init__(self, top_k_per_intent: int = 3):
        self.top_k_per_intent = top_k_per_intent

    def run(self, user_query: str) -> tuple:
        logger.info("VerifiedAgent.run() - Starting classification for query.")
        classification = classify_intents(user_query)
        intents = classification.get("intents", [])
        drug_source = classification.get("drug_source", None)
        logger.info("Classified intents: %s", intents)
        logger.info("Identified drug source: %s", drug_source)

        if drug_source == "other":
            logger.info("Drug source 'other' detected, running direct verification chain.")
            baseline_answer = generate_answer.invoke({
                "user_query": user_query,
                "context": "None",
                "references": "None"
            })
            verif_questions = plan_verification_questions.invoke({
                "query": user_query,
                "baseline": baseline_answer
            })
            verif_answers = execute_verifications.invoke({
                "verification_questions": verif_questions
            })
            final_verified_answer = generate_verified_answer.invoke({
                "query": user_query,
                "baseline": baseline_answer,
                "verif_questions": verif_questions,
                "verif_answers": verif_answers,
                "context": "None",
                "references": "None"
            })
            yes_count = sum(1 for answer in verif_answers if answer.lower() in ["yes", "true", "correct"])
            total = len(verif_answers)
            evaluation_ratio = f"{yes_count}/{total}"
            logger.info("Verification complete (ratio: %s).", evaluation_ratio)
            return final_verified_answer, None, evaluation_ratio

        logger.info("Retrieving context chunks (allowed_sources=%s).", drug_source)
        relevant_chunks = retrieve_relevant_chunks.invoke({
            "query": user_query,
            "intents": intents,
            "top_k_per_intent": self.top_k_per_intent,
            "allowed_sources": drug_source
        })
        context, references = generate_citations.invoke({"chunks": relevant_chunks})
        modified_query = f"{user_query}\n(Note: The relevant drug is {drug_source})"
        baseline_answer = generate_answer.invoke({
            "user_query": modified_query,
            "context": context,
            "references": references
        })

        logger.info("Planning verification questions.")
        verif_questions = plan_verification_questions.invoke({
            "query": user_query,
            "baseline": baseline_answer
        })
        logger.info("Executing verification questions.")
        verif_answers = execute_verifications.invoke({
            "verification_questions": verif_questions
        })
        logger.info("Verification answers: %s", verif_answers)

        final_verified_answer = generate_verified_answer.invoke({
            "query": user_query,
            "baseline": baseline_answer,
            "verif_questions": verif_questions,
            "verif_answers": verif_answers,
            "context": context,
            "references": references
        })
        yes_count = sum(1 for answer in verif_answers if answer.lower() in ["yes", "true", "correct"])
        total = len(verif_answers)
        evaluation_ratio = f"{yes_count}/{total}"
        logger.info("Final verified answer generated (ratio: %s).", evaluation_ratio)

        return final_verified_answer, references, evaluation_ratio


class ImageAgent:
    """
    Agent to retrieve an image path and its caption.
    """
    def __init__(self, top_k: int = 1):
        self.top_k = top_k

    def run(self, query: str) -> tuple:
        logger.info("ImageAgent.run() - Retrieving image for query.")
        image_path, caption = retrieve_image_path.invoke({
            "query": query,
            "top_k": self.top_k
        })
        logger.info("Image retrieved: %s, Caption: %s", image_path, caption)
        return image_path, caption


if __name__ == "__main__":
    query = "What are the side effects of Lipitor in elderly patients?"

    verified_agent = VerifiedAgent()
    final_verified_answer, references, eval_ratio = verified_agent.run(query)
    print("\n✅ Final Verified Answer:\n", final_verified_answer)
    print("\n📚 References:\n", references)
    print("\nEval Ratio:", eval_ratio)

    basic_agent = Agent()
    final_answer, references = basic_agent.run(query)
    print("\n✅ Final Answer:\n", final_answer)
    print("\n📚 References:\n", references)

    image_agent = ImageAgent()
    img_path, caption = image_agent.run(query)
    print("Image Path:", img_path, "Caption:", caption)
