# from typing import List

# import torch
# from google import genai
# from jaxtyping import Float
# from torch import Tensor

# from tropt.models.base import EncoderBaseModel, LossTextAccessMixin


# class GeminiEncoderModel(EncoderBaseModel, LossTextAccessMixin):
#     """
#     Google Gemini Encoder model wrapper, with text-query access.
#     https://ai.google.dev/gemini-api/docs/embeddings
#     """

#     def __init__(
#         self, model_name="gemini-embedding-001", d_model: int = 3072, **kwargs
#     ):
#         """
#         Initializes the Gemini Encoder Model wrapper.

#         Args:
#             model_name: The name of the Gemini embedding model to use.
#             d_model: The dimensionality of the embeddings (e.g., 768, 3072).
#         """
#         # os.environ["GOOGLE_API_KEY"] = ...  # required to be set externally

#         self.client = genai.Client()  # reads GOOGLE_API_KEY from the environment
#         self.model_name = model_name
#         self.d_model = d_model  # for gemini-embedding-001: could be 768, 1536, or 3072
#         self.text_to_task_type = {
#             "document": "RETRIEVAL_DOCUMENT",
#             "query": "RETRIEVAL_QUERY",
#         }

#     def __call__(
#         self, texts: List[str], text_type: str = None
#     ) -> Float[Tensor, "n_texts d_model"]:
#         """
#         Generates embeddings for the given texts using the Gemini API.

#         Args:
#             texts: A list of strings to embed.
#             text_type: The type of text (e.g., "document" or "query") to guide the embedding generation.

#         Returns:
#             A tensor containing the generated embeddings.
#         """
#         assert text_type in (
#             None,
#             "document",
#             "query",
#         ), f"Unsupported text_type {text_type}"
#         task_type = self.text_to_task_type.get(text_type, None)

#         result = self.client.models.embed_content(
#             contents=texts,
#             model=self.model_name,
#             config=genai.types.EmbedContentConfig(
#                 task_type=task_type,
#                 output_dimensionality=self.d_model,
#             ),
#         )

#         result = torch.stack(
#             [torch.tensor(emb.values) for emb in result.embeddings], dim=0
#         )  # shape: (n_texts, d_model)

#         return result




from typing import List
import logging

import torch
from google import genai
from jaxtyping import Float
from torch import Tensor
import time 

from tropt.models.base import EncoderBaseModel, LossTextAccessMixin


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)   # change to DEBUG for previews


class GeminiEncoderModel(EncoderBaseModel, LossTextAccessMixin):
    """
    Google Gemini Encoder model wrapper, with text-query access.
    https://ai.google.dev/gemini-api/docs/embeddings
    """

    def __init__(
        self, model_name="gemini-embedding-001", d_model: int = 3072, **kwargs
    ):
        """
        Initializes the Gemini Encoder Model wrapper.

        Args:
            model_name: The name of the Gemini embedding model to use.
            d_model: The dimensionality of the embeddings (e.g., 768, 3072).
        """
        self.client = genai.Client()  # reads GOOGLE_API_KEY from the environment

        self.model_name = model_name
        self.d_model = d_model  # for gemini-embedding-001: could be 768, 1536, or 3072
        self.text_to_task_type = {
            "document": "RETRIEVAL_DOCUMENT",
            "query": "RETRIEVAL_QUERY",
        }

        logger.info(
            "Initialized GeminiEncoderModel(model_name=%s, d_model=%s)",
            self.model_name,
            self.d_model,
        )

    def __call__(
        self, texts: List[str], text_type: str = None
    ) -> Float[Tensor, "n_texts d_model"]:
        """
        Generates embeddings for the given texts using the Gemini API.

        Args:
            texts: A list of strings to embed.
            text_type: The type of text (e.g., "document" or "query") to guide the embedding generation.

        Returns:
            A tensor containing the generated embeddings.
        """
        assert text_type in (None, "document", "query"), f"Unsupported text_type {text_type}"
        task_type = self.text_to_task_type.get(text_type, None)

        # ---- logging: input inspection (no behavior changes) ----
        n = len(texts) if texts is not None else 0
        logger.info(
            "GeminiEncoderModel.__call__: n_texts=%s text_type=%s task_type=%s model=%s out_dim=%s",
            n, text_type, task_type, self.model_name, self.d_model
        )

        if texts is None:
            logger.warning("texts is None")
        else:
            # types / shape hints
            first_type = type(texts[0]).__name__ if n > 0 else None
            nested = (n > 0 and isinstance(texts[0], (list, tuple)))
            logger.info("texts[0] type=%s nested_like=%s", first_type, nested)

            # character length stats (safe, no content)
            if n > 0 and all(isinstance(t, str) for t in texts):
                lengths = [len(t) for t in texts]
                logger.info(
                    "char_lens: min=%d max=%d avg=%.1f total=%d",
                    min(lengths), max(lengths), (sum(lengths) / len(lengths)), sum(lengths)
                )

                # short previews (truncated)
                def _preview(s: str, k: int = 120) -> str:
                    s = s.replace("\n", "\\n")
                    return s[:k] + ("…" if len(s) > k else "")

                if n >= 1:
                    logger.debug("preview[0]=%r", _preview(texts[0]))
                if n >= 2:
                    logger.debug("preview[1]=%r", _preview(texts[1]))
            else:
                # If not all strings, log a small sample of element types
                sample_types = [type(x).__name__ for x in texts[: min(n, 5)]]
                logger.info("sample element types (first up to 5): %s", sample_types)

        # ---- actual call (unchanged), with error logging ----
        try:
            logger.info("Calling Gemini API embed_content - Sleeping for 0.1 seconds...")
            time.sleep(0.1)
            result = self.client.models.embed_content(
                contents=texts,
                model=self.model_name,
                config=genai.types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.d_model,
                ),
            )
        except Exception as e:
            logger.exception(
                "embed_content failed: n_texts=%s text_type=%s task_type=%s model=%s out_dim=%s",
                n, text_type, task_type, self.model_name, self.d_model
            )
            raise

        # Log response size before tensor conversion
        try:
            n_emb = len(result.embeddings)
        except Exception:
            n_emb = None
        logger.info("embed_content returned embeddings_count=%s", n_emb)

        result = torch.stack([torch.tensor(emb.values) for emb in result.embeddings], dim=0)
        logger.info("output tensor shape=%s dtype=%s device=%s", tuple(result.shape), result.dtype, result.device)

        return result
