"""
Pipeline code for LLM-based classification of German social media posts as
prosecutable under §§185-187 StGB.

This module provides:
  - LM: wrapper around HuggingFace causal language models for batched inference
  - KnowledgeBase: retrieval store for few-shot demonstration examples (dense, sparse, fusion)
  - MultiStepKnowledgeBase: per-step retrieval stores for the 'explicit' multi-step pipeline
  - PromptConstructor: builds chat-formatted prompts, optionally with retrieved demonstrations
  - single_step_generation / multi_step_generation: the two classification pipelines
  - Helpers shared by all runner scripts: config validation, data loading,
    label extraction, result filenames, proportional demonstration ratios
"""

import json
import os
import random
import re
from datetime import datetime

import jinja2
import numpy as np
import pandas as pd
import torch
import yaml
from huggingface_hub import repo_exists
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from dataclasses import dataclass
from typing import Literal


# Allow PyTorch to use expandable memory segments to reduce fragmentation.
# Must be set before the CUDA allocator is first used.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# load up .env file variables to pull in secrets and keys
load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Language Model
# ---------------------------------------------------------------------------

TRANSIENT_OPENAI_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)


def openai_backoff(max_retries: int = 6):
    return retry(
        retry=retry_if_exception_type(TRANSIENT_OPENAI_ERRORS),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(max_retries),
        reraise=True,
    )


@openai_backoff(max_retries=6)
def completion_with_backoff(client, **kwargs):
    return client.responses.create(**kwargs)


@dataclass
class API_CONFIG:
    model: str = "gpt-5.5"
    # temperature: float = 0
    max_output_tokens: int = 160
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh"
    ] = "none"
    # top_logprobs (maximum number of tokens to return per position w. probability)
    # top_p (alternative to temperature, filters tokens with top_p probability)


class LM_API:
    """
     Wrapper for an external API-based language model. Implemented w.r.t OpenAI
    model family. Now an OpenAI Responses API wrapper with the same public
    generate() contract as LM.
    """

    def __init__(self, api_config: API_CONFIG):
        self.config = api_config
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def update_kv_cache(self) -> None:
        """Compatibility no-op so LM_API can be used where LM is expected."""
        return None

    def split_responses_instructions(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[dict[str, str]]]:
        instructions = []
        input_messages = []

        for message in messages:
            if message["role"] == "system":
                instructions.append(message["content"])
            else:
                input_messages.append(message)

        return ("\n\n".join(instructions) if instructions else None), input_messages

    def generate(
        self,
        prompts: list[list[dict[str, str]]],
        max_tokens: int = 10,
        thinking_mode: bool = False,
        do_sample: bool = False,
    ) -> list[str]:
        outputs = []

        for prompt in tqdm(
            prompts, total=len(prompts), mininterval=1.0, dynamic_ncols=True
        ):
            instructions, input_messages = self.split_responses_instructions(prompt)

            api_kwargs = {
                "model": self.config.model,
                "input": input_messages,
                "max_output_tokens": max_tokens,
                # "temperature": self.config.temperature if do_sample else 0,
            }

            if instructions is not None:
                api_kwargs["instructions"] = instructions

            if thinking_mode:
                api_kwargs["reasoning"] = {"effort": self.config.reasoning_effort}

            api_response = completion_with_backoff(self.client, **api_kwargs)
            outputs.append(api_response.output_text)

        return outputs


class LM:
    """
    Wrapper around any HuggingFace causal language model loadable via
    AutoModelForCausalLM. Handles batched prompt generation, automatic
    batch-size estimation from available GPU memory, and OOM recovery.
    """

    def __init__(self,repo_id:str,quantisation:bool=False):
        """
        Load a model from the HuggingFace Hub and estimate the maximum
        number of KV-cache tokens that fit in free GPU memory.

        Args:
            repo_id: HuggingFace Hub model identifier (e.g. "meta-llama/Llama-3-8B").
        """

        self.model_name = repo_id
        self.load(quantisation)
        self.update_kv_cache()


    def load(self,quantisation):

        """Load the model and tokenizer from HuggingFace Hub onto available devices."""

        if quantisation:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                device_map='auto',
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                device_map='auto',
                )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            device_map="auto",
            # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
        )
        # Use EOS as the padding token so the model can handle variable-length batches
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        chat_template = getattr(self.tokenizer, "chat_template", None) or ""
        self.thinking_capable = "enable_thinking" in chat_template

    def update_kv_cache(self):
        """
        Re-estimate how many KV-cache tokens fit in the GPU memory that is
        free right now. Called before each generation run so batch sizing
        adapts to whatever else (e.g. the embedding model) occupies the GPUs.
        """
        # Sum free memory across all visible GPUs
        free_memory = sum(
            torch.cuda.mem_get_info(device)[0]
            for device in range(torch.cuda.device_count())
        )

        # Multimodal wrappers nest the text model's dimensions in text_config
        model_config = self.model.config
        if "num_hidden_layers" not in model_config and "text_config" in model_config:
            model_config = model_config.text_config

        if "num_hidden_layers" not in model_config or "hidden_size" not in model_config:
            raise RuntimeError(
                f"Cannot estimate KV-cache size for {self.model_name}: its config "
                "exposes neither num_hidden_layers/hidden_size nor a text_config."
            )

        # Estimate KV-cache capacity: each token occupies
        # 2 (key+value) × 2 bytes (bfloat16) × num_layers × hidden_size bytes
        # Reference: https://www.baseten.co/blog/llm-transformer-inference-guide/#3500759-estimating-total-generation-time-on-each-gpu
        self.kv_cache_tokens = (
            free_memory
            / (2 * 2 * model_config.num_hidden_layers * model_config.hidden_size))
        if (
            "num_hidden_layers" in self.model.config
            and "hidden_size" in self.model.config
        ):
            self.kv_cache_tokens = free_memory / (
                2
                * 2
                * self.model.config.num_hidden_layers
                * self.model.config.hidden_size
            )
        elif (
            "text_config" in self.model.config
            and "num_hidden_layers" in self.model.config.text_config
            and "hidden_size" in self.model.config.text_config
        ):
            self.kv_cache_tokens = free_memory / (
                2
                * 2
                * self.model.config.text_config.num_hidden_layers
                * self.model.config.text_config.hidden_size
            )

    def get_batches(self, prompts, max_tokens):
        """
        Partition prompts into batches whose total token footprint fits in GPU memory.

        Batch size is estimated from KV-cache capacity, the longest prompt in the
        list, and the requested number of new tokens.

        Args:
            prompts:    List of prompts (any type whose str() gives a length proxy).
            max_tokens: Maximum new tokens to generate per prompt.

        Returns:
            A list of batches, where each batch is a sub-list of prompts.
        """

        # Per-prompt token footprint, estimated from the longest prompt:
        # chars * 1.3 deliberately overestimates the token count (German text
        # averages well above 1 char per token), leaving headroom in the
        # KV cache, plus max_tokens for the generated continuation.
        max_prompt_chars = max(len(str(x)) for x in prompts)
        batch_size = int(
            np.floor(self.kv_cache_tokens / (max_prompt_chars * 1.3 + max_tokens))
        )
        batch_size = max(1, batch_size)  # Always process at least one prompt at a time

        batches = []

        for i in range(0, len(prompts), batch_size):
            batches.append(prompts[i : i + batch_size])

        return batches

    def get_output(self, batch, max_tokens=50, thinking_mode=False) -> list[str]:
        """
        Run greedy generation (do_sample=False) on a single batch of
        chat-formatted prompts.

        On a RuntimeError during generation (assumed to be CUDA OOM) the
        batch is split in half and each half is retried recursively until
        it fits.

        Args:
            batch:         List of chat message lists (each in HuggingFace chat format).
            max_tokens:    Maximum new tokens to generate per prompt.
            thinking_mode: Enable chain-of-thought thinking (ignored by models
                           whose chat template has no enable_thinking flag).

        Returns:
            List of decoded output strings (one per prompt in the batch).
        """

        try:
            # some models support an explicit thinking/reasoning mode
            if self.thinking_capable:
                inputs = self.tokenizer.apply_chat_template(
                    batch,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=thinking_mode,
                    padding=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to("cuda")
            else:
                inputs = self.tokenizer.apply_chat_template(
                    batch,
                    tokenize=True,
                    add_generation_prompt=True,
                    padding=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to("cuda")

        except jinja2.exceptions.TemplateError as e:
            print("Error in applying chat template: ", e)
            print("Batch that caused the error: ", batch)
            raise

        input_length = inputs["input_ids"].shape[1]  # prompt length

        # CUDA out-of-memory surfaces as a generic RuntimeError. The recovery
        # (halving the batch) must happen *outside* the except block: while it
        # is active, the exception's traceback pins the failed generate() call's
        # stack frames — and with them its GPU allocations.
        try:
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # Slice off the prompt tokens so only the newly generated tokens are decoded
            new_tokens = generated_ids[:, input_length:]
            return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        except RuntimeError as e:
            print(e)
            if len(batch) == 1:
                # A single prompt that still does not fit cannot be split any
                # further — return an empty reply to keep outputs aligned
                print("Warning: inference failed for one sample that does not fit into GPU memory")
                return [""]
            print("OOM error — splitting batch in half and retrying...")

            # Each half is re-tokenized in the recursive call, so the full-batch
            # tensors can be released before retrying
            del inputs
            mid = len(batch) // 2
            return (
                self.get_output(batch[:mid], max_tokens=max_tokens, thinking_mode=thinking_mode)
                + self.get_output(batch[mid:], max_tokens=max_tokens, thinking_mode=thinking_mode)
            )

    def generate(
        self, prompts: list[list[dict[str, str]]], max_tokens=10, thinking_mode=False
    ) -> list[str]:
        """
        Generate text for a list of prompts, automatically batching to fit GPU memory.

        Args:
            prompts:       List of chat message lists.
            max_tokens:    Maximum new tokens per prompt.
            thinking_mode: Enable chain-of-thought thinking.

        Returns:
            List of output strings in the same order as the input prompts.
        """
        batches = self.get_batches(prompts, max_tokens)
        outputs = []

        for batch in tqdm(
            batches, total=len(batches), mininterval=1.0, dynamic_ncols=True
        ):
            outputs.extend(
                self.get_output(
                    batch, max_tokens=max_tokens, thinking_mode=thinking_mode
                )
            )

        return outputs


# ---------------------------------------------------------------------------
# Knowledge Base (retrieval store for few-shot demonstrations)
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """
    Stores labelled training examples and retrieves the most relevant ones for
    few-shot prompting. Labels are binary: True = prosecutable under
    §§185-187 StGB, False = not prosecutable (the DEF column).

    Embedding modes:
      - dense   : cosine-similarity search over text embeddings (via LangChain InMemoryVectorStore)
      - sparse  : keyword-based BM25 retrieval (via LangChain BM25Retriever)
      - fusion  : interleave demonstrations retrieved by both dense and sparse search

    Demonstration selection modes:
      - dynamic : retrieve examples per query text at inference time
      - static  : return a fixed, pre-optimised set of examples

    Retrieval modes for dynamic selection:
      - similarity : retrieve most similar examples (cosine similarity for dense, BM25 score for sparse)
      - diversity  : KMeans-cluster the training embeddings, randomly pick one example per cluster
      - mmr        : maximal marginal relevance, balances similarity and diversity
      - random     : sample uniformly from the training set
    """

    def __init__(
        self, data: list[str], labels: list[bool], config: dict, embedding_model=None
    ):
        # An already-loaded embedding model can be shared between knowledge bases
        self.embedding_model = embedding_model
        self.update_config(config)

        # Keep the raw data so the index structures can be built lazily in
        # query() if the demonstration mode only becomes 'dynamic' after
        # construction (via update_config)
        self.data = data
        self.labels = labels
        self.built = False

        # Skip indexing when no data is supplied (configured but empty knowledge base)
        if data is not None and labels is not None:
            self.build(data, labels)

    def update_config(self, config):
        """
        Adopt the retrieval-related settings of a (changed) config without
        re-indexing the data. Loads the embedding model on first call for
        dynamic mode; subsequent calls (and knowledge bases sharing the
        model) reuse it.
        """
        self.embedding_mode = config["embedding_mode"]
        self.demonstration_size = config["demonstration_size"]
        self.demonstration_mode = config["demonstration_mode"]
        self.retrieval_mode = config["retrieval_mode"]
        self.prompt_mode = config["prompt_mode"]
        self.embedding_model_name = config["embedding_model"]
        self.template_path = config["template_path"]
        # Keep the BM25 retrievers in sync with a changed demonstration size
        # (the diversity clusters stay fixed at their build-time size)
        if getattr(self, "built", False):
            self.retriever_hate.k = self.demonstration_size//2
            self.retriever_non_hate.k = self.demonstration_size//2
        # Only dynamic mode embeds the training data (dense index + KMeans
        # clusters); static demonstrations never need the embedding model
        if self.embedding_model is None and self.demonstration_mode == "dynamic":
            self.retriever_hate.k = self.demonstration_size // 2
            self.retriever_non_hate.k = self.demonstration_size // 2
        if self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=config["embedding_model"], 
                model_kwargs={"device": "cuda"}
            )

    def build(self, data: list[str], labels: list[bool]):
        """
        Index the training data for retrieval.

        For dynamic demonstration selection, all three index structures are
        built up front — a dense vector store, per-class BM25 retrievers, and
        per-class KMeans clusters — so that update_config() can switch between
        embedding/retrieval modes without re-indexing. Note that the BM25 k and
        the number of clusters are fixed here from the demonstration_size at
        build time.

        Args:
            data:   List of text samples.
            labels: Corresponding binary labels (True = prosecutable).
        """
        docs = [
            Document(page_content=text, metadata={"label": label})
            for text, label in zip(data, labels)
        ]

        if self.demonstration_mode == "dynamic":

            # Lazy fallback: the embedding model is not loaded at construction
            # time when the knowledge base starts out non-dynamic (see
            # update_config) but is required for all dynamic index structures
            if self.embedding_model is None:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={"device": "cuda"})

            # Dense index: one vector store over both classes; class filtering
            # happens at query time via a metadata filter
            vector_store = InMemoryVectorStore(self.embedding_model)
            vector_store.add_documents(docs)
            self.vector_store = vector_store

            # Sparse index: BM25Retriever does not support metadata filtering,
            # so a separate retriever is built per class, each returning
            # half of the demonstrations
            docs_hate = [doc for doc in docs if doc.metadata["label"] == 1]
            docs_non_hate = [doc for doc in docs if doc.metadata["label"] == 0]

            self.retriever_hate = BM25Retriever.from_documents(docs_hate)
            self.retriever_hate.k = self.demonstration_size // 2

            self.retriever_non_hate = BM25Retriever.from_documents(docs_non_hate)
            self.retriever_non_hate.k = self.demonstration_size // 2

            # Kept for random sampling in query()
            self.documents = docs

            # Diversity index: per-class KMeans clustering of the embeddings
            # already computed by the vector store; one cluster per
            # demonstration slot (demonstration_size // 2 per class)
            embeddings_hate = {
                doc["text"]: doc["vector"]
                for _, doc in self.vector_store.store.items()
                if doc["metadata"]["label"] == 1
            }
            embeddings_non_hate = {
                doc["text"]: doc["vector"]
                for _, doc in self.vector_store.store.items()
                if doc["metadata"]["label"] == 0
            }
            num_clusters = self.demonstration_size // 2
            self.cluster_hate = {num: [] for num in range(0, num_clusters)}
            clustering_hate = KMeans(n_clusters=num_clusters).fit_predict(
                list(embeddings_hate.values())
            )
            for cluster_id, text in zip(clustering_hate, embeddings_hate.keys()):
                self.cluster_hate[cluster_id].append(text)

            self.cluster_non_hate = {num: [] for num in range(0, num_clusters)}
            clustering_non_hate = KMeans(n_clusters=num_clusters).fit_predict(
                list(embeddings_non_hate.values())
            )
            for cluster_id, text in zip(
                clustering_non_hate, embeddings_non_hate.keys()
            ):
                self.cluster_non_hate[cluster_id].append(text)

            self.built = True

        elif self.demonstration_mode == "static":

            if self.prompt_mode == "title":

                with open(os.path.join(self.template_path,"static_title.yaml")) as stream:
                    self.static_demonstrations = yaml.safe_load(stream)


            elif self.prompt_mode == "implicit":

                with open(os.path.join(self.template_path,"static_implicit.yaml")) as stream:
                    self.static_demonstrations = yaml.safe_load(stream)      



    def retrieve_dense(self, query:str, label:bool, k:int=None) -> dict[str, bool]:
        """
        Retrieve k examples of one class from the dense index, using the
        configured retrieval_mode (similarity, mmr, or diversity).
        k defaults to demonstration_size // 2.
        """
        k = k if k is not None else self.demonstration_size // 2
        class_filter = lambda doc: doc.metadata.get("label") == label

        if self.retrieval_mode == "similarity":
            closest_docs = self.vector_store.similarity_search(
                query,
                k=k,
                filter=class_filter
                )
            return {doc.page_content.strip(): label for doc in closest_docs}
        elif self.retrieval_mode == "mmr":
            closest_docs = self.vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=k*4,
                filter=class_filter
                )
            return {doc.page_content.strip(): label for doc in closest_docs}
        elif self.retrieval_mode == "diversity":
            # One randomly drawn example per KMeans cluster of this class;
            # clusters are fixed at build time (demonstration_size//2), so k
            # is respected up to the number of clusters available
            clustering = self.cluster_hate if label else self.cluster_non_hate
            docs = {}

            for cluster in list(clustering.values()):
                docs[random.choice(cluster)] = label
                if len(docs) == k:
                    break

            return docs


    def retrieve_sparse(self, query:str, label:bool, k:int=None) -> dict[str, bool]:
        """Retrieve the top BM25 matches from the given class's own retriever."""
        retriever = self.retriever_hate if label else self.retriever_non_hate
        if k is not None:
            retriever.k = k
        closest_docs = retriever.invoke(query)
        return {doc.page_content.strip(): label for doc in closest_docs}

    def retrieve(self, query: str, label: bool, k:int=None) -> dict[str, bool]:
        """
        Retrieve k examples of one class, dispatching to dense, sparse, or
        fusion retrieval per the config. k defaults to demonstration_size // 2.

        Args:
            query: The input text to find similar examples for.
            label: Class label to filter by (True = prosecutable).
            k:     Number of examples to retrieve. Defaults to demonstration_size // 2.

        Returns:
            Dict mapping retrieved text → label.
        """
        k = k if k is not None else self.demonstration_size // 2

        if self.embedding_mode == "dense":
            return self.retrieve_dense(query,label,k)

        elif self.embedding_mode == "sparse":
            return self.retrieve_sparse(query,label,k)

        elif self.embedding_mode == "fusion":
            # Alternate between dense and sparse results until k examples
            # are collected; the dict deduplicates texts found by both
            dense_demos = self.retrieve_dense(query,label,k)
            sparse_demos = self.retrieve_sparse(query,label,k)

            demos = {}
            for dense_demo, sparse_demo in zip(dense_demos.keys(), sparse_demos.keys()):
                demos[dense_demo] = label
                if len(demos) == k:
                    break
                demos[sparse_demo] = label
                if len(demos) == k:
                    break

            return demos


    def query(self, query: str, demonstration_mode, ratio:dict=None) -> dict[str, bool]:
        """
        Return a set of demonstration examples covering both classes.

        Args:
            query:              The input text used to select relevant demonstrations.
            demonstration_mode: 'dynamic' or 'static'.
            ratio:              Optional per-class counts {"pos": int, "neg": int}
                                (must sum to demonstration_size). Defaults to an
                                equal split between the classes.

        Returns:
            Dict mapping example text → label.
        """

        if ratio is not None:
            num_pos = ratio["pos"]
            num_neg = ratio["neg"]
        else:
            num_pos = self.demonstration_size//2
            num_neg = self.demonstration_size//2

        assert num_pos+num_neg == self.demonstration_size, "Number of neg/pos demos does not add up to set demonstration size"


        if demonstration_mode == "dynamic" and not self.built:
            # The index structures are missing when the knowledge base was
            # constructed under a non-dynamic config — build them now
            self.demonstration_mode = "dynamic"
            self.build(self.data, self.labels)

        if demonstration_mode == "dynamic" and self.retrieval_mode != "random":
            # Retrieve the most relevant examples from each class
            return self.retrieve(query,False,num_neg) | self.retrieve(query,True,num_pos)

        elif self.retrieval_mode == "random":
            # Sample an equal number of examples from each class, ignoring the query
            negatives = [doc for doc in self.documents if not doc.metadata["label"]]
            positives = [doc for doc in self.documents if doc.metadata["label"]]
            return (
                {doc.page_content.strip(): doc.metadata["label"]
                 for doc in random.sample(negatives, num_neg)}
                | {doc.page_content.strip(): doc.metadata["label"]
                   for doc in random.sample(positives, num_pos)}
            )

        elif demonstration_mode == "static":
            # Fixed demonstration set selected once in build(); only defined
            # for the 'title' and 'implicit' prompt modes
            return self.static_demonstrations


# ---------------------------------------------------------------------------
# Multi-step Knowledge Base (per-step retrieval for the 'explicit' pipeline)
# ---------------------------------------------------------------------------


# Maps each step of the explicit template to its annotation column in
# data/single_step_annotation.csv (column letters follow Zufall et al. 2019)
STEP_ANNOTATION_COLUMNS = {
    "step1": "step_c_d_choice",  # defamatory object
    "step2": "step_e_choice",  # disparaging statement
    "step3": "step_f_choice",  # value judgement
    "step4": "step_g_choice",  # abusive insult
    "step5": "step_h_choice",  # public interest
    "step6": "step_i_choice",  # abusive criticism
}


def parse_step_label(value) -> bool | None:
    """
    Normalise a per-step annotation value to a binary label.

    The annotation columns mix booleans and the strings 'True'/'False'/'none';
    'none' (or NaN) means the example never reached that step of the decision
    schema and is therefore not a valid demonstration for it.

    Args:
        value: Raw cell value from single_step_annotation.csv.

    Returns:
        True/False, or None when the example carries no label for that step.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "true":
            return True
        if value == "false":
            return False
    return None


class MultiStepKnowledgeBase:
    """
    Per-step retrieval stores for few-shot prompting in the 'explicit' pipeline.

    Each of the 6 legal decision steps has its own binary labels (from
    data/single_step_annotation.csv), so a separate KnowledgeBase — with its own
    vector store, BM25 retrievers and KMeans clusters — is built per step from
    the examples annotated for that step. Examples are restricted to ids in the
    current train split to avoid test leakage. A single embedding model is
    shared across all per-step knowledge bases.

    Because some steps have very few examples of one class, the per-step
    demonstration size is balanced down to
    2 * min(demonstration_size // 2, #True, #False)
    so demonstrations stay class-balanced. A step where one class has no
    examples at all gets no demonstrations (zero-shot for that step).
    """

    def __init__(
        self, annotations: pd.DataFrame, train_ids, config: dict, embedding_model=None
    ):
        """
        Build one KnowledgeBase per decision step.

        Args:
            annotations:     DataFrame loaded from single_step_annotation.csv.
            train_ids:       Ids of the current train split; annotation rows
                             outside it are discarded.
            config:          Pipeline configuration dictionary.
            embedding_model: Optional pre-loaded embedding model to share.
        """
        self.demonstration_mode = config["demonstration_mode"]
        self.embedding_model = embedding_model
        self.step_kbs = {}
        self.step_class_counts = {}

        annotations = annotations[annotations["id"].isin(set(train_ids))]

        for step, column in STEP_ANNOTATION_COLUMNS.items():
            labels = annotations[column].map(parse_step_label)
            mask = labels.notna()
            texts = annotations.loc[mask, "description"].tolist()
            step_labels = labels[mask].astype(bool).tolist()

            n_true = sum(step_labels)
            n_false = len(step_labels) - n_true
            self.step_class_counts[step] = (n_false, n_true)

            step_size = self.step_demonstration_size(step, config["demonstration_size"])
            if step_size == 0:
                print(
                    f"Warning: no annotated examples for one class at {step} — running this step zero-shot."
                )
                self.step_kbs[step] = None
                continue

            step_config = {**config, "demonstration_size": step_size}
            kb = KnowledgeBase(
                texts, step_labels, step_config, embedding_model=self.embedding_model
            )
            self.embedding_model = kb.embedding_model
            self.step_kbs[step] = kb

    def step_demonstration_size(self, step: str, demonstration_size: int) -> int:
        """Balance the demonstration size down to what both classes can supply at this step."""
        n_false, n_true = self.step_class_counts[step]
        return 2 * min(demonstration_size // 2, n_true, n_false)

    def update_config(self, config: dict):
        """Propagate a config change to all per-step knowledge bases, re-capping their sizes."""
        self.demonstration_mode = config["demonstration_mode"]
        for step, kb in self.step_kbs.items():
            if kb is None:
                continue
            step_size = self.step_demonstration_size(step, config["demonstration_size"])
            kb.update_config({**config, "demonstration_size": step_size})

    def query(self, query: str, demonstration_mode, step: str) -> dict[str, bool]:
        """
        Return a balanced set of demonstrations labelled for the given step.

        Args:
            query:              The input text used to select relevant demonstrations.
            demonstration_mode: 'dynamic' or 'static'.
            step:               Template step name ('step1' … 'step6').

        Returns:
            Dict mapping example text → step label; empty when the step has no usable examples.
        """
        kb = self.step_kbs.get(step)
        if kb is None:
            return {}
        return kb.query(query, demonstration_mode)


# ---------------------------------------------------------------------------
# Prompt Constructor
# ---------------------------------------------------------------------------


class PromptConstructor:
    """
    Builds chat-formatted prompts from templates, optionally injecting
    few-shot demonstrations retrieved from a KnowledgeBase.

    Supported prompt modes (matching template filenames in template_path):
      - title       : minimal prompt — just the label names and a True/False answer format
      - description : adds a short description of the classification criteria
      - implicit    : all 6 legal decision steps inline in a single prompt
      - explicit    : one template block per decision step, used by
                      multi_step_generation() with one inference call per step
    """

    def __init__(self, knowledge_base: KnowledgeBase, config: dict):
        self.prompt_mode = config["prompt_mode"]
        self.embedding_mode = config["embedding_mode"]
        self.knowledge_base = knowledge_base
        self.demonstration_mode = config["demonstration_mode"]
        self.demonstration_size = config["demonstration_size"]
        self.retrieval_mode = config["retrieval_mode"]
        self.load_template(config["template_path"] + "/" + self.prompt_mode)

    def load_template(self, path: str):
        """
        Read and parse a prompt template file.

        For 'explicit' mode the template contains numbered <stepN> blocks that
        are parsed into a dict {"step1": ..., "step2": ..., ...}.
        For all other modes, a single <task> block is expected.
        A <system> block is always required for the system prompt.

        Args:
            path: Path to the template file.
        """

        with open(path, "r") as f:
            template = f.read()

        if self.prompt_mode == "explicit":
            self.template = {
                f"step{num}": content.strip()
                for num, content in re.findall(
                    r"<step(\d+)>(.*?)</step\d+>", template, re.DOTALL
                )
            }
        else:
            self.template = re.findall(r"<task>(.*?)</task>", template, re.DOTALL)[
                0
            ].strip()

        self.system_prompt = re.findall(r"<system>(.*?)</system>", template, re.DOTALL)[
            0
        ].strip()

    def create_message(
            self,
            text,
            task,
            demonstrations=None,
            system_prompt=False,
            order:str="random"
            ) -> list[dict[str,str]]:
        """
        Assemble a list of chat messages for a single classification example.

        The conversation structure is:
          [system]  →  task instruction
          [user/assistant] pairs for each demonstration (shuffled)
          [user]    →  the text to classify

        Args:
            text:          The input text to classify.
            task:          The task instruction string.
            demonstrations: Dict mapping example text → label (False/True). If provided, these are inserted as few-shot user/assistant turns.
            system_prompt: Whether to include a system message.

        Returns:
            List of message dicts in HuggingFace chat format.
        """

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": task})

        if demonstrations is not None:
            demo_list = list(demonstrations.items())
            if order == "random":
                random.shuffle(demo_list) # Randomise order to avoid position bias
            elif order == "true_first":
                demo_list.sort(key=lambda x: x[1], reverse=True)
            elif order == "true_last":
                demo_list.sort(key=lambda x: x[1])

            random.shuffle(demo_list)  # Randomise order to avoid position bias
            for demo_text, label in demo_list:
                messages.append({"role": "user", "content": demo_text})
                messages.append(
                    {"role": "assistant", "content": "True" if label else "False"}
                )

        messages.append({"role": "user", "content": text})
        return messages

    def construct(
            self,
            text: str,
            task:str | None=None,
            system_prompt: bool =False,
            step: str | None=None,
            order : str = "random",
            ratio : dict = None
            ) -> list[dict[str,str]]:

        """
        Build a complete prompt for the given text.

        Retrieves demonstrations from the KnowledgeBase when demonstration_size > 0.

        Args:
            text:          The input text to classify.
            task:          Override for the task instruction; defaults to the loaded template.
            system_prompt: Whether to prepend a system message.
            step:          Step name ('step1' … 'step6') for the explicit pipeline;
                           demonstrations are then retrieved from that step's knowledge base.

        Returns:
            List of message dicts ready for the tokenizer's apply_chat_template.
        """

        if task is None:
            task = self.template
        if self.demonstration_size > 0:
            if step is not None:
                demonstrations = self.knowledge_base.query(
                    text, self.demonstration_mode, step=step
                )
            else:
                demonstrations = self.knowledge_base.query(text, self.demonstration_mode,ratio=ratio)
        else:
            demonstrations = None
        return self.create_message(text, task, demonstrations, system_prompt=system_prompt, order=order)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def validate_config(config: dict, check_model: bool = True) -> dict:
    """
    Validate all fields of a pipeline configuration dict and rewrite
    inconsistent combinations to their canonical form (in place).

    run_all.py exploits the rewriting for grid deduplication: a combination is
    only executed when validate_config() leaves it unchanged.

    Raises AssertionError on invalid paths, unknown model identifiers, or
    out-of-range parameter values.

    Args:
        config:      Raw configuration dictionary (mutated and returned).
        check_model: Whether to verify model_name on the HuggingFace Hub
                     (skip for offline / repeated calls).

    Returns:
        The (potentially patched) configuration dictionary.
    """

    # --- Path checks ---
    for key in ("data_path", "template_path", "results_path"):
        assert os.path.exists(config[key]) and os.path.isdir(config[key]), (
            f"{key} '{config[key]}' does not exist or is not a directory."
        )

    # --- Model existence ---
    if check_model:
        assert repo_exists(config["model_name"]), (
            f"Model '{config['model_name']}' does not exist on the HuggingFace Hub."
        )

    for key in config.keys():
        if config[key] == "None":
            config[key] = None

    # --- Parameter range / allowed-value checks ---
    assert config["max_tokens"] > 0, "max_tokens must be a positive integer."
    assert config["thinking_mode"] in (True, False)
    assert config["embedding_mode"] in ("dense", "sparse", "fusion", None)
    assert config["prompt_mode"] in ("title", "description", "implicit", "explicit")
    assert config["demonstration_mode"] in ("dynamic", "static", None)
    assert config["demonstration_size"] >= 0
    assert config["retrieval_mode"] in (
        "similarity",
        "diversity",
        "mmr",
        "random",
        None,
    )

    # --- Cross-parameter consistency ---
    # Each rule rewrites an inconsistent combination to its canonical form.
    # (Rewrites happen silently; run_all.py calls this once per grid cell.)

    if config["demonstration_size"] == 0:
        # Zero-shot: demonstration/retrieval settings are meaningless
        for key in ("embedding_mode", "demonstration_mode", "retrieval_mode"):
            if config[key] is not None:
                config[key] = None

    else:
        # Static demonstrations are fixed up front, so no retrieval is involved
        if config["demonstration_mode"] == "static":
            if config["embedding_mode"] is not None:
                config["embedding_mode"] = None
            if config["retrieval_mode"] is not None:
                config["retrieval_mode"] = None

        # Demonstrations requested but no mode given → default to dynamic
        if config["demonstration_mode"] is None:
            config["demonstration_mode"] = "dynamic"

        if config["demonstration_mode"] == "dynamic":
            if (
                config["retrieval_mode"] == "random"
                and config["embedding_mode"] is not None
            ):
                # Random sampling needs no index
                config["embedding_mode"] = None
            elif config["embedding_mode"] is None and config["retrieval_mode"] is None:
                # Nothing specified → cheapest strategy
                config["retrieval_mode"] = "random"
            elif (
                config["embedding_mode"] is None
                and config["retrieval_mode"] != "random"
            ):
                # similarity/diversity/mmr need an index → default to dense
                config["embedding_mode"] = "dense"
            elif (
                config["embedding_mode"] is not None
                and config["retrieval_mode"] is None
            ):
                config["retrieval_mode"] = "similarity"

            if (
                config["embedding_mode"] == "dense"
                and config["embedding_model"] is None
            ):
                config["embedding_model"] = "codefuse-ai/F2LLM-0.6B"

            # diversity and mmr are only supported with dense embeddings
            if (
                config["embedding_mode"] in ["sparse", "fusion"]
                and config["retrieval_mode"] != "similarity"
            ):
                config["retrieval_mode"] = "similarity"

    return config


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def result_filename(config: dict, fold: int | None = None) -> str:
    """
    Build the standard result filename (see CLAUDE.md):
    {model}_{prompt}_{demo_mode}_{demo_size}_{embedding}_{retrieval}[_fold-N][_thinking].csv

    None values appear literally as "None". The model name is shortened to the
    part after the last "/" of the HuggingFace repo id.

    Args:
        config: Pipeline configuration dictionary.
        fold:   Cross-validation fold index; omitted from the name when None
                (e.g. competition runs without CV).
    """
    model_short = config["model_name"].split("/")[-1]
    fold_suffix = f"_fold-{fold}" if fold is not None else ""
    thinking_suffix = "_thinking" if config["thinking_mode"] else ""

    return (
        f"{model_short}_"
        f"{config['prompt_mode']}_"
        f"{config['demonstration_mode']}_"
        f"{config['demonstration_size']}_"
        f"{config['embedding_mode']}_"
        f"{config['retrieval_mode']}"
        f"{fold_suffix}"
        f"{thinking_suffix}.csv"
    )


def proportional_demo_ratio(labels: pd.Series, demonstration_size: int) -> dict[str, int]:
    """
    Per-class demonstration counts that mirror the class distribution of the
    train split (used by runs with ratio == 'proportional').

    Each class gets at least one demonstration. Because the two per-class
    counts are rounded independently, their sum can be one off — in that case
    a single count is adjusted so the total is exactly demonstration_size.

    Args:
        labels:             Binary labels of the train split (the DEF column).
        demonstration_size: Total number of demonstrations (k).

    Returns:
        {"pos": <count for True>, "neg": <count for False>}
    """
    num_neg = int(max(1, round(demonstration_size * (labels == False).sum() / len(labels))))
    num_pos = int(max(1, round(demonstration_size * (labels == True).sum() / len(labels))))

    diff = demonstration_size - (num_pos + num_neg)
    if diff == 1:
        # One short: top up the under-represented class
        if num_pos <= num_neg:
            num_pos += 1
        else:
            num_neg += 1
    elif diff == -1:
        # One over: take it from the over-represented class
        if num_pos >= num_neg:
            num_pos -= 1
        else:
            num_neg -= 1

    assert num_pos + num_neg == demonstration_size, (
        "Demo ratio does not add up to set demonstration size")
    return {"pos": num_pos, "neg": num_neg}


def read_labels_from_answer(answer: str) -> bool | None:
    """
    Extract a binary True/False label from a free-text model answer via
    case-insensitive substring matching on "true"/"false".

    Args:
        answer: Raw string output from the language model.

    Returns:
        True or False, or None when the answer is ambiguous
        (contains both words or neither).
    """

    lower = answer.lower()
    has_false = "false" in lower
    has_true = "true" in lower

    if has_false and not has_true:
        return False
    if has_true and not has_false:
        return True
    return None  # Ambiguous


def load_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset from a semicolon-delimited CSV and split into
    train (70 %) and test (30 %) sets with a fixed random seed.

    Args:
        file_path: Path to the CSV file.

    Returns:
        (train_df, test_df) tuple of DataFrames.
    """
    df = pd.read_csv(file_path, sep=";")
    return train_test_split(df, random_state=42, test_size=0.3)


# ---------------------------------------------------------------------------
# Generation pipelines
# ---------------------------------------------------------------------------


def single_step_generation(
        test: pd.DataFrame,
        lm: LM,
        pc: PromptConstructor,
        config: dict,
        order: str = "random",
        ratio: dict | None = None,
        ) -> pd.DataFrame:
    """
    Run the single-prompt classification pipeline ('title', 'description',
    'implicit') over the test set: build one prompt per instance (including
    demonstration retrieval if configured), generate, and parse the labels.

    Args:
        test:   Test DataFrame with "id" and "description" columns; a "DEF"
                column is carried over when present (absent for the
                unlabelled competition test set).
        lm:     Loaded LM instance for text generation.
        pc:     PromptConstructor for a single-step prompt mode.
        config: Pipeline configuration dictionary.
        order:  Demonstration ordering ('random', 'true_first', 'true_last').
        ratio:  Optional per-class demonstration counts (see proportional_demo_ratio).

    Returns:
        DataFrame with id, description, [DEF,] predicted_label (None where the
        model's reply was ambiguous), reply, and the JSON-serialised prompt.
        Persisting the result is left to the caller.
    """
    if config["embedding_mode"] is not None:
        print(f"Retrieving demonstrations {datetime.now()}...")
    prompts = [pc.construct(text, order=order, ratio=ratio) for text in tqdm(test["description"])]

    # Re-estimate the batch size from the GPU memory that is free right now
    # (the embedding model may have been loaded since the LM was)
    lm.update_kv_cache()

    print(f"Classifying {datetime.now()}...")
    generated_answers = lm.generate(
        prompts,
        max_tokens=config["max_tokens"],
        thinking_mode=config["thinking_mode"],
    )
    y_pred = [read_labels_from_answer(a) for a in generated_answers]

    results = pd.DataFrame({
        "id": test["id"],
        "description": test["description"],
        "predicted_label": y_pred,
        "reply": generated_answers,
        # Serialise the chat-format prompts so the CSV stays one row per instance
        "prompt": [json.dumps(p, ensure_ascii=False) for p in prompts],
    })
    if "DEF" in test.columns:
        results.insert(2, "DEF", test["DEF"])
    return results


def multi_step_generation(
        test:pd.DataFrame,
        lm:LM,
        pc:PromptConstructor,
        config:dict,
        order:str="random",
        ) -> pd.DataFrame:
    """
    Run the multi-step ('explicit') classification pipeline over the test set.

    Each step in the template is executed sequentially as an independent
    conversation: system prompt, the step's task, its few-shot demonstrations,
    and the post — previous steps' prompts and replies are not included.
    templates/explicit_decisions.yaml controls whether each step's True/False
    answer leads to an early final label or continues to the next step.

    Few-shot prompting: when demonstration_size > 0 the PromptConstructor must
    hold a MultiStepKnowledgeBase, so each step retrieves demonstrations that
    are labelled for that step's own decision criterion.

    Args:
        test:   Test DataFrame with a "description" column.
        lm:     Loaded LM instance for text generation.
        pc:     PromptConstructor with an 'explicit' multi-step template.
        config: Pipeline configuration dictionary.

    Returns:
        A copy of the test DataFrame extended with the last step's prompt
        (JSON-serialised), per-step label and reply columns (stepN,
        stepN_reply), and the final "predicted_label" (None where the model's
        reply was ambiguous). Persisting the result is left to the caller.
    """

    # Load the per-step decision logic (maps True/False label → "continue" | final label)
    with open(f"{config['template_path']}/explicit_decisions.yaml") as stream:
        decisions = yaml.safe_load(stream)

    # Initialise tracking columns on a deep copy so the original test set is unchanged
    all_replies = test.copy(deep=True)
    all_replies["prompt"] = [None] * len(all_replies)
    all_replies["continue"] = [True] * len(all_replies)
    all_replies["predicted_label"] = [None] * len(all_replies)

    print("Starting multi-step generations")

    for step, task in tqdm(pc.template.items()):
        print("Starting step ", step)

        # Initialise per-step result columns
        all_replies[step] = [None] * len(all_replies)
        all_replies[f"{step}_reply"] = [None] * len(all_replies)

        # Each step is an independent conversation: system prompt, the step's
        # task, its few-shot demonstrations and the post
        for i, row in all_replies.iterrows():
            if not row["continue"]:
                continue
            all_replies.at[i, "prompt"] = pc.construct(row["description"], task, system_prompt=True, step=step,order=order)


        # --- Generate answers for all still-active examples ---
        active_mask = all_replies["continue"]
        active_prompts = all_replies.loc[active_mask, "prompt"].tolist()
        active_ids = all_replies.loc[active_mask].index.tolist()

        # update KV-cache before generating answers for the current step
        lm.update_kv_cache()

        print(f"Generating answers ({datetime.now()})…")
        generated_answers = lm.generate(
            active_prompts,
            max_tokens=config["max_tokens"],
            thinking_mode=config["thinking_mode"],
        )

        print(f"Extracting labels from answers {datetime.now()}...")
        y_pred = list(map(read_labels_from_answer, generated_answers))

        for i, pred, answer in zip(active_ids, y_pred, generated_answers):
            all_replies.loc[i, step] = pred
            all_replies.loc[i, f"{step}_reply"] = answer

        # --- Apply decision logic to determine next action for each example ---
        decision = decisions[step]

        for i, row in all_replies.iterrows():
            pred = row[step]
            if pred is not None:
                next_action = decision[pred]
                if next_action == "continue":
                    all_replies.loc[i, "continue"] = True
                else:
                    # A definitive label has been reached; stop processing this example
                    all_replies.loc[i, "continue"] = False
                    all_replies.loc[i, "predicted_label"] = next_action

            else:
                # Ambiguous model output — mark as unresolved and halt
                if row["continue"]:
                    print(
                        f"Ambiguous reply at step {step}, row {i}: '{row[f'{step}_reply']}'"
                    )
                    all_replies.loc[i, "continue"] = False
                    all_replies.loc[i, "predicted_label"] = None

    # Serialise the chat-format prompts so the CSV stays one row per instance
    all_replies["prompt"] = all_replies["prompt"].apply(
        lambda x: json.dumps(x, ensure_ascii=False) if x is not None else None)

    return all_replies
