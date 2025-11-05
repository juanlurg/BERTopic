import time
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any, Union, Callable
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import (
    retry_with_exponential_backoff,
    truncate_document,
    validate_truncate_document_parameters,
)


DEFAULT_CHAT_PROMPT = """You will extract a short topic label from given documents and keywords.
Here are two examples of topics you created before:

# Example 1
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the worst food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
topic: Environmental impacts of eating meat

# Example 2
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
topic: Shipping and delivery issues

# Your task
Sample texts from this topic:
[DOCUMENTS]

Keywords: [KEYWORDS]

Based on the information above, extract a short topic label (three words at most) in the following format:
topic: <topic_label>
"""

DEFAULT_SYSTEM_PROMPT = "You are an assistant that extracts high-level topics from texts."


class Gemini(BaseRepresentation):
    r"""Using Google's Gemini API to generate topic labels based
    on their generative models.

    For an overview see:
    https://ai.google.dev/gemini-api/docs/models/gemini

    Arguments:
        client: A `google.generativeai.GenerativeModel` instance or model name string.
                If a string is provided, a GenerativeModel will be created with that name.
        model: Model to use within Gemini, defaults to `"gemini-1.5-flash"`.
               This parameter is ignored if `client` is already a GenerativeModel instance.
        generator_kwargs: Kwargs passed to `model.generate_content()`
                          for fine-tuning the output.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `self.default_system_prompt_` is used instead.
        delay_in_seconds: The delay in seconds between consecutive prompts
                          in order to prevent RateLimitErrors.
        exponential_backoff: Retry requests with a random exponential backoff.
                             A short sleep is used when a rate limit error is hit,
                             then the request is retried. Increase the sleep length
                             if errors are hit until 10 unsuccessful requests.
                             If True, overrides `delay_in_seconds`.
        nr_docs: The number of documents to pass to Gemini if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to Gemini.
                   Accepts values between 0 and 1. A higher
                   value results in passing more diverse documents
                   whereas lower values passes more similar documents.
        doc_length: The maximum length of each document. If a document is longer,
                    it will be truncated. If None, the entire document is passed.
        tokenizer: The tokenizer used to calculate to split the document into segments
                   used to count the length of a document.
                       * If tokenizer is 'char', then the document is split up
                         into characters which are counted to adhere to `doc_length`
                       * If tokenizer is 'whitespace', the document is split up
                         into words separated by whitespaces. These words are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is 'vectorizer', then the internal CountVectorizer
                         is used to tokenize the document. These tokens are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is a callable, then that callable is used to tokenize
                         the document. These tokens are counted and truncated depending
                         on `doc_length`

    Usage:

    To use this, you will need to install the google-generativeai package first:

    `pip install google-generativeai`

    Then, get yourself an API key and use Gemini's API as follows:

    ```python
    import google.generativeai as genai
    from bertopic.representation import Gemini
    from bertopic import BERTopic

    # Configure the API key
    genai.configure(api_key=MY_API_KEY)

    # Create your representation model
    representation_model = Gemini(model="gemini-1.5-flash", delay_in_seconds=1)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also pass a pre-configured GenerativeModel instance:

    ```python
    import google.generativeai as genai
    from bertopic.representation import Gemini

    # Configure and create model
    genai.configure(api_key=MY_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    representation_model = Gemini(client=model, delay_in_seconds=1)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS] \nThese documents are about the following topic: '"
    representation_model = Gemini(model="gemini-1.5-flash", prompt=prompt, delay_in_seconds=1)
    ```

    You can use generator_kwargs to pass additional parameters:

    ```python
    generator_kwargs = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 100,
    }
    representation_model = Gemini(
        model="gemini-1.5-flash",
        generator_kwargs=generator_kwargs,
        delay_in_seconds=1
    )
    ```
    """

    def __init__(
        self,
        client=None,
        model: str = "gemini-1.5-flash",
        prompt: str = None,
        system_prompt: str = None,
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float = None,
        exponential_backoff: bool = False,
        nr_docs: int = 4,
        diversity: float = None,
        doc_length: int = None,
        tokenizer: Union[str, Callable] = None,
        **kwargs,
    ):
        # Import here to handle optional dependency
        try:
            import google.generativeai as genai
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "google.generativeai is not installed. "
                "Please install it with: pip install google-generativeai"
            )

        # Handle client parameter - can be a model instance or None
        if client is None:
            # Create a new GenerativeModel with the specified model name
            self.model = genai.GenerativeModel(model)
            self.model_name = model
        elif isinstance(client, str):
            # If client is a string, treat it as model name
            self.model = genai.GenerativeModel(client)
            self.model_name = client
        else:
            # Assume client is already a GenerativeModel instance
            self.model = client
            self.model_name = model

        if prompt is None:
            self.prompt = DEFAULT_CHAT_PROMPT
        else:
            self.prompt = prompt

        if system_prompt is None:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
        else:
            self.system_prompt = system_prompt

        self.default_prompt_ = DEFAULT_CHAT_PROMPT
        self.default_system_prompt_ = DEFAULT_SYSTEM_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.exponential_backoff = exponential_backoff
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        validate_truncate_document_parameters(self.tokenizer, self.doc_length)

        self.prompts_ = []

        self.generator_kwargs = generator_kwargs.copy() if generator_kwargs else {}

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics.

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top n representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        # Generate using Gemini's Language Model
        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
            prompt = self._create_prompt(truncated_docs, topic, topics)
            self.prompts_.append(prompt)

            # Delay
            if self.delay_in_seconds:
                time.sleep(self.delay_in_seconds)

            # Combine system prompt with user prompt for Gemini
            full_prompt = f"{self.system_prompt}\n\n{prompt}"

            if self.exponential_backoff:
                response = generate_content_with_backoff(self.model, full_prompt, **self.generator_kwargs)
            else:
                response = self.model.generate_content(full_prompt, **self.generator_kwargs)

            # Extract the generated text
            # Handle cases where content might be blocked or empty
            try:
                if response and hasattr(response, "text") and response.text:
                    label = response.text.strip().replace("topic: ", "")
                else:
                    label = "No label returned"
            except (ValueError, AttributeError):
                # ValueError can occur if content is blocked
                label = "No label returned"

            updated_topics[topic] = [(label, 1)]

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = list(zip(*topics[topic]))[0]

        # Use the Default Chat Prompt
        if self.prompt == DEFAULT_CHAT_PROMPT:
            prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
            prompt = self._replace_documents(prompt, docs)

        # Use a custom prompt that leverages keywords, documents or both using
        # custom tags, namely [KEYWORDS] and [DOCUMENTS] respectively
        else:
            prompt = self.prompt
            if "[KEYWORDS]" in prompt:
                prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
            if "[DOCUMENTS]" in prompt:
                prompt = self._replace_documents(prompt, docs)

        return prompt

    @staticmethod
    def _replace_documents(prompt, docs):
        to_replace = ""
        for doc in docs:
            to_replace += f"- {doc}\n"
        prompt = prompt.replace("[DOCUMENTS]", to_replace)
        return prompt


def generate_content_with_backoff(model, prompt, **kwargs):
    """Generate content with exponential backoff for rate limit errors."""
    try:
        from google.api_core import exceptions as google_exceptions
        errors = (google_exceptions.ResourceExhausted,)
    except ImportError:
        # Fallback if google-api-core is not available
        errors = (Exception,)

    return retry_with_exponential_backoff(
        model.generate_content,
        errors=errors,
    )(prompt, **kwargs)
