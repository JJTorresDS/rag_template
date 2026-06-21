INSTRUCTIONS = """
Your task is to answer questions based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="nvidia/nemotron-3-nano-30b-a3b:free",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model
    
    def search(self, query, num_results=5):
        """Compatible with search methods that have an ElasticSearch like interface"""
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        """Builds a context string from search results"""
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        """Builds a prompt string from a query and search results"""
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )
    
    def llm(self, prompt):
        """Calls the LLM client with the given prompt"""
        input_messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response
    
    def rag(self, query):
        """Performs a RAG query (Main method)"""
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer



from agents import Agent, Runner, function_tool, trace

AGENTIC_INSTRUCTIONS = """You are a helpful teaching assistant for an online course.
You have access to a search tool to look up relevant Q&A pairs from the course knowledge base.
Always search before answering. You may search multiple times with different queries if needed."""


class RAGAgentic(RAGBase):

    def __init__(
        self,
        index,
        llm_client=None,
        num_results=5,
        instructions=AGENTIC_INSTRUCTIONS,
        model="gpt-4o-mini",
        **kwargs
    ):
        super().__init__(index, llm_client, instructions=instructions, model=model, **kwargs)
        self.num_results = num_results
        self._agent = self._build_agent()

    def _build_agent(self):
        _search = self.search
        _build_context = self.build_context
        _num_results = self.num_results

        @function_tool
        def search_knowledge_base(query: str) -> str:
            """Search the course knowledge base for relevant Q&A pairs."""
            results = _search(query, num_results=_num_results)
            return _build_context(results)

        return Agent(
            name="RAG Agent",
            instructions=self.instructions,
            model=self.model,
            tools=[search_knowledge_base],
        )

    def rag(self, query: str) -> str:
        import asyncio
        import concurrent.futures

        async def _run():
            with trace("RAGAgentic", metadata={"query": query, "course": self.course}):
                result = await Runner.run(self._agent, query)
                return result.final_output

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: asyncio.run(_run()))
            return future.result()


class RAGGithub(RAGBase):

    def search(self, query, num_results=5):
        boost_dict = {"filename": 2.0, "content": 1.0}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(f"File: {doc['filename']}")
            lines.append(doc["content"])
            lines.append("")

        return "\n".join(lines).strip()


GITHUB_INSTRUCTIONS = """You're a course teaching assistant. 
Answer the student's question using the search tool. 
Make multiple searches with different keywords before answering."""

class RAGGithubAgentic(RAGGithub):

    def __init__(
        self,
        index,
        llm_client=None,
        num_results=5,
        instructions=GITHUB_INSTRUCTIONS,
        model="gpt-5.4-mini",
        **kwargs
    ):
        super().__init__(index, llm_client, instructions=instructions, model=model, **kwargs)
        self.num_results = num_results
        self._agent = self._build_agent()

    def _build_agent(self):
        _search = self.search
        _build_context = self.build_context
        _num_results = self.num_results

        @function_tool
        def search_knowledge_base(query: str) -> str:
            """Search the course knowledge base for relevant Q&A pairs."""
            results = _search(query, num_results=_num_results)
            return _build_context(results)

        return Agent(
            name="RAG Agent",
            instructions=self.instructions,
            model=self.model,
            tools=[search_knowledge_base],
        )

    def rag(self, query: str) -> str:
        import asyncio
        import concurrent.futures

        async def _run():
            with trace("RAGGithubAgentic", metadata={"query": query, "course": self.course}):
                result = await Runner.run(self._agent, query)
                return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: asyncio.run(_run()))
            return future.result()

