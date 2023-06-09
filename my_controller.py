from typing import Dict, List, Optional, Any
from langchain import LLMChain
from langchain.chains.base import Chain
from pydantic import BaseModel, Field
from collections import deque
from my_chain import TaskCreationChain, TaskPrioritizationChain, ExecutionChain
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore


def get_next_task(
    task_creation_chain: LLMChain,
    result: Dict,
    task_description: str,
    task_list: List[str],
    objective: str,
) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(
        result=result,
        task_description=task_description,
        incomplete_tasks=incomplete_tasks,
        objective=objective,
    )
    new_tasks = response.split("\n")
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


def prioritize_tasks(
    task_prioritization_chain: LLMChain,
    this_task_id: int,
    task_list: List[Dict],
    objective: str,
) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(
        task_names=task_names, next_task_id=next_task_id, objective=objective
    )
    new_tasks = response.split("\n")
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list


def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata["task"]) for item in sorted_results]


def execute_task(vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5) -> str:
    """Execute a task."""
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execution_chain.run(objective=objective, context=context, task=task)


class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: ExecutionChain = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****CURRENT TASK LIST (TODO) *****" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n***** EXECUTING NEXT TASK : " + "\033[0m\033[0m", end='')
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n***** TASK EXECUTION RESULT : " + "\033[0m\033[0m", end='')
        print(result)

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a todo list of up to 6 steps necessary to achieve objective")
        self.add_task({"task_id": 1, "task_name": first_task})

        # Step 1: Pull the first task
        task = self.task_list.popleft()
        self.print_next_task(task)

        # Step 2: Execute the first task
        result = execute_task(self.vectorstore, self.execution_chain, objective, task["task_name"])
        this_task_id = int(task["task_id"])
        self.print_task_result(result)

        # add tasks to the task list
        new_tasks = result.split("\n")

        for task in new_tasks:
            if task.strip():
                new_task = {"task_id": int(task[0]), "task_name": task[3:]}
                self.add_task(new_task)


        while self.task_list:
            # Step 1: Pull the first task
            task = self.task_list.popleft()
            self.print_next_task(task)

            # Step 2: Execute the task
            result = execute_task(self.vectorstore, self.execution_chain, objective, task["task_name"])
            this_task_id = int(task["task_id"])
            self.print_task_result(result)

            # Step 3: Store the result in Pinecone vectorstore (this will be our context, i.e previously executed tasks)
            result_id = f"result_{task['task_id']}"
            self.vectorstore.add_texts(
                texts=[result],
                metadatas=[{"task": task["task_name"]}],
                ids=[result_id],
            )

        return {}

    @classmethod
    def from_llm(cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(llm, verbose=verbose)
        execution_chain = ExecutionChain.from_llm(llm, verbose=verbose)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
            vectorstore=vectorstore,
            **kwargs,
        )

