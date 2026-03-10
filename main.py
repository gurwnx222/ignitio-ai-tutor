from graph.builder import create_app
from graph.state import graph_state

def main():
    app = create_app()
    result = app.invoke(graph_state(
        user_query="I want to learn LangChain", 
        code_example={}, 
        concept_map={}, 
        explaination={}, 
        roadmap={}, 
        sub_tasks={}, 
        final_response="")
)

if __name__ == "__main__":
    main()
