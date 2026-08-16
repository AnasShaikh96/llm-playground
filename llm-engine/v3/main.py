from llm_client import chat


def main():
    print("Local LLM")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ")

        if prompt.lower() == "exit":
            break

        try:
            response = chat(prompt)
            print(f"\nLLM: {response}\n")
        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()